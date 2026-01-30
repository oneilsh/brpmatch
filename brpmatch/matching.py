"""
LSH-based distance matching for BRPMatch cohort matching.

This module performs locality-sensitive hashing (LSH) based matching
between treated and control cohorts.
"""

import time
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import BucketedRandomProjectionLSH, Normalizer
from pyspark.ml.functions import vector_to_array
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import DataFrame

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from sklearn.neighbors import NearestNeighbors

from .utils import (
    _discover_exact_match_column,
    _discover_id_column,
    _discover_treatment_column,
)


def match(
    features_df: DataFrame,
    feature_space: Literal["euclidean", "mahalanobis"] = "euclidean",
    n_neighbors: int = 5,
    bucket_length_multiplier: float = 1.0,
    num_hash_tables: int = 4,
    num_patients_trigger_rebucket: int = 10000,
    features_col: str = "features",
    verbose: bool = True,
    # New parameters for 1-to-k matching:
    ratio_k: int = 1,
    with_replacement: bool = False,
    reuse_max: Optional[int] = None,
    require_k: bool = True,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Perform LSH-based distance matching between treated and control cohorts.

    This function implements a multi-stage matching algorithm:
    1. Hash patients into buckets using Locality-Sensitive Hashing (LSH)
    2. Within each bucket, use k-NN to find candidate control patients
    3. Apply matching algorithm (greedy 1-to-1 per round for without replacement,
       or independent selection for with replacement)
    4. Return three DataFrames: units (analysis-ready), pairs (detailed matches), and bucket_stats (diagnostics)

    The algorithm uses 4 bucket levels with progressively smaller bucket lengths
    to adaptively handle varying bucket sizes.

    Parameters
    ----------
    features_df : DataFrame
        Output from generate_features() containing 'features' column
    feature_space : Literal["euclidean", "mahalanobis"]
        Feature space for bucketing and matching. "euclidean" uses original
        features with Euclidean distance. "mahalanobis" applies a whitening
        transform so that Euclidean distance in transformed space equals
        Mahalanobis distance in original space.
    n_neighbors : int
        Number of nearest neighbors to consider per treated patient.
        Should be >= ratio_k to ensure enough candidates.
    bucket_length_multiplier : float
        Multiplier for auto-computed bucket length. Default 1.0.
        Values > 1.0 create larger buckets (more candidates, slower).
        Values < 1.0 create smaller buckets (fewer candidates, faster).
    num_hash_tables : int
        Number of hash tables for LSH
    num_patients_trigger_rebucket : int
        Threshold for bucket size triggering finer bucketing
    features_col : str
        Name of the feature vector column
    verbose : bool
        If True, print progress and summary information
    ratio_k : int
        Number of controls to match per treated patient (k in k:1 matching).
        Default is 1 for standard 1:1 matching.
    with_replacement : bool
        If False (default), each control can only be matched to one treated
        patient. Uses round-robin algorithm for fairness (all treated get
        round 1 matches before any get round 2).
        If True, controls can be reused across different treated patients.
    reuse_max : Optional[int]
        Maximum times a control can be reused. Only applies when
        with_replacement=True. None means unlimited reuse.
    require_k : bool
        If True (default), treated patients who cannot get exactly ratio_k
        matches are marked as unmatched. If False, treated patients may
        receive fewer than ratio_k matches.

    Returns
    -------
    Tuple[DataFrame, DataFrame, DataFrame]
        A tuple of three DataFrames:

        units : DataFrame
            One row per patient (treated + control, matched + unmatched).
            Columns:
            - id: Patient ID (same as input {id_col})
            - subclass: Match group identifier (treated ID for matched, None for unmatched)
            - weight: ATT estimation weight (1.0 for treated, 1/k for controls, 0.0 for unmatched)
            - is_treated: Boolean treatment indicator

        pairs : DataFrame
            One row per (treated, control) match pair.
            Columns:
            - {id_col_base}: Treated patient ID
            - match_{id_col_base}: Matched control patient ID
            - match_round: Which round (1=best match, 2=second best, etc.)
            - treated_k: Number of controls matched to this treated
            - control_usage_count: Times this control was matched
            - pair_weight: Analysis weight = 1/(treated_k * control_usage_count)

        bucket_stats : DataFrame
            One row per LSH bucket with processing statistics.
            Columns:
            - bucket_id: Bucket identifier
            - num_patients: Total patients in bucket
            - num_treated: Treated patients in bucket
            - num_control: Control patients in bucket
            - num_matches: Match pairs produced from bucket
            - seconds: Processing time for bucket

        Note: Column names are auto-discovered from features_df using suffix conventions:
        - ID column: ends with __id
        - Treatment column: ends with __treat
        - Exact match grouping: ends with __group

    Notes
    -----
    Weighting: For ATT estimation, pair_weight accounts for both k:1 matching
    (each control contributes 1/k to its treated patient's estimate) and
    replacement (controls matched multiple times are down-weighted).

    Round-robin fairness: When with_replacement=False, the algorithm ensures
    all treated patients receive their 1st-choice match before any receive
    their 2nd-choice. This prevents early patients from "hoarding" the best
    controls.

    References
    ----------
    Inspired by R's MatchIt package:
    https://cran.r-project.org/web/packages/MatchIt/vignettes/matching-methods.html
    """
    # Auto-discover columns from naming convention
    id_col = _discover_id_column(features_df)
    treatment_col = _discover_treatment_column(features_df)
    exact_match_col = _discover_exact_match_column(features_df)

    # Extract base ID name for output columns (e.g., "person_id" from "person_id__id")
    id_col_base = id_col.replace("__id", "")

    # Add feature array column for easier manipulation
    persons_features_cohorts = features_df.withColumn(
        "feature_array", vector_to_array(features_col)
    )

    # Validate 1-to-k matching parameters
    if ratio_k < 1:
        raise ValueError(f"ratio_k must be >= 1, got {ratio_k}")

    if reuse_max is not None:
        if reuse_max < 1:
            raise ValueError(f"reuse_max must be >= 1, got {reuse_max}")
        if not with_replacement:
            print("Warning: reuse_max is ignored when with_replacement=False")

    if n_neighbors < ratio_k:
        print(
            f"Warning: n_neighbors ({n_neighbors}) < ratio_k ({ratio_k}). "
            f"Consider increasing n_neighbors to ensure enough candidates."
        )

    # Logging
    if verbose:
        print("Pre-filter")
        print(persons_features_cohorts.groupBy(treatment_col).count().toPandas())

    # Compute whitening transform for mahalanobis feature space
    whitening_matrix = None
    whitening_info = None  # Store for summary (original_dims, retained_dims)
    if feature_space == "mahalanobis":
        # Convert ML vectors to MLlib vectors for RowMatrix
        vec_converter_udf = F.udf(lambda v: Vectors.dense(v.toArray()), VectorUDT())
        features_for_cov = persons_features_cohorts.withColumn(
            "features_mllib", vec_converter_udf(features_col)
        ).select("features_mllib")

        # Compute covariance matrix using distributed RowMatrix
        # Note: computeCovariance() handles centering internally
        cov_matrix = RowMatrix(features_for_cov.rdd.map(lambda row: row[0])).computeCovariance().toArray()

        # Eigendecomposition of symmetric covariance matrix
        # eigh returns eigenvalues in ascending order, eigenvectors as columns
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Filter near-zero eigenvalues for numerical stability
        # These represent directions with no variance (collinearity)
        eigenvalue_threshold = 1e-10
        valid_mask = eigenvalues > eigenvalue_threshold
        eigenvalues_filtered = eigenvalues[valid_mask]
        eigenvectors_filtered = eigenvectors[:, valid_mask]

        # Compute whitening matrix: W = Λ^(-1/2) @ V^T
        # This transforms features so Euclidean distance = Mahalanobis distance
        inv_sqrt_eigenvalues = np.diag(1.0 / np.sqrt(eigenvalues_filtered))
        whitening_matrix = inv_sqrt_eigenvalues @ eigenvectors_filtered.T

        # Store whitening info for summary
        whitening_info = (cov_matrix.shape[0], whitening_matrix.shape[0])

        if verbose:
            print(f"Whitening transform: {cov_matrix.shape[0]} features -> {whitening_matrix.shape[0]} components")
            print(f"Filtered {np.sum(~valid_mask)} near-zero eigenvalues")

        # Broadcast whitening matrix to all workers
        spark_context = persons_features_cohorts.sparkSession.sparkContext
        whitening_matrix_bc = spark_context.broadcast(whitening_matrix)

        # UDF to apply whitening transform: z = W @ x
        from pyspark.ml.linalg import Vectors as MLVectors, VectorUDT as MLVectorUDT

        @F.udf(MLVectorUDT())
        def apply_whitening(feature_vector):
            x = np.array(feature_vector.toArray())
            z = whitening_matrix_bc.value @ x
            return MLVectors.dense(z.tolist())

        # Transform features and update feature_array
        persons_features_cohorts = persons_features_cohorts.withColumn(
            features_col, apply_whitening(F.col(features_col))
        ).withColumn(
            "feature_array", vector_to_array(features_col)
        )

    # L2 normalize feature vectors to unit length for better LSH bucket sizing
    # Use Spark's built-in Normalizer for efficiency at scale
    normalizer = Normalizer(inputCol=features_col, outputCol="_normalized_features", p=2.0)
    persons_features_cohorts = (
        normalizer.transform(persons_features_cohorts)
        .drop(features_col)
        .withColumnRenamed("_normalized_features", features_col)
        .withColumn("feature_array", vector_to_array(features_col))
    )

    # Compute bucket length: N^(-1/d) * multiplier
    if persons_features_cohorts.count() == 0:
        raise ValueError("Input DataFrame is empty - no patients to match")

    feature_cnt = (
        persons_features_cohorts.limit(1)
        .select(F.size(F.col("feature_array")))
        .collect()[0][0]
    )
    if feature_cnt == 0:
        raise ValueError("Feature array is empty - no features to match on")
    # https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/feature/BucketedRandomProjectionLSH.html
    # "If input vectors are normalized, 1-10 times of pow(numRecords, -1/inputDim) would be a reasonable value"
    bucket_length = 5 * pow(persons_features_cohorts.count(), (-1 / feature_cnt)) * bucket_length_multiplier

    # Create 4 levels of LSH buckets with progressively smaller bucket lengths
    bucket_stages = [
        BucketedRandomProjectionLSH(
            inputCol=features_col,
            outputCol="bucket_hashes1",
            seed=42,
            numHashTables=num_hash_tables,
            bucketLength=bucket_length,
        ),
        BucketedRandomProjectionLSH(
            inputCol=features_col,
            outputCol="bucket_hashes2",
            seed=42,
            numHashTables=num_hash_tables,
            bucketLength=bucket_length / 4,
        ),
        BucketedRandomProjectionLSH(
            inputCol=features_col,
            outputCol="bucket_hashes3",
            seed=42,
            numHashTables=num_hash_tables,
            bucketLength=bucket_length / 16,
        ),
        BucketedRandomProjectionLSH(
            inputCol=features_col,
            outputCol="bucket_hashes4",
            seed=42,
            numHashTables=num_hash_tables,
            bucketLength=bucket_length / 64,
        ),
    ]

    # Apply LSH and create bucket IDs at all levels
    persons_bucketed = (
        Pipeline(stages=bucket_stages)
        .fit(persons_features_cohorts)
        .transform(persons_features_cohorts)
        .withColumn(
            "bucket_id1",
            F.concat_ws(
                ":",
                F.col(exact_match_col),
                F.lit("b1"),
                F.xxhash64(F.col("bucket_hashes1")),
            ),
        )
        .withColumn(
            "bucket_id2",
            F.concat_ws(
                ":",
                F.col(exact_match_col),
                F.lit("b2"),
                F.xxhash64(F.col("bucket_hashes2")),
            ),
        )
        .withColumn(
            "bucket_id3",
            F.concat_ws(
                ":",
                F.col(exact_match_col),
                F.lit("b3"),
                F.xxhash64(F.col("bucket_hashes3")),
            ),
        )
        .withColumn(
            "bucket_id4",
            F.concat_ws(
                ":",
                F.col(exact_match_col),
                F.lit("b4"),
                F.xxhash64(F.col("bucket_hashes4")),
            ),
        )
    )

    # Count patients in each bucket at each level
    bucket_counts1 = persons_bucketed.groupBy("bucket_id1").agg(
        F.countDistinct(id_col).alias("num_patients1_raw")
    )
    bucket_counts2 = persons_bucketed.groupBy("bucket_id2").agg(
        F.countDistinct(id_col).alias("num_patients2_raw")
    )
    bucket_counts3 = persons_bucketed.groupBy("bucket_id3").agg(
        F.countDistinct(id_col).alias("num_patients3_raw")
    )
    bucket_counts4 = persons_bucketed.groupBy("bucket_id4").agg(
        F.countDistinct(id_col).alias("num_patients4_raw")
    )

    # Join counts back to main dataframe
    persons_bucketed = (
        persons_bucketed.join(bucket_counts1, "bucket_id1", how="full")
        .join(bucket_counts2, "bucket_id2", how="full")
        .join(bucket_counts3, "bucket_id3", how="full")
        .join(bucket_counts4, "bucket_id4", how="full")
    )


    # Select finest bucket level that keeps bucket size below threshold
    persons_bucketed = (
        persons_bucketed.withColumn(
            "bucket_id",
            F.when(
                F.col("num_patients1_raw") < num_patients_trigger_rebucket,
                F.col("bucket_id1"),
            )
            .when(
                F.col("num_patients2_raw") < num_patients_trigger_rebucket,
                F.col("bucket_id2"),
            )
            .when(
                F.col("num_patients3_raw") < num_patients_trigger_rebucket,
                F.col("bucket_id3"),
            )
            .when(
                F.col("num_patients4_raw") < num_patients_trigger_rebucket,
                F.col("bucket_id4"),
            )
            .otherwise(F.lit(None)),
        )
        .withColumn(
            "bucket_id_source",
            F.when(
                F.col("num_patients1_raw") < num_patients_trigger_rebucket, "bucket_1"
            )
            .when(
                F.col("num_patients2_raw") < num_patients_trigger_rebucket, "bucket_2"
            )
            .when(
                F.col("num_patients3_raw") < num_patients_trigger_rebucket, "bucket_3"
            )
            .when(
                F.col("num_patients4_raw") < num_patients_trigger_rebucket, "bucket_4"
            )
            .otherwise(F.lit(None)),
        )
        .drop(
            "num_patients1_raw",
            "num_patients2_raw",
            "num_patients3_raw",
            "num_patients4_raw",
            "bucket_id1",
            "bucket_id2",
            "bucket_id3",
            "bucket_id4",
        )
    )

    # Identify buckets which have patients from both cohorts (viable buckets)
    viable_buckets = (
        persons_bucketed.groupBy("bucket_id")
        .agg(F.countDistinct(treatment_col).alias("types"))
        .filter(F.col("types") == 2)
        .select("bucket_id")
        .distinct()
    )

    # Drop non-viable buckets without patients from both cohorts
    persons_bucketed = persons_bucketed.join(viable_buckets, "bucket_id")

    # Log bucket statistics for parameter tuning
    num_buckets = _log_bucket_stats(persons_bucketed, id_col, verbose=verbose)

    # Schema for return from pandas UDF
    # Use base name without __id suffix for cleaner output
    schema_potential_matches_arrays = StructType(
        [
            StructField(id_col_base, StringType()),
            StructField("match_" + id_col_base, StringType()),
            StructField("match_round", IntegerType()),
            StructField("treated_k", IntegerType()),
            StructField("control_usage_count", IntegerType()),
            StructField("pair_weight", DoubleType()),
            StructField("bucket_id", StringType()),
            StructField("bucket_num_patients", IntegerType()),
            StructField("bucket_num_treated", IntegerType()),
            StructField("bucket_num_control", IntegerType()),
            StructField("bucket_seconds", DoubleType()),
        ]
    )

    def find_neighbors(group_df):
        """
        Given a group of input rows (those with the same bucket_id),
        returns a dataframe of source-target matches.

        Supports 1-to-k matching with or without replacement.
        """
        bucket_start_time = time.perf_counter()
        bucket_size = group_df.shape[0]

        # Extract the individual cohorts
        needs_matching = group_df.loc[group_df[treatment_col] == 1]
        match_to = group_df.loc[group_df[treatment_col] == 0]

        # Capture bucket-level counts for stats
        num_treated_in_bucket = len(needs_matching)
        num_control_in_bucket = len(match_to)
        # Get bucket_id from first row (all rows in group have same bucket_id)
        bucket_id_value = group_df["bucket_id"].iloc[0]

        # Return empty DataFrame if either cohort is empty
        if len(needs_matching) == 0 or len(match_to) == 0:
            return pd.DataFrame(columns=schema_potential_matches_arrays.fieldNames())

        # Determine how many neighbors to fetch from k-NN
        # Need enough candidates for ratio_k rounds
        n_fetch = min(len(match_to), max(n_neighbors, ratio_k))

        # Extract arrays for k-NN
        # Note: id_col has __id suffix, id_col_base doesn't
        # We use id_col for reading from group_df (which has original features_df columns)
        # But write using id_col_base in output
        person_ids_need_matching = list(needs_matching[id_col])
        features_need_matching = np.array(list(needs_matching["feature_array"]))
        person_ids_match_to = list(match_to[id_col])
        features_match_to = np.array(list(match_to["feature_array"]))

        # Find nearest neighbors
        model = NearestNeighbors(metric="euclidean").fit(features_match_to)
        distances, indices = model.kneighbors(
            features_need_matching, n_neighbors=n_fetch, return_distance=True
        )

        # Build candidate DataFrame: one row per (treated, control, distance)
        # Use id_col_base for output column names
        candidates_list = []
        for i, treated_id in enumerate(person_ids_need_matching):
            for j in range(n_fetch):
                control_idx = indices[i, j]
                candidates_list.append({
                    id_col_base: treated_id,
                    "match_" + id_col_base: person_ids_match_to[control_idx],
                    "_distance": distances[i, j],
                })

        if not candidates_list:
            return pd.DataFrame(columns=schema_potential_matches_arrays.fieldNames())

        match_candidates = pd.DataFrame(candidates_list)

        # Rank candidates: for each treated, rank controls by distance
        match_candidates["_treated_rank"] = match_candidates.groupby(id_col_base)["_distance"].rank(method="first")
        # For each control, rank treated by distance (for tie-breaking in without-replacement)
        match_candidates["_control_rank"] = match_candidates.groupby("match_" + id_col_base)["_distance"].rank(method="first")

        # ===== MATCHING LOGIC =====

        if with_replacement:
            # ----- WITH REPLACEMENT -----
            # Each treated independently selects their k best controls
            # Controls can be reused (subject to reuse_max if set)

            all_matches = []
            control_usage = {}  # control_id -> usage count

            # Process each treated patient
            for treated_id in person_ids_need_matching:
                treated_candidates = match_candidates[
                    match_candidates[id_col_base] == treated_id
                ].sort_values("_treated_rank")

                matches_for_treated = []
                for _, row in treated_candidates.iterrows():
                    control_id = row["match_" + id_col_base]

                    # Check reuse_max constraint
                    current_usage = control_usage.get(control_id, 0)
                    if reuse_max is not None and current_usage >= reuse_max:
                        continue

                    # Record match
                    matches_for_treated.append({
                        id_col_base: treated_id,
                        "match_" + id_col_base: control_id,
                        "match_round": len(matches_for_treated) + 1,
                        "_distance": row["_distance"],
                    })
                    control_usage[control_id] = current_usage + 1

                    if len(matches_for_treated) >= ratio_k:
                        break

                all_matches.extend(matches_for_treated)

        else:
            # ----- WITHOUT REPLACEMENT (ROUND-ROBIN) -----
            # Round 1: all treated get best available match
            # Round 2: all treated get second best from remaining
            # ... ensures fairness

            all_matches = []
            available_controls = set(person_ids_match_to)
            matched_treated = set()  # Track treated who have been fully matched or failed

            for round_num in range(1, ratio_k + 1):
                if not available_controls:
                    break

                # Filter candidates to available controls
                round_candidates = match_candidates[
                    match_candidates["match_" + id_col_base].isin(available_controls) &
                    ~match_candidates[id_col_base].isin(matched_treated)
                ].copy()

                if round_candidates.empty:
                    break

                # Re-rank within available controls for this round
                round_candidates["_round_treated_rank"] = round_candidates.groupby(id_col_base)["_distance"].rank(method="first")
                round_candidates["_round_control_rank"] = round_candidates.groupby("match_" + id_col_base)["_distance"].rank(method="first")

                # Sort by treated's preference, then control's preference for tie-breaking
                round_candidates = round_candidates.sort_values(
                    ["_round_treated_rank", "_round_control_rank"]
                )

                # Greedy 1-to-1 matching for this round
                round_matches = []
                used_treated_this_round = set()
                used_controls_this_round = set()

                for _, row in round_candidates.iterrows():
                    treated_id = row[id_col_base]
                    control_id = row["match_" + id_col_base]

                    if treated_id in used_treated_this_round:
                        continue
                    if control_id in used_controls_this_round:
                        continue

                    round_matches.append({
                        id_col_base: treated_id,
                        "match_" + id_col_base: control_id,
                        "match_round": round_num,
                        "_distance": row["_distance"],
                    })
                    used_treated_this_round.add(treated_id)
                    used_controls_this_round.add(control_id)

                all_matches.extend(round_matches)

                # Remove used controls from pool
                available_controls -= used_controls_this_round

                # Track treated who didn't get a match this round (they won't in future rounds either)
                treated_without_match = set(person_ids_need_matching) - used_treated_this_round - matched_treated
                # If a treated patient couldn't get a match this round, they're done
                for tid in treated_without_match:
                    # Check if they had any candidates
                    had_candidates = round_candidates[round_candidates[id_col_base] == tid].shape[0] > 0
                    if not had_candidates:
                        matched_treated.add(tid)

        # ===== POST-PROCESSING =====

        if not all_matches:
            return pd.DataFrame(columns=schema_potential_matches_arrays.fieldNames())

        result_df = pd.DataFrame(all_matches)

        # Compute treated_k: how many matches each treated patient got
        result_df["treated_k"] = result_df.groupby(id_col_base)[id_col_base].transform("count")

        # Compute control_usage_count: how many times each control was used
        result_df["control_usage_count"] = result_df.groupby("match_" + id_col_base)["match_" + id_col_base].transform("count")

        # Compute pair_weight for downstream analysis
        result_df["pair_weight"] = 1.0 / (result_df["treated_k"] * result_df["control_usage_count"])

        # Handle require_k: filter out treated patients with fewer than k matches
        if require_k and ratio_k > 1:
            result_df = result_df[result_df["treated_k"] >= ratio_k]
            # Recompute control_usage_count after filtering
            if len(result_df) > 0:
                result_df["control_usage_count"] = result_df.groupby("match_" + id_col_base)["match_" + id_col_base].transform("count")
                result_df["pair_weight"] = 1.0 / (result_df["treated_k"] * result_df["control_usage_count"])

        # Add bucket metadata
        result_df["bucket_id"] = bucket_id_value
        result_df["bucket_num_patients"] = bucket_size
        result_df["bucket_num_treated"] = num_treated_in_bucket
        result_df["bucket_num_control"] = num_control_in_bucket

        bucket_end_time = time.perf_counter()
        result_df["bucket_seconds"] = bucket_end_time - bucket_start_time

        # Select and order output columns
        output_cols = [
            id_col_base,
            "match_" + id_col_base,
            "match_round",
            "treated_k",
            "control_usage_count",
            "pair_weight",
            "bucket_id",
            "bucket_num_patients",
            "bucket_num_treated",
            "bucket_num_control",
            "bucket_seconds",
        ]

        return result_df[output_cols]

    # Find matches for each bucket
    matches = (
        persons_bucketed.select(
            id_col, treatment_col, "bucket_id", "bucket_id_source", "feature_array"
        )
        .groupBy("bucket_id")
        .applyInPandas(find_neighbors, schema=schema_potential_matches_arrays)
    )

    # Print match summary
    if verbose:
        # Count treated, control, and matched
        cohort_counts = persons_features_cohorts.groupBy(treatment_col).count().toPandas()
        n_treated = int(cohort_counts[cohort_counts[treatment_col] == 1]["count"].iloc[0]) if len(cohort_counts[cohort_counts[treatment_col] == 1]) > 0 else 0
        n_control = int(cohort_counts[cohort_counts[treatment_col] == 0]["count"].iloc[0]) if len(cohort_counts[cohort_counts[treatment_col] == 0]) > 0 else 0

        # Count pairs, unique treated, and unique controls
        n_matched_pairs = matches.count()
        n_matched_treated = matches.select(id_col_base).distinct().count()
        n_unique_controls_used = matches.select("match_" + id_col_base).distinct().count()

        _print_match_summary(
            feature_space=feature_space,
            n_treated=n_treated,
            n_control=n_control,
            n_matched_pairs=n_matched_pairs,
            n_matched_treated=n_matched_treated,
            bucket_length=bucket_length,
            num_hash_tables=num_hash_tables,
            whitening_info=whitening_info,
            num_buckets=num_buckets,
            ratio_k=ratio_k,
            with_replacement=with_replacement,
            reuse_max=reuse_max,
            n_unique_controls_used=n_unique_controls_used,
        )

    # Build bucket_stats by aggregating from matches (one row per bucket)
    bucket_stats = matches.groupBy("bucket_id").agg(
        F.first("bucket_num_patients").alias("num_patients"),
        F.first("bucket_num_treated").alias("num_treated"),
        F.first("bucket_num_control").alias("num_control"),
        F.count("*").alias("num_matches"),
        F.first("bucket_seconds").alias("seconds"),
    ).select(
        "bucket_id",
        "num_patients",
        "num_treated",
        "num_control",
        "num_matches",
        "seconds"
    )

    # Build pairs DataFrame (drop bucket columns - they're in bucket_stats)
    pairs = matches.drop(
        "bucket_id", "bucket_num_patients", "bucket_num_treated",
        "bucket_num_control", "bucket_seconds"
    )

    # Build units DataFrame
    units = _build_units_df(features_df, matches, id_col, id_col_base, treatment_col)

    return units, pairs, bucket_stats


def _build_units_df(
    features_df: DataFrame,
    matches: DataFrame,
    id_col: str,
    id_col_base: str,
    treatment_col: str,
) -> DataFrame:
    """
    Build the units DataFrame with one row per patient.

    Parameters
    ----------
    features_df : DataFrame
        Input features DataFrame
    matches : DataFrame
        Match pairs DataFrame
    id_col : str
        Full ID column name (e.g., "person_id__id")
    id_col_base : str
        Base ID column name (e.g., "person_id")
    treatment_col : str
        Treatment column name (e.g., "treat__treat")

    Returns
    -------
    DataFrame
        Units DataFrame with columns: id, subclass, weight, is_treated
    """
    match_id_col = f"match_{id_col_base}"

    # Extract unique patients from features_df
    all_patients = features_df.select(
        F.col(id_col).alias("id"),
        F.col(treatment_col).cast("boolean").alias("is_treated")
    )

    # Compute treated weights (always 1.0 for matched treated)
    # Subclass = treated ID
    treated_weights = matches.select(
        F.col(id_col_base).alias("id"),
        F.col(id_col_base).alias("subclass"),
        F.lit(1.0).alias("weight"),
    ).distinct()

    # Compute control weights
    # Weight = sum of (1/treated_k) for each match the control appears in
    # Subclass = first treated ID they're matched to (arbitrary but consistent)
    control_weights = matches.withColumn(
        "_weight_contribution", 1.0 / F.col("treated_k")
    ).groupBy(match_id_col).agg(
        F.sum("_weight_contribution").alias("weight"),
        F.first(id_col_base).alias("subclass"),  # Use first treated as subclass
    ).select(
        F.col(match_id_col).alias("id"),
        F.col("subclass"),
        F.col("weight"),
    )

    # Combine treated and control weights
    matched_weights = treated_weights.unionByName(control_weights)

    # Join to all patients
    units = all_patients.join(
        matched_weights,
        on="id",
        how="left"
    )

    # Fill unmatched with defaults
    units = units.withColumn(
        "weight", F.coalesce(F.col("weight"), F.lit(0.0))
    ).withColumn(
        "subclass", F.coalesce(F.col("subclass"), F.lit(None))
    )

    # Sort by subclass descending (matched units first, unmatched at end with null subclass)
    return units.select("id", "subclass", "weight", "is_treated").orderBy(F.desc("subclass"))


def _print_match_summary(
    feature_space: str,
    n_treated: int,
    n_control: int,
    n_matched_pairs: int,
    n_matched_treated: int,
    bucket_length: float,
    num_hash_tables: int,
    whitening_info: Optional[Tuple[int, int]] = None,
    num_buckets: int = 0,
    warn_threshold: float = 0.5,
    # New parameters:
    ratio_k: int = 1,
    with_replacement: bool = False,
    reuse_max: Optional[int] = None,
    n_unique_controls_used: int = 0,
) -> None:
    """Print MatchIt-style summary of matching results."""

    # Header with ratio
    print(f"\nBRPMatch: 1:{ratio_k} nearest neighbor matching via LSH")

    # Replacement info
    if with_replacement:
        reuse_str = f", reuse_max={reuse_max}" if reuse_max else ", unlimited reuse"
        print(f" - replacement: with replacement{reuse_str}")
    else:
        print(f" - replacement: without replacement (round-robin)")

    # Feature space info
    if feature_space == "mahalanobis" and whitening_info:
        orig, retained = whitening_info
        print(f" - feature space: mahalanobis (whitened to {retained} components from {orig} features)")
    else:
        print(f" - feature space: {feature_space}")

    # LSH parameters
    print(f" - LSH bucket_length: {bucket_length:.4f}, num_hash_tables: {num_hash_tables}")
    print(f" - num_buckets: {num_buckets}")

    # Sample sizes
    print(f" - sample sizes:")
    print(f"     treated: {n_treated}")
    print(f"     control: {n_control}")

    # Match results
    match_rate = (n_matched_treated / n_treated) if n_treated > 0 else 0
    avg_controls = (n_matched_pairs / n_matched_treated) if n_matched_treated > 0 else 0

    print(f" - matched: {n_matched_pairs} pairs across {n_matched_treated} treated ({match_rate*100:.1f}% of treated)")
    if ratio_k > 1:
        print(f"     mean controls per treated: {avg_controls:.2f}")
    print(f"     unique controls used: {n_unique_controls_used} (of {n_control} available)")

    # Warn if match rate is low
    if match_rate < warn_threshold:
        print(
            f"Warning: Low match rate: only {match_rate*100:.1f}% of treated units were matched. "
            f"Consider adjusting bucket_length_multiplier or num_hash_tables parameters."
        )


def _log_bucket_stats(persons_bucketed: DataFrame, id_col: str, verbose: bool = True) -> int:
    """Log bucket statistics for parameter tuning. Returns bucket count."""
    bucket_counts = persons_bucketed.groupBy("bucket_id").agg(
        F.count(F.col("bucket_id")).alias("bucket_num_persons")
    )
    num_buckets = bucket_counts.count()

    if verbose:
        print(f"Num buckets: {num_buckets}")
        print("Bucket stats:")
        stats = bucket_counts.select(
            F.min(F.col("bucket_num_persons")).alias("min"),
            F.percentile_approx(F.col("bucket_num_persons"), 0.05).alias("percentile_5"),
            F.percentile_approx(F.col("bucket_num_persons"), 0.25).alias("percentile_25"),
            F.percentile_approx(F.col("bucket_num_persons"), 0.5).alias("percentile_50"),
            F.percentile_approx(F.col("bucket_num_persons"), 0.75).alias("percentile_75"),
            F.percentile_approx(F.col("bucket_num_persons"), 0.95).alias("percentile_95"),
            F.max(F.col("bucket_num_persons")).alias("max"),
            F.mean(F.col("bucket_num_persons")).alias("mean"),
        ).toPandas()
        print(stats.to_string())

    return num_buckets
