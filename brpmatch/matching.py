"""
LSH-based distance matching for BRPMatch cohort matching.

This module performs locality-sensitive hashing (LSH) based matching
between treated and control cohorts.
"""

import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import BucketedRandomProjectionLSH
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


def match(
    features_df: DataFrame,
    feature_space: Literal["euclidean", "mahalanobis"] = "euclidean",
    n_neighbors: int = 5,
    bucket_length: Optional[float] = None,
    num_hash_tables: int = 4,
    num_patients_trigger_rebucket: int = 10000,
    features_col: str = "features",
    treatment_col: str = "treat",
    id_col: str = "person_id",
    exact_match_col: str = "exact_match_id",
) -> DataFrame:
    """
    Perform LSH-based distance matching between treated and control cohorts.

    This function implements a multi-stage matching algorithm:
    1. Hash patients into buckets using Locality-Sensitive Hashing (LSH)
    2. Within each bucket, use k-NN to find similar control patients for each treated patient
    3. Apply greedy 1-to-1 matching to ensure unique matches
    4. Return matched pairs with distances

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
        Number of nearest neighbors to consider per treated patient
    bucket_length : Optional[float]
        Base bucket length for LSH. If None, computed as N^(-1/d)
    num_hash_tables : int
        Number of hash tables for LSH
    num_patients_trigger_rebucket : int
        Threshold for bucket size triggering finer bucketing
    features_col : str
        Name of the feature vector column
    treatment_col : str
        Name of the treatment indicator column (1=treated, 0=control)
    id_col : str
        Patient identifier column name
    exact_match_col : str
        Column for exact matching stratification

    Returns
    -------
    DataFrame
        DataFrame with columns:
        - {id_col}: treated patient ID
        - match_{id_col}: matched control patient ID
        - bucket_num_input_patients: size of bucket where match was found
        - bucket_seconds: time to process bucket
    """
    # Add feature array column for easier manipulation
    persons_features_cohorts = features_df.withColumn(
        "feature_array", vector_to_array(features_col)
    )

    # Logging
    print("Pre-filter")
    print(persons_features_cohorts.groupBy(treatment_col).count().toPandas())

    # Compute whitening transform for mahalanobis feature space
    whitening_matrix = None
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

    # Compute bucket length if not provided: N^(-1/d)
    if bucket_length is None:
        feature_cnt = (
            persons_features_cohorts.limit(1)
            .select(F.size(F.col("feature_array")))
            .collect()[0][0]
        )
        bucket_length = pow(persons_features_cohorts.count(), (-1 / feature_cnt))

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

    print("two")
    print(persons_bucketed.count())

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
    _log_bucket_stats(persons_bucketed, id_col)

    # Schema for return from pandas UDF
    schema_potential_matches_arrays = StructType(
        [
            StructField(id_col, StringType()),
            StructField("match_" + id_col, StringType()),
            StructField("bucket_num_input_patients", IntegerType()),
            StructField("bucket_seconds", DoubleType()),
        ]
    )

    def find_neighbors(group_df):
        """
        Given a group of input rows (those with the same bucket_id),
        returns a dataframe of source-target matches.
        """
        bucket_start_time = time.perf_counter()
        n = n_neighbors

        # Extract the individual cohorts
        needs_matching = group_df.loc[group_df[treatment_col] == 1]
        match_to = group_df.loc[group_df[treatment_col] == 0]

        # Return empty DataFrame if either cohort is empty
        if len(needs_matching) == 0 or len(match_to) == 0:
            return pd.DataFrame(columns=schema_potential_matches_arrays.fieldNames())

        # Update n to size(match_to) if smaller than n
        n = min(len(match_to), n)

        # Always use Euclidean distance (features are pre-transformed for mahalanobis)
        metric = "euclidean"

        # Extract input pandas dataframe columns to useful types
        person_ids_need_matching = list(needs_matching[id_col])
        features_need_matching = np.array(list(needs_matching["feature_array"]))  # 2D array
        person_ids_match_to = list(match_to[id_col])
        features_match_to = np.array(list(match_to["feature_array"]))  # 2D array

        # Find nearest neighbors for the features needing matching
        model = NearestNeighbors(metric=metric).fit(features_match_to)
        neighbors = model.kneighbors(
            features_need_matching, n_neighbors=n, return_distance=True
        )

        # Lists of neighbors and distances for each person_id
        match_person_ids = list(
            map(
                lambda indices: [person_ids_match_to[i] for i in indices],
                neighbors[1].tolist(),
            )
        )
        match_distances = neighbors[0].tolist()

        results = None
        for person_id, match_ids, match_dists in zip(
            person_ids_need_matching, match_person_ids, match_distances
        ):
            # Broadcast person_id
            person_id = [person_id] * len(match_ids)

            # Explode matches to one per row
            person_matches = np.stack([person_id, match_ids, match_dists], axis=1)

            # Iteratively stack results for each person_id
            results = (
                np.vstack([results, person_matches])
                if results is not None
                else person_matches
            )

        # Rank candidate matches in both directions and sort
        match_candidates = pd.DataFrame(
            results, columns=[id_col, "match_" + id_col, "_distance"]
        )
        match_candidates["_distance"] = pd.to_numeric(
            match_candidates["_distance"]
        )
        match_candidates["source_target_rank"] = match_candidates.groupby([id_col])[
            "_distance"
        ].rank()
        match_candidates["target_source_rank"] = match_candidates.groupby(
            "match_" + id_col
        )["_distance"].rank(method="first")
        match_candidates = match_candidates.sort_values(
            ["source_target_rank", "target_source_rank"]
        )

        # Greedy 1-to-1 matching
        i = 0
        while len(match_candidates) > i:
            # Ordering and XOR drop ensures that the ith record always greedily
            # chooses best source=>target match with ties broken by target=>source match
            person_id, match_person_id = match_candidates.iloc[i][
                [id_col, "match_" + id_col]
            ]

            # Drop records matching person_id XOR match_person_id
            match_candidates = match_candidates[
                ~(
                    (match_candidates[id_col] == person_id)
                    ^ (match_candidates["match_" + id_col] == match_person_id)
                )
            ]
            i += 1

        match_candidates["bucket_num_input_patients"] = group_df.shape[0]

        bucket_end_time = time.perf_counter()
        match_candidates["bucket_seconds"] = bucket_end_time - bucket_start_time
        return match_candidates.drop(columns=["source_target_rank", "target_source_rank", "_distance"])

    # Find matches for each bucket
    matches = (
        persons_bucketed.select(
            id_col, treatment_col, "bucket_id", "bucket_id_source", "feature_array"
        )
        .groupBy("bucket_id")
        .applyInPandas(find_neighbors, schema=schema_potential_matches_arrays)
    )

    return matches


def _log_bucket_stats(persons_bucketed: DataFrame, id_col: str) -> None:
    """Log bucket statistics for parameter tuning."""
    bucket_counts = persons_bucketed.groupBy("bucket_id").agg(
        F.count(F.col("bucket_id")).alias("bucket_num_persons")
    )
    print(f"Num buckets: {bucket_counts.count()}")
    print("Bucket stats:")
    stats = bucket_counts.select(
        F.min(F.col("bucket_num_persons")).alias("min"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.05).alias("percentile_5"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.25).alias("percentil_25"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.5).alias("percentile_50"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.75).alias("percentile_75"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.95).alias("percentil_95"),
        F.max(F.col("bucket_num_persons")).alias("max"),
        F.mean(F.col("bucket_num_persons")).alias("mean"),
    ).toPandas()
    print(stats.to_string())
