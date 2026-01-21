# Detailed Implementation Plan: 1-to-k Matching

This document provides line-by-line implementation guidance for extending `brpmatch/matching.py` to support 1-to-k matching. An implementing agent should be able to follow this plan to completion without additional context.

## File: `brpmatch/matching.py`

---

## 1. Function Signature Changes (lines 31-42)

### Current:
```python
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
    verbose: bool = True,
) -> DataFrame:
```

### New:
```python
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
    verbose: bool = True,
    # New parameters for 1-to-k matching:
    ratio_k: int = 1,
    with_replacement: bool = False,
    reuse_max: Optional[int] = None,
    require_k: bool = True,
) -> DataFrame:
```

---

## 2. Parameter Validation (insert after line 96, before "# Logging")

Insert the following validation block:

```python
    # Validate 1-to-k matching parameters
    if ratio_k < 1:
        raise ValueError(f"ratio_k must be >= 1, got {ratio_k}")

    if reuse_max is not None:
        if reuse_max < 1:
            raise ValueError(f"reuse_max must be >= 1, got {reuse_max}")
        if not with_replacement:
            warnings.warn(
                "reuse_max is ignored when with_replacement=False",
                UserWarning
            )

    if n_neighbors < ratio_k:
        warnings.warn(
            f"n_neighbors ({n_neighbors}) < ratio_k ({ratio_k}). "
            f"Consider increasing n_neighbors to ensure enough candidates.",
            UserWarning
        )
```

---

## 3. Output Schema Changes (lines 344-352)

### Current:
```python
    schema_potential_matches_arrays = StructType(
        [
            StructField(id_col, StringType()),
            StructField("match_" + id_col, StringType()),
            StructField("bucket_num_input_patients", IntegerType()),
            StructField("bucket_seconds", DoubleType()),
        ]
    )
```

### New:
```python
    schema_potential_matches_arrays = StructType(
        [
            StructField(id_col, StringType()),
            StructField("match_" + id_col, StringType()),
            StructField("match_round", IntegerType()),
            StructField("treated_k", IntegerType()),
            StructField("control_usage_count", IntegerType()),
            StructField("pair_weight", DoubleType()),
            StructField("bucket_num_input_patients", IntegerType()),
            StructField("bucket_seconds", DoubleType()),
        ]
    )
```

---

## 4. Rewrite `find_neighbors` Function (lines 354-453)

This is the core change. Replace the entire `find_neighbors` function with the following implementation.

### Key Design Decisions:
- **Without replacement**: Use round-robin approach (all treated get round 1 before round 2)
- **With replacement**: Each treated independently gets k nearest (with optional reuse_max)
- Both paths share the same candidate generation (k-NN) but differ in selection logic

```python
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

        # Return empty DataFrame if either cohort is empty
        if len(needs_matching) == 0 or len(match_to) == 0:
            return pd.DataFrame(columns=schema_potential_matches_arrays.fieldNames())

        # Determine how many neighbors to fetch from k-NN
        # Need enough candidates for ratio_k rounds
        n_fetch = min(len(match_to), max(n_neighbors, ratio_k))

        # Extract arrays for k-NN
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
        candidates_list = []
        for i, treated_id in enumerate(person_ids_need_matching):
            for j in range(n_fetch):
                control_idx = indices[i, j]
                candidates_list.append({
                    id_col: treated_id,
                    "match_" + id_col: person_ids_match_to[control_idx],
                    "_distance": distances[i, j],
                })

        if not candidates_list:
            return pd.DataFrame(columns=schema_potential_matches_arrays.fieldNames())

        match_candidates = pd.DataFrame(candidates_list)

        # Rank candidates: for each treated, rank controls by distance
        match_candidates["_treated_rank"] = match_candidates.groupby(id_col)["_distance"].rank(method="first")
        # For each control, rank treated by distance (for tie-breaking in without-replacement)
        match_candidates["_control_rank"] = match_candidates.groupby("match_" + id_col)["_distance"].rank(method="first")

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
                    match_candidates[id_col] == treated_id
                ].sort_values("_treated_rank")

                matches_for_treated = []
                for _, row in treated_candidates.iterrows():
                    control_id = row["match_" + id_col]

                    # Check reuse_max constraint
                    current_usage = control_usage.get(control_id, 0)
                    if reuse_max is not None and current_usage >= reuse_max:
                        continue

                    # Record match
                    matches_for_treated.append({
                        id_col: treated_id,
                        "match_" + id_col: control_id,
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
                    match_candidates["match_" + id_col].isin(available_controls) &
                    ~match_candidates[id_col].isin(matched_treated)
                ].copy()

                if round_candidates.empty:
                    break

                # Re-rank within available controls for this round
                round_candidates["_round_treated_rank"] = round_candidates.groupby(id_col)["_distance"].rank(method="first")
                round_candidates["_round_control_rank"] = round_candidates.groupby("match_" + id_col)["_distance"].rank(method="first")

                # Sort by treated's preference, then control's preference for tie-breaking
                round_candidates = round_candidates.sort_values(
                    ["_round_treated_rank", "_round_control_rank"]
                )

                # Greedy 1-to-1 matching for this round
                round_matches = []
                used_treated_this_round = set()
                used_controls_this_round = set()

                for _, row in round_candidates.iterrows():
                    treated_id = row[id_col]
                    control_id = row["match_" + id_col]

                    if treated_id in used_treated_this_round:
                        continue
                    if control_id in used_controls_this_round:
                        continue

                    round_matches.append({
                        id_col: treated_id,
                        "match_" + id_col: control_id,
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
                    had_candidates = round_candidates[round_candidates[id_col] == tid].shape[0] > 0
                    if not had_candidates:
                        matched_treated.add(tid)

        # ===== POST-PROCESSING =====

        if not all_matches:
            return pd.DataFrame(columns=schema_potential_matches_arrays.fieldNames())

        result_df = pd.DataFrame(all_matches)

        # Compute treated_k: how many matches each treated patient got
        result_df["treated_k"] = result_df.groupby(id_col)[id_col].transform("count")

        # Compute control_usage_count: how many times each control was used
        result_df["control_usage_count"] = result_df.groupby("match_" + id_col)["match_" + id_col].transform("count")

        # Compute pair_weight for downstream analysis
        result_df["pair_weight"] = 1.0 / (result_df["treated_k"] * result_df["control_usage_count"])

        # Handle require_k: filter out treated patients with fewer than k matches
        if require_k and ratio_k > 1:
            result_df = result_df[result_df["treated_k"] >= ratio_k]
            # Recompute control_usage_count after filtering
            if len(result_df) > 0:
                result_df["control_usage_count"] = result_df.groupby("match_" + id_col)["match_" + id_col].transform("count")
                result_df["pair_weight"] = 1.0 / (result_df["treated_k"] * result_df["control_usage_count"])

        # Add bucket metadata
        result_df["bucket_num_input_patients"] = bucket_size

        bucket_end_time = time.perf_counter()
        result_df["bucket_seconds"] = bucket_end_time - bucket_start_time

        # Select and order output columns
        output_cols = [
            id_col,
            "match_" + id_col,
            "match_round",
            "treated_k",
            "control_usage_count",
            "pair_weight",
            "bucket_num_input_patients",
            "bucket_seconds",
        ]

        return result_df[output_cols]
```

---

## 5. Update `_print_match_summary` Function (lines 486-526)

### Changes needed:

1. Add new parameters to function signature
2. Update header line to show ratio
3. Add replacement info
4. Update statistics to show pairs vs unique treated

### New function signature (line 486-496):
```python
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
```

### New function body:
```python
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
        warnings.warn(
            f"Low match rate: only {match_rate*100:.1f}% of treated units were matched. "
            f"Consider adjusting bucket_length or num_hash_tables parameters.",
            UserWarning
        )
```

---

## 6. Update Call to `_print_match_summary` (lines 464-481)

### Current:
```python
    if verbose:
        # Count treated, control, and matched
        cohort_counts = persons_features_cohorts.groupBy(treatment_col).count().toPandas()
        n_treated = int(cohort_counts[cohort_counts[treatment_col] == 1]["count"].iloc[0]) if len(cohort_counts[cohort_counts[treatment_col] == 1]) > 0 else 0
        n_control = int(cohort_counts[cohort_counts[treatment_col] == 0]["count"].iloc[0]) if len(cohort_counts[cohort_counts[treatment_col] == 0]) > 0 else 0
        n_matched = matches.count()

        _print_match_summary(
            feature_space=feature_space,
            n_treated=n_treated,
            n_control=n_control,
            n_matched=n_matched,
            bucket_length=bucket_length,
            num_hash_tables=num_hash_tables,
            whitening_info=whitening_info,
            num_buckets=num_buckets,
        )
```

### New:
```python
    if verbose:
        # Count treated, control, and matched
        cohort_counts = persons_features_cohorts.groupBy(treatment_col).count().toPandas()
        n_treated = int(cohort_counts[cohort_counts[treatment_col] == 1]["count"].iloc[0]) if len(cohort_counts[cohort_counts[treatment_col] == 1]) > 0 else 0
        n_control = int(cohort_counts[cohort_counts[treatment_col] == 0]["count"].iloc[0]) if len(cohort_counts[cohort_counts[treatment_col] == 0]) > 0 else 0

        # Count pairs, unique treated, and unique controls
        n_matched_pairs = matches.count()
        n_matched_treated = matches.select(id_col).distinct().count()
        n_unique_controls_used = matches.select("match_" + id_col).distinct().count()

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
```

---

## 7. Update Docstring (lines 44-92)

Replace the docstring with:

```python
    """
    Perform LSH-based distance matching between treated and control cohorts.

    This function implements a multi-stage matching algorithm:
    1. Hash patients into buckets using Locality-Sensitive Hashing (LSH)
    2. Within each bucket, use k-NN to find candidate control patients
    3. Apply matching algorithm (greedy 1-to-1 per round for without replacement,
       or independent selection for with replacement)
    4. Return matched pairs with distances and weights

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
        matches are excluded from output. If False, treated patients may
        receive fewer than ratio_k matches.

    Returns
    -------
    DataFrame
        DataFrame with columns:
        - {id_col}: treated patient ID
        - match_{id_col}: matched control patient ID
        - match_round: which round this match came from (1=best, 2=second best, etc.)
        - treated_k: total matches for this treated patient
        - control_usage_count: times this control was matched (always 1 if without replacement)
        - pair_weight: weight for analysis = 1/(treated_k * control_usage_count)
        - bucket_num_input_patients: size of bucket where match was found
        - bucket_seconds: time to process bucket

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
```

---

## 8. Summary of All Changes

| Section | Lines | Change Type |
|---------|-------|-------------|
| Function signature | 31-42 | Add 4 new parameters |
| Parameter validation | after 96 | Insert new block |
| Output schema | 344-352 | Add 4 new fields |
| `find_neighbors` | 354-453 | Complete rewrite |
| `_print_match_summary` | 486-526 | Update signature and body |
| Summary call | 464-481 | Update to compute and pass new stats |
| Docstring | 44-92 | Update with new parameters and return fields |

---

## 9. Testing Checklist

After implementation, verify:

1. **Backward compatibility**: `ratio_k=1, with_replacement=False` produces same results as before (except new columns)

2. **Round-robin fairness**: With `ratio_k=3, with_replacement=False`:
   - All treated patients should have `match_round=1` before any have `match_round=2`
   - Verify by checking that count of `match_round=1` >= count of `match_round=2` >= count of `match_round=3`

3. **With replacement**: With `ratio_k=3, with_replacement=True`:
   - Same control can appear multiple times in output
   - `control_usage_count` > 1 for reused controls

4. **reuse_max**: With `ratio_k=3, with_replacement=True, reuse_max=2`:
   - No control should have `control_usage_count` > 2

5. **require_k=False**: With insufficient controls:
   - Treated patients with fewer than k matches should be included
   - `treated_k` should reflect actual count

6. **require_k=True** (default): With insufficient controls:
   - Treated patients with fewer than k matches should be excluded

7. **pair_weight correctness**:
   - Verify `pair_weight = 1 / (treated_k * control_usage_count)` for all rows

8. **Edge cases**:
   - Empty buckets
   - Single treated patient
   - Single control patient
   - `ratio_k` > number of controls in bucket

---

## 10. Example Usage

```python
# Standard 1:1 matching (current behavior)
matches = match(features_df, ratio_k=1)

# 1:3 matching without replacement (round-robin)
matches = match(features_df, ratio_k=3, with_replacement=False)

# 1:3 matching with replacement
matches = match(features_df, ratio_k=3, with_replacement=True)

# 1:3 matching with replacement, max 5 uses per control
matches = match(features_df, ratio_k=3, with_replacement=True, reuse_max=5)

# 1:3 matching, allow fewer than 3 matches
matches = match(features_df, ratio_k=3, require_k=False)
```
