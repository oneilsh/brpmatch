# Detailed Implementation Plan: Match Output Restructuring

This document provides implementation details sufficient for a follow-up agent to execute the refactor with just these two plan files as context.

## Overview

Change `match()` to return a tuple of three DataFrames: `(units, pairs, bucket_stats)`.

## File-by-File Changes

---

### 1. `brpmatch/matching.py`

#### 1.1 Update function signature and docstring

**Location**: Lines 37-51

Change return type annotation:

```python
def match(
    features_df: DataFrame,
    ...
) -> Tuple[DataFrame, DataFrame, DataFrame]:
```

Update docstring Returns section:

```python
"""
Returns
-------
Tuple[DataFrame, DataFrame, DataFrame]
    A tuple of three DataFrames:

    units : DataFrame
        One row per patient (treated + control, matched + unmatched).
        Columns:
        - {id_col}: Patient ID (same as input)
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
"""
```

#### 1.2 Update schema to include bucket stats columns

**Location**: Around lines 417-430 (schema definition)

Add new columns to the output schema for bucket-level statistics:

```python
schema_potential_matches_arrays = StructType(
    [
        StructField(id_col_base, StringType()),
        StructField("match_" + id_col_base, StringType()),
        StructField("match_round", IntegerType()),
        StructField("treated_k", IntegerType()),
        StructField("control_usage_count", IntegerType()),
        StructField("pair_weight", DoubleType()),
        StructField("bucket_id", StringType()),           # NEW
        StructField("bucket_num_patients", IntegerType()),  # Renamed from bucket_num_input_patients
        StructField("bucket_num_treated", IntegerType()),   # NEW
        StructField("bucket_num_control", IntegerType()),   # NEW
        StructField("bucket_seconds", DoubleType()),
    ]
)
```

#### 1.3 Capture bucket stats inside `find_neighbors` function

**Location**: Inside `find_neighbors` function (around lines 432-641)

The function already has access to the separated cohorts:
- `needs_matching` = treated patients in bucket
- `match_to` = control patients in bucket

Capture these counts early in the function (after line 444):

```python
# Capture bucket-level counts for stats
num_treated_in_bucket = len(needs_matching)
num_control_in_bucket = len(match_to)
```

Then when building result_df (around line 624-641), add all bucket stats:

```python
# Add bucket metadata
result_df["bucket_id"] = group_df["bucket_id"].iloc[0]  # All rows have same bucket_id
result_df["bucket_num_patients"] = bucket_size
result_df["bucket_num_treated"] = num_treated_in_bucket
result_df["bucket_num_control"] = num_control_in_bucket

bucket_end_time = time.perf_counter()
result_df["bucket_seconds"] = bucket_end_time - bucket_start_time
```

Update the output columns list:

```python
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
```

#### 1.4 Build the three output DataFrames

**Location**: After line 650 (after the `find_neighbors` is applied)

Replace the final return statement with logic to build all three DataFrames:

```python
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
```

#### 1.4 Add helper function `_build_units_df`

**Location**: Add as new function after `match()` (around line 680)

```python
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

    return units.select("id", "subclass", "weight", "is_treated")
```

#### 1.5 Update imports

Add `Tuple` to typing imports if not already present:

```python
from typing import Literal, Optional, Tuple
```

---

### 2. `brpmatch/stratify.py`

#### Option A: Remove entirely

Delete the file and remove from `__init__.py`.

#### Option B: Keep as internal utility (recommended for this phase)

Rename to `_stratify.py` or keep but remove from `__all__` in `__init__.py`. The `match_summary()` function may still use it internally during transition.

---

### 3. `brpmatch/summary.py`

#### 3.1 Update `match_summary()` to accept units DataFrame

**Location**: Lines 40-143

Change signature to accept the new format:

```python
def match_summary(
    features_df: DataFrame,
    units_df: DataFrame,  # Changed from matched_df
    sample_frac: float = 0.05,
    ...
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, matplotlib.figure.Figure]]:
```

Update the internal logic:

```python
# OLD: Uses stratify_for_plot()
# stratified_df = stratify_for_plot(features_df, matched_df)

# NEW: Join features_df with units_df directly
stratified_df = features_df.join(
    units_df.select("id", "subclass", "weight"),
    features_df[id_col] == units_df["id"],
    "left"
).drop(units_df["id"])

# Rename subclass to strata for compatibility with existing balance computation
stratified_df = stratified_df.withColumnRenamed("subclass", "strata")
```

The rest of the balance computation should work unchanged since it just looks for a `strata` column.

#### 3.2 Update `match_data()` to use units DataFrame

**Location**: Lines 304-426

Simplify significantly - now just joins units to original data:

```python
def match_data(
    original_df: DataFrame,
    units_df: DataFrame,
    id_col: str,
) -> DataFrame:
    """
    Join match information to original DataFrame for outcome analysis.

    Parameters
    ----------
    original_df : DataFrame
        Original input DataFrame (before generate_features)
    units_df : DataFrame
        Units DataFrame from match() output
    id_col : str
        Name of the ID column in original_df

    Returns
    -------
    DataFrame
        Original DataFrame with added columns: weight, subclass, matched
    """
    # Add matched boolean
    units_with_matched = units_df.withColumn(
        "matched", F.col("subclass").isNotNull()
    )

    # Join to original data
    result = original_df.join(
        units_with_matched.select("id", "weight", "subclass", "matched"),
        original_df[id_col] == units_with_matched["id"],
        "left"
    ).drop(units_with_matched["id"])

    # Fill any unjoined rows (shouldn't happen, but defensive)
    result = result.withColumn(
        "weight", F.coalesce(F.col("weight"), F.lit(0.0))
    ).withColumn(
        "matched", F.coalesce(F.col("matched"), F.lit(False))
    )

    return result
```

#### 3.3 Remove import of stratify_for_plot

**Location**: Line 17

Remove or comment out:
```python
# from .stratify import stratify_for_plot  # No longer needed
```

---

### 4. `brpmatch/__init__.py`

#### 4.1 Update exports

**Location**: Lines 1-22

```python
from .features import generate_features
from .loveplot import love_plot
from .matching import match
from .summary import match_summary, match_data
# Remove: from .stratify import stratify_for_plot

__version__ = "0.1.0"
__all__ = [
    "generate_features",
    "match",
    "match_summary",
    "match_data",
    # Remove: "stratify_for_plot",
    "love_plot",
]
```

---

### 5. `brpmatch/loveplot.py`

#### 5.1 Update to work with new strata column name

The love plot expects a `strata_col` parameter. Ensure it defaults to `"strata"` and works correctly when the column contains treated IDs (the subclass values).

**Check**: The current implementation should work, but verify it handles `None` values for unmatched rows.

---

### 6. Example Files

Update all 6 example files in `example/*/example.py`:

#### 6.1 Update imports

```python
# Remove stratify_for_plot from imports
from brpmatch import generate_features, match, match_summary
```

#### 6.2 Update match() call

```python
# OLD:
matched = match(features_df, ...)

# NEW:
units, pairs, bucket_stats = match(features_df, ...)
```

#### 6.3 Remove stratify_for_plot() call

```python
# OLD:
stratified = stratify_for_plot(features_df, matched)

# NEW:
# No longer needed - units already contains the stratification info
# For saving stratified data, join units back to features_df if needed
stratified = features_df.join(
    units.select("id", "subclass", "weight"),
    features_df["id__id"] == units["id"],
    "left"
).drop("id")
```

#### 6.4 Update match_summary() call

```python
# OLD:
summary, fig = match_summary(features_df, matched, ...)

# NEW:
summary, fig = match_summary(features_df, units, ...)
```

#### 6.5 Update output saving

```python
# OLD:
matched.toPandas().to_csv(..., "matched.csv", ...)

# NEW:
pairs.toPandas().to_csv(..., "pairs.csv", ...)  # Renamed for clarity
units.toPandas().to_csv(..., "units.csv", ...)
bucket_stats.toPandas().to_csv(..., "bucket_stats.csv", ...)
```

#### 6.6 Update print_matching_stats() in example/utils.py

This utility function may need updates to work with the new output format.

---

### 7. Test Files

#### 7.1 `tests/test_matching.py`

**Update all tests that call `match()`** to unpack the tuple:

```python
# OLD:
matched_df = match(features_df, ...)

# NEW:
units, pairs, bucket_stats = match(features_df, ...)
# Most existing tests should use 'pairs' as it has the same structure
```

**Add new tests for the units DataFrame**:

```python
def test_units_df_structure(features_df):
    """Test that units DataFrame has expected structure."""
    units, pairs, bucket_stats = match(features_df, n_neighbors=5)

    # Check columns
    assert "id" in units.columns
    assert "subclass" in units.columns
    assert "weight" in units.columns
    assert "is_treated" in units.columns

    # Check all patients are included
    assert units.count() == features_df.count()


def test_units_df_weights(features_df):
    """Test weight computation in units DataFrame."""
    units, pairs, bucket_stats = match(features_df, n_neighbors=5)
    pdf = units.toPandas()

    # Matched treated should have weight=1
    matched_treated = pdf[(pdf["is_treated"]) & (pdf["subclass"].notna())]
    assert all(matched_treated["weight"] == 1.0)

    # Unmatched should have weight=0
    unmatched = pdf[pdf["subclass"].isna()]
    assert all(unmatched["weight"] == 0.0)


def test_bucket_stats_structure(features_df):
    """Test that bucket_stats DataFrame has expected structure."""
    units, pairs, bucket_stats = match(features_df, n_neighbors=5)

    # Check columns
    assert "bucket_id" in bucket_stats.columns
    assert "num_patients" in bucket_stats.columns
    assert "num_treated" in bucket_stats.columns
    assert "num_control" in bucket_stats.columns
    assert "num_matches" in bucket_stats.columns
    assert "seconds" in bucket_stats.columns

    # Verify consistency: num_patients = num_treated + num_control
    pdf = bucket_stats.toPandas()
    assert all(pdf["num_patients"] == pdf["num_treated"] + pdf["num_control"])
```

**Update existing tests** to use `pairs` where they previously used `matched_df`:

```python
def test_one_to_one_constraint(features_df):
    """Test that matching satisfies 1-to-1 constraint."""
    units, pairs, bucket_stats = match(features_df, n_neighbors=5)
    pdf = pairs.toPandas()  # Changed from matched_df
    # ... rest unchanged
```

#### 7.2 `tests/test_summary.py`

**Update fixture**:

```python
@pytest.fixture
def matched_data(spark, lalonde_df):
    """Generate features and matched data for testing."""
    # ... feature generation unchanged ...

    # Match returns tuple now
    units, pairs, bucket_stats = match(features_df, n_neighbors=5)

    return features_df, units  # Return units instead of matched_df
```

**Update test function calls**:

```python
def test_match_summary_basic(matched_data):
    features_df, units = matched_data  # Changed from matched_df

    summary_df = match_summary(
        features_df,
        units,  # Changed from matched_df
        ...
    )
```

---

## Implementation Order

1. **`brpmatch/matching.py`**: Core changes to return tuple
2. **`brpmatch/summary.py`**: Update to accept units DataFrame
3. **`brpmatch/__init__.py`**: Update exports
4. **`brpmatch/stratify.py`**: Remove from public API
5. **Tests**: Update all test files
6. **Examples**: Update all example files

## Testing Strategy

1. Run existing tests after each file change to catch regressions
2. Add new tests for the new output DataFrames
3. Run all examples manually to verify end-to-end workflow
4. Verify output CSVs have correct structure

## Rollback Plan

If issues arise, the changes are isolated to:
- `matching.py`: Return type change
- `summary.py`: Parameter type change
- `__init__.py`: Export list change

These can be reverted independently. Keep `stratify.py` as internal (don't delete) until the refactor is confirmed working.
