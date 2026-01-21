# Detailed Implementation Plan: Feature Space Transformations

This document provides step-by-step implementation instructions for adding whitening-based Mahalanobis support to the LSH bucketing system. Read `PLAN_HIGH_LEVEL.md` first for context.

## References

- [PySpark BucketedRandomProjectionLSH](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.BucketedRandomProjectionLSH.html) - LSH for Euclidean distance
- [PySpark RowMatrix.computeCovariance](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.linalg.distributed.RowMatrix.html) - Distributed covariance computation
- [NumPy linalg.eigh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) - Eigendecomposition for symmetric matrices

## Overview of Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `brpmatch/matching.py` | Modify | Rename parameter, add whitening transform, remove `match_distance` from output |
| `tests/test_matching.py` | Modify | Update parameter name, remove `match_distance` assertions |
| `scripts/example.py` | Modify | Update parameter name, remove `match_distance` display |

---

## Part 1: Changes to `brpmatch/matching.py`

### Step 1.1: Update imports

**Location**: Lines 1-29

**Current imports** (lines 11-14):
```python
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from numpy.linalg import pinv
```

**Action**: No new imports needed. `np.linalg.eigh` is accessed via the existing `numpy` import. Remove `pinv` import since it will no longer be used.

**Change**:
```python
# Line 14: Remove this import
from numpy.linalg import pinv  # DELETE THIS LINE
```

Also remove the unused `mahalanobis` import:
```python
# Line 28: Remove this import
from scipy.spatial.distance import mahalanobis  # DELETE THIS LINE
```

---

### Step 1.2: Rename function parameter

**Location**: Line 34

**Current**:
```python
distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean",
```

**Change to**:
```python
feature_space: Literal["euclidean", "mahalanobis"] = "euclidean",
```

---

### Step 1.3: Update docstring

**Location**: Lines 44-88

**Current docstring excerpt** (lines 60-61):
```python
distance_metric : Literal["euclidean", "mahalanobis"]
    Distance metric for k-NN within buckets
```

**Replace with**:
```python
feature_space : Literal["euclidean", "mahalanobis"]
    Feature space for bucketing and matching. "euclidean" uses original
    features with Euclidean distance. "mahalanobis" applies a whitening
    transform so that Euclidean distance in transformed space equals
    Mahalanobis distance in original space.
```

**Also update the Returns section** (lines 79-87). Remove the `match_distance` bullet:

**Current**:
```python
Returns
-------
DataFrame
    DataFrame with columns:
    - {id_col}: treated patient ID
    - match_{id_col}: matched control patient ID
    - match_distance: distance between matched pair
    - bucket_num_input_patients: size of bucket where match was found
    - bucket_seconds: time to process bucket
```

**Change to**:
```python
Returns
-------
DataFrame
    DataFrame with columns:
    - {id_col}: treated patient ID
    - match_{id_col}: matched control patient ID
    - bucket_num_input_patients: size of bucket where match was found
    - bucket_seconds: time to process bucket
```

---

### Step 1.4: Add whitening transform computation

**Location**: After line 92 (after adding `feature_array` column), BEFORE the bucket_length computation (line 98)

**Why before bucket_length?** The bucket_length heuristic `N^(-1/d)` should use the dimensionality of the space being bucketed. For mahalanobis, that's the whitened space (which may have fewer dimensions if eigenvalues are filtered).

**Insert the following new code block**:

```python
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
        cov_matrix = RowMatrix(features_for_cov).computeCovariance().toArray()

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
```

---

### Step 1.5: The whitening code includes feature transformation

**Note**: The code block in Step 1.4 already includes the feature transformation (broadcasting the whitening matrix and applying it via UDF). The `feature_array` column is updated with transformed features.

After this block executes:
- `features_col` contains whitened vectors
- `feature_array` contains whitened arrays
- The bucket_length computation (which follows) will use the transformed feature dimensionality

---

### Step 1.6: Update output schema to remove match_distance

**Location**: Lines 287-296

**Current schema**:
```python
    schema_potential_matches_arrays = StructType(
        [
            StructField(id_col, StringType()),
            StructField("match_" + id_col, StringType()),
            StructField("match_distance", DoubleType()),
            StructField("bucket_num_input_patients", IntegerType()),
            StructField("bucket_seconds", DoubleType()),
        ]
    )
```

**Change to**:
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

---

### Step 1.7: Simplify find_neighbors function - always use Euclidean

**Location**: Lines 298-405 (the `find_neighbors` function)

Since we now always work in the appropriate feature space (original for euclidean, whitened for mahalanobis), we always use Euclidean distance for the k-NN matching.

**Remove the distance metric branching** (lines 317-326):

**Current**:
```python
        # Set distance metric for NN model
        if distance_metric == "euclidean":
            metric = distance_metric
        elif distance_metric == "mahalanobis":
            metric = mahal_dist
        else:
            raise ValueError(
                f"Within-bucket distance metric must be one of 'euclidean' or "
                f"'mahalanobis'. Got '{distance_metric}'."
            )
```

**Replace with**:
```python
        # Always use Euclidean distance (features are pre-transformed for mahalanobis)
        metric = "euclidean"
```

---

### Step 1.8: Update find_neighbors to not return match_distance

The `find_neighbors` function builds results with `match_distance`. We need to keep distances internally for ranking, but not return them.

**Location**: Lines 366-371

**Current**:
```python
        match_candidates = pd.DataFrame(
            results, columns=[id_col, "match_" + id_col, "match_distance"]
        )
        match_candidates["match_distance"] = pd.to_numeric(
            match_candidates["match_distance"]
        )
```

**Change to** (rename internal column to avoid confusion):
```python
        match_candidates = pd.DataFrame(
            results, columns=[id_col, "match_" + id_col, "_distance"]
        )
        match_candidates["_distance"] = pd.to_numeric(
            match_candidates["_distance"]
        )
```

**Location**: Lines 373-378 (ranking code)

**Current**:
```python
        match_candidates["source_target_rank"] = match_candidates.groupby([id_col])[
            "match_distance"
        ].rank()
        match_candidates["target_source_rank"] = match_candidates.groupby(
            "match_" + id_col
        )["match_distance"].rank(method="first")
```

**Change to**:
```python
        match_candidates["source_target_rank"] = match_candidates.groupby([id_col])[
            "_distance"
        ].rank()
        match_candidates["target_source_rank"] = match_candidates.groupby(
            "match_" + id_col
        )["_distance"].rank(method="first")
```

**Location**: Line 405

**Current**:
```python
        return match_candidates.drop(columns=["source_target_rank", "target_source_rank"])
```

**Change to**:
```python
        return match_candidates.drop(columns=["source_target_rank", "target_source_rank", "_distance"])
```

---

### Step 1.9: Remove the old Mahalanobis distance computation

**Location**: Lines 272-285

**Current code** (no longer needed):
```python
    # Define covariance matrix and distance function for Mahalanobis
    if distance_metric == "mahalanobis":
        # Compute the pseudoinverse of the covariance matrix for features
        vec_converter_udf = F.udf(lambda v: Vectors.dense(v.toArray()), VectorUDT())
        features_converted = persons_features_cohorts.withColumn(
            "features_converted", vec_converter_udf(features_col)
        ).select("features_converted")
        inverse_covariance_mat = pinv(
            RowMatrix(features_converted).computeCovariance().toArray()
        )

        def mahal_dist(vec1, vec2):
            """Custom distance function using globally-computed inverse covariance matrix"""
            return mahalanobis(vec1, vec2, inverse_covariance_mat)
```

**Action**: DELETE this entire block. The whitening transform (Step 1.4-1.5) replaces this functionality.

---

## Part 2: Changes to `tests/test_matching.py`

### Step 2.1: Update parameter name in all test functions

**Location**: Lines 31, 41, 64, 74, 86

**Change all occurrences of**:
```python
distance_metric="euclidean"
```
**to**:
```python
feature_space="euclidean"
```

**And change**:
```python
distance_metric="mahalanobis"
```
**to**:
```python
feature_space="mahalanobis"
```

---

### Step 2.2: Remove match_distance assertions

**Location**: Line 36

**Current**:
```python
    assert "match_distance" in matched_df.columns
```

**Action**: DELETE this line.

**Location**: Lines 72-80 (entire `test_distances_positive` function)

**Current**:
```python
def test_distances_positive(features_df):
    """Test that all match distances are non-negative."""
    matched_df = match(features_df, distance_metric="euclidean", n_neighbors=5, id_col="id")

    # Check that all distances are >= 0
    from pyspark.sql import functions as F

    min_distance = matched_df.select(F.min("match_distance")).first()[0]
    assert min_distance >= 0
```

**Action**: DELETE this entire function.

**Location**: Line 95

**Current**:
```python
    assert "match_distance" in matched_df.columns
```

**Action**: DELETE this line.

---

### Step 2.3: Add new test for whitening transform

**Location**: At end of file, add new test:

```python
def test_mahalanobis_uses_whitening(features_df):
    """Test that mahalanobis feature space produces valid matches using whitening."""
    matched_df = match(
        features_df, feature_space="mahalanobis", n_neighbors=5, id_col="id"
    )

    # Should produce valid matches
    assert matched_df.count() > 0

    # Check that required columns exist (no match_distance)
    assert "id" in matched_df.columns
    assert "match_id" in matched_df.columns
    assert "match_distance" not in matched_df.columns

    # Verify 1-to-1 constraint still holds
    pdf = matched_df.toPandas()
    id_counts = pdf["id"].value_counts()
    assert all(id_counts <= 1), "Some treated patients matched multiple times"
    match_id_counts = pdf["match_id"].value_counts()
    assert all(match_id_counts <= 1), "Some control patients matched multiple times"
```

---

## Part 3: Changes to `scripts/example.py`

### Step 3.1: Update parameter name

**Location**: Line 77

**Current**:
```python
        distance_metric="euclidean",
```

**Change to**:
```python
        feature_space="euclidean",
```

---

### Step 3.2: Remove match_distance from display

**Location**: Line 86

**Current**:
```python
    matched_df.select("id", "match_id", "match_distance").show(5)
```

**Change to**:
```python
    matched_df.select("id", "match_id").show(5)
```

---

## Part 4: Complete Code for Key Sections

### 4.1: Complete whitening transform block (insert after bucket_length computation)

This is the complete code block to insert in `matching.py` after line 105:

```python
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
        cov_matrix = RowMatrix(features_for_cov).computeCovariance().toArray()

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
        whitening_matrix_bc = persons_features_cohorts.sql_ctx.sparkSession.sparkContext.broadcast(whitening_matrix)

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
```

---

## Verification Checklist

After implementation, verify:

1. **Tests pass**: Run `pytest tests/test_matching.py -v`
2. **Example runs**: Run `python scripts/example.py`
3. **Euclidean behavior unchanged**: Compare output before/after for `feature_space="euclidean"`
4. **Mahalanobis produces matches**: Verify `feature_space="mahalanobis"` produces non-empty results
5. **No match_distance in output**: Verify output schema has only `id`, `match_id`, `bucket_num_input_patients`, `bucket_seconds`

---

## Mathematical Reference

### Why Whitening Works

The Mahalanobis distance between vectors x and y is:
```
d_M(x, y) = sqrt((x - y)^T Σ^(-1) (x - y))
```

Where Σ is the covariance matrix. If we decompose Σ = VΛV^T (eigendecomposition), then:
```
Σ^(-1) = V Λ^(-1) V^T
```

Let W = Λ^(-1/2) V^T be the whitening matrix. Then for transformed vectors z = Wx:
```
||z_x - z_y||² = (Wx - Wy)^T (Wx - Wy)
              = (x - y)^T W^T W (x - y)
              = (x - y)^T V Λ^(-1/2) Λ^(-1/2) V^T (x - y)
              = (x - y)^T V Λ^(-1) V^T (x - y)
              = (x - y)^T Σ^(-1) (x - y)
              = d_M(x, y)²
```

Therefore: **Euclidean distance in whitened space = Mahalanobis distance in original space**

---

## Order of Operations Summary

1. Load features DataFrame, add `feature_array` column
2. **[NEW]** If `feature_space="mahalanobis"`:
   - Compute covariance matrix
   - Compute eigendecomposition
   - Filter small eigenvalues
   - Compute whitening matrix W = Λ^(-1/2) V^T
   - Transform all features: z = Wx
   - Update `feature_array` with transformed features
3. Compute bucket_length if not provided (uses current feature dimensionality)
4. Apply LSH bucketing (on original or whitened features)
5. Within each bucket, use Euclidean k-NN (always)
6. Apply greedy 1-to-1 matching
7. Return matches (without distances)
