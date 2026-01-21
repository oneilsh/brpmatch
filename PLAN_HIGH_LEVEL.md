# High-Level Plan: Feature Space Transformations for LSH Bucketing

## Problem Statement

The current implementation supports two distance metrics for within-bucket matching: Euclidean and Mahalanobis. However, the LSH bucketing step **always operates in Euclidean space**, regardless of which distance metric is selected.

This creates an inconsistency when `distance_metric="mahalanobis"`:
- **Bucketing**: Uses random projections in Euclidean space (ignores feature correlations)
- **Within-bucket matching**: Uses Mahalanobis distance (accounts for covariance)

Patients who are "close" in the Mahalanobis sense may land in different buckets because LSH doesn't account for the covariance structure. This can lead to suboptimal matches or missed matches entirely.

## Proposed Solution

Replace the `distance_metric` parameter with a `feature_space` parameter that controls a **pre-transformation** applied to features before both bucketing AND within-bucket matching.

**Key insight**: Mahalanobis distance in original space = Euclidean distance in whitened (PCA-transformed) space.

By whitening features before LSH, we get Mahalanobis-aware bucketing "for free" using the existing Euclidean LSH machinery.

## High-Level Design

### API Change

```python
# Current API
match(..., distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean", ...)

# Proposed API
match(..., feature_space: Literal["euclidean", "mahalanobis"] = "euclidean", ...)
```

### Behavior by Feature Space

| `feature_space` | Transform Applied | Bucketing | Within-bucket Distance |
|-----------------|-------------------|-----------|------------------------|
| `"euclidean"` | None | Euclidean LSH on original features | Euclidean |
| `"mahalanobis"` | Whitening (PCA-based) | Euclidean LSH on whitened features | Euclidean (equivalent to Mahalanobis in original space) |

### Whitening Transform

Given covariance matrix Σ with eigendecomposition Σ = VΛV^T:

1. Compute whitening matrix: W = Λ^(-1/2) V^T
2. Transform features: z = Wx
3. Result: Euclidean distance ||z_i - z_j|| = Mahalanobis distance in original space

## Key Decision Points

### 1. Where to implement the transform

**Decision**: In `matching.py`, not `features.py`

**Rationale**:
- The covariance computation for Mahalanobis is already in `matching.py`
- This is a matching concern, not a feature engineering concern
- Keeps `generate_features()` focused on preprocessing (one-hot encoding, scaling)

### 2. Handling singular/near-singular covariance matrices

**Decision**: Filter eigenvalues below a threshold (e.g., 1e-10)

**Rationale**:
- The code already uses `pinv()` for the inverse covariance matrix
- Near-zero eigenvalues indicate directions with no variance (constant features or collinearity)
- These dimensions contribute nothing to distance and can be safely dropped

### 3. Distance values in output

**Current state**: `match_distance` is returned in output and used internally for ranking candidates during greedy matching.

**Decision**: Remove `match_distance` from output schema for all modes

**Rationale**:
- Transformed-space distances have no meaningful interpretation to users
- Consistent API across all `feature_space` options (cleaner than conditional output)
- Balance statistics (SMD, variance ratios) are what truly matter for quality assessment
- Distances are still computed and used internally for candidate ranking

**Breaking change**: Tests and example code reference `match_distance`. These will need updates.

### 4. Interaction with existing StandardScaler

**Current state**: Features are scaled to unit variance but NOT centered (`withMean=False`)

**Decision**: No change needed

**Rationale**:
- `withMean=False` preserves sparsity for one-hot encoded features
- PySpark's `RowMatrix.computeCovariance()` handles centering internally
- The whitening transform will work correctly on variance-scaled features

### 5. Bucket length computation

**Current formula**: `bucket_length = N^(-1/d)` where N = patients, d = features

**Decision**: Compute bucket_length AFTER transformation, using the transformed feature dimensionality

**Rationale**:
- The heuristic should use the dimensionality of the space being bucketed
- For mahalanobis, LSH operates on whitened features, so d = number of retained eigenvalues
- This naturally handles collinearity (filtered eigenvalues reduce d)
- Users can still override with explicit `bucket_length` parameter

## Potential Gotchas

### 1. Eigenvalue computation at scale

**Issue**: Computing full eigendecomposition of a large covariance matrix can be expensive.

**Mitigation**:
- Covariance matrix size is d×d where d = number of features (typically 10s-100s, not millions)
- Already computing covariance for Mahalanobis; eigendecomposition adds minimal overhead
- NumPy's `eigh()` is efficient for symmetric matrices

### 2. Numerical stability

**Issue**: Very small eigenvalues can cause numerical issues when computing Λ^(-1/2).

**Mitigation**:
- Filter eigenvalues below threshold (1e-10)
- Use `np.linalg.eigh()` which is numerically stable for symmetric positive semi-definite matrices

### 3. Sparse vs Dense features

**Issue**: One-hot encoded categoricals create sparse vectors. Whitening produces dense vectors.

**Mitigation**:
- The existing code already converts to dense for Mahalanobis computation (`matching.py:276-278`)
- No additional changes needed, but memory usage may increase
- Document this behavior for users with very high-dimensional categorical features

### 4. Broadcasting the transform matrix

**Issue**: The whitening matrix must be applied to every row in a distributed Spark DataFrame.

**Mitigation**:
- Broadcast the (small) whitening matrix to all workers
- Apply via UDF, similar to existing dense vector conversion
- Matrix is d×d where d is typically small

### 5. Backward compatibility

**Issue**: Renaming `distance_metric` to `feature_space` breaks existing code.

**Options**:
- A) Clean break: rename parameter, update docs
- B) Deprecation: accept both names, warn if old name used
- C) Keep old name: just change the behavior description

**Recommendation**: Option A (clean break) since this is pre-1.0 software and the semantic meaning changes (it's no longer just about distance metric, it's about the entire feature space).

## Future Directions (Not in Scope)

### UMAP Transform
- Would add `feature_space="umap"` option
- Requires `spark-rapids-ml` dependency
- Spreads clumpy patient data for better bucketing efficiency
- UMAP output would need normalization (arbitrary scale)
- Mutually exclusive with Mahalanobis (different use cases)

### Propensity Score Matching
- Different matching paradigm entirely
- Would likely be a separate `method` parameter
- Orthogonal to feature space transforms

### 1-to-N Matching
- Extension to within-bucket matching logic
- Would add `match_ratio` parameter
- Independent of feature space transforms

## Files to Modify

| File | Changes |
|------|---------|
| `matching.py` | Rename parameter, add whitening transform logic, update LSH pipeline |
| `__init__.py` | Update exports if any signature changes |
| Tests | Update for new parameter name, add whitening-specific tests |
| README/docs | Update parameter documentation |

## Success Criteria

1. When `feature_space="mahalanobis"`, bucketing respects covariance structure
2. Matching quality improves for correlated features (can test with synthetic data)
3. Performance overhead is acceptable (eigendecomposition is fast for typical feature counts)
4. Existing `feature_space="euclidean"` behavior is unchanged
