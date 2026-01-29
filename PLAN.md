# Code Review Fixes Plan

This plan addresses issues identified in the code review. For documentation updates, follow the style guide in [DOCUMENTATION_STYLE.md](./DOCUMENTATION_STYLE.md).

---

## 1. Critical Documentation Fixes (README.md)

These issues would cause runtime errors for users following the README.

### 1.1 Fix `distance_metric` → `feature_space` Parameter Name

**Locations:** Lines 66, 199, 332

**Current (broken):**
```python
matched_df = match(features_df, distance_metric="euclidean", ...)
```

**Fix:**
```python
matched_df = match(features_df, feature_space="euclidean", ...)
```

Update all three occurrences plus the API reference description.

---

### 1.2 Remove Non-Existent `love_plot()` Parameters

**Locations:** Lines 75-81, 206-212, 245-247, 362-363

The README shows `feature_cols` and `treatment_col` as parameters, but these are auto-discovered.

**Current (broken):**
```python
fig = love_plot(
    stratified_df,
    feature_cols,              # This parameter doesn't exist
    treatment_col="treat",     # This parameter doesn't exist
    sample_frac=0.05,
)
```

**Fix:**
```python
fig = love_plot(
    stratified_df,
    sample_frac=0.05,
)
```

Update API reference to document auto-discovery:
```markdown
**Auto-discovered columns:**
- Feature columns: end with `__cat`, `__num`, `__date`, or `__exact` suffixes
- Treatment column: standardized as `treat__treat`
```

---

### 1.3 Fix `date_reference` Default Value

**Location:** Line 279

**Current:** `(default: "2018-01-01")`
**Fix:** `(default: "1970-01-01")`

---

### 1.4 Add Missing 1-to-k Matching Parameters

**Location:** API Reference section (after line 340)

Add to `match()` parameters:
```markdown
- `ratio_k`: Number of controls to match per treated patient (default: 1)
- `with_replacement`: Allow control reuse across treated patients (default: False)
- `reuse_max`: Maximum times a control can be reused when with_replacement=True (default: unlimited)
- `require_k`: If True, only keep treated patients with exactly k matches (default: False)
```

---

### 1.5 Remove Incorrect `match()` Parameters from API Reference

**Location:** Lines 337-340

Remove these lines (they're auto-discovered, not parameters):
```markdown
- `treatment_col`: Treatment indicator column (default: "treat")
- `id_col`: Patient ID column (default: "person_id")
- `exact_match_col`: Exact match column (default: "exact_match_id")
```

Replace with auto-discovery documentation.

---

## 2. Code Bug Fixes

### 2.1 Floating-Point Zero Comparison (High Priority)

**Files:** `brpmatch/loveplot.py:270,281`

**Current:**
```python
if pooled_std == 0:
    return 0.0
```

**Fix:**
```python
if pooled_std < 1e-10:
    return 0.0
```

Apply same pattern to `var_c == 0` check at line 281.

---

### 2.2 Empty DataFrame Guard (High Priority)

**File:** `brpmatch/matching.py:263-271`

**Current:** Unsafe `.collect()[0][0]` on potentially empty DataFrame.

**Fix:** Add empty DataFrame check:
```python
if persons_features_cohorts.count() == 0:
    raise ValueError("Input DataFrame is empty - no patients to match")

if bucket_length is None:
    feature_cnt = (
        persons_features_cohorts.limit(1)
        .select(F.size(F.col("feature_array")))
        .collect()[0][0]
    )
    if feature_cnt == 0:
        raise ValueError("Feature array is empty - no features to match on")
    bucket_length = 5 * pow(persons_features_cohorts.count(), (-1 / feature_cnt))
```

---

### 2.3 Consistent Error Types (Medium Priority)

**File:** `brpmatch/features.py:180,190`

**Current:** Uses `RuntimeError` for validation
**Fix:** Change to `ValueError`

---

### 2.4 Remove Unused `verbose` Parameter (Low Priority)

**File:** `brpmatch/features.py:101`

Either implement verbose logging or remove the parameter entirely. Recommend removal since it's documented as unused.

---

## 3. Code Organization Improvements

### 3.1 Create Shared Utilities Module

**New file:** `brpmatch/utils.py`

Extract duplicated functions:
```python
# From matching.py, summary.py, stratify.py
def _discover_id_column(df: DataFrame) -> str: ...
def _discover_treatment_column(df: DataFrame) -> str: ...
def _discover_exact_match_column(df: DataFrame) -> str: ...
def _discover_feature_columns(df: DataFrame) -> List[str]: ...

# From loveplot.py, summary.py
def _strip_suffix(col_name: str) -> str: ...

# From loveplot.py (make public for cross-module use)
def compute_smd(treated_values: np.ndarray, control_values: np.ndarray) -> float: ...
def compute_variance_ratio(treated_values: np.ndarray, control_values: np.ndarray) -> float: ...
```

Update imports in all affected modules.

---

### 3.2 Consolidate Balance Computation (Optional)

`_compute_balance_stats()` in loveplot.py and `_compute_comprehensive_balance()` in summary.py have overlapping logic. Consider unifying into a single implementation in utils.py.

---

## 4. Testing Updates

After code changes, verify:

1. **Run existing tests:**
   ```bash
   make test
   ```

2. **Run examples:**
   ```bash
   make example
   ```

3. **Verify README examples work:**
   - Copy Quick Start example, run in fresh environment
   - Verify all examples use correct parameter names

---

## Priority Order

1. **Immediate (breaks users):** Section 1 (README fixes)
2. **High (potential runtime errors):** Sections 2.1, 2.2
3. **Medium (code quality):** Sections 2.3, 3.1
4. **Low (nice to have):** Sections 2.4, 3.2

---

## Files to Modify

| File | Changes |
|------|---------|
| `README.md` | Parameter name fixes, API reference updates |
| `brpmatch/loveplot.py` | Float comparison fix (lines 270, 281) |
| `brpmatch/matching.py` | Empty DataFrame guard (line 263) |
| `brpmatch/features.py` | Error types, verbose param |
| `brpmatch/utils.py` | New file with shared utilities |
| `brpmatch/summary.py` | Import from utils |
| `brpmatch/stratify.py` | Import from utils |
