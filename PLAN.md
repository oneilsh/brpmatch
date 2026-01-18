# BRPMatch Refactoring Plan

## Overview

BRPMatch is an Apache Spark-based cohort matching tool that uses distance-based methods (LSH + k-NN) for large-scale propensity score-like matching. The current implementation was built for Palantir Foundry and needs to be refactored into a standalone Python package.

## Current State

### Existing Files
- `pipeline.py` (481 lines) - Main Spark code with two functions:
  - `brp_generate_features()` - Converts patient data into feature vectors using Spark ML (StringIndexer, OneHotEncoder, Imputer, StandardScaler, optional GBT imputation)
  - `brp_match_features()` - LSH-based matching algorithm using BucketedRandomProjectionLSH and scikit-learn's NearestNeighbors via Pandas UDFs
- `pipeline.sql` (46 lines) - SQL for stratifying matched pairs for love plot analysis
- `pipeline.R` (88 lines) - R code using `cobalt` and `ggplot2` for love plot visualization
- `pyproject.toml` - Poetry-based, but dependencies not populated
- `Makefile` - Has PyPI publishing targets, needs testing/zip targets

### Issues to Address
1. Code uses `@transform_pandas` decorators from `foundry_ml` - must be removed
2. Functions have hardcoded column names - need parameterization
3. `brp_match_features()` has early returns (lines ~290, ~303) that should be removed
4. Dependencies not listed in pyproject.toml
5. No package structure (code is in root)
6. SQL needs conversion to PySpark
7. R visualization needs conversion to Python/matplotlib

---

## Target State

### Package Configuration
| Setting | Value |
|---------|-------|
| Python version | >=3.10 |
| Spark version | >=3.3 |
| Visualization | matplotlib |
| API style | Function-based with explicit arguments |

### Dependencies
```
pyspark>=3.3
numpy
scipy
scikit-learn
pandas
matplotlib
pytest (dev)
```

### Target API

```python
from brpmatch import generate_features, match, stratify_for_plot, love_plot

# 1. Generate feature vectors
features_df = generate_features(
    spark,
    df,
    categorical_cols=["state", "smoker", "diabetes"],
    numeric_cols=["age", "bmi"],
    date_cols=["diagnosis_date"],
    exact_match_cols=["gender"],           # Pre-stratification
    treatment_col="cohort",
    treatment_value="treated",
    gbt_impute_cols=["bmi"],               # Optional GBT imputation
)

# 2. Perform matching
matched_df = match(
    features_df,
    distance_metric="euclidean",           # or "mahalanobis"
    n_neighbors=5,
    bucket_length=2.0,
)

# 3. Prepare data for visualization
stratified_df = stratify_for_plot(features_df, matched_df)

# 4. Generate love plot
fig = love_plot(
    stratified_df,
    treatment_col="treat",
    sample_frac=0.05,
)
fig.savefig("balance.png")
```

### Target Directory Structure
```
brpmatch/
├── pyproject.toml
├── Makefile
├── PLAN.md                  # This file
├── README.md
├── brpmatch/
│   ├── __init__.py          # Package exports, version
│   ├── features.py          # generate_features()
│   ├── matching.py          # match()
│   ├── stratify.py          # stratify_for_plot()
│   └── loveplot.py          # love_plot()
├── tests/
│   ├── data/
│   │   └── lalonde.csv      # Test dataset (treat, age, educ, black, hispan, married, nodegree, re74, re75, re78, id)
│   ├── conftest.py          # pytest fixtures (SparkSession)
│   ├── test_features.py
│   ├── test_matching.py
│   └── test_loveplot.py
└── scripts/
    └── example.py           # Example usage
```

### Makefile Targets
| Target | Purpose |
|--------|---------|
| `install` | Poetry install |
| `dev` | Install with dev dependencies (pytest, build, twine) |
| `test` | Run pytest with local Spark |
| `build` | Build wheel and sdist |
| `zip` | Create importable .zip for offline use (just the package, no compiled deps) |
| `clean` | Remove build artifacts |
| `publish-test` | Upload to TestPyPI |
| `publish` | Upload to PyPI |

---

## Implementation Phases

### Phase 1: Project Structure
- Create `brpmatch/` package directory
- Create `tests/` and `tests/data/` directories
- Create `scripts/` directory
- Update `pyproject.toml`:
  - Change `requires-python` to `>=3.10`
  - Add all dependencies
  - Add dev dependencies (pytest)
  - Configure package to find `brpmatch/` directory
- Update `Makefile`:
  - Add `test` target
  - Add `zip` target

### Phase 2: Refactor Feature Generation (pipeline.py → brpmatch/features.py)
- Remove `@transform_pandas` decorator and Foundry imports
- Create `generate_features()` function with parameters:
  - `spark: SparkSession`
  - `df: DataFrame`
  - `categorical_cols: List[str]`
  - `numeric_cols: List[str]`
  - `date_cols: List[str] = None`
  - `exact_match_cols: List[str] = None`
  - `treatment_col: str`
  - `treatment_value: str`
  - `gbt_impute_cols: List[str] = None`
  - `date_reference: str = "2018-01-01"` (for converting dates to numeric)
- Extract hardcoded column lists into parameters
- Keep Spark ML pipeline logic (StringIndexer, OneHotEncoder, Imputer, StandardScaler)
- Keep GBT imputation logic as optional feature

### Phase 3: Refactor Matching (pipeline.py → brpmatch/matching.py)
- Remove `@transform_pandas` decorator and Foundry imports
- Remove early returns (~lines 290, 303)
- Create `match()` function with parameters:
  - `features_df: DataFrame` (output from generate_features)
  - `distance_metric: str = "euclidean"` (or "mahalanobis")
  - `n_neighbors: int = 5`
  - `bucket_length: float = 2.0`
  - `num_hash_tables: int = 4`
  - `features_col: str = "features"`
  - `treatment_col: str = "treat"`
  - `id_col: str = "person_id"`
- Keep LSH bucketing logic (BucketedRandomProjectionLSH with multiple bucket levels)
- Keep Pandas UDF for k-NN within buckets
- Keep greedy 1-to-1 matching logic

### Phase 4: Convert SQL to PySpark (pipeline.sql → brpmatch/stratify.py)
- Create `stratify_for_plot()` function with parameters:
  - `features_df: DataFrame`
  - `matched_df: DataFrame`
  - `id_col: str = "person_id"`
  - `match_id_col: str = "match_person_id"`
- Implement the SQL logic using DataFrame API:
  - Create strata identifiers from matched pairs
  - Left join to identify which patients are matched
  - Coalesce to create unified matched indicators

### Phase 5: Convert R to Python (pipeline.R → brpmatch/loveplot.py)
- Create `love_plot()` function with parameters:
  - `stratified_df: DataFrame`
  - `treatment_col: str = "treat"`
  - `strata_col: str = "strata"`
  - `categorical_suffix: str = "_index"`
  - `numeric_suffix: str = "_imputed"`
  - `sample_frac: float = 0.05`
  - `figsize: tuple = (10, 12)`
- Implement balance statistics calculation:
  - Standardized Mean Difference (SMD): `(mean_treated - mean_control) / pooled_std`
  - Variance Ratio: `var_treated / var_control`
  - Calculate for both unadjusted (all data) and adjusted (matched pairs only)
- Create matplotlib visualization:
  - Point plot showing SMD and variance ratio
  - Color-coded by unadjusted vs adjusted
  - Faceted or multi-panel layout

### Phase 6: Testing
- Create `tests/conftest.py` with SparkSession fixture:
  ```python
  @pytest.fixture(scope="session")
  def spark():
      return SparkSession.builder \
          .master("local[*]") \
          .appName("brpmatch-test") \
          .getOrCreate()
  ```
- Create `tests/data/lalonde.csv` (user will provide)
- Write tests for each module:
  - `test_features.py`: Test feature generation with lalonde data
  - `test_matching.py`: Test matching algorithm
  - `test_loveplot.py`: Test balance calculation and plot generation

### Phase 7: Package Finalization
- Create `brpmatch/__init__.py` with exports:
  ```python
  from .features import generate_features
  from .matching import match
  from .stratify import stratify_for_plot
  from .loveplot import love_plot

  __version__ = "0.1.0"
  __all__ = ["generate_features", "match", "stratify_for_plot", "love_plot"]
  ```
- Create `scripts/example.py` demonstrating full pipeline
- Add docstrings to all public functions
- Test PyPI publishing workflow (TestPyPI first)

---

## Technical Notes

### Balance Statistics (for Love Plot)

**Standardized Mean Difference (SMD):**
```
SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)
```

**Variance Ratio:**
```
VR = var_treated / var_control
```

For adjusted statistics, compute within matched strata only.

### LSH Bucketing Strategy

The current code uses 4 bucket levels with progressively smaller bucket lengths:
- Level 1: `bucket_length / 1`
- Level 2: `bucket_length / 4`
- Level 3: `bucket_length / 16`
- Level 4: `bucket_length / 64`

This allows adaptive selection based on bucket sizes.

### Offline .zip Distribution

The .zip should contain only the `brpmatch/` package code. All dependencies (pyspark, numpy, scipy, sklearn, pandas, matplotlib) are either:
- Assumed installed (pyspark)
- Compiled with C extensions (cannot be bundled portably)

Users import via:
```python
import sys
sys.path.insert(0, "/path/to/brpmatch.zip")
from brpmatch import generate_features, match
```

---

## Instructions for Implementation Agent

When developing a detailed implementation plan from this document:

1. **Read the existing code first**: Thoroughly review `pipeline.py`, `pipeline.sql`, and `pipeline.R` to understand the exact logic being refactored.

2. **Preserve algorithm correctness**: The LSH bucketing, k-NN matching, and greedy 1-to-1 assignment logic must be preserved exactly. Do not simplify or "improve" the algorithm without explicit approval.

3. **Handle edge cases**: The current code has logic for:
   - Empty buckets
   - Buckets with only treated or only control patients
   - Mahalanobis distance with singular covariance matrices (uses pseudo-inverse)

4. **Maintain Spark best practices**:
   - Avoid collecting large DataFrames to driver
   - Use broadcast joins where appropriate
   - Keep Pandas UDF processing within partitions

5. **For the love plot conversion**:
   - The R code uses `cobalt::bal.tab()` which computes SMD and variance ratios
   - Implement these statistics manually in Python
   - Match the visual style: point plot with before/after comparison

6. **Testing priorities**:
   - Feature generation: Verify one-hot encoding, imputation, scaling produce expected shapes
   - Matching: Verify 1-to-1 constraint is satisfied, distances are reasonable
   - Love plot: Verify balance statistics match expected values for lalonde dataset

7. **Keep changes minimal**: Only implement what's specified. Don't add logging, don't add CLI interfaces, don't add configuration files unless explicitly requested.
