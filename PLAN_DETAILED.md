# BRPMatch Detailed Implementation Plan

This document provides a step-by-step implementation plan for refactoring BRPMatch from a Foundry-based Spark cohort matching tool into a standalone Python package.

---

## Phase 1: Project Structure Setup

### Step 1.1: Create Package Directory Structure

Create the following directories:
```
brpmatch/brpmatch/
brpmatch/tests/
brpmatch/tests/data/       # lalonde.csv already here
brpmatch/scripts/
```

### Step 1.2: Update pyproject.toml

**File:** `/Users/oneilsh/Documents/projects/tislab/n3c/brpmatch/pyproject.toml`

**Changes required:**
- Line 9: Change `requires-python = ">=3.13"` to `requires-python = ">=3.10"`
- Line 10: Replace empty `dependencies = []` with full dependency list
- Add dev dependencies section
- Add pytest configuration

**Target content:**
```toml
[project]
name = "brpmatch"
version = "0.1.0"
description = "Large-scale distance-based cohort matching on Apache Spark"
authors = [
    {name = "Shawn T O'Neil", email = "shawn@tislab.org"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "pyspark>=3.3",
    "numpy>=1.21",
    "scipy>=1.7",
    "scikit-learn>=1.0",
    "pandas>=1.3",
    "matplotlib>=3.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "build>=0.10",
    "twine>=4.0",
]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### Step 1.3: Update Makefile

**File:** `/Users/oneilsh/Documents/projects/tislab/n3c/brpmatch/Makefile`

**Add these targets after the `clean:` target (line 22):**

```makefile
test:
	poetry run pytest tests/ -v

test-quick:
	poetry run pytest tests/ -v -x --tb=short

zip: clean build
	mkdir -p dist
	cd brpmatch && zip -r ../dist/brpmatch.zip . -x "*.pyc" -x "__pycache__/*" -x "*.egg-info/*"
	@echo "Created dist/brpmatch.zip"
```

---

## Phase 2: Feature Generation Module

### Step 2.1: Create brpmatch/features.py

**Source:** Extract from `pipeline.py` lines 31-174

**Function signature:**
```python
from typing import List, Optional
from pyspark.sql import SparkSession, DataFrame

def generate_features(
    spark: SparkSession,
    df: DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    treatment_col: str,
    treatment_value: str,
    date_cols: Optional[List[str]] = None,
    exact_match_cols: Optional[List[str]] = None,
    gbt_impute_cols: Optional[List[str]] = None,
    date_reference: str = "2018-01-01",
    id_col: str = "person_id",
) -> DataFrame:
    """
    Convert patient data into feature vectors for matching.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    df : DataFrame
        Input DataFrame with patient data
    categorical_cols : List[str]
        Columns to treat as categorical (will be one-hot encoded)
    numeric_cols : List[str]
        Columns to treat as numeric (will be mean-imputed)
    treatment_col : str
        Column name containing treatment/cohort indicator
    treatment_value : str
        Value in treatment_col that indicates "treated" group
    date_cols : Optional[List[str]]
        Date columns to convert to numeric (days from date_reference)
    exact_match_cols : Optional[List[str]]
        Categorical columns to use for exact matching stratification
    gbt_impute_cols : Optional[List[str]]
        Numeric/date columns to impute using Gradient Boosted Trees
    date_reference : str
        Reference date for converting date columns to numeric
    id_col : str
        Patient identifier column name

    Returns
    -------
    DataFrame
        DataFrame with:
        - 'features' column (scaled feature vector)
        - 'treat' column (1 for treated, 0 for control)
        - 'exact_match_id' column (for stratification)
        - Original columns with suffixes (_index, _imputed, etc.)
    """
```

**Required imports:**
```python
from typing import List, Optional

import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame, SparkSession
```

**Key modifications from original code:**

1. **Remove Foundry imports and decorator** (lines 3-9):
   - Delete `@transform_pandas` decorator
   - Delete `from foundry_ml import Model, Stage`

2. **Parameterize hardcoded values** (lines 42-58):
   - Replace `cohort_column = 'gender_concept_name'` with `treatment_col` parameter
   - Replace `source_cohort = "MALE"` with `treatment_value` parameter
   - Replace hardcoded column lists with parameters

3. **Treatment column creation** (line 159):
   - Change: `F.when(F.col("gender_concept_name")==source_cohort, 1)`
   - To: `F.when(F.col(treatment_col)==treatment_value, 1)`

4. **Handle type conversion for treatment comparison**:
   - Convert `treatment_value` to match column type before comparison

**Algorithm to preserve (from lines 60-174):**
1. Cast categorical columns to string
2. Convert date columns to days-since-reference
3. Create exact_match_id by concatenating exact_match_cols
4. Build ML Pipeline with StringIndexer → OneHotEncoder → Imputer → VectorAssembler
5. Optionally apply GBT imputation for specified columns
6. Apply StandardScaler to final features
7. Create treat column (1=treated, 0=control)

---

## Phase 3: Matching Module

### Step 3.1: Create brpmatch/matching.py

**Source:** Extract from `pipeline.py` lines 176-481

**Critical fixes required:**
1. **Remove early return at line 290:** `return persons_bucketed`
2. **Remove early return at line 303:** `return persons_bucketed`
3. **Remove early return at line 460:** `return all_persons_bucketed`
4. **Remove limit() at line 203:** `persons_features = brp_generate_features.limit(100000)`

**Function signature:**
```python
from typing import Literal, Optional
from pyspark.sql import DataFrame

def match(
    features_df: DataFrame,
    distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean",
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

    Parameters
    ----------
    features_df : DataFrame
        Output from generate_features() containing 'features' column
    distance_metric : Literal["euclidean", "mahalanobis"]
        Distance metric for k-NN within buckets
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
        - match_distance: distance between matched pair
        - bucket_num_input_patients: size of bucket where match was found
        - bucket_seconds: time to process bucket
    """
```

**Required imports:**
```python
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from numpy.linalg import pinv
from pyspark.ml import Pipeline
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.functions import vector_to_array
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors
```

**Critical algorithm sections to preserve:**

1. **LSH Bucketing (lines 243-261):**
   - 4 levels with bucket lengths: `[base, base/4, base/16, base/64]`
   - Uses `BucketedRandomProjectionLSH` with `numHashTables=4`
   - Bucket IDs combine `exact_match_id` with hash values

2. **Adaptive bucket selection (lines 305-322):**
   - Selects finest bucket level that keeps count < `num_patients_trigger_rebucket`
   - Falls back to coarsest if all buckets too large

3. **Viable bucket filtering (lines 324-336):**
   - Only keeps buckets with both treated AND control patients

4. **Mahalanobis distance (lines 341-353):**
   - Computes pseudoinverse of covariance matrix globally
   - Uses `scipy.spatial.distance.mahalanobis`
   - Uses `numpy.linalg.pinv` for singular matrices

5. **Pandas UDF for k-NN (lines 365-440):**
   - `find_neighbors()` processes each bucket partition
   - Uses `sklearn.neighbors.NearestNeighbors`
   - Handles empty cohorts gracefully

6. **Greedy 1-to-1 matching (lines 419-433):**
   ```
   Ranks candidates by source->target distance, then target->source for ties
   Iteratively:
     1. Take row i (best remaining source->target match)
     2. Remove all rows where XOR(person_id==i, match_id==j) is True
     3. Increment i
   This ensures 1-to-1 matching with greedy distance minimization
   ```

---

## Phase 4: Stratification Module

### Step 4.1: Create brpmatch/stratify.py

**Source:** Convert `pipeline.sql` (46 lines) to PySpark DataFrame API

**Function signature:**
```python
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

def stratify_for_plot(
    features_df: DataFrame,
    matched_df: DataFrame,
    id_col: str = "person_id",
    match_id_col: str = "match_person_id",
) -> DataFrame:
    """
    Prepare matched data for love plot visualization.

    Joins features with match information to create strata identifiers
    for computing balance statistics.

    Parameters
    ----------
    features_df : DataFrame
        Output from generate_features()
    matched_df : DataFrame
        Output from match()
    id_col : str
        Patient identifier column in features_df
    match_id_col : str
        Matched patient identifier column in matched_df

    Returns
    -------
    DataFrame
        Features DataFrame with additional columns:
        - is_matched: non-null if patient is part of a matched pair
        - strata: unique identifier for each matched pair (format: "id:match_id")
    """
```

**Implementation (converts SQL logic):**
```python
def stratify_for_plot(
    features_df: DataFrame,
    matched_df: DataFrame,
    id_col: str = "person_id",
    match_id_col: str = "match_person_id",
) -> DataFrame:
    import pyspark.sql.functions as F

    # Create strata identifier for each match pair
    with_strata = matched_df.withColumn(
        "strata",
        F.concat_ws(":", F.col(id_col), F.col(match_id_col))
    )

    # Join to identify treated patients who are matched
    matched_from = features_df.join(
        with_strata.select(
            F.col(id_col),
            F.col(id_col).alias("matched_from"),
            F.col("strata").alias("strata_from")
        ),
        on=id_col,
        how="left"
    )

    # Join to identify control patients who are matched
    matched_to = matched_from.join(
        with_strata.select(
            F.col(match_id_col),
            F.col(match_id_col).alias("matched_to"),
            F.col("strata").alias("strata_to")
        ),
        matched_from[id_col] == with_strata[match_id_col],
        how="left"
    ).drop(match_id_col)

    # Coalesce to single is_matched and strata columns
    result = matched_to.withColumn(
        "is_matched",
        F.coalesce(F.col("matched_from"), F.col("matched_to"))
    ).withColumn(
        "strata",
        F.coalesce(F.col("strata_from"), F.col("strata_to"))
    ).drop("matched_from", "matched_to", "strata_from", "strata_to")

    return result
```

---

## Phase 5: Love Plot Module

### Step 5.1: Create brpmatch/loveplot.py

**Source:** Convert `pipeline.R` (88 lines) to Python/matplotlib

**Function signature:**
```python
from typing import List, Optional, Tuple
from pyspark.sql import DataFrame
import matplotlib.pyplot as plt
import matplotlib.figure

def love_plot(
    stratified_df: DataFrame,
    treatment_col: str = "treat",
    strata_col: str = "strata",
    categorical_suffix: str = "_index",
    numeric_suffix: str = "_imputed",
    sample_frac: float = 0.05,
    figsize: Tuple[int, int] = (10, 12),
    feature_cols: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """
    Generate a love plot showing covariate balance before and after matching.

    Parameters
    ----------
    stratified_df : DataFrame
        Output from stratify_for_plot()
    treatment_col : str
        Column indicating treatment (1) vs control (0)
    strata_col : str
        Column identifying matched pairs
    categorical_suffix : str
        Suffix identifying categorical feature columns
    numeric_suffix : str
        Suffix identifying numeric feature columns
    sample_frac : float
        Fraction of data to sample for plotting (for large datasets)
    figsize : Tuple[int, int]
        Figure size (width, height) in inches
    feature_cols : Optional[List[str]]
        Specific columns to include. If None, auto-detect by suffix.

    Returns
    -------
    matplotlib.figure.Figure
        Love plot figure with two panels:
        - Left: Absolute Standardized Mean Difference
        - Right: Variance Ratio
    """
```

### Step 5.2: Balance Statistics Formulas

**Standardized Mean Difference (SMD):**
```python
def compute_smd(treated_values: np.ndarray, control_values: np.ndarray) -> float:
    """Compute standardized mean difference."""
    mean_t = np.nanmean(treated_values)
    mean_c = np.nanmean(control_values)
    var_t = np.nanvar(treated_values, ddof=1)
    var_c = np.nanvar(control_values, ddof=1)
    pooled_std = np.sqrt((var_t + var_c) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean_t - mean_c) / pooled_std
```

**Variance Ratio (VR):**
```python
def compute_variance_ratio(treated_values: np.ndarray, control_values: np.ndarray) -> float:
    """Compute variance ratio."""
    var_t = np.nanvar(treated_values, ddof=1)
    var_c = np.nanvar(control_values, ddof=1)
    if var_c == 0:
        return np.inf if var_t > 0 else 1.0
    return var_t / var_c
```

### Step 5.3: Helper Functions

```python
def _identify_feature_columns(
    columns: List[str],
    categorical_suffix: str,
    numeric_suffix: str,
) -> List[str]:
    """Identify feature columns by suffix."""
    features = []
    for col in columns:
        if col.endswith(categorical_suffix) or col.endswith(numeric_suffix):
            features.append(col)
    return features


def _compute_balance_stats(
    pdf: pd.DataFrame,
    feature_cols: List[str],
    treatment_col: str,
    strata_col: str,
) -> pd.DataFrame:
    """
    Compute balance statistics for all features.

    Returns DataFrame with columns:
    - covariate: feature name
    - smd_unadjusted: SMD on all data
    - smd_adjusted: SMD on matched pairs only
    - vr_unadjusted: Variance ratio on all data
    - vr_adjusted: Variance ratio on matched pairs only
    """
    results = []

    treated = pdf[pdf[treatment_col] == 1]
    control = pdf[pdf[treatment_col] == 0]

    matched = pdf[pdf[strata_col].notna()]
    treated_matched = matched[matched[treatment_col] == 1]
    control_matched = matched[matched[treatment_col] == 0]

    for col in feature_cols:
        smd_un = compute_smd(treated[col].values, control[col].values)
        vr_un = compute_variance_ratio(treated[col].values, control[col].values)
        smd_adj = compute_smd(treated_matched[col].values, control_matched[col].values)
        vr_adj = compute_variance_ratio(treated_matched[col].values, control_matched[col].values)

        results.append({
            "covariate": col,
            "smd_unadjusted": smd_un,
            "smd_adjusted": smd_adj,
            "vr_unadjusted": vr_un,
            "vr_adjusted": vr_adj,
        })

    return pd.DataFrame(results)
```

---

## Phase 6: Testing

### Step 6.1: Create tests/conftest.py

```python
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    session = (
        SparkSession.builder
        .master("local[*]")
        .appName("brpmatch-test")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="session")
def lalonde_df(spark):
    """Load the lalonde test dataset."""
    import os
    data_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "lalonde.csv"
    )
    return spark.read.csv(data_path, header=True, inferSchema=True)
```

### Step 6.2: Lalonde Dataset Structure

**File:** `tests/data/lalonde.csv`

**Columns:**
- `id`: Patient identifier (e.g., "NSW1", "PSID1")
- `treat`: Treatment indicator (1 = treated/NSW, 0 = control/PSID)
- `age`: Age in years (numeric, range 16-55)
- `educ`: Years of education (numeric, range 0-18)
- `race`: Race category (categorical: "black", "hispan", "white")
- `married`: Married indicator (binary: 0 or 1)
- `nodegree`: No high school degree indicator (binary: 0 or 1)
- `re74`: Real earnings in 1974 (numeric, many zeros)
- `re75`: Real earnings in 1975 (numeric, many zeros)
- `re78`: Real earnings in 1978 (numeric, outcome variable)

**Size:** 614 rows (185 treated, 429 control)

### Step 6.3: Test Cases for test_features.py

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_basic_feature_generation` | Generate features with lalonde data | DataFrame has 'features', 'treat', 'exact_match_id' columns |
| `test_feature_vector_dimensions` | Check feature vector size | dim = n_categorical_onehot + n_numeric |
| `test_exact_match_cols` | Test exact matching stratification | Distinct exact_match_id values = distinct values of exact_match_cols |
| `test_treatment_column_creation` | Verify treat column | 185 treated, 429 control |
| `test_numeric_imputation` | Test null handling | No nulls in *_imputed columns |

### Step 6.4: Test Cases for test_matching.py

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_basic_matching` | Match with euclidean distance | Output has id, match_id, match_distance columns |
| `test_one_to_one_constraint` | Verify 1-to-1 matching | Each id and match_id appears at most once |
| `test_match_count_bounded` | Check match count | count <= min(treated, control) = 185 |
| `test_distances_positive` | Verify distances | All match_distance >= 0 |
| `test_mahalanobis_distance` | Match with Mahalanobis | Produces valid matches (count > 0) |

### Step 6.5: Test Cases for test_loveplot.py

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_love_plot_returns_figure` | Generate plot | Returns matplotlib.figure.Figure |
| `test_balance_statistics_computation` | Compute balance stats | One row per feature; most features show improvement |
| `test_smd_calculation` | SMD with known values | SMD = 1.0 for mean_diff=2, pooled_std=2 |
| `test_variance_ratio_calculation` | VR with known values | VR > 1 when treated has higher variance |

---

## Phase 7: Package Finalization

### Step 7.1: Create brpmatch/__init__.py

```python
"""
BRPMatch: Large-scale distance-based cohort matching on Apache Spark.
"""

from .features import generate_features
from .matching import match
from .stratify import stratify_for_plot
from .loveplot import love_plot

__version__ = "0.1.0"
__all__ = [
    "generate_features",
    "match",
    "stratify_for_plot",
    "love_plot",
]
```

### Step 7.2: Create scripts/example.py

Example script demonstrating full pipeline with lalonde data:
1. Load CSV
2. Generate features
3. Perform matching
4. Stratify for plot
5. Generate and save love plot

---

## Known Issues and Decisions

| Issue | Resolution |
|-------|------------|
| Treatment column type mismatch | Convert treatment_value to match column dtype before comparison |
| Early returns in pipeline.py | Remove lines 290, 303, 460 |
| Limit call in matching | Remove line 203 |
| Hardcoded column names in SQL | Parameterize with id_col, derive match_{id_col} |
| SparkSession management | Require as parameter to generate_features(); derive from DataFrame for other functions |
| Large dataset memory for love plot | Use sample_frac parameter (default 0.05) |
| Categorical vs numeric column detection | Use naming convention: _index for categorical, _imputed for numeric |

---

## Implementation Order

1. Phase 1: Project structure (directories, pyproject.toml, Makefile)
2. Phase 2: features.py
3. Phase 6.1-6.3: conftest.py and test_features.py
4. Phase 3: matching.py
5. Phase 6.4: test_matching.py
6. Phase 4: stratify.py
7. Phase 5: loveplot.py
8. Phase 6.5: test_loveplot.py
9. Phase 7: __init__.py, example.py

---

## Source File Reference

| Target File | Source | Lines |
|-------------|--------|-------|
| brpmatch/features.py | pipeline.py | 31-174 |
| brpmatch/matching.py | pipeline.py | 176-481 |
| brpmatch/stratify.py | pipeline.sql | 1-46 |
| brpmatch/loveplot.py | pipeline.R | 1-88 |
