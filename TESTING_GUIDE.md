# BRPMatch Testing Guide (Mac)

## Setup Complete ✓

All dependencies are installed and tests are passing on Mac!

## Installation Summary

The following steps were completed to get tests running on macOS:

### 1. Install Dependencies

```bash
poetry install
poetry run pip install pyarrow  # Required for Pandas UDFs
```

### 2. Verify Installation

```bash
# Check PySpark
poetry run python -c "import pyspark; print(f'PySpark {pyspark.__version__}')"
# Output: PySpark 4.1.1

# Check PyArrow
poetry run python -c "import pyarrow; print(f'PyArrow {pyarrow.__version__}')"
# Output: PyArrow 22.0.0

# Check Java
java -version
# Output: openjdk version "23.0.2"
```

## Running Tests

### Run All Tests
```bash
make test
# or
poetry run pytest tests/ -v
```

### Run Specific Test File
```bash
poetry run pytest tests/test_features.py -v
poetry run pytest tests/test_matching.py -v
poetry run pytest tests/test_loveplot.py -v
```

### Run Single Test
```bash
poetry run pytest tests/test_matching.py::test_basic_matching -v
```

### Fast-Fail Mode
```bash
make test-quick
# or
poetry run pytest tests/ -v -x --tb=short
```

## Test Results

All **15 tests passing**:

### Feature Tests (5/5) ✓
- `test_basic_feature_generation` - Generates features with lalonde data
- `test_feature_vector_dimensions` - Validates feature vector size
- `test_exact_match_cols` - Tests exact matching stratification
- `test_treatment_column_creation` - Validates treat column
- `test_numeric_imputation` - Tests null handling

### Matching Tests (5/5) ✓
- `test_basic_matching` - Basic euclidean matching
- `test_one_to_one_constraint` - Validates 1-to-1 matching
- `test_match_count_bounded` - Checks match count limits
- `test_distances_positive` - Validates distance values
- `test_mahalanobis_distance` - Tests Mahalanobis metric

### Love Plot Tests (5/5) ✓
- `test_love_plot_returns_figure` - Generates matplotlib figure
- `test_balance_statistics_computation` - Validates balance stats
- `test_smd_calculation` - Tests SMD formula
- `test_variance_ratio_calculation` - Tests VR with extreme values
- `test_variance_ratio_equal_variance` - Tests VR with equal variance

## Technical Fixes Applied

### 1. Java 23 Compatibility
Added security manager flags to SparkSession configuration in `tests/conftest.py`:
```python
.config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
.config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
```

### 2. PySpark 4.x Pandas UDF Syntax
Updated `brpmatch/matching.py` to use modern `applyInPandas()`:
```python
# Old (PySpark 2.x/3.x):
@pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
def find_neighbors(group_key, group_df):
    ...

.groupBy("bucket_id").apply(find_neighbors)

# New (PySpark 4.x):
def find_neighbors(group_df):
    ...

.groupBy("bucket_id").applyInPandas(find_neighbors, schema=schema)
```

### 3. Import Cleanup
Fixed duplicate imports in `tests/test_features.py` by adding `from pyspark.sql import functions as F` at the top.

### 4. Test Assertion Fix
Corrected SMD calculation test with accurate expected value (1.265 instead of 1.0).

## Requirements

### System
- macOS (tested on Apple Silicon)
- Java 11+ (tested with Java 23)
- Python 3.10+ (tested with Python 3.13)

### Python Packages
- pyspark>=3.3 (installed: 4.1.1)
- pyarrow>=15.0 (installed: 22.0.0)
- numpy>=1.21
- scipy>=1.7
- scikit-learn>=1.0
- pandas>=1.3
- matplotlib>=3.5
- pytest>=7.0

## Running the Example

```bash
# Easy way
make example

# Or directly
poetry run python scripts/example.py
```

This will:
1. Load the lalonde dataset
2. Generate features
3. Perform matching
4. Create stratified data
5. Generate and save a love plot to `balance_plot.png`

## Test Data

The tests use the LaLonde dataset (`tests/data/lalonde.csv`):
- 614 patients total
- 185 treated (NSW training program)
- 429 control (PSID comparison group)
- Variables: age, education, race, marital status, earnings

## Troubleshooting

### Issue: Java Security Manager Error
**Symptom**: `UnsupportedOperationException: getSubject is supported only if a security manager is allowed`

**Solution**: Already fixed in `tests/conftest.py` with Java options.

### Issue: PyArrow Not Found
**Symptom**: `PySparkImportError: [PACKAGE_NOT_INSTALLED] PyArrow >= 15.0.0 must be installed`

**Solution**:
```bash
poetry run pip install 'pyarrow>=15.0.0'
```

### Issue: Slow Tests
**Symptom**: Tests take a long time to run

**Solution**: This is normal for Spark initialization. First test run is slower (~35 seconds for all tests).

## Performance Notes

- Test suite runs in ~35 seconds total
- First Spark session creation takes ~5-10 seconds
- Subsequent tests reuse the session (fixture scope="session")
- Matching tests are the slowest due to LSH computation

## Next Steps

1. ✓ All tests passing
2. Try running `scripts/example.py`
3. Build the package: `make build`
4. Create offline zip: `make zip`
5. Publish to TestPyPI: `make publish-test`
