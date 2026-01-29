# BRPMatch

Large-scale distance-based cohort matching on Apache Spark.

## Features

- **Scalable**: Uses Apache Spark for distributed processing of large datasets
- **Flexible**: Supports categorical, numeric, and date features
- **Advanced matching**: LSH-based bucketing with k-NN matching within buckets
- **Distance metrics**: Euclidean and Mahalanobis distance
- **Ratio matching**: 1-to-k matching with or without replacement
- **Exact matching**: Pre-stratification on categorical variables
- **Visualization**: Love plots for covariate balance assessment

## Installation

```bash
pip install brpmatch

# Or from source
git clone https://github.com/yourusername/brpmatch.git
cd brpmatch && poetry install
```

### Managed Environments

If you are working in a managed environment where you cannot install packages but can upload files,
the `dist` directory contains `.zip` archives of the package for import. Upload the archive to the
working directory of your environment, insert it into `sys.path`, and distribute it to spark executors
for use as well:

```
import sys

sys.path.insert(0, "brpmatch.zip")
import brpmatch

spark.sparkContext.addPyFile("brpmatch.zip")
```

## Quick Start

You will need a *Spark* data frame with columns indicating:

*. An ID column indicating individual patients, one per row (e.g. `"person_id"`).
*. The treatment, which should have two distinct values defining the cohorts that need matching (e.g. a `cohort` column with values `"treated"` and `"control"`).
*. The treatment value, defining which cohort needs matches (e.g. `"treated"`).
*. Lists of numeric (e.g. `age`, `bmi`), categorical (e.g. `state` or boolean `smoker`), and/or date columns (e.g. `date_of_vaccination`) to use as matching features. Dates are converted to numeric represenation as days from a reference date (default `1/1/1970`; as these are internally normalized the reference date typically does not matter).

You may also optionally include categorical columns to enforce exact matching on via `exact_cols`.

We start by generating features, which performs one-hot encoding for categorical features, date-to-numeric transformation. 

```python
from brpmatch import generate_features, match, match_summary

# assuming df contains columns for 

# 1. Generate features
features_df = generate_features(
    spark, df,
    id_col = "person_id",
    treatment_col="cohort",           # Column with treatment/control labels
    treatment_value="treated",        # Value indicating treated group
    categorical_cols=["state", "smoker"],
    numeric_cols=["age", "bmi"],
    date_cols=["date_of_vaccination"]
)
```

The resulting features dataframe includes these transformed features with derived column names.
These features are not normalized (normalized versions will be derived for matching), so they may be useful
for your own modeling purposes as well. It also includes a `features` column with the same information as an array column.

Next we perform the matching. To start we'll use the default, 1-to-1 nearest-neighbor matching in Euclidean space.

```python
# 2. Match treated to controls
matched_df = match(features_df)
```

The resulting matched dataframe contains information about how patients 

```python
# 3. Assess balance (prints summary table and optionally generates love plot)
summary, fig = match_summary(features_df, matched_df, plot=True)
fig.savefig("balance.png")
```

## Usage Examples

### Distance Metrics

Euclidean distance (default) works well when features are on similar scales.
Mahalanobis distance accounts for feature correlations.

```python
# Euclidean distance (default)
matched_df = match(features_df, feature_space="euclidean")

# Mahalanobis distance - whitens features to account for correlations
matched_df = match(features_df, feature_space="mahalanobis")
```

### Ratio Matching (1-to-k)

Match each treated patient to multiple controls for increased statistical power.

```python
# 1:3 matching without replacement (default)
# Each control matched to at most one treated patient
matched_df = match(features_df, ratio_k=3)

# 1:3 matching with replacement
# Controls can be reused across treated patients
matched_df = match(features_df, ratio_k=3, with_replacement=True)

# Limit control reuse
matched_df = match(features_df, ratio_k=3, with_replacement=True, reuse_max=5)

# Only keep treated patients who got exactly k matches
matched_df = match(features_df, ratio_k=3, require_k=True)
```

### Exact Matching (Pre-Stratification)

Force matches within groups defined by categorical variables.

```python
features_df = generate_features(
    spark, df,
    treatment_col="cohort",
    treatment_value="treated",
    categorical_cols=["state", "smoker"],
    numeric_cols=["age", "bmi"],
    exact_match_cols=["gender"],  # Match within same gender only
)
```

### Date Features

Convert date columns to numeric (days from reference date) for matching.

```python
features_df = generate_features(
    spark, df,
    treatment_col="cohort",
    treatment_value="treated",
    date_cols=["diagnosis_date", "enrollment_date"],
    date_reference="2020-01-01",  # Default: "1970-01-01"
)
```

### Assessing Balance

The `match_summary()` function computes balance statistics and optionally generates a love plot.

```python
# Get balance statistics only
summary = match_summary(features_df, matched_df)

# Get statistics and love plot
summary, fig = match_summary(features_df, matched_df, plot=True)

# For large datasets, sample for faster computation
summary, fig = match_summary(features_df, matched_df, plot=True, sample_frac=0.05)
```

Balance statistics include:
- **SMD (Standardized Mean Difference)**: Goal is |SMD| < 0.1
- **Variance Ratio**: Goal is VR between 0.5 and 2.0
- **eCDF statistics**: Mean and max empirical CDF differences

### Creating Analysis-Ready Datasets

Use `match_data()` to create weighted datasets for outcome analysis.

```python
from brpmatch import match_data

# Create dataset with ATT weights for outcome analysis
analysis_df = match_data(df, matched_df, id_col="person_id")

# Columns added:
#   - weights: ATT estimation weights (0 for unmatched)
#   - subclass: Match group identifier
#   - matched: Boolean indicating if row was matched

# Use for weighted regression
matched_only = analysis_df.filter("matched = true")
```

## Spark Environments

### Local Mode (Development)

Installing `pyspark` includes everything needed. No separate cluster required.

```python
spark = SparkSession.builder.master("local[*]").getOrCreate()
```

For Java 17+, add security manager configuration:

```python
spark = (
    SparkSession.builder
    .master("local[*]")
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
    .getOrCreate()
)
```

### Managed Environments (Databricks, EMR, etc.)

The `spark` session is already configured. Just import and use.

```python
# spark is already available
from brpmatch import generate_features, match, match_summary

df = spark.table("database.patients")
features_df = generate_features(spark, df, ...)
matched_df = match(features_df)
```

### Performance Tips

```python
# Cache feature DataFrame if reusing
features_df = generate_features(...).cache()

# Larger bucket_length = faster matching, potentially lower quality
matched_df = match(features_df, bucket_length=3.0)

# Sample for visualization on large datasets
summary, fig = match_summary(features_df, matched_df, plot=True, sample_frac=0.01)
```

## Algorithm Details

### LSH Bucketing

BRPMatch uses 4 levels of Locality-Sensitive Hashing with progressively smaller bucket lengths:
- Level 1: `bucket_length / 1`
- Level 2: `bucket_length / 4`
- Level 3: `bucket_length / 16`
- Level 4: `bucket_length / 64`

The algorithm adaptively selects the finest bucket level that keeps bucket size manageable.

### k-NN Matching

Within each bucket, k-NN finds the k most similar controls for each treated patient. Greedy matching ensures unique assignments (without replacement) or tracks reuse (with replacement).

### Balance Statistics

- **SMD**: `(mean_treated - mean_control) / pooled_std`
- **Variance Ratio**: `var_treated / var_control`

Both computed before (unadjusted) and after (adjusted) matching.

## API Summary

### `generate_features()`

Converts patient data into feature vectors for matching.

```python
features_df = generate_features(
    spark,                              # Active Spark session
    df,                                 # Input DataFrame
    treatment_col="cohort",             # Column with treatment labels
    treatment_value="treated",          # Value indicating treated group
    categorical_cols=["a", "b"],        # One-hot encoded
    numeric_cols=["x", "y"],            # Cast to double (must not contain nulls)
    date_cols=["d"],                    # Converted to days from reference
    exact_match_cols=["gender"],        # Pre-stratification groups
    date_reference="1970-01-01",        # Reference date for date conversion
    id_col="person_id",                 # Patient identifier
)
```

At least one of `categorical_cols`, `numeric_cols`, or `date_cols` must be provided.

### `match()`

Performs LSH-based distance matching between treated and control cohorts.

```python
matched_df = match(
    features_df,                        # Output from generate_features()
    feature_space="euclidean",          # "euclidean" or "mahalanobis"
    n_neighbors=5,                      # Neighbors to consider per bucket
    ratio_k=1,                          # Controls per treated patient
    with_replacement=False,             # Allow control reuse
    reuse_max=None,                     # Max reuse per control (None = unlimited)
    require_k=False,                    # Require exactly k matches
    bucket_length=None,                 # LSH bucket length (auto-computed if None)
    num_hash_tables=4,                  # Number of hash tables
)
```

### `match_summary()`

Generates balance statistics and optional love plot.

```python
# Statistics only
summary = match_summary(features_df, matched_df)

# Statistics and plot
summary, fig = match_summary(
    features_df,
    matched_df,
    sample_frac=0.05,                   # Sample fraction for large datasets
    plot=True,                          # Generate love plot
    figsize=(10, 12),                   # Figure size
    verbose=True,                       # Print summary table
)
```

### `stratify_for_plot()`

Prepares matched data with strata identifiers.

```python
stratified_df = stratify_for_plot(features_df, matched_df)
# Adds: is_matched, strata columns
```

### `love_plot()`

Generates love plot directly from stratified data.

```python
fig = love_plot(
    stratified_df,                      # Output from stratify_for_plot()
    sample_frac=0.05,                   # Sample fraction
    figsize=(10, 12),                   # Figure size
)
```

### `match_data()`

Creates weighted dataset for outcome analysis.

```python
analysis_df = match_data(
    original_df,                        # Original input DataFrame
    matched_df,                         # Output from match()
    id_col="person_id",                 # ID column name
)
# Adds: weights, subclass, matched columns
```

## Development

```bash
make dev          # Install development dependencies
make test         # Run tests
make test-quick   # Run tests in fast-fail mode
make example      # Run example pipeline with LaLonde dataset
make build        # Build package
make zip          # Create offline zip distribution
make clean        # Clean build artifacts
```

## Requirements

- **Python** >= 3.12
- **Java** (for PySpark)
  - Local development: Java 17 recommended (`brew install openjdk@17`)
  - Managed environments: Handled automatically

Python packages (installed automatically):
- PySpark >= 3.5
- PyArrow >= 15.0
- NumPy >= 1.21
- SciPy >= 1.7
- scikit-learn >= 1.0
- pandas >= 1.3
- matplotlib >= 3.5

## Changelog

### 1.0.0
- Initial release
- LSH-based matching with Euclidean and Mahalanobis distance
- 1-to-k matching with and without replacement
- Exact match pre-stratification
- Love plot visualization
- Balance statistics (SMD, VR, eCDF)

## License

MIT License

## Contributing

Contributions welcome. Please open an issue or submit a pull request.
