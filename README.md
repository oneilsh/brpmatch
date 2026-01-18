# BRPMatch

Large-scale distance-based cohort matching on Apache Spark.

BRPMatch is a Spark-based cohort matching tool that uses distance-based methods (LSH + k-NN) for large-scale propensity score-like matching.

## Features

- **Scalable**: Uses Apache Spark for distributed processing of large datasets
- **Flexible**: Supports categorical, numeric, and date features
- **Advanced matching**: LSH-based bucketing with k-NN matching within buckets
- **Distance metrics**: Euclidean and Mahalanobis distance supported
- **Visualization**: Generates love plots to assess covariate balance
- **GBT imputation**: Optional gradient boosted tree imputation for missing values

## Installation

```bash
# Install from PyPI (when published)
pip install brpmatch

# Install from source
git clone https://github.com/yourusername/brpmatch.git
cd brpmatch
poetry install
```

## Quick Start

```python
from pyspark.sql import SparkSession
from brpmatch import generate_features, match, stratify_for_plot, love_plot

# Create Spark session (with Java 17+ compatibility)
spark = (
    SparkSession.builder
    .appName("matching")
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
    .getOrCreate()
)

# Load your data
df = spark.read.csv("data.csv", header=True, inferSchema=True)

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

## Usage in Managed Spark Environments

If you're working in an environment where Spark is already running (Databricks, AWS EMR, Palantir Foundry, etc.), you can use the existing `spark` session directly. No need to create or configure a SparkSession.

### Databricks Example

```python
# spark is already available as a global variable in Databricks notebooks
from brpmatch import generate_features, match, stratify_for_plot, love_plot

# Load your data (spark is already configured)
df = spark.table("your_database.patient_cohorts")

# Or read from files
df = spark.read.parquet("/mnt/data/cohorts/")

# Use BRPMatch functions directly - they work with the existing spark session
features_df = generate_features(
    spark,  # Use the existing spark session
    df,
    categorical_cols=["state", "race", "smoker"],
    numeric_cols=["age", "bmi", "baseline_glucose"],
    treatment_col="treatment_group",
    treatment_value="intervention",
    exact_match_cols=["study_site"],  # Match within same site
)

matched_df = match(features_df, n_neighbors=5)
stratified_df = stratify_for_plot(features_df, matched_df)

# Generate and display plot in notebook
fig = love_plot(stratified_df, sample_frac=0.1)
display(fig)  # Databricks display function
```

### AWS EMR / Jupyter Notebook Example

```python
# spark is typically available via %%configure magic or SparkSession.getActiveSession()
from brpmatch import generate_features, match, stratify_for_plot, love_plot

# Get existing session if needed
# spark = SparkSession.getActiveSession()

# Read from S3
df = spark.read.parquet("s3://my-bucket/cohort-data/")

# Run matching pipeline
features_df = generate_features(
    spark,
    df,
    categorical_cols=["region", "diagnosis"],
    numeric_cols=["age", "lab_value"],
    treatment_col="arm",
    treatment_value="1",
)

matched_df = match(features_df)
stratified_df = stratify_for_plot(features_df, matched_df)

# Save plot to S3
fig = love_plot(stratified_df)
fig.savefig("/tmp/balance.png")
# Then copy to S3 using boto3 or aws cli
```

### Palantir Foundry Example

BRPMatch was originally developed for Foundry! Here's how to use the refactored version:

```python
from transforms.api import transform_df, Input, Output
from brpmatch import generate_features, match, stratify_for_plot, love_plot

@transform_df(
    Output("ri.foundry.main.dataset.matched-cohorts"),
    patients=Input("ri.foundry.main.dataset.patient-data"),
)
def compute_matches(patients):
    # Use the spark session from the transform context
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()

    features_df = generate_features(
        spark,
        patients,
        categorical_cols=["state", "smoker"],
        numeric_cols=["age", "bmi"],
        treatment_col="cohort",
        treatment_value="treated",
    )

    matched_df = match(features_df)
    return matched_df


@transform_df(
    Output("ri.foundry.main.dataset.balance-stats"),
    features=Input("ri.foundry.main.dataset.patient-data"),
    matches=Input("ri.foundry.main.dataset.matched-cohorts"),
)
def create_balance_plot(features, matches):
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()

    # Regenerate features (or read from a cached dataset)
    features_df = generate_features(
        spark,
        features,
        categorical_cols=["state", "smoker"],
        numeric_cols=["age", "bmi"],
        treatment_col="cohort",
        treatment_value="treated",
    )

    stratified_df = stratify_for_plot(features_df, matches)

    # For visualization, save plot to a Foundry dataset or export
    # Note: Foundry doesn't directly support matplotlib figures in datasets
    # You'll need to either:
    # 1. Save to a temporary location and upload as a resource
    # 2. Return the stratified_df and create the plot in a notebook
    return stratified_df
```

### Key Points for Managed Environments

1. **No SparkSession configuration needed** - Use the existing session
2. **Data location flexibility** - Read from tables, S3, HDFS, etc.
3. **Scalability** - BRPMatch leverages your cluster's resources automatically
4. **Memory management** - Use `sample_frac` in `love_plot()` for very large datasets
5. **Checkpointing** - Save intermediate results (features_df, matched_df) to avoid recomputation

### Performance Tips

For large datasets (>10M rows):
```python
# Cache feature DataFrame if reusing
features_df = generate_features(...).cache()

# Use larger bucket_length for faster matching (but potentially lower quality)
matched_df = match(features_df, bucket_length=3.0)

# Sample for visualization
fig = love_plot(stratified_df, sample_frac=0.01)  # Use 1% of data
```

## API Reference

### `generate_features()`

Converts patient data into feature vectors for matching.

**Parameters:**
- `spark`: Active Spark session
- `df`: Input DataFrame with patient data
- `categorical_cols`: Columns to one-hot encode
- `numeric_cols`: Columns to treat as numeric (mean imputation)
- `treatment_col`: Column indicating treatment/control status
- `treatment_value`: Value indicating "treated" group
- `date_cols`: Date columns to convert to numeric (optional)
- `exact_match_cols`: Columns for exact matching stratification (optional)
- `gbt_impute_cols`: Columns to impute using GBT (optional)
- `date_reference`: Reference date for date conversion (default: "2018-01-01")
- `id_col`: Patient identifier column (default: "person_id")

**Returns:** DataFrame with features, treatment indicator, and exact match ID

### `match()`

Performs LSH-based distance matching between treated and control cohorts.

**Parameters:**
- `features_df`: Output from generate_features()
- `distance_metric`: "euclidean" or "mahalanobis" (default: "euclidean")
- `n_neighbors`: Number of neighbors to consider (default: 5)
- `bucket_length`: Base LSH bucket length (default: auto-computed)
- `num_hash_tables`: Number of hash tables (default: 4)
- `num_patients_trigger_rebucket`: Threshold for finer bucketing (default: 10000)
- `features_col`: Feature vector column name (default: "features")
- `treatment_col`: Treatment indicator column (default: "treat")
- `id_col`: Patient ID column (default: "person_id")
- `exact_match_col`: Exact match column (default: "exact_match_id")

**Returns:** DataFrame with matched pairs and distances

### `stratify_for_plot()`

Prepares matched data for love plot visualization.

**Parameters:**
- `features_df`: Output from generate_features()
- `matched_df`: Output from match()
- `id_col`: Patient ID column (default: "person_id")
- `match_id_col`: Matched patient ID column (default: "match_person_id")

**Returns:** Features DataFrame with strata identifiers

### `love_plot()`

Generates a love plot showing covariate balance.

**Parameters:**
- `stratified_df`: Output from stratify_for_plot()
- `treatment_col`: Treatment indicator column (default: "treat")
- `strata_col`: Strata identifier column (default: "strata")
- `categorical_suffix`: Suffix for categorical features (default: "_index")
- `numeric_suffix`: Suffix for numeric features (default: "_imputed")
- `sample_frac`: Sampling fraction for large datasets (default: 0.05)
- `figsize`: Figure size (default: (10, 12))
- `feature_cols`: Specific features to plot (optional)

**Returns:** matplotlib Figure object

## Development

```bash
# Install development dependencies
make dev

# Run tests
make test

# Run tests in fast-fail mode
make test-quick

# Run example pipeline with lalonde dataset
make example

# Build package
make build

# Create offline zip distribution
make zip

# Clean build artifacts
make clean
```

The `make example` command runs the complete BRPMatch pipeline on the included LaLonde dataset, demonstrating:
- Feature generation from patient data
- LSH-based matching between treated and control cohorts
- Stratification for visualization
- Love plot generation showing covariate balance

Output: Creates `balance_plot.png` showing standardized mean differences and variance ratios before/after matching.

## Algorithm Details

### LSH Bucketing

BRPMatch uses 4 levels of Locality-Sensitive Hashing with progressively smaller bucket lengths:
- Level 1: `bucket_length / 1`
- Level 2: `bucket_length / 4`
- Level 3: `bucket_length / 16`
- Level 4: `bucket_length / 64`

The algorithm adaptively selects the finest bucket level that keeps bucket size manageable.

### k-NN Matching

Within each bucket, k-NN is used to find the k most similar control patients for each treated patient. Greedy 1-to-1 matching ensures unique matches.

### Balance Statistics

- **Standardized Mean Difference (SMD)**: `(mean_treated - mean_control) / pooled_std`
- **Variance Ratio (VR)**: `var_treated / var_control`

Both statistics are computed before (unadjusted) and after (adjusted) matching.

## Requirements

- Python >= 3.10
- Java 11+ (for PySpark)
- PySpark >= 3.3
- PyArrow >= 15.0
- NumPy >= 1.21
- SciPy >= 1.7
- scikit-learn >= 1.0
- pandas >= 1.3
- matplotlib >= 3.5

**Note for Java 17+**: When using Java 17 or newer, add these configuration options to your SparkSession:
```python
.config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
.config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
```

## License

MIT License

## Citation

If you use BRPMatch in your research, please cite:

```
[Citation details to be added]
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions, please open an issue on GitHub.
