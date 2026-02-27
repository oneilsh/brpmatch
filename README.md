# BRPMatch
Shawn T. O'Neil, UNC Chapel Hill, Evan French, VCU

Large-scale nearest-neighbor patient cohort matching on Apache Spark via Bucketed Random Projection.

## Features

- Built for Apache Spark distributed processing of large datasets
- Euclidean or Mahalanobis distance matching
- 1-to-k matching with or without replacement
- Exact matching on categorical variables
- Love plot and tabular covariate balance assessment

## Installation

```bash
pip install brpmatch

# Or from source
git clone https://github.com/oneilsh/brpmatch.git
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

Please see the [example.py](example/example.py) and example outputs in the [example](example) directory. Here's an example, 
assuming an active Spark environment and the [lalonde](https://search.r-project.org/CRAN/refmans/designmatch/html/lalonde.html) data in a Spark dataframe called `lalonde_spark_df`:

```python
from brpmatch import generate_features, match, match_summary

# generate features from input spark dataframe
features_df = generate_features(
    spark,
    lalonde_spark_df,
    id_col="id",
    treatment_col="treat",
    treatment_value="1",
    categorical_cols=["race", "married", "nodegree"],
    numeric_cols=["age", "educ", "re74", "re75"],
)

# perform matching; here Mahalanobis 2-to-1 with replacement
units, pairs, bucket_stats = match(
    features_df,
    feature_space = "mahalanobis",
    ratio_k = 2,
    with_replacement = True
)

# assess balance via dataframe summaries (Pandas) and Love plot (Matplotlib)
balance_pandas_df, summary_pandas_df, fig = match_summary(
    features_df,
    units,
    sample_frac=1.0
)

# filter out unmatched patients from units (weight == 0)
units_filtered = units.filter(units.weight > 0)

# re-join to input data and optionally collect to Pandas for analysis
lalonde_matched_pandas = (
    lalonde_spark_df
       .join(ununits_filteredits, lalonde_spark_df.id == units_filtered.id, "inner")
       .drop(units_filtered.id)
       .orderBy("subclass")
       .toPandas()
)
```

## Requirements

- **Python** >= 3.10
- **PySpark** >= 3.5
- **Java** (for PySpark)
  - Local development: Java 17 recommended (`brew install openjdk@17`)
  - Managed environments: should be handled by the hosting environment

## Changelog

### 1.0
- Initial release: Euclidean and Mahalanobis distance, 1-to-k with and without replacement, exact matching requirements, supporting numeric, categorical, and date (converted to numeric) input features.

git test