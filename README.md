# BRPMatch

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

Please see the [example.py](example/example.py) and example outputs in the [example](example) directory.

## Requirements

- **Python** >= 3.12
- **Java** (for PySpark)
  - Local development: Java 17 recommended (`brew install openjdk@17`)
  - Managed environments: should be handled by the hosting environment

Python packages (installed automatically):
- PySpark >= 3.5
- PyArrow >= 15.0
- NumPy >= 1.21
- SciPy >= 1.7
- scikit-learn >= 1.0
- pandas >= 1.3
- matplotlib >= 3.5

## Changelog

### 1.0
- Initial release: Euclidean and Mahalanobis distance, 1-to-k with and without replacement, exact matching requirements, supporting numeric, categorical, and date (converted to numeric) input features.