"""
Feature generation for BRPMatch cohort matching.

This module converts patient data into feature vectors for matching.
"""

import re
from typing import List, Optional, Tuple

import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession

# Default maximum categories (can be overridden via parameter)
DEFAULT_MAX_CATEGORIES = 20


def _sanitize_value(value: str) -> str:
    """
    Sanitize a categorical value for use in column names.

    - Lowercase
    - Replace spaces and special chars with underscore
    - Collapse multiple underscores
    - Strip leading/trailing underscores
    """
    result = str(value).lower()
    # Replace spaces and problematic characters with underscore
    result = re.sub(r'[\s/\\.\-\(\)\[\]\{\}:;,\'"]+', '_', result)
    # Collapse multiple underscores
    result = re.sub(r'_+', '_', result)
    # Strip leading/trailing underscores
    result = result.strip('_')
    return result


def _create_onehot_columns(
    df: DataFrame,
    col: str,
    suffix: str = "__cat",
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> Tuple[DataFrame, List[str]]:
    """
    Create one-hot encoded columns for a categorical column.

    Args:
        df: Input DataFrame
        col: Categorical column name
        suffix: Suffix for generated columns (__cat or __exact)
        max_categories: Maximum allowed distinct values (raises error if exceeded)

    Returns:
        Tuple of (DataFrame with new columns, list of new column names)

    Raises:
        ValueError: If the column has more distinct values than max_categories
    """
    # Get distinct values (sorted for deterministic ordering)
    distinct_values = [
        row[col] for row in df.select(col).distinct().collect()
        if row[col] is not None
    ]
    distinct_values = sorted([str(v) for v in distinct_values])

    # Check cardinality
    if len(distinct_values) > max_categories:
        raise ValueError(
            f"Column '{col}' has {len(distinct_values)} distinct values, "
            f"which exceeds max_categories={max_categories}. "
            f"High-cardinality categorical columns are not suitable for matching. "
            f"Consider binning the values, using it as a numeric column, "
            f"or increasing max_categories if you're sure this is intended."
        )

    new_cols = []
    for val in distinct_values:
        sanitized = _sanitize_value(val)
        new_col_name = f"{col}_{sanitized}{suffix}"
        df = df.withColumn(
            new_col_name,
            F.when(F.col(col) == val, 1.0).otherwise(0.0)
        )
        new_cols.append(new_col_name)

    return df, new_cols


def generate_features(
    spark: SparkSession,
    df: DataFrame,
    id_col: str,
    treatment_col: str,
    treatment_value: str,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    exact_match_cols: Optional[List[str]] = None,
    date_reference: str = "1970-01-01",
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> DataFrame:
    """
    Convert patient data into feature vectors for matching.

    This function processes input data through multiple transformations:
    1. Converts dates to numeric (days from reference date)
    2. Creates exact match stratification groups
    3. One-hot encodes categorical variables
    4. Creates feature vectors for matching

    Output DataFrame columns use a suffix-based naming convention:
    - {id_col}__id: Patient identifier
    - treat__treat: Treatment indicator (0/1)
    - {cat_col}_{value}__cat: One-hot encoded categorical features
    - {exact_col}_{value}__exact: One-hot encoded exact match features
    - {num_col}__num: Numeric features
    - {date_col}__date: Date features (days from reference)
    - exact_match__group: Composite exact match grouping ID
    - features: Assembled feature vector for LSH

    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    df : DataFrame
        Input DataFrame with patient data
    id_col : str
        Patient identifier column name
    treatment_col : str
        Column name containing treatment/cohort indicator
    treatment_value : str
        Value in treatment_col that indicates "treated" group
    categorical_cols : Optional[List[str]]
        Columns to treat as categorical (will be one-hot encoded). Optional if
        numeric_cols or date_cols is provided.
    numeric_cols : Optional[List[str]]
        Columns to treat as numeric (must not contain nulls). Optional if
        categorical_cols or date_cols is provided.
    date_cols : Optional[List[str]]
        Date columns to convert to numeric (days from date_reference). Optional.
    exact_match_cols : Optional[List[str]]
        Categorical columns to use for exact matching stratification. Optional.
    date_reference : str
        Reference date for converting date columns to numeric
    max_categories : int
        Maximum number of distinct values allowed per categorical column.
        High-cardinality columns (e.g., zip codes) are not suitable for matching.
        Default: 20. Can be increased if needed.

    Returns
    -------
    DataFrame
        DataFrame with feature columns using suffix-based naming convention.
        Downstream functions (match, match_summary, etc.) auto-discover columns
        from these suffixes and do not require explicit column parameters.
    """
    # Set defaults for optional parameters
    if categorical_cols is None:
        categorical_cols = []
    if numeric_cols is None:
        numeric_cols = []
    if date_cols is None:
        date_cols = []
    if exact_match_cols is None:
        exact_match_cols = []

    # Validate that at least one feature type is provided
    if not categorical_cols and not numeric_cols and not date_cols:
        raise ValueError(
            "At least one of categorical_cols, numeric_cols, or date_cols must be provided"
        )

    # Validate that each column is only listed once
    all_cols = categorical_cols + numeric_cols + date_cols
    for col in all_cols:
        if all_cols.count(col) != 1:
            raise ValueError(
                f"The column {col} is used multiple times across "
                "categorical_cols, numeric_cols, and date_cols."
            )

    # Validate exact_match_cols are in categorical_cols or are additional columns
    for col in exact_match_cols:
        if col not in categorical_cols and col not in all_cols:
            # If exact match col not in categorical_cols, it must be in the DataFrame
            if col not in df.columns:
                raise ValueError(
                    f"The exact match column '{col}' must be present in the DataFrame."
                )

    # Create ID column with __id suffix
    id_col_internal = f"{id_col}__id"
    df = df.withColumn(id_col_internal, F.col(id_col).cast("string"))

    # Create treatment column with __treat suffix
    df = df.withColumn(
        "treat__treat",
        F.when(
            F.col(treatment_col) == F.lit(treatment_value).cast(df.schema[treatment_col].dataType),
            1
        ).otherwise(0)
    )

    # Convert categorical columns to strings
    for c in categorical_cols:
        df = df.withColumn(c, F.col(c).cast("string"))

    # Convert exact match columns to strings (if not already in categorical_cols)
    for c in exact_match_cols:
        if c not in categorical_cols:
            df = df.withColumn(c, F.col(c).cast("string"))

    # Process categorical columns (excluding exact match cols which are handled separately)
    categorical_feature_cols = []
    for c in categorical_cols:
        # Skip if this is an exact match column (handled separately)
        if c in exact_match_cols:
            continue
        df, new_cols = _create_onehot_columns(df, c, suffix="__cat", max_categories=max_categories)
        categorical_feature_cols.extend(new_cols)

    # Process exact match columns
    exact_match_feature_cols = []
    if exact_match_cols:
        for c in exact_match_cols:
            df, new_cols = _create_onehot_columns(df, c, suffix="__exact", max_categories=max_categories)
            exact_match_feature_cols.extend(new_cols)

        # Create composite exact match grouping column
        df = df.withColumn(
            "exact_match__group",
            F.concat_ws("_", *[F.col(c) for c in exact_match_cols])
        )
    else:
        df = df.withColumn("exact_match__group", F.lit("all"))

    # Process numeric columns - rename with __num suffix
    numeric_feature_cols = []
    for c in numeric_cols:
        new_col = f"{c}__num"
        df = df.withColumn(new_col, F.col(c).cast("double"))
        numeric_feature_cols.append(new_col)

    # Process date columns - convert to days from reference with __date suffix
    date_feature_cols = []
    for c in date_cols:
        new_col = f"{c}__date"
        df = df.withColumn(
            new_col,
            F.datediff(F.col(c), F.lit(date_reference)).cast("double")
        )
        date_feature_cols.append(new_col)

    # Assemble feature vector from all feature columns
    assembler_cols = (
        categorical_feature_cols +    # race_white__cat, race_black__cat, ...
        exact_match_feature_cols +    # gender_male__exact, gender_female__exact, ...
        numeric_feature_cols +        # age__num, bmi__num, ...
        date_feature_cols             # diagnosis_date__date, ...
    )

    # Create unscaled features
    assembler = VectorAssembler(inputCols=assembler_cols, outputCol="unscaled_features")
    df = assembler.transform(df)

    # Scale features (standardization happens here, not in match())
    scaler = StandardScaler(
        inputCol="unscaled_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )
    df = Pipeline(stages=[scaler]).fit(df).transform(df)

    # Select output columns
    output_cols = (
        [id_col_internal] +           # person_id__id
        ["treat__treat"] +            # treatment indicator
        categorical_feature_cols +    # race_white__cat, ...
        exact_match_feature_cols +    # gender_male__exact, ...
        numeric_feature_cols +        # age__num, ...
        date_feature_cols +           # diagnosis_date__date
        ["exact_match__group"] +      # composite grouping
        ["features"]                  # assembled feature vector
    )
    df = df.select(output_cols)

    return df
