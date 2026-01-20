"""
Feature generation for BRPMatch cohort matching.

This module converts patient data into feature vectors for matching.
"""

from typing import List, Optional

import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame, SparkSession


def generate_features(
    spark: SparkSession,
    df: DataFrame,
    treatment_col: str,
    treatment_value: str,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    exact_match_cols: Optional[List[str]] = None,
    date_reference: str = "1970-01-01",
    id_col: str = "person_id",
) -> DataFrame:
    """
    Convert patient data into feature vectors for matching.

    This function processes input data through multiple transformations:
    1. Converts dates to numeric (days from reference date)
    2. Creates exact match stratification IDs
    3. One-hot encodes categorical variables
    4. Standardizes features
    5. Creates treatment indicator

    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    df : DataFrame
        Input DataFrame with patient data
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
        Categorical columns to use for exact matching stratification. Requires
        categorical_cols to be provided. Optional.
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
        - Original columns with suffixes (_index, etc.)
        - treatment_col column
        - id_col column
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

    # Validate that exact_match_cols requires categorical_cols
    if exact_match_cols and not categorical_cols:
        raise ValueError(
            "exact_match_cols can only be used when categorical_cols is provided. "
            "All exact match columns must be categorical."
        )

    # Validate that each column is only listed once
    all_cols = categorical_cols + numeric_cols + date_cols
    for col in all_cols:
        if all_cols.count(col) != 1:
            raise RuntimeError(
                f"The column {col} is used multiple times across "
                "categorical_cols, numeric_cols, and date_cols."
            )

    # Validate exact_match_cols are categorical
    for col in exact_match_cols:
        if col not in categorical_cols:
            raise RuntimeError(
                f"The column {col} must be listed as a categorical column "
                "to be used for exact matching."
            )

    # Convert categorical columns to strings
    for c in categorical_cols:
        df = df.withColumn(c, F.col(c).cast("string"))

    # Transform date columns to numeric (days from reference)
    for c in date_cols:
        df = df.withColumn(
            f"{c}_days_from_2018", F.datediff(F.col(c), F.lit(date_reference))
        )
        numeric_cols.append(f"{c}_days_from_2018")

    # Create exact_match_id column as concatenation of exact match column values
    # NULL values are replaced with "NULL" string
    if len(exact_match_cols) > 0:
        df = df.withColumn(
            "exact_match_id",
            F.when(df[exact_match_cols[0]].isNull(), "NULL").otherwise(
                df[exact_match_cols[0]]
            ),
        )
        # Concatenate remaining columns
        for col in exact_match_cols[1:]:
            df = df.withColumn(
                "exact_match_id",
                F.concat_ws(
                    ":",
                    df["exact_match_id"],
                    F.when(df[col].isNull(), "NULL").otherwise(df[col]),
                ),
            )
    else:
        df = df.withColumn("exact_match_id", F.lit(1))

    # Remove exact match columns from categorical columns
    # (they won't be used for feature-based matching)
    categorical_cols = list(filter(lambda x: x not in exact_match_cols, categorical_cols))

    # Build preprocessing pipeline
    preprocessing_stages = []

    # Convert categorical variables to one-hot encodings
    print("categorical_cols", categorical_cols)
    categorical_index_cols = []
    for c in categorical_cols:
        preprocessing_stages += [
            StringIndexer(
                inputCol=c, outputCol=f"{c}_index", handleInvalid="keep"
            )
        ]
        preprocessing_stages += [
            OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_onehot", dropLast=True)
        ]
        categorical_index_cols.append(f"{c}_index")

    # Create feature vector from onehot and numeric columns
    print("numeric_cols", numeric_cols)
    feature_cols = [f"{c}_onehot" for c in categorical_cols] + numeric_cols
    preprocessing_stages += [
        VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")
    ]

    df = Pipeline(stages=preprocessing_stages).fit(df).transform(df)

    # Scale features
    finish_features_p = Pipeline(
        stages=[
            StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True)
        ]
    )

    df = finish_features_p.fit(df).transform(df)

    # Create treat column (1 for treated, 0 for control)
    # Cast treatment_value to match the column type
    df = df.withColumn(
        "treat",
        F.when(F.col(treatment_col) == F.lit(treatment_value).cast(df.schema[treatment_col].dataType), 1).otherwise(0),
    )

    # Select relevant columns to return
    df = df.select(
        categorical_cols
        + numeric_cols
        + date_cols
        + exact_match_cols
        + categorical_index_cols
        + ["exact_match_id"]
        + ["features"]
        + ["treat"]
        + [treatment_col]
        + [id_col]
    )

    return df
