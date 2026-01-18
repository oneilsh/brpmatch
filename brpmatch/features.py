"""
Feature generation for BRPMatch cohort matching.

This module converts patient data into feature vectors for matching.
"""

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

    This function processes input data through multiple transformations:
    1. Converts dates to numeric (days from reference date)
    2. Creates exact match stratification IDs
    3. One-hot encodes categorical variables
    4. Imputes missing numeric values (mean or GBT)
    5. Standardizes features
    6. Creates treatment indicator

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
        - treatment_col column
        - id_col column
    """
    # Set defaults for optional parameters
    if date_cols is None:
        date_cols = []
    if exact_match_cols is None:
        exact_match_cols = []
    if gbt_impute_cols is None:
        gbt_impute_cols = []

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

    # Validate gbt_impute_cols are numeric or date columns
    for col in gbt_impute_cols:
        if col not in numeric_cols + date_cols:
            raise RuntimeError(
                f"The column {col} must be listed as a numeric or date column "
                "to be used for GBT imputation."
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

    # Remove GBT imputation columns from numeric columns
    # (so they are not imputed with mean imputation)
    numeric_cols = list(filter(lambda x: x not in gbt_impute_cols, numeric_cols))

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

    # Impute missing numeric values with mean imputation
    print("numeric_cols", numeric_cols)
    for c in numeric_cols:
        preprocessing_stages += [
            Imputer(inputCol=c, outputCol=f"{c}_imputed", strategy="mean")
        ]

    # Create feature vector from onehot/imputed columns
    feature_cols = [f"{c}_onehot" for c in categorical_cols] + [
        f"{c}_imputed" for c in numeric_cols
    ]
    preprocessing_stages += [
        VectorAssembler(inputCols=feature_cols, outputCol="gbt_features")
    ]

    df = Pipeline(stages=preprocessing_stages).fit(df).transform(df)

    # Apply GBT imputation for specified columns
    if len(gbt_impute_cols) > 0:
        for c in gbt_impute_cols:
            values = df.filter(F.col(c).isNotNull())
            gbt_model = Pipeline(
                stages=[
                    GBTRegressor(
                        featuresCol="gbt_features",
                        labelCol=c,
                        predictionCol=f"{c}_imputed",
                        seed=42,
                    )
                ]
            )

            # Predict column value, but use actual value when available
            df = (
                gbt_model.fit(values)
                .transform(df)
                .withColumn(f"{c}_imputed", F.coalesce(F.col(c), F.col(f"{c}_imputed")))
            )

        # Add GBT features to features vector
        print(feature_cols)
        feature_cols = ["gbt_features"] + [f"{c}_imputed" for c in gbt_impute_cols]
        print(feature_cols)
        finish_features_p = Pipeline(
            stages=[
                VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features"),
                StandardScaler(
                    inputCol="unscaled_features", outputCol="features", withStd=True
                ),
            ]
        )
    else:
        # No GBT imputation, just scale existing features
        finish_features_p = Pipeline(
            stages=[
                StandardScaler(inputCol="gbt_features", outputCol="features", withStd=True)
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
        + gbt_impute_cols
        + exact_match_cols
        + categorical_index_cols
        + feature_cols
        + ["exact_match_id"]
        + ["features"]
        + ["treat"]
        + [treatment_col]
        + [id_col]
    )

    return df
