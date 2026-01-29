"""
Tests for feature generation in BRPMatch.
"""

import pytest
from pyspark.sql import functions as F

from brpmatch import generate_features


def test_basic_feature_generation(spark, lalonde_df):
    """Test basic feature generation with lalonde data."""
    # Add an id column if not present
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race", "married", "nodegree"],
        numeric_cols=["age", "educ", "re74", "re75"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
    )

    # Check that required columns exist with new naming convention
    assert "features" in features_df.columns
    assert "treat__treat" in features_df.columns
    assert "exact_match__group" in features_df.columns
    assert "id__id" in features_df.columns


def test_column_naming_convention(spark, lalonde_df):
    """Test that columns follow the new suffix-based naming convention."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race", "married"],
        numeric_cols=["age", "educ"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
    )

    # Check for categorical one-hot columns (should end with __cat)
    cat_cols = [c for c in features_df.columns if c.endswith("__cat")]
    assert len(cat_cols) > 0, "Should have categorical columns with __cat suffix"
    # Should have columns like race_black__cat, race_white__cat, married_0__cat, married_1__cat
    assert any(c.startswith("race_") and c.endswith("__cat") for c in cat_cols)
    assert any(c.startswith("married_") and c.endswith("__cat") for c in cat_cols)

    # Check for numeric columns (should end with __num)
    num_cols = [c for c in features_df.columns if c.endswith("__num")]
    assert "age__num" in num_cols
    assert "educ__num" in num_cols

    # Check that old-style columns don't exist
    assert "race_index" not in features_df.columns
    assert "race_onehot" not in features_df.columns


def test_exact_match_cols(spark, lalonde_df):
    """Test exact matching stratification with new naming."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race", "married"],
        numeric_cols=["age"],
        treatment_col="treat",
        treatment_value="1",
        exact_match_cols=["married"],
        id_col="id",
    )

    # Check that exact_match__group exists and has distinct values
    assert "exact_match__group" in features_df.columns
    distinct_count = features_df.select(F.countDistinct("exact_match__group")).first()[0]
    assert distinct_count > 0

    # Check that exact match columns have __exact suffix
    exact_cols = [c for c in features_df.columns if c.endswith("__exact")]
    assert len(exact_cols) > 0
    assert any(c.startswith("married_") and c.endswith("__exact") for c in exact_cols)

    # Check that married columns are NOT in __cat (since they're exact match)
    cat_cols = [c for c in features_df.columns if c.endswith("__cat")]
    assert not any(c.startswith("married_") for c in cat_cols), "Exact match cols should use __exact, not __cat"


def test_feature_vector_dimensions(spark, lalonde_df):
    """Test that feature vector has correct dimensions."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race"],
        numeric_cols=["age", "educ"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
    )

    # Get feature vector size
    from pyspark.ml.functions import vector_to_array

    features_df = features_df.withColumn("feature_array", vector_to_array("features"))
    feature_size = features_df.select(F.size("feature_array")).first()[0]

    # Should have: race categories (one-hot) + 2 (numeric) > 0
    assert feature_size > 0


def test_treatment_column_creation(spark, lalonde_df):
    """Test that treat__treat column is correctly created."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race"],
        numeric_cols=["age"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
    )

    # Check that treat__treat column has 0 and 1 values
    assert "treat__treat" in features_df.columns
    treat_counts = features_df.groupBy("treat__treat").count().collect()
    assert len(treat_counts) == 2
    assert all(row["treat__treat"] in [0, 1] for row in treat_counts)


def test_only_categorical_features(spark, lalonde_df):
    """Test that function works with only categorical features."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race", "married", "nodegree"],
        numeric_cols=None,
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
    )

    assert "features" in features_df.columns
    assert features_df.count() > 0
    # Should have __cat columns
    cat_cols = [c for c in features_df.columns if c.endswith("__cat")]
    assert len(cat_cols) > 0


def test_only_numeric_features(spark, lalonde_df):
    """Test that function works with only numeric features."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=None,
        numeric_cols=["age", "educ", "re74", "re75"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
    )

    assert "features" in features_df.columns
    assert features_df.count() > 0
    # Should have __num columns
    num_cols = [c for c in features_df.columns if c.endswith("__num")]
    assert len(num_cols) == 4


def test_no_features_raises_error(spark, lalonde_df):
    """Test that providing no features raises ValueError."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    with pytest.raises(ValueError, match="At least one of.*must be provided"):
        generate_features(
            spark,
            lalonde_df,
            categorical_cols=None,
            numeric_cols=None,
            date_cols=None,
            treatment_col="treat",
            treatment_value="1",
            id_col="id",
        )


def test_max_categories_enforcement(spark, lalonde_df):
    """Test that max_categories parameter is enforced."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    # Create a high-cardinality column
    lalonde_with_zip = lalonde_df.withColumn(
        "fake_zip", F.monotonically_increasing_id().cast("string")
    )

    # Should raise error if categories exceed max_categories
    with pytest.raises(ValueError, match="exceeds max_categories"):
        generate_features(
            spark,
            lalonde_with_zip,
            categorical_cols=["fake_zip"],  # Has many unique values
            numeric_cols=["age"],
            treatment_col="treat",
            treatment_value="1",
            id_col="id",
            max_categories=5,  # Set low threshold
        )


def test_max_categories_override(spark, lalonde_df):
    """Test that max_categories can be increased."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    # This should work with a higher max_categories
    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race"],
        numeric_cols=["age"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
        max_categories=100,  # High threshold
    )

    assert "features" in features_df.columns


def test_value_sanitization(spark):
    """Test that categorical values with special characters are sanitized."""
    # Create a test DataFrame with special characters in categorical values
    test_data = spark.createDataFrame([
        ("1", "African American", 25, 1),
        ("2", "White/Caucasian", 30, 0),
        ("3", "Hispanic-Latino", 35, 1),
    ], ["id", "ethnicity", "age", "treat"])

    features_df = generate_features(
        spark,
        test_data,
        categorical_cols=["ethnicity"],
        numeric_cols=["age"],
        treatment_col="treat",
        treatment_value=1,
        id_col="id",
    )

    # Check that sanitized column names exist
    cat_cols = [c for c in features_df.columns if c.endswith("__cat")]
    # Should have columns like ethnicity_african_american__cat (spaces->underscores, lowercase)
    assert any("african_american" in c for c in cat_cols)
    assert any("white_caucasian" in c for c in cat_cols)
    assert any("hispanic_latino" in c for c in cat_cols)


def test_exact_match_not_in_categorical(spark, lalonde_df):
    """Test that exact match columns can be specified without being in categorical_cols."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    # This should work - exact_match_cols don't need to be in categorical_cols
    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race"],
        numeric_cols=["age"],
        treatment_col="treat",
        treatment_value="1",
        exact_match_cols=["married"],  # Not in categorical_cols
        id_col="id",
    )

    # Should have exact match columns
    exact_cols = [c for c in features_df.columns if c.endswith("__exact")]
    assert len(exact_cols) > 0
    assert any(c.startswith("married_") for c in exact_cols)
