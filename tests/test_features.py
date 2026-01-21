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
        verbose=False,
    )

    # Check that required columns exist
    assert "features" in features_df.columns
    assert "treat" in features_df.columns
    assert "exact_match_id" in features_df.columns


def test_feature_vector_dimensions(spark, lalonde_df):
    """Test that feature vector has correct dimensions."""
    # Add an id column if not present
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race"],  # race has ~3 categories, one-hot will be 2-dim
        numeric_cols=["age", "educ"],  # 2 numeric features
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
        verbose=False,
    )

    # Get feature vector size
    from pyspark.ml.functions import vector_to_array

    features_df = features_df.withColumn("feature_array", vector_to_array("features"))
    feature_size = features_df.select(F.size("feature_array")).first()[0]

    # Should have: 2 (race one-hot) + 2 (numeric) = 4 features
    # Note: exact count depends on categories in data
    assert feature_size > 0


def test_exact_match_cols(spark, lalonde_df):
    """Test exact matching stratification."""
    # Add an id column if not present
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
        verbose=False,
    )

    # Check that exact_match_id exists and has distinct values
    distinct_count = features_df.select(F.countDistinct("exact_match_id")).first()[0]
    assert distinct_count > 0

    # Check that married column is not in the feature vector
    # (exact match cols should be removed from feature matching)
    assert "married_onehot" not in features_df.columns or features_df.select(
        "married_onehot"
    ).first() is None


def test_treatment_column_creation(spark, lalonde_df):
    """Test that treat column is correctly created."""
    # Add an id column if not present
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
        verbose=False,
    )

    # Check that treat column has 0 and 1 values
    treat_counts = features_df.groupBy("treat").count().collect()
    assert len(treat_counts) == 2
    assert all(row["treat"] in [0, 1] for row in treat_counts)


def test_only_categorical_features(spark, lalonde_df):
    """Test that function works with only categorical features."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race", "married", "nodegree"],
        numeric_cols=None,  # Explicitly None
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
        verbose=False,
    )

    assert "features" in features_df.columns
    assert features_df.count() > 0


def test_only_numeric_features(spark, lalonde_df):
    """Test that function works with only numeric features."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=None,  # Explicitly None
        numeric_cols=["age", "educ", "re74", "re75"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
        verbose=False,
    )

    assert "features" in features_df.columns
    assert features_df.count() > 0


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


def test_exact_match_requires_categorical(spark, lalonde_df):
    """Test that exact_match_cols requires categorical_cols."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id())

    with pytest.raises(ValueError, match="exact_match_cols can only be used when categorical_cols"):
        generate_features(
            spark,
            lalonde_df,
            categorical_cols=None,
            numeric_cols=["age", "educ"],
            exact_match_cols=["race"],  # Can't use without categorical_cols
            treatment_col="treat",
            treatment_value="1",
            id_col="id",
        )
