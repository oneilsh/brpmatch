"""
Tests for matching functionality in BRPMatch.
"""

import pytest
from pyspark.sql import functions as F

from brpmatch import generate_features, match


@pytest.fixture
def features_df(spark, lalonde_df):
    """Generate features from lalonde data for testing."""
    # Add an id column if not present
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    return generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race", "married", "nodegree"],
        numeric_cols=["age", "educ", "re74", "re75"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
        verbose=False,
    )


def test_basic_matching(features_df):
    """Test basic matching with euclidean distance."""
    matched_df = match(features_df, feature_space="euclidean", n_neighbors=5, id_col="id", verbose=False)

    # Check that required columns exist
    assert "id" in matched_df.columns
    assert "match_id" in matched_df.columns


def test_one_to_one_constraint(features_df):
    """Test that matching satisfies 1-to-1 constraint."""
    matched_df = match(features_df, feature_space="euclidean", n_neighbors=5, id_col="id", verbose=False)

    # Collect to pandas for easier checking
    pdf = matched_df.toPandas()

    # Each id should appear at most once
    id_counts = pdf["id"].value_counts()
    assert all(id_counts <= 1), "Some treated patients matched multiple times"

    # Each match_id should appear at most once
    match_id_counts = pdf["match_id"].value_counts()
    assert all(match_id_counts <= 1), "Some control patients matched multiple times"


def test_match_count_bounded(features_df):
    """Test that match count is bounded by min(treated, control)."""
    # Count treated and control
    from pyspark.sql import functions as F

    treat_counts = features_df.groupBy("treat").count().collect()
    treat_count = next(row["count"] for row in treat_counts if row["treat"] == 1)
    control_count = next(row["count"] for row in treat_counts if row["treat"] == 0)

    matched_df = match(features_df, feature_space="euclidean", n_neighbors=5, id_col="id", verbose=False)

    match_count = matched_df.count()

    # Should have at most min(treated, control) matches
    assert match_count <= min(treat_count, control_count)


def test_mahalanobis_distance(features_df):
    """Test matching with Mahalanobis distance."""
    matched_df = match(
        features_df, feature_space="mahalanobis", n_neighbors=5, id_col="id", verbose=False
    )

    # Should produce valid matches
    assert matched_df.count() > 0

    # Check that required columns exist
    assert "id" in matched_df.columns
    assert "match_id" in matched_df.columns


def test_mahalanobis_uses_whitening(features_df):
    """Test that mahalanobis feature space produces valid matches using whitening."""
    matched_df = match(
        features_df, feature_space="mahalanobis", n_neighbors=5, id_col="id", verbose=False
    )

    # Should produce valid matches
    assert matched_df.count() > 0

    # Check that required columns exist (no match_distance)
    assert "id" in matched_df.columns
    assert "match_id" in matched_df.columns
    assert "match_distance" not in matched_df.columns

    # Verify 1-to-1 constraint still holds
    pdf = matched_df.toPandas()
    id_counts = pdf["id"].value_counts()
    assert all(id_counts <= 1), "Some treated patients matched multiple times"
    match_id_counts = pdf["match_id"].value_counts()
    assert all(match_id_counts <= 1), "Some control patients matched multiple times"
