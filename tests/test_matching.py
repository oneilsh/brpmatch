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
    )


def test_basic_matching(features_df):
    """Test basic matching with euclidean distance."""
    matched_df = match(features_df, distance_metric="euclidean", n_neighbors=5, id_col="id")

    # Check that required columns exist
    assert "id" in matched_df.columns
    assert "match_id" in matched_df.columns
    assert "match_distance" in matched_df.columns


def test_one_to_one_constraint(features_df):
    """Test that matching satisfies 1-to-1 constraint."""
    matched_df = match(features_df, distance_metric="euclidean", n_neighbors=5, id_col="id")

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

    matched_df = match(features_df, distance_metric="euclidean", n_neighbors=5, id_col="id")

    match_count = matched_df.count()

    # Should have at most min(treated, control) matches
    assert match_count <= min(treat_count, control_count)


def test_distances_positive(features_df):
    """Test that all match distances are non-negative."""
    matched_df = match(features_df, distance_metric="euclidean", n_neighbors=5, id_col="id")

    # Check that all distances are >= 0
    from pyspark.sql import functions as F

    min_distance = matched_df.select(F.min("match_distance")).first()[0]
    assert min_distance >= 0


def test_mahalanobis_distance(features_df):
    """Test matching with Mahalanobis distance."""
    matched_df = match(
        features_df, distance_metric="mahalanobis", n_neighbors=5, id_col="id"
    )

    # Should produce valid matches
    assert matched_df.count() > 0

    # Check that required columns exist
    assert "id" in matched_df.columns
    assert "match_id" in matched_df.columns
    assert "match_distance" in matched_df.columns
