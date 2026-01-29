"""
Tests for summary statistics functionality in BRPMatch.
"""

import pytest
from pyspark.sql import functions as F

from brpmatch import generate_features, match, match_summary


@pytest.fixture
def matched_data(spark, lalonde_df):
    """Generate features and matched data for testing."""
    # Add an id column if not present
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    # Generate features
    features_df = generate_features(
        spark,
        lalonde_df,
        categorical_cols=["race", "married", "nodegree"],
        numeric_cols=["age", "educ", "re74", "re75"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
    )

    # Match (no id_col parameter - auto-discovered)
    matched_df = match(features_df, feature_space="euclidean", n_neighbors=5)

    return features_df, matched_df


def test_match_summary_basic(matched_data):
    """Test basic match_summary functionality."""
    features_df, matched_df = matched_data

    # No feature_cols parameter - auto-discovered
    summary_df = match_summary(
        features_df,
        matched_df,
        sample_frac=1.0,
        plot=False,
    )

    # Check that required columns exist
    assert "covariate" in summary_df.columns
    assert "smd_unadjusted" in summary_df.columns
    assert "smd_adjusted" in summary_df.columns
    assert "vr_unadjusted" in summary_df.columns
    assert "vr_adjusted" in summary_df.columns


def test_match_summary_auto_discovers_features(matched_data):
    """Test that match_summary auto-discovers feature columns."""
    features_df, matched_df = matched_data

    summary_df = match_summary(
        features_df,
        matched_df,
        sample_frac=1.0,
        plot=False,
    )

    # Should have discovered features from __cat, __num, __date, __exact suffixes
    # Check that we have some features
    assert len(summary_df) > 0


def test_match_summary_with_plot(matched_data):
    """Test that match_summary can generate plots."""
    features_df, matched_df = matched_data

    result = match_summary(
        features_df,
        matched_df,
        sample_frac=1.0,
        plot=True,
    )

    # Should return tuple of (summary_df, figure)
    assert isinstance(result, tuple)
    assert len(result) == 2
    summary_df, fig = result

    # Check summary DataFrame
    assert "covariate" in summary_df.columns

    # Check figure
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_match_summary_display_names(matched_data):
    """Test that match_summary includes display names."""
    features_df, matched_df = matched_data

    summary_df = match_summary(
        features_df,
        matched_df,
        sample_frac=1.0,
        plot=False,
    )

    # Should have display_name column with stripped suffixes
    assert "display_name" in summary_df.columns
    assert "feature_type" in summary_df.columns


def test_match_summary_balance_improvement(matched_data):
    """Test that matching improves balance (reduced SMD)."""
    features_df, matched_df = matched_data

    summary_df = match_summary(
        features_df,
        matched_df,
        sample_frac=1.0,
        plot=False,
    )

    # On average, adjusted SMD should be smaller than unadjusted
    mean_smd_unadj = summary_df["smd_unadjusted"].abs().mean()
    mean_smd_adj = summary_df["smd_adjusted"].abs().mean()

    # Adjusted should be less than or equal to unadjusted
    # (matching should improve balance)
    assert mean_smd_adj <= mean_smd_unadj


def test_match_summary_sampling(matched_data):
    """Test that sampling parameter works."""
    features_df, matched_df = matched_data

    # Should work with different sample fractions
    summary_full = match_summary(
        features_df,
        matched_df,
        sample_frac=1.0,
        plot=False,
    )

    summary_sample = match_summary(
        features_df,
        matched_df,
        sample_frac=0.5,
        plot=False,
    )

    # Should have same structure (columns)
    assert set(summary_full.columns) == set(summary_sample.columns)


def test_match_summary_ecdf_statistics(matched_data):
    """Test that eCDF statistics are computed."""
    features_df, matched_df = matched_data

    summary_df = match_summary(
        features_df,
        matched_df,
        sample_frac=1.0,
        plot=False,
    )

    # Should have eCDF columns
    assert "ecdf_mean_unadj" in summary_df.columns
    assert "ecdf_mean_adj" in summary_df.columns
    assert "ecdf_max_unadj" in summary_df.columns
    assert "ecdf_max_adj" in summary_df.columns

    # eCDF values should be between 0 and 1
    assert all(summary_df["ecdf_mean_unadj"].between(0, 1))
    assert all(summary_df["ecdf_mean_adj"].between(0, 1))
    assert all(summary_df["ecdf_max_unadj"].between(0, 1))
    assert all(summary_df["ecdf_max_adj"].between(0, 1))
