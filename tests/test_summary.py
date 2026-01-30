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
    # match() now returns a tuple of (units, pairs, bucket_stats)
    units, pairs, bucket_stats = match(features_df, feature_space="euclidean", n_neighbors=5)

    return features_df, units


def test_match_summary_basic(matched_data):
    """Test basic match_summary functionality."""
    features_df, units = matched_data

    # No feature_cols parameter - auto-discovered
    balance_df, aggregate_df, fig = match_summary(
        features_df,
        units,
        sample_frac=1.0,
    )

    # Check that required columns exist in balance_df
    assert "display_name" in balance_df.columns
    assert "smd_unadjusted" in balance_df.columns
    assert "smd_adjusted" in balance_df.columns
    assert "vr_unadjusted" in balance_df.columns
    assert "vr_adjusted" in balance_df.columns

    # Check aggregate_df structure
    assert "statistic" in aggregate_df.columns
    assert "value" in aggregate_df.columns

    # Check figure
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_match_summary_auto_discovers_features(matched_data):
    """Test that match_summary auto-discovers feature columns."""
    features_df, units = matched_data

    balance_df, aggregate_df, fig = match_summary(
        features_df,
        units,
        sample_frac=1.0,
    )

    # Should have discovered features from __cat, __num, __date, __exact suffixes
    # Check that we have some features
    assert len(balance_df) > 0


def test_match_summary_display_names(matched_data):
    """Test that match_summary includes display names."""
    features_df, units = matched_data

    balance_df, aggregate_df, fig = match_summary(
        features_df,
        units,
        sample_frac=1.0,
    )

    # Should have display_name column with stripped suffixes
    assert "display_name" in balance_df.columns
    # display_name should be first column
    assert balance_df.columns[0] == "display_name"


def test_match_summary_balance_improvement(matched_data):
    """Test that matching improves balance (reduced SMD)."""
    features_df, units = matched_data

    balance_df, aggregate_df, fig = match_summary(
        features_df,
        units,
        sample_frac=1.0,
    )

    # On average, adjusted SMD should be smaller than unadjusted
    mean_smd_unadj = balance_df["smd_unadjusted"].abs().mean()
    mean_smd_adj = balance_df["smd_adjusted"].abs().mean()

    # Adjusted should be less than or equal to unadjusted
    # (matching should improve balance)
    assert mean_smd_adj <= mean_smd_unadj


def test_match_summary_sampling(matched_data):
    """Test that sampling parameter works."""
    features_df, units = matched_data

    # Should work with different sample fractions
    balance_full, aggregate_full, fig_full = match_summary(
        features_df,
        units,
        sample_frac=1.0,
    )

    balance_sample, aggregate_sample, fig_sample = match_summary(
        features_df,
        units,
        sample_frac=0.5,
    )

    # Should have same structure (columns)
    assert set(balance_full.columns) == set(balance_sample.columns)
    assert set(aggregate_full.columns) == set(aggregate_sample.columns)

    # Verify sample_frac is correctly reported in aggregate stats
    stats_full = dict(zip(aggregate_full["statistic"], aggregate_full["value"]))
    stats_sample = dict(zip(aggregate_sample["statistic"], aggregate_sample["value"]))
    assert stats_full["sample_frac"] == 1.0
    assert stats_sample["sample_frac"] == 0.5


def test_match_summary_ecdf_excluded_by_default(matched_data):
    """Test that eCDF statistics are excluded by default."""
    features_df, units = matched_data

    balance_df, aggregate_df, fig = match_summary(
        features_df,
        units,
        sample_frac=1.0,
    )

    # Should NOT have eCDF columns by default
    assert "ecdf_mean_unadj" not in balance_df.columns
    assert "ecdf_mean_adj" not in balance_df.columns
    assert "ecdf_max_unadj" not in balance_df.columns
    assert "ecdf_max_adj" not in balance_df.columns


def test_match_summary_ecdf_statistics(matched_data):
    """Test that eCDF statistics are computed when include_ecdf=True."""
    features_df, units = matched_data

    balance_df, aggregate_df, fig = match_summary(
        features_df,
        units,
        sample_frac=1.0,
        include_ecdf=True,
    )

    # Should have eCDF columns
    assert "ecdf_mean_unadj" in balance_df.columns
    assert "ecdf_mean_adj" in balance_df.columns
    assert "ecdf_max_unadj" in balance_df.columns
    assert "ecdf_max_adj" in balance_df.columns

    # eCDF values should be between 0 and 1
    assert all(balance_df["ecdf_mean_unadj"].between(0, 1))
    assert all(balance_df["ecdf_mean_adj"].between(0, 1))
    assert all(balance_df["ecdf_max_unadj"].between(0, 1))
    assert all(balance_df["ecdf_max_adj"].between(0, 1))


def test_match_summary_aggregate_statistics(matched_data):
    """Test that aggregate statistics DataFrame contains expected statistics."""
    features_df, units = matched_data

    balance_df, aggregate_df, fig = match_summary(
        features_df,
        units,
        sample_frac=1.0,
    )

    # Convert to dict for easier lookup
    stats = dict(zip(aggregate_df["statistic"], aggregate_df["value"]))

    # Check all expected statistics are present
    expected_stats = [
        "sample_frac",
        "n_treated_total",
        "n_control_total",
        "n_treated_matched",
        "n_control_matched",
        "n_treated_unmatched",
        "n_control_unmatched",
        "pct_treated_matched",
        "pct_control_matched",
        "mean_controls_per_treated",
        "min_controls_per_treated",
        "max_controls_per_treated",
        "effective_sample_size_treated",
        "effective_sample_size_control",
    ]
    for stat_name in expected_stats:
        assert stat_name in stats, f"Missing statistic: {stat_name}"

    # Basic sanity checks
    assert stats["sample_frac"] == 1.0
    assert stats["n_treated_total"] > 0
    assert stats["n_control_total"] > 0
    assert stats["n_treated_matched"] <= stats["n_treated_total"]
    assert stats["n_control_matched"] <= stats["n_control_total"]
    assert stats["n_treated_unmatched"] == stats["n_treated_total"] - stats["n_treated_matched"]
    assert stats["n_control_unmatched"] == stats["n_control_total"] - stats["n_control_matched"]
    assert 0 <= stats["pct_treated_matched"] <= 100
    assert 0 <= stats["pct_control_matched"] <= 100
    assert stats["mean_controls_per_treated"] >= 0
    assert stats["effective_sample_size_treated"] >= 0
    assert stats["effective_sample_size_control"] >= 0
