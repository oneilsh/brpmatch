"""
Tests for love plot functionality in BRPMatch.
"""

import matplotlib.figure
import numpy as np
import pytest
from pyspark.sql import functions as F

from brpmatch import generate_features, love_plot, match, stratify_for_plot
from brpmatch.loveplot import _compute_smd, _compute_variance_ratio


@pytest.fixture
def stratified_df(spark, lalonde_df):
    """Generate stratified data for testing."""
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
        verbose=False,
    )

    # Match (no id_col parameter - auto-discovered)
    matched_df = match(features_df, feature_space="euclidean", n_neighbors=5, verbose=False)

    # Stratify (no column parameters - auto-discovered)
    return stratify_for_plot(features_df, matched_df)


def test_love_plot_returns_figure(stratified_df):
    """Test that love_plot returns a matplotlib figure."""
    # No feature_cols parameter - auto-discovered from suffixes
    fig = love_plot(stratified_df, sample_frac=1.0)

    assert isinstance(fig, matplotlib.figure.Figure)


def test_balance_statistics_computation(stratified_df):
    """Test that balance statistics are computed correctly."""
    # Just ensure the plot can be generated
    # (balance stats are computed internally)
    fig = love_plot(stratified_df, sample_frac=1.0)

    # Figure should have 2 subplots (SMD and VR)
    assert len(fig.axes) == 2


def test_love_plot_auto_discovers_features(stratified_df):
    """Test that love_plot auto-discovers feature columns from suffixes."""
    fig = love_plot(stratified_df, sample_frac=1.0)

    # Should successfully generate plot with auto-discovered features
    assert isinstance(fig, matplotlib.figure.Figure)
    # Should have 2 axes (SMD and VR)
    assert len(fig.axes) == 2


def test_smd_calculation():
    """Test SMD calculation with known values."""
    # Create simple test arrays
    treated = np.array([3.0, 4.0, 5.0, 6.0, 7.0])  # mean=5, var=2.5
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # mean=3, var=2.5

    smd = _compute_smd(treated, control)

    # SMD = (5 - 3) / sqrt((2.5 + 2.5) / 2) = 2 / sqrt(2.5) = 1.265
    assert abs(smd - 1.265) < 0.01


def test_variance_ratio_calculation():
    """Test variance ratio calculation with known values."""
    # Create arrays with different variances
    treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # var = 2.5
    control = np.array([2.5, 2.5, 2.5, 2.5, 2.5])  # var = 0.0

    vr = _compute_variance_ratio(treated, control)

    # VR should be inf when control variance is 0
    assert vr == np.inf


def test_variance_ratio_equal_variance():
    """Test variance ratio when variances are equal."""
    # Create arrays with same variance
    treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    vr = _compute_variance_ratio(treated, control)

    # VR should be 1.0 when variances are equal
    assert abs(vr - 1.0) < 0.01
