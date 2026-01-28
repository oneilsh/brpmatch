"""
Tests for matching functionality in BRPMatch.
"""

import pytest
from pyspark.sql import functions as F

from brpmatch import generate_features, match, match_data


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
    # No id_col parameter needed - auto-discovered
    matched_df = match(features_df, feature_space="euclidean", n_neighbors=5, verbose=False)

    # Check that required columns exist (using base name without __id suffix)
    assert "id" in matched_df.columns
    assert "match_id" in matched_df.columns


def test_one_to_one_constraint(features_df):
    """Test that matching satisfies 1-to-1 constraint."""
    matched_df = match(features_df, feature_space="euclidean", n_neighbors=5, verbose=False)

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
    # Count treated and control using new column name
    treat_counts = features_df.groupBy("treat__treat").count().collect()
    treat_count = next(row["count"] for row in treat_counts if row["treat__treat"] == 1)
    control_count = next(row["count"] for row in treat_counts if row["treat__treat"] == 0)

    matched_df = match(features_df, feature_space="euclidean", n_neighbors=5, verbose=False)

    match_count = matched_df.count()

    # Should have at most min(treated, control) matches
    assert match_count <= min(treat_count, control_count)


def test_mahalanobis_distance(features_df):
    """Test matching with Mahalanobis distance."""
    matched_df = match(features_df, feature_space="mahalanobis", n_neighbors=5, verbose=False)

    # Should produce valid matches
    assert matched_df.count() > 0

    # Check that required columns exist
    assert "id" in matched_df.columns
    assert "match_id" in matched_df.columns


def test_mahalanobis_uses_whitening(features_df):
    """Test that mahalanobis feature space produces valid matches using whitening."""
    matched_df = match(features_df, feature_space="mahalanobis", n_neighbors=5, verbose=False)

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


# =============================================================================
# Tests for 1-to-k Matching Features
# =============================================================================


def test_new_output_columns_exist(features_df):
    """Test that new output columns are present in matched output."""
    matched_df = match(features_df, n_neighbors=5, verbose=False)

    expected_cols = [
        "id", "match_id", "match_round", "treated_k",
        "control_usage_count", "pair_weight",
        "bucket_num_input_patients", "bucket_seconds"
    ]
    for col in expected_cols:
        assert col in matched_df.columns, f"Missing column: {col}"


def test_ratio_k_default_backward_compatible(features_df):
    """Test that default ratio_k=1 produces 1-to-1 matching (backward compatible)."""
    matched_df = match(features_df, n_neighbors=5, verbose=False)
    pdf = matched_df.toPandas()

    # Each treated should have exactly 1 match
    assert all(pdf["treated_k"] == 1), "Default should produce 1:1 matching"

    # Each control should be used exactly once
    assert all(pdf["control_usage_count"] == 1), "Without replacement, controls used once"

    # pair_weight should be 1.0 for all (1/(1*1))
    assert all(pdf["pair_weight"] == 1.0), "pair_weight should be 1.0 for 1:1 matching"

    # match_round should be 1 for all
    assert all(pdf["match_round"] == 1), "All matches should be round 1"


def test_ratio_k_multiple_without_replacement(features_df):
    """Test 1-to-k matching without replacement."""
    ratio_k = 3
    matched_df = match(
        features_df, n_neighbors=10, ratio_k=ratio_k,
        with_replacement=False, verbose=False
    )
    pdf = matched_df.toPandas()

    # With require_k=True (default), all treated should have exactly k matches
    treated_counts = pdf.groupby("id").size()
    assert all(treated_counts == ratio_k), f"All treated should have {ratio_k} matches"

    # Each control should appear at most once (no replacement)
    control_counts = pdf["match_id"].value_counts()
    assert all(control_counts == 1), "Without replacement, each control used once"

    # match_round should range from 1 to ratio_k
    assert set(pdf["match_round"].unique()) == set(range(1, ratio_k + 1))


def test_ratio_k_with_replacement(features_df):
    """Test 1-to-k matching with replacement."""
    ratio_k = 3
    matched_df = match(
        features_df, n_neighbors=10, ratio_k=ratio_k,
        with_replacement=True, verbose=False
    )
    pdf = matched_df.toPandas()

    # All treated should have exactly k matches
    treated_counts = pdf.groupby("id").size()
    assert all(treated_counts == ratio_k), f"All treated should have {ratio_k} matches"

    # Controls may be reused (count can be > 1)
    # Just verify the column exists and values are >= 1
    assert all(pdf["control_usage_count"] >= 1)


def test_reuse_max_constraint(features_df):
    """Test that reuse_max limits control reuse."""
    ratio_k = 3
    reuse_max = 2
    matched_df = match(
        features_df, n_neighbors=10, ratio_k=ratio_k,
        with_replacement=True, reuse_max=reuse_max, verbose=False
    )
    pdf = matched_df.toPandas()

    # No control should be used more than reuse_max times
    control_counts = pdf["match_id"].value_counts()
    assert all(control_counts <= reuse_max), f"No control should exceed reuse_max={reuse_max}"


def test_require_k_false_allows_partial_matches(features_df):
    """Test that require_k=False allows treated with fewer than k matches."""
    # Use a high ratio_k that might not be achievable for all treated
    ratio_k = 5
    matched_df = match(
        features_df, n_neighbors=10, ratio_k=ratio_k,
        with_replacement=False, require_k=False, verbose=False
    )
    pdf = matched_df.toPandas()

    # treated_k should reflect actual match count (may be < ratio_k)
    assert all(pdf["treated_k"] >= 1), "All matched treated should have at least 1 match"
    assert all(pdf["treated_k"] <= ratio_k), f"treated_k should not exceed {ratio_k}"


def test_round_robin_fairness(features_df):
    """Test that round-robin ensures all treated get round 1 before round 2."""
    ratio_k = 3
    matched_df = match(
        features_df, n_neighbors=10, ratio_k=ratio_k,
        with_replacement=False, verbose=False
    )
    pdf = matched_df.toPandas()

    # Count matches per round
    round_counts = pdf["match_round"].value_counts().sort_index()

    # Round 1 should have >= Round 2 >= Round 3 (fairness)
    for i in range(1, len(round_counts)):
        assert round_counts.iloc[i-1] >= round_counts.iloc[i], \
            "Earlier rounds should have at least as many matches as later rounds"


def test_pair_weight_computation(features_df):
    """Test that pair_weight = 1 / (treated_k * control_usage_count)."""
    ratio_k = 2
    matched_df = match(
        features_df, n_neighbors=10, ratio_k=ratio_k,
        with_replacement=True, verbose=False
    )
    pdf = matched_df.toPandas()

    # Verify pair_weight formula
    expected_weight = 1.0 / (pdf["treated_k"] * pdf["control_usage_count"])
    assert all(abs(pdf["pair_weight"] - expected_weight) < 1e-10), \
        "pair_weight should equal 1/(treated_k * control_usage_count)"


def test_ratio_k_validation(features_df):
    """Test that invalid ratio_k raises ValueError."""
    with pytest.raises(ValueError, match="ratio_k must be >= 1"):
        match(features_df, n_neighbors=5, ratio_k=0, verbose=False)


def test_reuse_max_validation(features_df):
    """Test that invalid reuse_max raises ValueError."""
    with pytest.raises(ValueError, match="reuse_max must be >= 1"):
        match(
            features_df, n_neighbors=5,
            with_replacement=True, reuse_max=0, verbose=False
        )


def test_reuse_max_warning_without_replacement(features_df):
    """Test that reuse_max with with_replacement=False issues warning."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        match(
            features_df, n_neighbors=5,
            with_replacement=False, reuse_max=5, verbose=False
        )
        # Check that a warning was issued
        assert any("reuse_max is ignored" in str(warning.message) for warning in w)


def test_n_neighbors_warning(features_df):
    """Test that n_neighbors < ratio_k issues warning."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        match(features_df, n_neighbors=2, ratio_k=5, verbose=False)
        assert any("n_neighbors" in str(warning.message) and "ratio_k" in str(warning.message)
                   for warning in w)


def test_match_round_ordering(features_df):
    """Test that match_round indicates preference order (1=best, 2=second best)."""
    ratio_k = 2
    matched_df = match(
        features_df, n_neighbors=10, ratio_k=ratio_k,
        with_replacement=False, verbose=False
    )
    pdf = matched_df.toPandas()

    # For each treated, round 1 match should exist
    for treated_id in pdf["id"].unique():
        treated_matches = pdf[pdf["id"] == treated_id]
        assert 1 in treated_matches["match_round"].values, \
            f"Treated {treated_id} should have a round 1 match"


def test_control_usage_count_without_replacement(features_df):
    """Test that control_usage_count is always 1 without replacement."""
    matched_df = match(
        features_df, n_neighbors=10, ratio_k=2,
        with_replacement=False, verbose=False
    )
    pdf = matched_df.toPandas()

    assert all(pdf["control_usage_count"] == 1), \
        "Without replacement, control_usage_count should always be 1"


# =============================================================================
# Tests for match_data() Function
# =============================================================================


def test_match_data_basic(spark, lalonde_df, features_df):
    """Test basic match_data functionality."""
    # Add id if not present
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    # Perform matching
    matched_df = match(features_df, n_neighbors=5, verbose=False)

    # Create matched data
    result_df = match_data(lalonde_df, matched_df, id_col="id")

    # Check that required columns exist
    assert "weights" in result_df.columns
    assert "subclass" in result_df.columns
    assert "matched" in result_df.columns

    # Check that all rows are present
    assert result_df.count() == lalonde_df.count()


def test_match_data_weights_1to1(spark, lalonde_df, features_df):
    """Test that 1:1 matching produces weight=1 for all matched rows."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    matched_df = match(features_df, n_neighbors=5, ratio_k=1, verbose=False)
    result_df = match_data(lalonde_df, matched_df, id_col="id")

    # All matched rows should have weight=1
    matched_rows = result_df.filter(F.col("matched") == True).select("weights").collect()
    assert all(row["weights"] == 1.0 for row in matched_rows), "1:1 matching should produce weight=1"


def test_match_data_weights_1to3_no_replacement(spark, lalonde_df, features_df):
    """Test that 1:3 matching without replacement produces weight=1/3 for controls."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    matched_df = match(features_df, n_neighbors=10, ratio_k=3, with_replacement=False, verbose=False)
    result_df = match_data(lalonde_df, matched_df, id_col="id")

    # Join with treatment info to separate treated/control
    result_with_treat = result_df.join(
        features_df.select("id__id", "treat__treat"),
        result_df["id"] == features_df["id__id"],
        "left"
    ).drop("id__id")

    # Treated should have weight=1
    treated_matched = result_with_treat.filter(
        (F.col("matched") == True) & (F.col("treat__treat") == 1)
    ).select("weights").collect()
    assert all(row["weights"] == 1.0 for row in treated_matched), "Treated should have weight=1"

    # Controls should have weight=1/3
    control_matched = result_with_treat.filter(
        (F.col("matched") == True) & (F.col("treat__treat") == 0)
    ).select("weights").collect()
    expected_weight = 1.0 / 3
    assert all(abs(row["weights"] - expected_weight) < 1e-10 for row in control_matched), \
        "Controls in 1:3 matching should have weight=1/3"


def test_match_data_unmatched_have_zero_weight(spark, lalonde_df, features_df):
    """Test that unmatched rows have weight=0."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    matched_df = match(features_df, n_neighbors=5, verbose=False)
    result_df = match_data(lalonde_df, matched_df, id_col="id")

    # Unmatched rows should have weight=0
    unmatched_rows = result_df.filter(F.col("matched") == False).select("weights").collect()
    assert all(row["weights"] == 0.0 for row in unmatched_rows), "Unmatched rows should have weight=0"


def test_match_data_subclass_assignment(spark, lalonde_df, features_df):
    """Test that subclass correctly identifies matched sets."""
    if "id" not in lalonde_df.columns:
        lalonde_df = lalonde_df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    matched_df = match(features_df, n_neighbors=5, verbose=False)
    result_df = match_data(lalonde_df, matched_df, id_col="id")

    # Matched rows should have non-null subclass
    matched_rows = result_df.filter(F.col("matched") == True)
    assert matched_rows.filter(F.col("subclass").isNull()).count() == 0, \
        "Matched rows should have subclass assigned"

    # Unmatched rows should have null subclass
    unmatched_rows = result_df.filter(F.col("matched") == False)
    assert unmatched_rows.filter(F.col("subclass").isNotNull()).count() == 0, \
        "Unmatched rows should have null subclass"
