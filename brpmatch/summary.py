"""
Summary statistics and balance reporting for BRPMatch.

This module provides MatchIt-style summary output for matched cohorts.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from .stratify import stratify_for_plot
from .loveplot import love_plot, _compute_smd, _compute_variance_ratio
from .warnings_util import warn


def _discover_feature_columns(df: DataFrame) -> List[str]:
    """Find all feature columns by suffix pattern."""
    suffixes = ("__cat", "__num", "__date", "__exact")
    return [c for c in df.columns if any(c.endswith(s) for s in suffixes)]


def _discover_id_column(df: DataFrame) -> str:
    """Find the ID column by looking for __id suffix."""
    id_cols = [c for c in df.columns if c.endswith("__id")]
    if len(id_cols) != 1:
        raise ValueError(
            f"Expected exactly one column ending with '__id', found: {id_cols}"
        )
    return id_cols[0]


def _strip_suffix(col_name: str) -> str:
    """Strip the __cat, __num, __date, __exact suffix for display."""
    for suffix in ("__cat", "__num", "__date", "__exact"):
        if col_name.endswith(suffix):
            return col_name[:-len(suffix)]
    return col_name


def _get_column_type(col_name: str) -> str:
    """Get the column type from its suffix."""
    if col_name.endswith("__exact"):
        return "exact"
    elif col_name.endswith("__cat"):
        return "categorical"
    elif col_name.endswith("__num"):
        return "numeric"
    elif col_name.endswith("__date"):
        return "date"
    return "unknown"


def match_summary(
    features_df: DataFrame,
    matched_df: DataFrame,
    sample_frac: float = 0.05,
    threshold_smd: float = 0.1,
    threshold_vr: Tuple[float, float] = (0.5, 2.0),
    plot: bool = False,
    figsize: Tuple[int, int] = (10, 12),
    verbose: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, matplotlib.figure.Figure]]:
    """
    Generate MatchIt-style balance summary for matched cohorts.

    Computes balance statistics (SMD, Variance Ratio, eCDF) for covariates
    before and after matching, with optional love plot generation.

    Parameters
    ----------
    features_df : DataFrame
        Output from generate_features()
    matched_df : DataFrame
        Output from match()
    sample_frac : float
        Fraction of data to sample (for large datasets). Default 0.05 (5%)
    threshold_smd : float
        Warn if any |SMD| exceeds this threshold. Default 0.1 (MatchIt convention)
    threshold_vr : Tuple[float, float]
        Warn if any VR outside this range. Default (0.5, 2.0) (MatchIt convention)
    plot : bool
        If True, also generate and return a love plot
    figsize : Tuple[int, int]
        Figure size if plot=True
    verbose : bool
        If True, print balance summary table and sample sizes (default: True)

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Figure]
        Balance summary table with columns:
        - covariate: feature name
        - mean_treated, mean_control: group means (unadjusted)
        - mean_treated_adj, mean_control_adj: group means (adjusted/matched)
        - smd_unadjusted, smd_adjusted: standardized mean difference
        - vr_unadjusted, vr_adjusted: variance ratio (continuous vars only)
        - ecdf_mean_unadj, ecdf_mean_adj: mean eCDF difference
        - ecdf_max_unadj, ecdf_max_adj: max eCDF difference (KS statistic)

        If plot=True, returns (summary_df, figure) tuple.

    Notes
    -----
    Column names are auto-discovered from features_df:
    - Feature columns: end with __cat, __num, __date, or __exact suffixes
    - ID column: ends with __id suffix
    - Treatment column: standardized as treat__treat
    """
    # Auto-discover columns
    feature_cols = _discover_feature_columns(features_df)
    id_col = _discover_id_column(features_df)
    treatment_col = "treat__treat"  # Standardized name

    # Derive match_id_col from id_col
    id_col_base = id_col.replace("__id", "")
    match_id_col = f"match_{id_col_base}"

    # Stratify data for balance computation
    stratified_df = stratify_for_plot(features_df, matched_df)

    # Sample if needed
    if sample_frac < 1.0:
        stratified_df = stratified_df.sample(
            withReplacement=False, fraction=sample_frac, seed=42
        )

    # Collect to pandas for computation
    pdf = stratified_df.select(
        feature_cols + [treatment_col, "strata"]
    ).toPandas()

    # Compute comprehensive balance statistics
    balance_df = _compute_comprehensive_balance(
        pdf, feature_cols, treatment_col, strata_col="strata"
    )

    # Add display names and feature types to balance DataFrame
    balance_df["display_name"] = balance_df["covariate"].apply(_strip_suffix)
    balance_df["feature_type"] = balance_df["covariate"].apply(_get_column_type)

    # Print summary table and sample sizes if verbose
    if verbose:
        _print_balance_table(balance_df, sample_frac)
        _print_sample_sizes(pdf, treatment_col, strata_col="strata")

    # Check thresholds and warn
    _check_balance_thresholds(balance_df, threshold_smd, threshold_vr)

    if plot:
        # Generate love plot using existing infrastructure
        fig = love_plot(
            stratified_df,
            strata_col="strata",
            sample_frac=1.0,  # Already sampled above
            figsize=figsize,
        )
        return balance_df, fig

    return balance_df


def _compute_comprehensive_balance(
    pdf: pd.DataFrame,
    feature_cols: List[str],
    treatment_col: str,
    strata_col: str,
) -> pd.DataFrame:
    """
    Compute comprehensive balance statistics including eCDF metrics.

    Returns DataFrame with all balance statistics for each covariate.
    """
    results = []

    treated = pdf[pdf[treatment_col] == 1]
    control = pdf[pdf[treatment_col] == 0]

    matched = pdf[pdf[strata_col].notna()]
    treated_matched = matched[matched[treatment_col] == 1]
    control_matched = matched[matched[treatment_col] == 0]

    for col in feature_cols:
        t_vals = treated[col].values
        c_vals = control[col].values
        t_vals_adj = treated_matched[col].values
        c_vals_adj = control_matched[col].values

        # Determine if binary (for VR display)
        is_binary = _is_binary_column(pdf[col])

        # Compute statistics
        result = {
            "covariate": col,
            "mean_treated": np.nanmean(t_vals),
            "mean_control": np.nanmean(c_vals),
            "mean_treated_adj": np.nanmean(t_vals_adj),
            "mean_control_adj": np.nanmean(c_vals_adj),
            "smd_unadjusted": _compute_smd(t_vals, c_vals),
            "smd_adjusted": _compute_smd(t_vals_adj, c_vals_adj),
            "vr_unadjusted": _compute_variance_ratio(t_vals, c_vals) if not is_binary else np.nan,
            "vr_adjusted": _compute_variance_ratio(t_vals_adj, c_vals_adj) if not is_binary else np.nan,
            "ecdf_mean_unadj": _compute_ecdf_mean(t_vals, c_vals),
            "ecdf_mean_adj": _compute_ecdf_mean(t_vals_adj, c_vals_adj),
            "ecdf_max_unadj": _compute_ecdf_max(t_vals, c_vals),
            "ecdf_max_adj": _compute_ecdf_max(t_vals_adj, c_vals_adj),
            "is_binary": is_binary,
        }
        results.append(result)

    return pd.DataFrame(results)


def _is_binary_column(series: pd.Series) -> bool:
    """Check if a column contains only binary (0/1) values."""
    unique_vals = series.dropna().unique()
    return len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})


def _compute_ecdf_mean(treated_values: np.ndarray, control_values: np.ndarray) -> float:
    """
    Compute mean absolute difference in empirical CDFs.

    This measures the average distance between the two distributions
    across all values.
    """
    # Remove NaN values
    t = treated_values[~np.isnan(treated_values)]
    c = control_values[~np.isnan(control_values)]

    if len(t) == 0 or len(c) == 0:
        return np.nan

    # Combine and sort all unique values
    all_vals = np.unique(np.concatenate([t, c]))

    # Compute eCDF at each point
    ecdf_t = np.array([np.mean(t <= v) for v in all_vals])
    ecdf_c = np.array([np.mean(c <= v) for v in all_vals])

    # Mean absolute difference
    return np.mean(np.abs(ecdf_t - ecdf_c))


def _compute_ecdf_max(treated_values: np.ndarray, control_values: np.ndarray) -> float:
    """
    Compute max absolute difference in empirical CDFs (Kolmogorov-Smirnov statistic).

    This is the maximum vertical distance between the two eCDFs.
    """
    # Remove NaN values
    t = treated_values[~np.isnan(treated_values)]
    c = control_values[~np.isnan(control_values)]

    if len(t) == 0 or len(c) == 0:
        return np.nan

    # Combine and sort all unique values
    all_vals = np.unique(np.concatenate([t, c]))

    # Compute eCDF at each point
    ecdf_t = np.array([np.mean(t <= v) for v in all_vals])
    ecdf_c = np.array([np.mean(c <= v) for v in all_vals])

    # Max absolute difference
    return np.max(np.abs(ecdf_t - ecdf_c))


def _print_balance_table(balance_df: pd.DataFrame, sample_frac: float) -> None:
    """Print formatted balance table with clean display names."""
    sample_note = f" (sampled {sample_frac*100:.0f}% of data)" if sample_frac < 1.0 else ""
    print(f"\nBalance Summary{sample_note}")
    print("=" * 100)

    # Format for display using display_name instead of raw column name
    display_df = balance_df[
        ["display_name", "mean_treated", "mean_control", "smd_unadjusted",
         "smd_adjusted", "vr_unadjusted", "vr_adjusted", "feature_type"]
    ].copy()

    # Add indicator for exact match columns
    display_df["Covariate"] = display_df.apply(
        lambda row: f"{row['display_name']} (exact)" if row["feature_type"] == "exact" else row["display_name"],
        axis=1
    )

    # Select and rename columns for final display
    display_df = display_df[
        ["Covariate", "mean_treated", "mean_control", "smd_unadjusted",
         "smd_adjusted", "vr_unadjusted", "vr_adjusted"]
    ].copy()

    display_df.columns = [
        "Covariate", "Mean Treated", "Mean Control",
        "SMD (Unadj)", "SMD (Adj)", "VR (Unadj)", "VR (Adj)"
    ]

    # Format numeric columns
    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) and not np.isinf(x) else "-"
        )

    print(display_df.to_string(index=False))
    print("=" * 100)


def _print_sample_sizes(pdf: pd.DataFrame, treatment_col: str, strata_col: str) -> None:
    """Print sample size summary."""
    treated = pdf[pdf[treatment_col] == 1]
    control = pdf[pdf[treatment_col] == 0]
    matched = pdf[pdf[strata_col].notna()]
    treated_matched = matched[matched[treatment_col] == 1]
    control_matched = matched[matched[treatment_col] == 0]

    print("\nSample sizes:")
    print(f"  Treated: {len(treated)} (matched: {len(treated_matched)}, unmatched: {len(treated) - len(treated_matched)})")
    print(f"  Control: {len(control)} (matched: {len(control_matched)}, unmatched: {len(control) - len(control_matched)})")


def _check_balance_thresholds(
    balance_df: pd.DataFrame,
    threshold_smd: float,
    threshold_vr: Tuple[float, float]
) -> None:
    """Check balance thresholds and emit warnings."""
    # Check SMD threshold
    poor_smd = balance_df[balance_df["smd_adjusted"].abs() > threshold_smd]
    if len(poor_smd) > 0:
        warn(
            f"Poor balance detected: {len(poor_smd)} covariate(s) have |SMD| > {threshold_smd} "
            f"after matching: {', '.join(poor_smd['covariate'].tolist())}"
        )

    # Check VR threshold (only for non-binary)
    vr_min, vr_max = threshold_vr
    continuous = balance_df[~balance_df["is_binary"]]
    poor_vr = continuous[
        (continuous["vr_adjusted"] < vr_min) | (continuous["vr_adjusted"] > vr_max)
    ]
    if len(poor_vr) > 0:
        warn(
            f"Variance ratio outside [{vr_min}, {vr_max}] for {len(poor_vr)} covariate(s) "
            f"after matching: {', '.join(poor_vr['covariate'].tolist())}"
        )


def match_data(
    original_df: DataFrame,
    matched_df: DataFrame,
    id_col: str,
) -> DataFrame:
    """
    Create a matched dataset with weights for downstream analysis.

    Analogous to R matchit's match.data() function. Computes ATT (Average Treatment
    Effect on the Treated) weights following the methodology in Greifer (2021) and
    the MatchIt package.

    Note: These weights estimate ATT, not ATE (Average Treatment Effect). This is
    inherent to matching: we keep treated units as-is and find similar controls,
    answering "what would have happened to the treated if they hadn't been treated?"
    ATT is appropriate when treatment has barriers to participation or when you want
    to evaluate the effect on those who actually received treatment. For ATE
    estimation, consider propensity score weighting (IPTW) instead of matching.

    Weight calculation:
    - Treated units: weight = 1
    - Control units: weight = sum of 1/k for each match, where k is the number
      of controls matched to that treated unit (treated_k in matched_df)

    This implements the "stratum propensity score" weighting approach where
    matched sets define strata, and inverse probability weights are computed
    based on the proportion of treated units in each stratum.

    References:
    - https://ngreifer.github.io/blog/matching-weights/
    - https://kosukeimai.github.io/MatchIt/reference/matchit.html

    Parameters
    ----------
    original_df : DataFrame
        Original input DataFrame (before generate_features)
    matched_df : DataFrame
        Output from match()
    id_col : str
        Name of the ID column in original_df

    Returns
    -------
    DataFrame
        DataFrame with original columns plus:
        - weights: ATT estimation weights (see formula above; 0 for unmatched)
        - subclass: Match group identifier (treated_id for matched pairs)
        - matched: Boolean indicating if row was matched

    Example
    -------
    >>> result = match_data(df, matched_df, id_col="person_id")
    >>> # Use for weighted regression:
    >>> result.filter(F.col("matched")).select("outcome", "treatment", "weights", ...)
    """
    # Get the match ID column name from matched_df
    match_id_col = None
    for c in matched_df.columns:
        if c.startswith("match_") and c != "match_round":
            match_id_col = c
            break

    if match_id_col is None:
        raise ValueError("Could not find match ID column in matched_df")

    # Extract base ID column name (e.g., "person_id" from "match_person_id")
    id_col_base = match_id_col.replace("match_", "")

    # Create subclass (match group) identifier
    # Each treated patient defines a subclass
    matched_with_subclass = matched_df.withColumn(
        "subclass", F.col(id_col_base)
    )

    # ----- Treated patient weights -----
    # Treated units always receive weight = 1
    treated_weights = matched_with_subclass.select(
        F.col(id_col_base).alias("_join_id"),
        F.lit(1.0).alias("weights"),
        F.col("subclass"),
        F.lit(True).alias("matched")
    ).distinct()

    # ----- Control patient weights -----
    # Weight = sum of 1/treated_k for each match this control appears in
    #
    # For each row in matched_df:
    #   - treated_k = number of controls matched to that treated unit
    #   - This control's contribution from this match = 1/treated_k
    #
    # If matching with replacement, a control may appear in multiple rows,
    # so we sum across all matches.
    #
    # Example: control C matched to T1 (treated_k=3) and T2 (treated_k=2)
    #   weight = 1/3 + 1/2 = 5/6
    control_weights = matched_with_subclass.withColumn(
        "_weight_contribution", 1.0 / F.col("treated_k")
    ).groupBy(match_id_col).agg(
        F.sum("_weight_contribution").alias("weights"),
        F.first("subclass").alias("subclass"),  # Use first subclass if matched to multiple treated
        F.lit(True).alias("matched")
    ).withColumnRenamed(match_id_col, "_join_id")

    # Combine treated and control weights
    all_weights = treated_weights.unionByName(control_weights)

    # Join with original data
    result = original_df.join(
        all_weights,
        original_df[id_col] == all_weights["_join_id"],
        "left"
    ).drop("_join_id")

    # Fill unmatched rows
    result = result.withColumn(
        "weights", F.coalesce(F.col("weights"), F.lit(0.0))
    ).withColumn(
        "matched", F.coalesce(F.col("matched"), F.lit(False))
    ).withColumn(
        "subclass", F.coalesce(F.col("subclass"), F.lit(None))
    )

    return result
