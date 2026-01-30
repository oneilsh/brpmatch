"""
Summary statistics and balance reporting for BRPMatch.

This module provides MatchIt-style summary output for matched cohorts.
"""

from typing import List, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from .loveplot import love_plot
# from .stratify import stratify_for_plot  # No longer needed
from .utils import (
    _discover_feature_columns,
    _discover_id_column,
    _strip_suffix,
    compute_smd,
    compute_variance_ratio,
)


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
    units_df: DataFrame,
    sample_frac: float = 0.05,
    figsize: Tuple[int, int] = (10, 12),
    include_ecdf: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, matplotlib.figure.Figure]:
    """
    Generate MatchIt-style balance summary for matched cohorts.

    Computes balance statistics (SMD, Variance Ratio, eCDF) for covariates
    before and after matching, with love plot generation.

    Parameters
    ----------
    features_df : DataFrame
        Output from generate_features()
    units_df : DataFrame
        Units DataFrame from match() output (first element of returned tuple)
    sample_frac : float
        Fraction of data to sample (for large datasets). Default 0.05 (5%)
    figsize : Tuple[int, int]
        Figure size for the love plot
    include_ecdf : bool
        If True, include eCDF statistics in output (default: False)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Figure]
        Returns (balance_df, aggregate_df, figure).

        balance_df : pd.DataFrame
            Balance summary table with columns:
            - display_name: feature name (cleaned)
            - mean_treated, mean_control: group means (unadjusted)
            - mean_treated_adj, mean_control_adj: group means (adjusted/matched)
            - smd_unadjusted, smd_adjusted: standardized mean difference
            - vr_unadjusted, vr_adjusted: variance ratio (continuous vars only)
            - ecdf_mean_unadj, ecdf_mean_adj: mean eCDF difference (if include_ecdf=True)
            - ecdf_max_unadj, ecdf_max_adj: max eCDF difference (if include_ecdf=True)

        aggregate_df : pd.DataFrame
            Aggregate matching statistics with columns:
            - statistic: name of the statistic
            - value: numeric value
            Includes: cohort sizes before/after matching, match rates,
            average controls per treated, effective sample sizes, etc.

        figure : matplotlib.figure.Figure
            Love plot visualization of balance statistics.

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

    # Join features_df with units_df directly
    stratified_df = features_df.join(
        units_df.select("id", "subclass", "weight"),
        features_df[id_col] == units_df["id"],
        "left"
    ).drop(units_df["id"])

    # Rename subclass to strata for compatibility with existing balance computation
    stratified_df = stratified_df.withColumnRenamed("subclass", "strata")

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
        pdf, feature_cols, treatment_col, strata_col="strata",
        include_ecdf=include_ecdf
    )

    # Add display names and feature types to balance DataFrame
    balance_df["display_name"] = balance_df["covariate"].apply(_strip_suffix)
    balance_df["feature_type"] = balance_df["covariate"].apply(_get_column_type)

    # Clean up the returned DataFrame: drop internal columns, reorder with display_name first
    columns_to_drop = ["covariate", "is_binary", "feature_type"]
    balance_df = balance_df.drop(columns=columns_to_drop)
    # Move display_name to first column
    cols = ["display_name"] + [c for c in balance_df.columns if c != "display_name"]
    balance_df = balance_df[cols]

    # Compute aggregate matching statistics
    aggregate_df = _compute_aggregate_stats(units_df, sample_frac=sample_frac)

    # Generate love plot
    fig = love_plot(
        stratified_df,
        strata_col="strata",
        sample_frac=1.0,  # Already sampled above
        figsize=figsize,
    )

    return balance_df, aggregate_df, fig


def _compute_comprehensive_balance(
    pdf: pd.DataFrame,
    feature_cols: List[str],
    treatment_col: str,
    strata_col: str,
    include_ecdf: bool = False,
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
            "smd_unadjusted": compute_smd(t_vals, c_vals),
            "smd_adjusted": compute_smd(t_vals_adj, c_vals_adj),
            "vr_unadjusted": compute_variance_ratio(t_vals, c_vals) if not is_binary else np.nan,
            "vr_adjusted": compute_variance_ratio(t_vals_adj, c_vals_adj) if not is_binary else np.nan,
            "is_binary": is_binary,
        }
        if include_ecdf:
            result["ecdf_mean_unadj"] = _compute_ecdf_mean(t_vals, c_vals)
            result["ecdf_mean_adj"] = _compute_ecdf_mean(t_vals_adj, c_vals_adj)
            result["ecdf_max_unadj"] = _compute_ecdf_max(t_vals, c_vals)
            result["ecdf_max_adj"] = _compute_ecdf_max(t_vals_adj, c_vals_adj)
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


def _compute_aggregate_stats(units_df: DataFrame, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Compute aggregate matching statistics from units DataFrame.

    Parameters
    ----------
    units_df : DataFrame
        Units DataFrame from match() with columns: id, subclass, weight, is_treated
    sample_frac : float
        Fraction of data sampled for analysis (default 1.0 = no sampling)

    Returns
    -------
    pd.DataFrame
        DataFrame with statistic names and values
    """
    # Collect units data for computation
    units_pdf = units_df.toPandas()

    # Split by treatment status
    treated = units_pdf[units_pdf["is_treated"] == True]  # noqa: E712
    control = units_pdf[units_pdf["is_treated"] == False]  # noqa: E712

    # Matched = has non-null subclass
    treated_matched = treated[treated["subclass"].notna()]
    control_matched = control[control["subclass"].notna()]

    # Cohort sizes
    n_treated_total = len(treated)
    n_control_total = len(control)
    n_treated_matched = len(treated_matched)
    n_control_matched = len(control_matched)
    n_treated_unmatched = n_treated_total - n_treated_matched
    n_control_unmatched = n_control_total - n_control_matched

    # Match rates
    pct_treated_matched = (n_treated_matched / n_treated_total * 100) if n_treated_total > 0 else 0.0
    pct_control_matched = (n_control_matched / n_control_total * 100) if n_control_total > 0 else 0.0

    # Average controls per treated (from subclass counts)
    # Each unique subclass corresponds to one treated patient
    if n_treated_matched > 0:
        # Count controls per subclass (treated ID)
        controls_per_treated = control_matched.groupby("subclass").size()
        mean_controls_per_treated = controls_per_treated.mean()
        min_controls_per_treated = controls_per_treated.min()
        max_controls_per_treated = controls_per_treated.max()
    else:
        mean_controls_per_treated = 0.0
        min_controls_per_treated = 0.0
        max_controls_per_treated = 0.0

    # Effective Sample Size (ESS) for treated and control
    # ESS = (sum of weights)^2 / sum of weights^2
    # For matched units only (weight > 0)
    treated_weights = treated_matched["weight"].values
    control_weights = control_matched["weight"].values

    if len(treated_weights) > 0 and np.sum(treated_weights) > 0:
        effective_sample_size_treated = (np.sum(treated_weights) ** 2) / np.sum(treated_weights ** 2)
    else:
        effective_sample_size_treated = 0.0

    if len(control_weights) > 0 and np.sum(control_weights) > 0:
        effective_sample_size_control = (np.sum(control_weights) ** 2) / np.sum(control_weights ** 2)
    else:
        effective_sample_size_control = 0.0

    # Build results DataFrame
    stats = [
        ("sample_frac", sample_frac),
        ("n_treated_total", n_treated_total),
        ("n_control_total", n_control_total),
        ("n_treated_matched", n_treated_matched),
        ("n_control_matched", n_control_matched),
        ("n_treated_unmatched", n_treated_unmatched),
        ("n_control_unmatched", n_control_unmatched),
        ("pct_treated_matched", pct_treated_matched),
        ("pct_control_matched", pct_control_matched),
        ("mean_controls_per_treated", mean_controls_per_treated),
        ("min_controls_per_treated", min_controls_per_treated),
        ("max_controls_per_treated", max_controls_per_treated),
        ("effective_sample_size_treated", effective_sample_size_treated),
        ("effective_sample_size_control", effective_sample_size_control),
    ]

    return pd.DataFrame(stats, columns=["statistic", "value"])


