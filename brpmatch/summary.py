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
    units_df : DataFrame
        Units DataFrame from match() output (first element of returned tuple)
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
        pdf, feature_cols, treatment_col, strata_col="strata"
    )

    # Add display names and feature types to balance DataFrame
    balance_df["display_name"] = balance_df["covariate"].apply(_strip_suffix)
    balance_df["feature_type"] = balance_df["covariate"].apply(_get_column_type)

    # Print summary table and sample sizes if verbose
    if verbose:
        _print_balance_table(balance_df, sample_frac)
        _print_sample_sizes(pdf, treatment_col, strata_col="strata")

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
            "smd_unadjusted": compute_smd(t_vals, c_vals),
            "smd_adjusted": compute_smd(t_vals_adj, c_vals_adj),
            "vr_unadjusted": compute_variance_ratio(t_vals, c_vals) if not is_binary else np.nan,
            "vr_adjusted": compute_variance_ratio(t_vals_adj, c_vals_adj) if not is_binary else np.nan,
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


