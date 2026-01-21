"""
Summary statistics and balance reporting for BRPMatch.

This module provides MatchIt-style summary output for matched cohorts.
"""

from typing import List, Optional, Tuple, Union
import warnings

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from .stratify import stratify_for_plot
from .loveplot import love_plot, _compute_smd, _compute_variance_ratio


def match_summary(
    features_df: DataFrame,
    matched_df: DataFrame,
    feature_cols: List[str],
    treatment_col: str = "treat",
    id_col: str = "person_id",
    match_id_col: Optional[str] = None,
    sample_frac: float = 0.05,
    threshold_smd: float = 0.1,
    threshold_vr: Tuple[float, float] = (0.5, 2.0),
    plot: bool = False,
    figsize: Tuple[int, int] = (10, 12),
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
    feature_cols : List[str]
        Columns to compute balance statistics for
    treatment_col : str
        Column indicating treatment (1) vs control (0)
    id_col : str
        Patient identifier column
    match_id_col : str, optional
        Matched patient ID column in matched_df. If None, defaults to f"match_{id_col}"
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
    """
    if match_id_col is None:
        match_id_col = f"match_{id_col}"

    # Stratify data for balance computation
    stratified_df = stratify_for_plot(
        features_df, matched_df, id_col=id_col, match_id_col=match_id_col
    )

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

    # Print summary table
    _print_balance_table(balance_df, sample_frac)

    # Print sample size summary
    _print_sample_sizes(pdf, treatment_col, strata_col="strata")

    # Check thresholds and warn
    _check_balance_thresholds(balance_df, threshold_smd, threshold_vr)

    if plot:
        # Generate love plot using existing infrastructure
        fig = love_plot(
            stratified_df,
            feature_cols,
            treatment_col=treatment_col,
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
    """Print formatted balance table to console."""
    sample_note = f" (sampled {sample_frac*100:.0f}% of data)" if sample_frac < 1.0 else ""
    print(f"\nBalance Summary{sample_note}")
    print("=" * 100)

    # Format for display
    display_df = balance_df[
        ["covariate", "mean_treated", "mean_control", "smd_unadjusted",
         "smd_adjusted", "vr_unadjusted", "vr_adjusted"]
    ].copy()

    # Rename columns for display
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
        warnings.warn(
            f"Poor balance detected: {len(poor_smd)} covariate(s) have |SMD| > {threshold_smd} "
            f"after matching: {', '.join(poor_smd['covariate'].tolist())}",
            UserWarning
        )

    # Check VR threshold (only for non-binary)
    vr_min, vr_max = threshold_vr
    continuous = balance_df[~balance_df["is_binary"]]
    poor_vr = continuous[
        (continuous["vr_adjusted"] < vr_min) | (continuous["vr_adjusted"] > vr_max)
    ]
    if len(poor_vr) > 0:
        warnings.warn(
            f"Variance ratio outside [{vr_min}, {vr_max}] for {len(poor_vr)} covariate(s) "
            f"after matching: {', '.join(poor_vr['covariate'].tolist())}",
            UserWarning
        )
