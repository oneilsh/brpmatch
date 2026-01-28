"""
Love plot visualization for BRPMatch covariate balance.

This module generates love plots showing covariate balance before and after matching.
"""

from typing import List, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame


def _strip_suffix(col_name: str) -> str:
    """Strip the __cat, __num, __date, __exact suffix for display."""
    for suffix in ("__cat", "__num", "__date", "__exact"):
        if col_name.endswith(suffix):
            return col_name[:-len(suffix)]
    return col_name


def _is_exact_match_column(col_name: str) -> bool:
    """Check if column is an exact match column."""
    return col_name.endswith("__exact")


def _is_categorical_column(col_name: str) -> bool:
    """Check if column is a categorical column."""
    return col_name.endswith("__cat")


def love_plot(
    stratified_df: DataFrame,
    strata_col: str = "strata",
    sample_frac: float = 0.05,
    figsize: Tuple[int, int] = (10, 12),
) -> matplotlib.figure.Figure:
    """
    Generate a love plot showing covariate balance before and after matching.

    Parameters
    ----------
    stratified_df : DataFrame
        Output from stratify_for_plot()
    strata_col : str
        Column identifying matched pairs
    sample_frac : float
        Fraction of data to sample for plotting (for large datasets)
    figsize : Tuple[int, int]
        Figure size (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
        Love plot figure with two panels:
        - Left: Absolute Standardized Mean Difference
        - Right: Variance Ratio

    Notes
    -----
    Column names are auto-discovered from stratified_df:
    - Feature columns: end with __cat, __num, __date, or __exact suffixes
    - Treatment column: standardized as treat__treat
    - Display names strip suffixes and mark exact match columns with "(exact)"
    """
    # Auto-discover feature columns
    suffixes = ("__cat", "__num", "__date", "__exact")
    feature_cols = [c for c in stratified_df.columns if any(c.endswith(s) for s in suffixes)]
    treatment_col = "treat__treat"

    # Sample data if needed
    if sample_frac < 1.0:
        stratified_df = stratified_df.sample(withReplacement=False, fraction=sample_frac, seed=42)

    # Collect data to pandas
    pdf = stratified_df.select(feature_cols + [treatment_col, strata_col]).toPandas()

    # Compute balance statistics
    balance_df = _compute_balance_stats(pdf, feature_cols, treatment_col, strata_col)

    # Calculate improvement for ordering
    balance_df["improvement"] = abs(balance_df["smd_unadjusted"]) - abs(
        balance_df["smd_adjusted"]
    )

    # Create display names
    balance_df["display_name"] = balance_df["covariate"].apply(
        lambda col: f"{_strip_suffix(col)} (exact)" if _is_exact_match_column(col) else _strip_suffix(col)
    )

    # Reshape for plotting
    plot_df = balance_df.melt(
        id_vars=["covariate", "display_name", "improvement"],
        value_vars=["smd_unadjusted", "smd_adjusted", "vr_unadjusted", "vr_adjusted"],
        var_name="eval_variable",
        value_name="eval_value",
    )

    # Parse variable names
    plot_df["test"] = plot_df["eval_variable"].apply(
        lambda x: (
            "Absolute Standardized Mean Difference"
            if x.startswith("smd")
            else "Variance Ratio"
        )
    )
    plot_df["set"] = plot_df["eval_variable"].apply(
        lambda x: "Unadjusted" if x.endswith("unadjusted") else "Adjusted"
    )

    # Take absolute value of SMD
    smd_mask = plot_df["test"] == "Absolute Standardized Mean Difference"
    plot_df.loc[smd_mask, "eval_value"] = plot_df.loc[smd_mask, "eval_value"].abs()

    # Sort by improvement
    plot_df = plot_df.sort_values("improvement")
    display_name_order = (
        balance_df.sort_values("improvement")["display_name"].unique().tolist()
    )

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Plot SMD
    smd_data = plot_df[plot_df["test"] == "Absolute Standardized Mean Difference"]
    for sample_type in ["Unadjusted", "Adjusted"]:
        subset = smd_data[smd_data["set"] == sample_type]
        # Check if exact match for marker style
        is_exact = subset["covariate"].apply(_is_exact_match_column)
        # Plot exact match with square markers
        if is_exact.any():
            exact_subset = subset[is_exact]
            axes[0].scatter(
                exact_subset["eval_value"],
                exact_subset["display_name"],
                label=sample_type if not subset[~is_exact].empty else sample_type,
                alpha=0.7,
                s=50,
                marker='s',
            )
        # Plot non-exact with circle markers
        if (~is_exact).any():
            non_exact_subset = subset[~is_exact]
            axes[0].scatter(
                non_exact_subset["eval_value"],
                non_exact_subset["display_name"],
                label=sample_type if is_exact.any() else sample_type,
                alpha=0.7,
                s=50,
                marker='o',
            )
    axes[0].set_xlabel("Absolute Standardized Mean Difference")
    axes[0].set_ylabel("Variable")
    axes[0].legend(title="Sample")
    axes[0].grid(True, alpha=0.3)
    # Add reference line at SMD = 0.1
    axes[0].axvline(x=0.1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    # Add annotation for the threshold
    axes[0].text(0.105, 0, "0.1 threshold", color='red', fontsize=10,
                 verticalalignment='bottom', horizontalalignment='left')

    # Plot Variance Ratio (excluding categorical variables with NaN values)
    vr_data = plot_df[
        (plot_df["test"] == "Variance Ratio") & (plot_df["eval_value"].notna())
    ]
    for sample_type in ["Unadjusted", "Adjusted"]:
        subset = vr_data[vr_data["set"] == sample_type]
        # Check if exact match for marker style
        is_exact = subset["covariate"].apply(_is_exact_match_column)
        # Plot exact match with square markers
        if is_exact.any():
            exact_subset = subset[is_exact]
            axes[1].scatter(
                exact_subset["eval_value"],
                exact_subset["display_name"],
                label=sample_type if not subset[~is_exact].empty else sample_type,
                alpha=0.7,
                s=50,
                marker='s',
            )
        # Plot non-exact with circle markers
        if (~is_exact).any():
            non_exact_subset = subset[~is_exact]
            axes[1].scatter(
                non_exact_subset["eval_value"],
                non_exact_subset["display_name"],
                label=sample_type if is_exact.any() else sample_type,
                alpha=0.7,
                s=50,
                marker='o',
            )
    axes[1].set_xlabel("Variance Ratio")
    axes[1].legend(title="Sample")
    axes[1].grid(True, alpha=0.3)
    # Add reference line at VR = 1.0
    axes[1].axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    # Add annotation for equal variance
    axes[1].text(1.02, 0, "equal variance", color='red', fontsize=10,
                 verticalalignment='bottom', horizontalalignment='left')

    # Set y-axis to show display names in order
    axes[0].set_yticks(range(len(display_name_order)))
    axes[0].set_yticklabels(display_name_order, fontsize=7)

    plt.tight_layout()

    return fig


def _compute_balance_stats(
    pdf: pd.DataFrame, feature_cols: List[str], treatment_col: str, strata_col: str
) -> pd.DataFrame:
    """
    Compute balance statistics for all features.

    Returns DataFrame with columns:
    - covariate: feature name
    - smd_unadjusted: SMD on all data
    - smd_adjusted: SMD on matched pairs only
    - vr_unadjusted: Variance ratio on all data
    - vr_adjusted: Variance ratio on matched pairs only
    """
    results = []

    treated = pdf[pdf[treatment_col] == 1]
    control = pdf[pdf[treatment_col] == 0]

    matched = pdf[pdf[strata_col].notna()]
    treated_matched = matched[matched[treatment_col] == 1]
    control_matched = matched[matched[treatment_col] == 0]

    for col in feature_cols:
        smd_un = _compute_smd(treated[col].values, control[col].values)
        smd_adj = _compute_smd(
            treated_matched[col].values, control_matched[col].values
        )

        # Skip variance ratio for categorical columns
        if _is_categorical_column(col):
            vr_un = np.nan
            vr_adj = np.nan
        else:
            vr_un = _compute_variance_ratio(treated[col].values, control[col].values)
            vr_adj = _compute_variance_ratio(
                treated_matched[col].values, control_matched[col].values
            )

        results.append(
            {
                "covariate": col,
                "smd_unadjusted": smd_un,
                "smd_adjusted": smd_adj,
                "vr_unadjusted": vr_un,
                "vr_adjusted": vr_adj,
            }
        )

    return pd.DataFrame(results)


def _compute_smd(treated_values: np.ndarray, control_values: np.ndarray) -> float:
    """Compute standardized mean difference."""
    mean_t = np.nanmean(treated_values)
    mean_c = np.nanmean(control_values)
    var_t = np.nanvar(treated_values, ddof=1)
    var_c = np.nanvar(control_values, ddof=1)
    pooled_std = np.sqrt((var_t + var_c) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean_t - mean_c) / pooled_std


def _compute_variance_ratio(
    treated_values: np.ndarray, control_values: np.ndarray
) -> float:
    """Compute variance ratio."""
    var_t = np.nanvar(treated_values, ddof=1)
    var_c = np.nanvar(control_values, ddof=1)
    if var_c == 0:
        return np.inf if var_t > 0 else 1.0
    return var_t / var_c
