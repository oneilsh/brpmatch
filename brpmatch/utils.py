"""
Shared utility functions for BRPMatch.

This module contains column discovery functions and statistical computations
used across multiple modules.
"""

from typing import List

import numpy as np
from pyspark.sql import DataFrame


def _discover_id_column(df: DataFrame) -> str:
    """Find the ID column by looking for __id suffix."""
    id_cols = [c for c in df.columns if c.endswith("__id")]
    if len(id_cols) != 1:
        raise ValueError(
            f"Expected exactly one column ending with '__id', found: {id_cols}"
        )
    return id_cols[0]


def _discover_treatment_column(df: DataFrame) -> str:
    """Find the treatment column by looking for __treat suffix."""
    treat_cols = [c for c in df.columns if c.endswith("__treat")]
    if len(treat_cols) != 1:
        raise ValueError(
            f"Expected exactly one column ending with '__treat', found: {treat_cols}"
        )
    return treat_cols[0]


def _discover_exact_match_column(df: DataFrame) -> str:
    """Find the exact match grouping column by looking for __group suffix."""
    group_cols = [c for c in df.columns if c.endswith("__group")]
    if len(group_cols) != 1:
        raise ValueError(
            f"Expected exactly one column ending with '__group', found: {group_cols}"
        )
    return group_cols[0]


def _discover_feature_columns(df: DataFrame) -> List[str]:
    """Find all feature columns by suffix pattern."""
    suffixes = ("__cat", "__num", "__date", "__exact")
    return [c for c in df.columns if any(c.endswith(s) for s in suffixes)]


def _strip_suffix(col_name: str) -> str:
    """Strip the __cat, __num, __date, __exact suffix for display."""
    for suffix in ("__cat", "__num", "__date", "__exact"):
        if col_name.endswith(suffix):
            return col_name[: -len(suffix)]
    return col_name


def compute_smd(treated_values: np.ndarray, control_values: np.ndarray) -> float:
    """
    Compute standardized mean difference.

    Parameters
    ----------
    treated_values : np.ndarray
        Values from the treated group
    control_values : np.ndarray
        Values from the control group

    Returns
    -------
    float
        Standardized mean difference (SMD)
    """
    mean_t = np.nanmean(treated_values)
    mean_c = np.nanmean(control_values)
    var_t = np.nanvar(treated_values, ddof=1)
    var_c = np.nanvar(control_values, ddof=1)
    pooled_std = np.sqrt((var_t + var_c) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return (mean_t - mean_c) / pooled_std


def compute_variance_ratio(
    treated_values: np.ndarray, control_values: np.ndarray
) -> float:
    """
    Compute variance ratio.

    Parameters
    ----------
    treated_values : np.ndarray
        Values from the treated group
    control_values : np.ndarray
        Values from the control group

    Returns
    -------
    float
        Variance ratio (treated/control)
    """
    var_t = np.nanvar(treated_values, ddof=1)
    var_c = np.nanvar(control_values, ddof=1)
    if var_c < 1e-10:
        return np.inf if var_t > 1e-10 else 1.0
    return var_t / var_c
