"""
Stratification for love plot visualization in BRPMatch.

This module prepares matched data for love plot visualization by creating
strata identifiers for computing balance statistics.
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from .utils import _discover_id_column


def stratify_for_plot(
    features_df: DataFrame,
    matched_df: DataFrame,
) -> DataFrame:
    """
    Prepare matched data for love plot visualization.

    Joins features with match information to create strata identifiers
    for computing balance statistics.

    Parameters
    ----------
    features_df : DataFrame
        Output from generate_features()
    matched_df : DataFrame
        Output from match()

    Returns
    -------
    DataFrame
        Features DataFrame with additional columns:
        - is_matched: non-null if patient is part of a matched pair
        - strata: unique identifier for each matched pair (format: "id:match_id")

    Notes
    -----
    Column names are auto-discovered from features_df:
    - ID column: ends with __id suffix
    - Matched ID column derived from ID column base name
    """
    # Auto-discover ID column
    id_col = _discover_id_column(features_df)

    # Derive base name and match column name
    id_col_base = id_col.replace("__id", "")
    match_id_col = f"match_{id_col_base}"

    # Create strata identifier for each match pair
    with_strata = matched_df.withColumn(
        "strata", F.concat_ws(":", F.col(id_col_base), F.col(match_id_col))
    )

    # Prepare treated side (id_col_base -> treated patients)
    treated_strata = with_strata.select(
        F.col(id_col_base).alias("_treated_id"),
        F.col(id_col_base).alias("matched_from"),
        F.col("strata").alias("strata_from")
    )

    # Join to identify treated patients who are matched
    matched_from = features_df.join(
        treated_strata,
        features_df[id_col] == treated_strata["_treated_id"],
        how="left",
    ).drop("_treated_id")

    # Prepare control side (match_id_col -> control patients)
    control_strata = with_strata.select(
        F.col(match_id_col).alias("_control_id"),
        F.col(match_id_col).alias("matched_to"),
        F.col("strata").alias("strata_to")
    )

    # Join to identify control patients who are matched
    matched_to = matched_from.join(
        control_strata,
        matched_from[id_col] == control_strata["_control_id"],
        how="left",
    ).drop("_control_id")

    # Coalesce to single is_matched and strata columns
    result = (
        matched_to.withColumn(
            "is_matched", F.coalesce(F.col("matched_from"), F.col("matched_to"))
        )
        .withColumn("strata", F.coalesce(F.col("strata_from"), F.col("strata_to")))
        .drop("matched_from", "matched_to", "strata_from", "strata_to")
    )

    return result
