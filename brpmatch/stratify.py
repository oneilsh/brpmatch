"""
Stratification for love plot visualization in BRPMatch.

This module prepares matched data for love plot visualization by creating
strata identifiers for computing balance statistics.
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def stratify_for_plot(
    features_df: DataFrame,
    matched_df: DataFrame,
    id_col: str = "person_id",
    match_id_col: str = "match_person_id",
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
    id_col : str
        Patient identifier column in features_df
    match_id_col : str
        Matched patient identifier column in matched_df

    Returns
    -------
    DataFrame
        Features DataFrame with additional columns:
        - is_matched: non-null if patient is part of a matched pair
        - strata: unique identifier for each matched pair (format: "id:match_id")
    """
    # Create strata identifier for each match pair
    with_strata = matched_df.withColumn(
        "strata", F.concat_ws(":", F.col(id_col), F.col(match_id_col))
    )

    # Join to identify treated patients who are matched
    matched_from = features_df.join(
        with_strata.select(
            F.col(id_col), F.col(id_col).alias("matched_from"), F.col("strata").alias("strata_from")
        ),
        on=id_col,
        how="left",
    )

    # Join to identify control patients who are matched
    matched_to = matched_from.join(
        with_strata.select(
            F.col(match_id_col),
            F.col(match_id_col).alias("matched_to"),
            F.col("strata").alias("strata_to"),
        ),
        matched_from[id_col] == with_strata[match_id_col],
        how="left",
    ).drop(match_id_col)

    # Coalesce to single is_matched and strata columns
    result = (
        matched_to.withColumn(
            "is_matched", F.coalesce(F.col("matched_from"), F.col("matched_to"))
        )
        .withColumn("strata", F.coalesce(F.col("strata_from"), F.col("strata_to")))
        .drop("matched_from", "matched_to", "strata_from", "strata_to")
    )

    return result
