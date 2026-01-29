"""
Example: 1:3 Matching WITHOUT Replacement - Euclidean Distance

Demonstrates 1:3 matching where each treated gets up to 3 controls.
Controls are used at most once. Round-robin algorithm ensures fairness.
"""

import os
import sys

from pyspark.sql import functions as F

# Add parent directory to path for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brpmatch import generate_features, match, match_summary
from utils import create_spark_session, load_lalonde, setup_pandas_display, print_matching_stats
from utils import section_header, subsection_header, highlight, value

setup_pandas_display()


def main():
    """
    Demonstrates 1:3 propensity score matching WITHOUT replacement using BRPMatch.

    Workflow:
        1. Load raw patient data
        2. Generate features (transform and standardize covariates)
        3. Match treated to control patients using LSH-based k-NN (ratio_k=3)
        4. Assess balance (compute SMD, VR, and generate love plot)
        5. Create stratified dataset for outcome analysis
        6. Save all outputs (matched pairs, stratified data, balance statistics)

    This example uses Euclidean distance in standardized feature space.
    Each treated patient is matched to UP TO 3 control patients.
    Without replacement: each control can only be used once (round-robin algorithm ensures fairness).
    """
    print(section_header("1:3 Matching WITHOUT Replacement - Euclidean Distance", "="))

    # Create Spark session
    spark = create_spark_session("brpmatch-1to3-no-repl-euclidean")
    output_dir = os.path.dirname(__file__) or "."

    # Load data
    print(f"\n{subsection_header('Loading data')}...")
    df = load_lalonde(spark)
    print(f"  Loaded {value(str(df.count()))} rows")

    # Generate matching features from raw data
    # This function transforms raw data into standardized features for matching:
    #   1. Categorical columns → one-hot encoded
    #   2. Numeric columns → cast to double (no transformation yet)
    #   3. Date columns → converted to days from reference date (if provided)
    #   4. All features assembled into vector → standardized (scaled by std dev)
    #
    # Column naming scheme: original columns preserved, new columns added with suffixes:
    #   - __cat: one-hot encoded categorical (e.g., race_black__cat, married_1__cat)
    #   - __num: numeric features cast to double (e.g., age__num, educ__num)
    #   - __date: date features as days from reference (e.g., diagnosis_date__date)
    #   - __treat: treatment indicator (0/1) (e.g., treat__treat)
    #   - __id: patient identifier (e.g., id__id)
    #   - features: assembled and standardized feature vector (used for matching)
    print(f"\n{subsection_header('Generating features')}...")
    features_df = generate_features(
        spark, df,
        treatment_col="treat",
        treatment_value="1",
        categorical_cols=["race", "married", "nodegree"],
        numeric_cols=["age", "educ", "re74", "re75"],
        id_col="id",
    )
    print(f"  Generated {value(str(len([c for c in features_df.columns if '__' in c])))} feature columns")

    # Match treated to control patients using LSH-based matching
    # The algorithm uses Locality-Sensitive Hashing (LSH) to bucket similar patients,
    # then finds nearest neighbors within buckets using k-NN in the specified feature space.
    # For Euclidean distance, uses standardized features directly.
    #
    # With ratio_k=3 and with_replacement=False:
    #   - Each treated patient gets UP TO 3 control matches
    #   - Each control can only be matched once (without replacement)
    #   - Round-robin algorithm: all treated get round 1 matches before any get round 2
    #   - Some treated may get fewer than 3 matches if controls are exhausted
    #
    # Output: tuple of (units, pairs, bucket_stats) DataFrames
    # - units: all units with match status, subclass, and weights
    # - pairs: treated-control pairs (PRIMARY OUTPUT for linking outcomes)
    # - bucket_stats: LSH bucketing statistics for diagnostics
    print(f"\n{subsection_header('Matching')}...")
    units, pairs, bucket_stats = match(
        features_df,
        feature_space="euclidean",
        n_neighbors=10,
        ratio_k=3,
        with_replacement=False,
        verbose=False,
    )
    print_matching_stats(features_df, pairs)

    # Show distribution of matches per treated (may vary when controls are limited)
    matches_per_treated = pairs.groupBy("id").count().toPandas()
    print(f"  Matches per treated: min={matches_per_treated['count'].min()}, "
          f"mean={matches_per_treated['count'].mean():.1f}, "
          f"max={matches_per_treated['count'].max()}")

    # Show sample matched pairs to inspect the matching results
    # Key columns to understand:
    #   - id: treated patient ID
    #   - match_id: matched control patient ID
    #   - pair_weight: weight for analysis = 1/(treated_k * control_usage_count)
    #   - match_round: which round this match came from (1=best match, 2=second best, etc.)
    #   - treated_k: total matches this treated patient has
    #   - control_usage_count: how many times this control was matched (always 1 without replacement)
    print(f"\n{subsection_header('Sample matched pairs')} (5 samples, dataframe transposed for readability):")
    sample_df = pairs.limit(5).toPandas().T
    sample_df.columns = [f"Sample {i+1}" for i in range(len(sample_df.columns))]
    print(sample_df.to_string())

    # Generate balance summary and diagnostic plot
    # Computes covariate balance statistics before and after matching:
    #   - SMD (Standardized Mean Difference): difference in means scaled by pooled std dev
    #     Goal: |SMD| < 0.1 (rule of thumb); closer to 0 is better
    #   - VR (Variance Ratio): ratio of variances between treated and control
    #     Goal: VR between 0.5 and 2.0 (rule of thumb); closer to 1 is better
    #     Note: VR only computed for continuous variables (__num, __date)
    #
    # The plot (love plot) visualizes balance improvements across all covariates
    print(f"\n{subsection_header('Generating balance summary and plot')}...")
    summary, fig = match_summary(features_df, units, sample_frac=1.0, plot=True, verbose=False)
    n_covariates = len(summary)
    n_improved = (summary['smd_unadjusted'].abs() > summary['smd_adjusted'].abs()).sum()
    print(f"  Assessed {value(str(n_covariates))} covariates")
    print(f"  Balance improved for {highlight(str(n_improved))}/{value(str(n_covariates))} covariates")

    # Display the balance summary table
    # Compare "Before" (unadjusted) vs "After" (adjusted) to assess matching quality
    # "-" indicates VR not applicable (categorical variables)
    print(f"\n{subsection_header('Balance summary')} (SMD = Standardized Mean Difference):")
    display_cols = ["display_name", "smd_unadjusted", "smd_adjusted", "vr_unadjusted", "vr_adjusted"]
    summary_display = summary[display_cols].copy()
    summary_display.columns = ["Covariate", "SMD (Before)", "SMD (After)", "VR (Before)", "VR (After)"]
    # Format numeric columns
    for col in summary_display.columns[1:]:
        summary_display[col] = summary_display[col].apply(
            lambda x: f"{x:.3f}" if abs(x) < 10 else "-"
        )
    print(summary_display.to_string(index=False))

    # Create analysis-ready dataset by joining match info to original data
    # The units DataFrame has columns: id, subclass, weight, is_treated
    # Simply join it to your original data to add match information:
    #   - subclass: unique identifier for each matched set (treated ID, None if unmatched)
    #   - weight: weight for analysis (1.0 for treated, 1/k for controls, 0.0 for unmatched)
    #   - is_treated: boolean treatment indicator
    #
    # Use this DataFrame for outcome analysis (regression, survival, etc.)
    # The subclass column enables stratified/paired analysis respecting the matched structure
    print(f"\n{subsection_header('Creating analysis dataset')}...")
    analysis_df = df.join(units, df["id"] == units["id"], "left").drop(units["id"])
    n_matched = analysis_df.filter("subclass is not null").count()
    print(f"  {value(str(analysis_df.count()))} total rows, {value(str(n_matched))} matched")

    # Show sample analysis data to inspect the analysis-ready dataset
    # Contains original columns (with original names) plus match information
    print(f"\n{subsection_header('Sample analysis data')} (5 matched samples, transposed for readability):")
    sample_df = analysis_df.filter("subclass is not null").limit(5).toPandas().T
    sample_df.columns = [f"Sample {i+1}" for i in range(len(sample_df.columns))]
    print(sample_df.to_string())

    # Save all outputs to disk for further analysis
    # File descriptions:
    #   - pairs.csv: Treated-control pairs with pair_weight for debugging/inspection
    #   - units.csv: All units with match status, subclass, and weights
    #                Minimal output - just IDs and match info
    #   - bucket_stats.csv: LSH bucketing statistics for diagnostics
    #   - analysis.csv: PRIMARY OUTPUT - original data with match info added
    #                   Use for outcome analysis (regression, survival, etc.)
    #   - summary.csv: Balance statistics table (SMD, VR, eCDF) before and after matching
    #   - balance.png: Love plot visualization of balance improvements
    print(f"\n{subsection_header('Saving outputs')} to {output_dir}/")
    fig.savefig(os.path.join(output_dir, "balance.png"), dpi=150, bbox_inches="tight")
    pairs.toPandas().to_csv(os.path.join(output_dir, "pairs.csv"), index=False)
    units.toPandas().to_csv(os.path.join(output_dir, "units.csv"), index=False)
    bucket_stats.toPandas().to_csv(os.path.join(output_dir, "bucket_stats.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    analysis_df.orderBy(F.desc("subclass")).toPandas().to_csv(os.path.join(output_dir, "analysis.csv"), index=False)

    print(f"  {highlight('✓')} balance.png - visual balance diagnostic")
    print(f"  {highlight('✓')} pairs.csv - matched pairs")
    print(f"  {highlight('✓')} units.csv - unit-level match info")
    print(f"  {highlight('✓')} bucket_stats.csv - LSH bucketing statistics")
    print(f"  {highlight('✓')} summary.csv - balance statistics")
    print(f"  {highlight('✓')} analysis.csv - analysis-ready dataset")

    spark.stop()
    print(f"\n{highlight('Done!')}")


if __name__ == "__main__":
    main()
