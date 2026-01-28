"""
Example: 1:3 Matching WITHOUT Replacement - Mahalanobis Distance

Demonstrates 1:3 matching where each treated gets up to 3 controls.
Controls are used at most once. Mahalanobis distance accounts for correlations.
"""

import os
import sys

# Add parent directory to path for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brpmatch import generate_features, match, match_summary, stratify_for_plot
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

    This example uses Mahalanobis distance, which accounts for feature correlations
    by applying a whitening transform before computing Euclidean distance.
    Each treated patient is matched to UP TO 3 control patients.
    Without replacement: each control can only be used once (round-robin algorithm ensures fairness).
    """
    print(section_header("1:3 Matching WITHOUT Replacement - Mahalanobis Distance", "="))

    # Create Spark session
    spark = create_spark_session("brpmatch-1to3-no-repl-mahalanobis")
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
    # For Mahalanobis distance, applies whitening transform to account for feature correlations,
    # then uses Euclidean distance in the transformed space (equivalent to Mahalanobis in original).
    #
    # With ratio_k=3 and with_replacement=False:
    #   - Each treated patient gets UP TO 3 control matches
    #   - Each control can only be matched once (without replacement)
    #   - Round-robin algorithm: all treated get round 1 matches before any get round 2
    #   - Some treated may get fewer than 3 matches if controls are exhausted
    #
    # Output columns: id (treated), match_id (control), pair_weight, match_round, and more
    # The matched DataFrame is the PRIMARY OUTPUT - use it to link outcomes to matched pairs
    print(f"\n{subsection_header('Matching')}...")
    matched = match(
        features_df,
        feature_space="mahalanobis",
        n_neighbors=10,
        ratio_k=3,
        with_replacement=False,
        verbose=False,
    )
    print_matching_stats(features_df, matched)

    # Show distribution of matches per treated (may vary when controls are limited)
    matches_per_treated = matched.groupBy("id").count().toPandas()
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
    sample_df = matched.limit(5).toPandas().T
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
    summary, fig = match_summary(features_df, matched, sample_frac=1.0, plot=True, verbose=False)
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

    # Create stratified dataset for outcome analysis
    # This joins features_df with matched pairs to add strata identifiers:
    #   - is_matched: indicates if patient was successfully matched
    #   - strata: unique identifier for each matched pair (format: "id:match_id")
    #
    # Use this DataFrame for outcome analysis (regression, survival, etc.)
    # The strata column enables stratified/paired analysis respecting the matched structure
    print(f"\n{subsection_header('Stratifying data for analysis')}...")
    stratified = stratify_for_plot(features_df, matched)
    print(f"  {value(str(stratified.count()))} total rows (treated + control)")

    # Show sample stratified data to inspect the analysis-ready dataset
    # Contains all feature columns (with __cat, __num, __id, __treat suffixes) plus strata identifiers
    # Note: Column names use double underscore notation - see feature generation step above for details
    print(f"\n{subsection_header('Sample stratified data')} (5 samples, features array column dropped and dataframe transposed for readability):")
    sample_df = stratified.limit(5).toPandas()
    # Drop the features column as it's too long for display
    sample_df = sample_df.drop(columns=['features'], errors='ignore')
    sample_df = sample_df.T
    sample_df.columns = [f"Sample {i+1}" for i in range(len(sample_df.columns))]
    print(sample_df.to_string())

    # Save all outputs to disk for further analysis
    # File descriptions:
    #   - matched.csv: PRIMARY OUTPUT - treated-control pairs with pair_weight column
    #                  Use this to link outcomes and for weighted analysis (e.g., weighted regression)
    #   - stratified.csv: Analysis-ready dataset with all features plus strata identifiers
    #                     Use for stratified/paired analysis (e.g., stratified Cox models)
    #   - summary.csv: Balance statistics table (SMD, VR, eCDF) before and after matching
    #                  Use to assess and report matching quality
    #   - balance.png: Love plot visualization of balance improvements
    #                  Use for visual diagnostics and publication figures
    print(f"\n{subsection_header('Saving outputs')} to {output_dir}/")
    fig.savefig(os.path.join(output_dir, "balance.png"), dpi=150, bbox_inches="tight")
    matched.toPandas().to_csv(os.path.join(output_dir, "matched.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    stratified.toPandas().to_csv(os.path.join(output_dir, "stratified.csv"), index=False)

    print(f"  {highlight('✓')} balance.png - visual balance diagnostic")
    print(f"  {highlight('✓')} matched.csv - matched pairs with weights")
    print(f"  {highlight('✓')} summary.csv - balance statistics")
    print(f"  {highlight('✓')} stratified.csv - analysis-ready dataset")

    spark.stop()
    print(f"\n{highlight('Done!')}")


if __name__ == "__main__":
    main()
