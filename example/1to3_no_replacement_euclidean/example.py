"""
Example: 1:3 Matching WITHOUT Replacement - Euclidean Distance

Demonstrates 1:3 matching where each treated gets up to 3 controls.
Controls are used at most once. Round-robin algorithm ensures fairness.
"""

import os
import sys

# Add parent directory to path for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brpmatch import generate_features, match, match_summary, stratify_for_plot
from utils import create_spark_session, load_lalonde, setup_pandas_display
from utils import section_header, subsection_header, highlight, value

setup_pandas_display()


def main():
    print(section_header("1:3 Matching WITHOUT Replacement - Euclidean Distance", "="))

    # Create Spark session
    spark = create_spark_session("brpmatch-1to3-no-repl-euclidean")
    output_dir = os.path.dirname(__file__) or "."

    # Load data
    print(f"\n{subsection_header('Loading data')}...")
    df = load_lalonde(spark)
    print(f"  Loaded {value(str(df.count()))} rows")

    # Generate features
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

    # Match
    print(f"\n{subsection_header('Matching')}...")
    matched = match(
        features_df,
        feature_space="euclidean",
        n_neighbors=10,
        ratio_k=3,
        with_replacement=False,
        verbose=False,
    )
    n_pairs = matched.count()
    n_treated = matched.select("id").distinct().count()
    print(f"  {highlight(str(n_pairs))} matched pairs across {value(str(n_treated))} treated ({n_pairs / n_treated:.2f} controls per treated)")

    # Show sample matched pairs (just key columns)
    print(f"\n{subsection_header('Sample matched pairs')} (showing key columns):")
    matched_sample = matched.select("id", "match_id", "pair_weight", "match_round").limit(5).toPandas()
    print(matched_sample.to_string(index=False))

    # Balance summary
    print(f"\n{subsection_header('Generating balance summary and plot')}...")
    summary, fig = match_summary(features_df, matched, sample_frac=1.0, plot=True, verbose=False)
    print(f"  Assessed {value(str(len(summary)))} covariates")

    # Show summary table
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

    # Stratify data for analysis
    print(f"\n{subsection_header('Stratifying data for analysis')}...")
    stratified = stratify_for_plot(features_df, matched)
    print(f"  {value(str(stratified.count()))} total rows (treated + control)")

    # Show sample stratified data (first 3 rows for compactness)
    print(f"\n{subsection_header('Sample stratified data')}:")
    strat_sample = stratified.select("id__id", "treat__treat", "age__num", "educ__num", "strata").limit(3).toPandas()
    # Rename for display
    strat_sample.columns = ["id", "treat", "age", "educ", "strata"]
    print(strat_sample.to_string(index=False))

    # Save outputs
    print(f"\n{subsection_header('Saving outputs')} to {output_dir}/")
    fig.savefig(os.path.join(output_dir, "balance.png"), dpi=150, bbox_inches="tight")
    matched.toPandas().to_csv(os.path.join(output_dir, "matched.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    stratified.toPandas().to_csv(os.path.join(output_dir, "stratified.csv"), index=False)

    print(f"  {highlight('✓')} balance.png")
    print(f"  {highlight('✓')} matched.csv")
    print(f"  {highlight('✓')} summary.csv")
    print(f"  {highlight('✓')} stratified.csv")

    spark.stop()
    print(f"\n{highlight('Done!')}")


if __name__ == "__main__":
    main()
