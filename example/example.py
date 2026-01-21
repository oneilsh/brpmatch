"""
Example usage of BRPMatch demonstrating feature spaces and 1-to-k matching.

This script runs the BRPMatch pipeline on the Lalonde dataset (real observational data)
demonstrating:
- Euclidean vs Mahalanobis feature spaces
- 1:1 vs 1:3 matching ratios
- Matching with and without replacement
- Balance assessment across different matching strategies

Outputs (6 matched pairs CSVs):
- lalonde_1to1_euclidean_matched.csv
- lalonde_1to1_mahalanobis_matched.csv
- lalonde_1to3_no_replacement_euclidean_matched.csv
- lalonde_1to3_no_replacement_mahalanobis_matched.csv
- lalonde_1to3_with_replacement_euclidean_matched.csv
- lalonde_1to3_with_replacement_mahalanobis_matched.csv

Outputs (6 balance plots):
- lalonde_1to1_euclidean_balance.png
- lalonde_1to1_mahalanobis_balance.png
- lalonde_1to3_no_replacement_euclidean_balance.png
- lalonde_1to3_no_replacement_mahalanobis_balance.png
- lalonde_1to3_with_replacement_euclidean_balance.png
- lalonde_1to3_with_replacement_mahalanobis_balance.png
"""

import os

from pyspark.sql import SparkSession, functions as F

from brpmatch import generate_features, love_plot, match, stratify_for_plot, match_summary


def main():
    """Run BRPMatch example on Lalonde dataset demonstrating various matching options."""

    print("=" * 80)
    print("BRPMatch Example: Feature Spaces and 1-to-k Matching")
    print("Dataset: Lalonde (Real Observational Data)")
    print("=" * 80)

    # Create Spark session
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("brpmatch-example")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )

    # Output to the example directory
    output_dir = os.path.dirname(__file__)
    if not output_dir:
        output_dir = "."

    # Load lalonde dataset
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "data", "lalonde.csv"
    )

    if not os.path.exists(data_path):
        print(f"\nError: Lalonde dataset not found at {data_path}")
        print("Please provide the lalonde.csv file in tests/data/")
        return

    print(f"\nLoading data from {data_path}...")
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Add id column if not present
    if "id" not in df.columns:
        df = df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    print(f"Loaded {df.count()} rows")

    # 1. Generate features
    print("\n" + "=" * 80)
    print("Step 1: Generating Features")
    print("=" * 80)
    features_df = generate_features(
        spark,
        df,
        treatment_col="treat",
        treatment_value="1",
        categorical_cols=["race", "married", "nodegree"],
        numeric_cols=["age", "educ", "re74", "re75"],
        id_col="id",
    )

    print(f"Generated features for {features_df.count()} patients")

    # Show treatment distribution
    print("\nTreatment distribution:")
    features_df.groupBy("treat").count().show()

    # 2. Standard 1:1 matching with Euclidean distance
    print("\n" + "=" * 80)
    print("Step 2: Standard 1:1 Matching (Euclidean)")
    print("=" * 80)
    print("Default behavior: ratio_k=1, without replacement")
    matched_1to1_euc = match(
        features_df,
        feature_space="euclidean",
        n_neighbors=10,
        id_col="id",
        # ratio_k=1 is the default
    )
    print(f"Generated {matched_1to1_euc.count()} matched pairs")

    # 3. Standard 1:1 matching with Mahalanobis distance
    print("\n" + "=" * 80)
    print("Step 3: Standard 1:1 Matching (Mahalanobis)")
    print("=" * 80)
    print("Whitens features to account for correlations")
    matched_1to1_mah = match(
        features_df,
        feature_space="mahalanobis",
        n_neighbors=10,
        id_col="id",
    )
    print(f"Generated {matched_1to1_mah.count()} matched pairs")

    # 4. 1:3 matching WITHOUT replacement - Euclidean
    print("\n" + "=" * 80)
    print("Step 4: 1:3 Matching WITHOUT Replacement (Euclidean)")
    print("=" * 80)
    print("Each treated gets 3 controls; controls used at most once")
    print("Round-robin ensures fairness: all treated get round 1 before round 2")
    matched_1to3_no_repl_euc = match(
        features_df,
        feature_space="euclidean",
        n_neighbors=10,
        id_col="id",
        ratio_k=3,
        with_replacement=False,
    )
    n_pairs = matched_1to3_no_repl_euc.count()
    n_treated = matched_1to3_no_repl_euc.select("id").distinct().count()
    print(f"Generated {n_pairs} matched pairs across {n_treated} treated")
    print(f"Mean controls per treated: {n_pairs / n_treated:.2f}")

    # Show match round distribution
    print("\nMatch round distribution (round-robin fairness):")
    matched_1to3_no_repl_euc.groupBy("match_round").count().orderBy("match_round").show()

    # 5. 1:3 matching WITHOUT replacement - Mahalanobis
    print("\n" + "=" * 80)
    print("Step 5: 1:3 Matching WITHOUT Replacement (Mahalanobis)")
    print("=" * 80)
    print("Each treated gets 3 controls; controls used at most once")
    print("Round-robin with whitened features")
    matched_1to3_no_repl_mah = match(
        features_df,
        feature_space="mahalanobis",
        n_neighbors=10,
        id_col="id",
        ratio_k=3,
        with_replacement=False,
    )
    n_pairs = matched_1to3_no_repl_mah.count()
    n_treated = matched_1to3_no_repl_mah.select("id").distinct().count()
    print(f"Generated {n_pairs} matched pairs across {n_treated} treated")
    print(f"Mean controls per treated: {n_pairs / n_treated:.2f}")

    # Show match round distribution
    print("\nMatch round distribution (round-robin fairness):")
    matched_1to3_no_repl_mah.groupBy("match_round").count().orderBy("match_round").show()

    # 6. 1:3 matching WITH replacement - Euclidean
    print("\n" + "=" * 80)
    print("Step 6: 1:3 Matching WITH Replacement (Euclidean)")
    print("=" * 80)
    print("Each treated gets their 3 nearest controls (controls can be reused)")
    matched_1to3_repl_euc = match(
        features_df,
        feature_space="euclidean",
        n_neighbors=10,
        id_col="id",
        ratio_k=3,
        with_replacement=True,
    )
    n_pairs = matched_1to3_repl_euc.count()
    n_treated = matched_1to3_repl_euc.select("id").distinct().count()
    n_controls = matched_1to3_repl_euc.select("match_id").distinct().count()
    print(f"Generated {n_pairs} matched pairs across {n_treated} treated")
    print(f"Unique controls used: {n_controls}")

    # Show control usage distribution
    print("\nControl usage distribution (with replacement):")
    control_usage = matched_1to3_repl_euc.groupBy("control_usage_count").count().orderBy("control_usage_count")
    control_usage.show()

    # 7. 1:3 matching WITH replacement - Mahalanobis
    print("\n" + "=" * 80)
    print("Step 7: 1:3 Matching WITH Replacement (Mahalanobis)")
    print("=" * 80)
    print("Each treated gets their 3 nearest controls (controls can be reused)")
    matched_1to3_repl_mah = match(
        features_df,
        feature_space="mahalanobis",
        n_neighbors=10,
        id_col="id",
        ratio_k=3,
        with_replacement=True,
    )
    n_pairs = matched_1to3_repl_mah.count()
    n_treated = matched_1to3_repl_mah.select("id").distinct().count()
    n_controls = matched_1to3_repl_mah.select("match_id").distinct().count()
    print(f"Generated {n_pairs} matched pairs across {n_treated} treated")
    print(f"Unique controls used: {n_controls}")

    # Show control usage distribution
    print("\nControl usage distribution (with replacement):")
    control_usage = matched_1to3_repl_mah.groupBy("control_usage_count").count().orderBy("control_usage_count")
    control_usage.show()

    # 8. Generate balance summaries
    print("\n" + "=" * 80)
    print("Step 8: Generating Balance Summaries")
    print("=" * 80)

    feature_cols = ["race_index", "married_index", "nodegree_index", "age", "educ", "re74", "re75"]

    print("\nEuclidean 1:1 Balance:")
    summary_euc, fig_euc = match_summary(
        features_df,
        matched_1to1_euc,
        feature_cols,
        id_col="id",
        sample_frac=1.0,
        plot=True,
    )
    fig_euc.savefig(os.path.join(output_dir, "lalonde_1to1_euclidean_balance.png"), dpi=150, bbox_inches="tight")
    print("\nSaved lalonde_1to1_euclidean_balance.png")

    print("\nMahalanobis 1:1 Balance:")
    summary_mah, fig_mah = match_summary(
        features_df,
        matched_1to1_mah,
        feature_cols,
        id_col="id",
        sample_frac=1.0,
        plot=True,
    )
    fig_mah.savefig(os.path.join(output_dir, "lalonde_1to1_mahalanobis_balance.png"), dpi=150, bbox_inches="tight")
    print("\nSaved lalonde_1to1_mahalanobis_balance.png")

    print("\n1:3 Without Replacement (Euclidean) Balance:")
    summary_1to3_no_repl_euc, fig_1to3_no_repl_euc = match_summary(
        features_df,
        matched_1to3_no_repl_euc,
        feature_cols,
        id_col="id",
        sample_frac=1.0,
        plot=True,
    )
    fig_1to3_no_repl_euc.savefig(os.path.join(output_dir, "lalonde_1to3_no_replacement_euclidean_balance.png"), dpi=150, bbox_inches="tight")
    print("\nSaved lalonde_1to3_no_replacement_euclidean_balance.png")

    print("\n1:3 Without Replacement (Mahalanobis) Balance:")
    summary_1to3_no_repl_mah, fig_1to3_no_repl_mah = match_summary(
        features_df,
        matched_1to3_no_repl_mah,
        feature_cols,
        id_col="id",
        sample_frac=1.0,
        plot=True,
    )
    fig_1to3_no_repl_mah.savefig(os.path.join(output_dir, "lalonde_1to3_no_replacement_mahalanobis_balance.png"), dpi=150, bbox_inches="tight")
    print("\nSaved lalonde_1to3_no_replacement_mahalanobis_balance.png")

    print("\n1:3 With Replacement (Euclidean) Balance:")
    summary_1to3_repl_euc, fig_1to3_repl_euc = match_summary(
        features_df,
        matched_1to3_repl_euc,
        feature_cols,
        id_col="id",
        sample_frac=1.0,
        plot=True,
    )
    fig_1to3_repl_euc.savefig(os.path.join(output_dir, "lalonde_1to3_with_replacement_euclidean_balance.png"), dpi=150, bbox_inches="tight")
    print("\nSaved lalonde_1to3_with_replacement_euclidean_balance.png")

    print("\n1:3 With Replacement (Mahalanobis) Balance:")
    summary_1to3_repl_mah, fig_1to3_repl_mah = match_summary(
        features_df,
        matched_1to3_repl_mah,
        feature_cols,
        id_col="id",
        sample_frac=1.0,
        plot=True,
    )
    fig_1to3_repl_mah.savefig(os.path.join(output_dir, "lalonde_1to3_with_replacement_mahalanobis_balance.png"), dpi=150, bbox_inches="tight")
    print("\nSaved lalonde_1to3_with_replacement_mahalanobis_balance.png")

    # 9. Save matched pairs
    print("\n" + "=" * 80)
    print("Step 9: Saving Matched Pairs")
    print("=" * 80)

    matched_1to1_euc.toPandas().to_csv(
        os.path.join(output_dir, "lalonde_1to1_euclidean_matched.csv"), index=False
    )
    print("Saved lalonde_1to1_euclidean_matched.csv")

    matched_1to1_mah.toPandas().to_csv(
        os.path.join(output_dir, "lalonde_1to1_mahalanobis_matched.csv"), index=False
    )
    print("Saved lalonde_1to1_mahalanobis_matched.csv")

    matched_1to3_no_repl_euc.toPandas().to_csv(
        os.path.join(output_dir, "lalonde_1to3_no_replacement_euclidean_matched.csv"), index=False
    )
    print("Saved lalonde_1to3_no_replacement_euclidean_matched.csv")

    matched_1to3_no_repl_mah.toPandas().to_csv(
        os.path.join(output_dir, "lalonde_1to3_no_replacement_mahalanobis_matched.csv"), index=False
    )
    print("Saved lalonde_1to3_no_replacement_mahalanobis_matched.csv")

    matched_1to3_repl_euc.toPandas().to_csv(
        os.path.join(output_dir, "lalonde_1to3_with_replacement_euclidean_matched.csv"), index=False
    )
    print("Saved lalonde_1to3_with_replacement_euclidean_matched.csv")

    matched_1to3_repl_mah.toPandas().to_csv(
        os.path.join(output_dir, "lalonde_1to3_with_replacement_mahalanobis_matched.csv"), index=False
    )
    print("Saved lalonde_1to3_with_replacement_mahalanobis_matched.csv")

    # Stop Spark
    spark.stop()

    # Summary
    print("\n" + "=" * 80)
    print("Example Completed Successfully!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated matched pairs CSV files:")
    print("  - lalonde_1to1_euclidean_matched.csv                    (1:1, Euclidean)")
    print("  - lalonde_1to1_mahalanobis_matched.csv                  (1:1, Mahalanobis)")
    print("  - lalonde_1to3_no_replacement_euclidean_matched.csv     (1:3 round-robin, Euclidean)")
    print("  - lalonde_1to3_no_replacement_mahalanobis_matched.csv   (1:3 round-robin, Mahalanobis)")
    print("  - lalonde_1to3_with_replacement_euclidean_matched.csv   (1:3 with replacement, Euclidean)")
    print("  - lalonde_1to3_with_replacement_mahalanobis_matched.csv (1:3 with replacement, Mahalanobis)")
    print("\nGenerated balance plots:")
    print("  - lalonde_1to1_euclidean_balance.png                    (1:1, Euclidean)")
    print("  - lalonde_1to1_mahalanobis_balance.png                  (1:1, Mahalanobis)")
    print("  - lalonde_1to3_no_replacement_euclidean_balance.png     (1:3 round-robin, Euclidean)")
    print("  - lalonde_1to3_no_replacement_mahalanobis_balance.png   (1:3 round-robin, Mahalanobis)")
    print("  - lalonde_1to3_with_replacement_euclidean_balance.png   (1:3 with replacement, Euclidean)")
    print("  - lalonde_1to3_with_replacement_mahalanobis_balance.png (1:3 with replacement, Mahalanobis)")
    print("\nMatched pair CSV schema:")
    print("  - id: treated patient ID")
    print("  - match_id: matched control patient ID")
    print("  - match_round: which round (1=best match, 2=second best, etc.)")
    print("  - treated_k: total controls matched to this treated")
    print("  - control_usage_count: times this control was matched globally")
    print("  - pair_weight: weight for analysis = 1/(treated_k * control_usage_count)")
    print("  - bucket_num_input_patients: bucket size")
    print("  - bucket_seconds: processing time")
    print("\nWeighting Notes:")
    print("  For ATT estimation, use pair_weight to properly weight matched pairs:")
    print("  - Adjusts for k:1 matching (each control contributes 1/k)")
    print("  - Adjusts for replacement (reused controls down-weighted)")
    print("\nComparison Guide:")
    print("  - Compare 1:1 vs 1:3 to see effect of matching ratio")
    print("  - Compare Euclidean vs Mahalanobis to see effect of feature space")
    print("  - Compare with/without replacement to see tradeoffs (match rate vs uniqueness)")
    print("=" * 80)


if __name__ == "__main__":
    main()
