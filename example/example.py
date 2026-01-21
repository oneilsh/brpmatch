"""
Example usage of BRPMatch demonstrating Euclidean vs Mahalanobis feature spaces.

This script runs the BRPMatch pipeline on the Lalonde dataset (real observational data)
using two different feature spaces:
- Euclidean: uses original features with Euclidean distance
- Mahalanobis: whitens features to account for correlations

Outputs:
- lalonde_euclidean_balance.png
- lalonde_euclidean_matched.csv
- lalonde_mahalanobis_balance.png
- lalonde_mahalanobis_matched.csv
"""

import os

from pyspark.sql import SparkSession, functions as F

from brpmatch import generate_features, love_plot, match, stratify_for_plot


def main():
    """Run BRPMatch example on Lalonde dataset with Euclidean and Mahalanobis feature spaces."""
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 80)
    print("BRPMatch Example: Euclidean vs Mahalanobis Feature Spaces")
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

    # 2. Perform matching with EUCLIDEAN feature space
    print("\n" + "=" * 80)
    print("Step 2: Matching with EUCLIDEAN Feature Space")
    print("=" * 80)
    print("Uses original features with Euclidean distance")
    matched_euclidean = match(
        features_df,
        feature_space="euclidean",
        n_neighbors=5,
        id_col="id",
    )
    print(f"Generated {matched_euclidean.count()} matches")

    # 3. Perform matching with MAHALANOBIS feature space
    print("\n" + "=" * 80)
    print("Step 3: Matching with MAHALANOBIS Feature Space")
    print("=" * 80)
    print("Whitens features to account for correlations")
    matched_mahalanobis = match(
        features_df,
        feature_space="mahalanobis",
        n_neighbors=5,
        id_col="id",
    )
    print(f"Generated {matched_mahalanobis.count()} matches")

    # 4. Stratify and visualize
    print("\n" + "=" * 80)
    print("Step 4: Generating Balance Plots")
    print("=" * 80)
    stratified_euclidean = stratify_for_plot(
        features_df, matched_euclidean, id_col="id", match_id_col="match_id"
    )
    stratified_mahalanobis = stratify_for_plot(
        features_df, matched_mahalanobis, id_col="id", match_id_col="match_id"
    )

    feature_cols = ["race_index", "married_index", "nodegree_index", "age", "educ", "re74", "re75"]

    fig_euc = love_plot(stratified_euclidean, feature_cols, treatment_col="treat", sample_frac=1.0)
    fig_euc.savefig(os.path.join(output_dir, "lalonde_euclidean_balance.png"), dpi=150, bbox_inches="tight")
    print("Saved lalonde_euclidean_balance.png")

    fig_mah = love_plot(stratified_mahalanobis, feature_cols, treatment_col="treat", sample_frac=1.0)
    fig_mah.savefig(os.path.join(output_dir, "lalonde_mahalanobis_balance.png"), dpi=150, bbox_inches="tight")
    print("Saved lalonde_mahalanobis_balance.png")

    # 5. Save matched pairs
    print("\n" + "=" * 80)
    print("Step 5: Saving Matched Pairs")
    print("=" * 80)
    matched_euclidean.toPandas().to_csv(os.path.join(output_dir, "lalonde_euclidean_matched.csv"), index=False)
    print("Saved lalonde_euclidean_matched.csv")

    matched_mahalanobis.toPandas().to_csv(os.path.join(output_dir, "lalonde_mahalanobis_matched.csv"), index=False)
    print("Saved lalonde_mahalanobis_matched.csv")

    # Stop Spark
    spark.stop()

    # Summary
    print("\n" + "=" * 80)
    print("Example Completed Successfully!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated files:")
    print("  - lalonde_euclidean_balance.png")
    print("  - lalonde_euclidean_matched.csv")
    print("  - lalonde_mahalanobis_balance.png")
    print("  - lalonde_mahalanobis_matched.csv")
    print("\nMatched pair CSV schema:")
    print("  - id: treated patient ID")
    print("  - match_id: matched control patient ID")
    print("  - bucket_num_input_patients: bucket size")
    print("  - bucket_seconds: processing time")
    print("=" * 80)


if __name__ == "__main__":
    main()
