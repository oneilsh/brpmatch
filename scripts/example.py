"""
Example usage of BRPMatch with the Lalonde dataset.

This script demonstrates the full BRPMatch pipeline:
1. Load data
2. Generate features
3. Perform matching
4. Stratify for visualization
5. Generate and save love plot
"""

import os

from pyspark.sql import SparkSession, functions as F

from brpmatch import generate_features, love_plot, match, stratify_for_plot


def main():
    """Run BRPMatch example with Lalonde data."""
    import warnings
    warnings.filterwarnings("ignore")

    # Create Spark session
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("brpmatch-example")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )

    # Load lalonde dataset
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "data", "lalonde.csv"
    )

    if not os.path.exists(data_path):
        print(f"Error: Lalonde dataset not found at {data_path}")
        print("Please provide the lalonde.csv file in tests/data/")
        return

    print(f"Loading data from {data_path}...")
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Add id column if not present
    if "id" not in df.columns:
        df = df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    print(f"Loaded {df.count()} rows")

    # 1. Generate features
    print("\n1. Generating features...")
    features_df = generate_features(
        spark,
        df,
        categorical_cols=["race", "married", "nodegree"],
        numeric_cols=["age", "educ", "re74", "re75"],
        treatment_col="treat",
        treatment_value="1",
        id_col="id",
    )

    print(f"Generated features for {features_df.count()} patients")

    # Show treatment distribution
    print("\nTreatment distribution:")
    features_df.groupBy("treat").count().show()

    # 2. Perform matching
    print("\n2. Performing matching...")
    matched_df = match(
        features_df,
        distance_metric="euclidean",
        n_neighbors=5,
        id_col="id",
    )

    print(f"Generated {matched_df.count()} matches")

    # Show sample matches
    print("\nSample matches:")
    matched_df.select("id", "match_id", "match_distance").show(5)

    # 3. Stratify for plot
    print("\n3. Stratifying for visualization...")
    stratified_df = stratify_for_plot(features_df, matched_df, id_col="id", match_id_col="match_id")

    # Show how many patients are matched
    matched_count = stratified_df.filter(F.col("is_matched").isNotNull()).count()
    print(f"{matched_count} patients are in matched pairs")

    # 4. Generate love plot
    print("\n4. Generating love plot...")
    fig = love_plot(
        stratified_df,
        treatment_col="treat",
        sample_frac=1.0,  # Use all data for small dataset
    )

    # Save plot
    output_path = "balance_plot.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved love plot to {output_path}")

    # Stop Spark
    spark.stop()

    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()
