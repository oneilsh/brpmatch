"""
Example usage of BRPMatch demonstrating Euclidean vs Mahalanobis feature spaces.

This script runs the BRPMatch pipeline on two datasets:
1. Lalonde dataset (real observational data)
2. Synthetic data with correlated features

For each dataset, matching is performed in both feature spaces:
- Euclidean: uses original features
- Mahalanobis: whitens features to account for correlations

Outputs 4 sets of results (PNG plots + CSV matched pairs):
- lalonde_euclidean_*
- lalonde_mahalanobis_*
- synthetic_euclidean_*
- synthetic_mahalanobis_*
"""

import os

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F

from brpmatch import generate_features, love_plot, match, stratify_for_plot


def generate_correlated_data(n_treated=200, n_control=800, n_features=6,
                              correlation_strength=0.7, treatment_effect_size=0.6, seed=42):
    """
    Generate synthetic data with correlated features and group differences.

    Parameters
    ----------
    n_treated : int
        Number of treated patients
    n_control : int
        Number of control patients
    n_features : int
        Number of features to generate
    correlation_strength : float
        Correlation between features (0-1)
    treatment_effect_size : float
        Standardized mean difference between groups
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        DataFrame with id, treat, and feature columns
    """
    np.random.seed(seed)

    # Create covariance matrix with specified correlation
    cov_matrix = np.full((n_features, n_features), correlation_strength)
    np.fill_diagonal(cov_matrix, 1.0)

    # Generate control group (mean=0)
    control_features = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=cov_matrix,
        size=n_control
    )

    # Generate treated group (shifted by treatment effect)
    treated_features = np.random.multivariate_normal(
        mean=np.ones(n_features) * treatment_effect_size,
        cov=cov_matrix,
        size=n_treated
    )

    # Combine into dataframe
    all_features = np.vstack([treated_features, control_features])
    treat = np.array([1] * n_treated + [0] * n_control)

    df = pd.DataFrame(
        all_features,
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    df["id"] = [f"P{i:04d}" for i in range(len(df))]
    df["treat"] = treat

    return df


def main():
    """Run BRPMatch example with both Lalonde and synthetic data."""
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 80)
    print("BRPMatch Example: Euclidean vs Mahalanobis Feature Spaces")
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
        .getOrCreate()
    )

    # Output to the example directory
    output_dir = os.path.dirname(__file__)
    if not output_dir:
        output_dir = "."

    # =========================================================================
    # PART 1: LALONDE DATASET (Real observational data)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Lalonde Dataset (Real Observational Data)")
    print("=" * 80)

    # Load lalonde dataset
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "data", "lalonde.csv"
    )

    if not os.path.exists(data_path):
        print(f"Error: Lalonde dataset not found at {data_path}")
        print("Please provide the lalonde.csv file in tests/data/")
        return

    print(f"\nLoading data from {data_path}...")
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Add id column if not present
    if "id" not in df.columns:
        df = df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    print(f"Loaded {df.count()} rows")

    # 1. Generate features
    print("\n1. Generating features for Lalonde data...")
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
    print("\n2. Performing matching with EUCLIDEAN feature space...")
    print("   (Uses original features with Euclidean distance)")
    lalonde_matched_euclidean = match(
        features_df,
        feature_space="euclidean",
        n_neighbors=5,
        id_col="id",
    )
    print(f"   Generated {lalonde_matched_euclidean.count()} matches")

    # 3. Perform matching with MAHALANOBIS feature space
    print("\n3. Performing matching with MAHALANOBIS feature space...")
    print("   (Whitens features to account for correlations)")
    lalonde_matched_mahalanobis = match(
        features_df,
        feature_space="mahalanobis",
        n_neighbors=5,
        id_col="id",
    )
    print(f"   Generated {lalonde_matched_mahalanobis.count()} matches")

    # 4. Stratify and visualize
    print("\n4. Generating balance plots...")
    lalonde_stratified_euclidean = stratify_for_plot(
        features_df, lalonde_matched_euclidean, id_col="id", match_id_col="match_id"
    )
    lalonde_stratified_mahalanobis = stratify_for_plot(
        features_df, lalonde_matched_mahalanobis, id_col="id", match_id_col="match_id"
    )

    feature_cols = ["race_index", "married_index", "nodegree_index", "age", "educ", "re74", "re75"]

    fig_lalonde_euc = love_plot(lalonde_stratified_euclidean, feature_cols, treatment_col="treat", sample_frac=1.0)
    fig_lalonde_euc.savefig(os.path.join(output_dir, "lalonde_euclidean_balance.png"), dpi=150, bbox_inches="tight")
    print("   ✓ Saved lalonde_euclidean_balance.png")

    fig_lalonde_mah = love_plot(lalonde_stratified_mahalanobis, feature_cols, treatment_col="treat", sample_frac=1.0)
    fig_lalonde_mah.savefig(os.path.join(output_dir, "lalonde_mahalanobis_balance.png"), dpi=150, bbox_inches="tight")
    print("   ✓ Saved lalonde_mahalanobis_balance.png")

    # 5. Save matched pairs
    print("\n5. Saving matched pairs...")
    lalonde_matched_euclidean.toPandas().to_csv(os.path.join(output_dir, "lalonde_euclidean_matched.csv"), index=False)
    print("   ✓ Saved lalonde_euclidean_matched.csv")

    lalonde_matched_mahalanobis.toPandas().to_csv(os.path.join(output_dir, "lalonde_mahalanobis_matched.csv"), index=False)
    print("   ✓ Saved lalonde_mahalanobis_matched.csv")

    # Get counts for summary
    lalonde_euc_count = lalonde_matched_euclidean.count()
    lalonde_mah_count = lalonde_matched_mahalanobis.count()

    # =========================================================================
    # PART 2: SYNTHETIC DATA (Correlated features)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Synthetic Data (Correlated Features)")
    print("=" * 80)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic data with correlated features...")
    print("   Configuration:")
    print("   - 100 treated, 1000 control patients")
    print("   - 6 features with moderate correlations (r=0.5)")
    print("   - Moderate group differences (d=0.6)")

    synthetic_pandas = generate_correlated_data(
        n_treated=1000,
        n_control=10000,
        n_features=6,
        correlation_strength=0.5,
        treatment_effect_size=0.6,
        seed=42
    )

    synthetic_df = spark.createDataFrame(synthetic_pandas)
    print(f"\n   Generated {synthetic_df.count()} patients")
    print(f"   Treated: {synthetic_df.filter(F.col('treat') == 1).count()}")
    print(f"   Control: {synthetic_df.filter(F.col('treat') == 0).count()}")

    # Show correlation structure
    feature_cols_syn = [c for c in synthetic_pandas.columns if c.startswith("feature_")]
    corr_matrix = synthetic_pandas[feature_cols_syn].corr()
    print(f"\n   Feature correlation matrix (sample):")
    print(corr_matrix.iloc[:3, :3].round(2).to_string())

    # 2. Generate features (already numeric, just need vector column)
    print("\n2. Preparing features...")
    synthetic_features = generate_features(
        spark,
        synthetic_df,
        treatment_col="treat",
        treatment_value="1",
        categorical_cols=[],
        numeric_cols=feature_cols_syn,
        id_col="id",
    )
    print(f"   Features ready for {synthetic_features.count()} patients")

    # 3. Perform matching with EUCLIDEAN feature space
    print("\n3. Performing matching with EUCLIDEAN feature space...")
    print("   (Ignores feature correlations)")
    synthetic_matched_euclidean = match(
        synthetic_features,
        feature_space="euclidean",
        n_neighbors=5,
        id_col="id",
    )
    print(f"   Generated {synthetic_matched_euclidean.count()} matches")

    # 4. Perform matching with MAHALANOBIS feature space
    print("\n4. Performing matching with MAHALANOBIS feature space...")
    print("   (Accounts for feature correlations via whitening)")
    print("   Note: Using larger bucket_length to compensate for data spreading")
    synthetic_matched_mahalanobis = match(
        synthetic_features,
        feature_space="mahalanobis",
        n_neighbors=5,
        bucket_length=0.5,  # Larger than default to get more buckets
        id_col="id",
    )
    print(f"   Generated {synthetic_matched_mahalanobis.count()} matches")

    # 5. Stratify and visualize
    print("\n5. Generating balance plots...")
    synthetic_stratified_euclidean = stratify_for_plot(
        synthetic_features, synthetic_matched_euclidean, id_col="id", match_id_col="match_id"
    )
    synthetic_stratified_mahalanobis = stratify_for_plot(
        synthetic_features, synthetic_matched_mahalanobis, id_col="id", match_id_col="match_id"
    )

    fig_synthetic_euc = love_plot(synthetic_stratified_euclidean, feature_cols_syn, treatment_col="treat", sample_frac=1.0)
    fig_synthetic_euc.savefig(os.path.join(output_dir, "synthetic_euclidean_balance.png"), dpi=150, bbox_inches="tight")
    print("   ✓ Saved synthetic_euclidean_balance.png")

    fig_synthetic_mah = love_plot(synthetic_stratified_mahalanobis, feature_cols_syn, treatment_col="treat", sample_frac=1.0)
    fig_synthetic_mah.savefig(os.path.join(output_dir, "synthetic_mahalanobis_balance.png"), dpi=150, bbox_inches="tight")
    print("   ✓ Saved synthetic_mahalanobis_balance.png")

    # 6. Save matched pairs
    print("\n6. Saving matched pairs...")
    synthetic_matched_euclidean.toPandas().to_csv(os.path.join(output_dir, "synthetic_euclidean_matched.csv"), index=False)
    print("   ✓ Saved synthetic_euclidean_matched.csv")

    synthetic_matched_mahalanobis.toPandas().to_csv(os.path.join(output_dir, "synthetic_mahalanobis_matched.csv"), index=False)
    print("   ✓ Saved synthetic_mahalanobis_matched.csv")

    # Get counts for summary
    synthetic_euc_count = synthetic_matched_euclidean.count()
    synthetic_mah_count = synthetic_matched_mahalanobis.count()

    # =========================================================================
    # SUMMARY
    # =========================================================================

    # Stop Spark
    spark.stop()

    print("\n" + "=" * 80)
    print("✓ Example completed successfully!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}/")
    print("\n" + "-" * 80)
    print("LALONDE DATASET (Real Data)")
    print("-" * 80)
    print("Matched pairs:")
    print(f"  - Euclidean:    {lalonde_euc_count} matches")
    print(f"  - Mahalanobis:  {lalonde_mah_count} matches")
    print("\nFiles:")
    print("  - lalonde_euclidean_balance.png")
    print("  - lalonde_mahalanobis_balance.png")
    print("  - lalonde_euclidean_matched.csv")
    print("  - lalonde_mahalanobis_matched.csv")

    print("\n" + "-" * 80)
    print("SYNTHETIC DATA (Correlated Features, r=0.5)")
    print("-" * 80)
    print("Matched pairs:")
    print(f"  - Euclidean:    {synthetic_euc_count} matches")
    print(f"  - Mahalanobis:  {synthetic_mah_count} matches")
    print("\nFiles:")
    print("  - synthetic_euclidean_balance.png")
    print("  - synthetic_mahalanobis_balance.png")
    print("  - synthetic_euclidean_matched.csv")
    print("  - synthetic_mahalanobis_matched.csv")

    print("\n" + "-" * 80)
    print("Matched pair CSV schema:")
    print("  - id: treated patient ID")
    print("  - match_id: matched control patient ID")
    print("  - bucket_num_input_patients: bucket size")
    print("  - bucket_seconds: processing time")
    print("=" * 80)


if __name__ == "__main__":
    main()
