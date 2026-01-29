"""Utility functions for BRPMatch examples."""

import os
import sys
import pandas as pd
from pyspark.sql import SparkSession, functions as F


class Colors:
    """ANSI color codes for terminal output."""
    # Only use colors if stdout is a TTY
    _use_color = sys.stdout.isatty()

    BLUE = '\033[94m' if _use_color else ''
    CYAN = '\033[96m' if _use_color else ''
    GREEN = '\033[92m' if _use_color else ''
    YELLOW = '\033[93m' if _use_color else ''
    MAGENTA = '\033[95m' if _use_color else ''
    BOLD = '\033[1m' if _use_color else ''
    RESET = '\033[0m' if _use_color else ''


def section_header(text: str, char: str = "=") -> str:
    """Create a colored section header."""
    line = char * len(text)
    return f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.RESET}\n{Colors.CYAN}{line}{Colors.RESET}"


def subsection_header(text: str) -> str:
    """Create a colored subsection header."""
    return f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.RESET}"


def highlight(text: str) -> str:
    """Highlight text in green."""
    return f"{Colors.GREEN}{text}{Colors.RESET}"


def value(text: str) -> str:
    """Format a value in bold."""
    return f"{Colors.BOLD}{text}{Colors.RESET}"


def create_spark_session(app_name: str) -> SparkSession:
    """
    Create a configured SparkSession for examples.

    Parameters
    ----------
    app_name : str
        Name for the Spark application

    Returns
    -------
    SparkSession
        Configured Spark session with logging suppressed
    """
    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow -Dorg.slf4j.simpleLogger.defaultLogLevel=ERROR")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_lalonde(spark: SparkSession):
    """
    Load the Lalonde dataset.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session

    Returns
    -------
    DataFrame
        Lalonde dataset with id column added if not present
    """
    # Find the data directory (2 levels up from example subdirectories)
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "lalonde.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Lalonde dataset not found at {data_path}")

    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Add id column if not present
    if "id" not in df.columns:
        df = df.withColumn("id", F.monotonically_increasing_id().cast("string"))

    return df


def setup_pandas_display():
    """Configure pandas display options for cleaner output in examples."""
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 120)  # Wider terminal
    pd.set_option('display.max_colwidth', 20)  # Truncate long values


def print_matching_stats(features_df, matched, treatment_col="treat__treat", id_col="id__id"):
    """
    Print statistics about matched vs unmatched patients.

    Parameters
    ----------
    features_df : DataFrame
        Features dataframe with treatment column
    matched : DataFrame
        Matched pairs dataframe with id and match_id columns
    treatment_col : str
        Name of treatment column in features_df
    id_col : str
        Name of ID column in features_df
    """
    from pyspark.sql import functions as F

    # Get treated and control counts from features
    treated_df = features_df.filter(F.col(treatment_col) == "1")
    control_df = features_df.filter(F.col(treatment_col) == "0")

    n_treated_total = treated_df.count()
    n_control_total = control_df.count()

    # Get matched counts
    n_treated_matched = matched.select("id").distinct().count()
    n_control_matched = matched.select("match_id").distinct().count()

    # Print statistics
    print(f"  Treated: {highlight(str(n_treated_matched))}/{value(str(n_treated_total))} matched")
    print(f"  Control: {highlight(str(n_control_matched))}/{value(str(n_control_total))} matched")
    print(f"  Total matched pairs: {highlight(str(matched.count()))}")
