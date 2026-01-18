"""
Pytest fixtures for BRPMatch testing.
"""

import os

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    import warnings
    warnings.filterwarnings("ignore")

    # Set Java options for compatibility with Java 17+
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    session = (
        SparkSession.builder.master("local[*]")
        .appName("brpmatch-test")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="session")
def lalonde_df(spark):
    """Load the lalonde test dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "lalonde.csv")

    # Check if file exists, if not skip tests that need it
    if not os.path.exists(data_path):
        pytest.skip(f"Lalonde dataset not found at {data_path}")

    return spark.read.csv(data_path, header=True, inferSchema=True)
