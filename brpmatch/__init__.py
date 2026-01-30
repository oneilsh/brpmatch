"""
BRPMatch: Large-scale distance-based cohort matching on Apache Spark.

BRPMatch is a Spark-based cohort matching tool that uses distance-based methods
(LSH + k-NN) for large-scale propensity score-like matching.
"""

from .features import generate_features
from .loveplot import love_plot
from .matching import match
from .summary import match_summary

__all__ = [
    "generate_features",
    "match",
    "match_summary",
    "love_plot",
]
