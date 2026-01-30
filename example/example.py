import os
import sys

#######################################################################
##
## 0. Setup
##
##    Data must live in a Spark dataframe. Here we load the lalonde 
##    data used by the MatchIt package in R.
##
##    We also load the 3 main functions from brpmatch:
##      - generate_features(): data pre-processing
##      - match(): perform matching
##      - match_summary(): matching stats, balance summaries, and Love plots
## 
#######################################################################

script_location = os.path.dirname(__file__)
sys.path.insert(0, script_location)
from example_utils import create_spark_session, load_lalonde

spark = create_spark_session("brpmatch-1to1-euclidean")
data_df = load_lalonde(spark)

from brpmatch import generate_features, match, match_summary



data_df.show(5)
# +----+-----+---+----+------+-------+--------+----+----+--------+
# |  id|treat|age|educ|  race|married|nodegree|re74|re75|    re78|
# +----+-----+---+----+------+-------+--------+----+----+--------+
# |NSW1|    1| 37|  11| black|      1|       1| 0.0| 0.0|9930.046|
# |NSW2|    1| 22|   9|hispan|      0|       1| 0.0| 0.0|3595.894|
# |NSW3|    1| 30|  12| black|      0|       0| 0.0| 0.0|24909.45|
# |NSW4|    1| 27|  11| black|      0|       1| 0.0| 0.0|7506.146|
# |NSW5|    1| 33|   8| black|      0|       1| 0.0| 0.0|289.7899|
# +----+-----+---+----+------+-------+--------+----+----+--------+




#######################################################################
##
## 2. Generate features for matching
##
##    Matching requires continuous features:
##      - Categorical and boolean columns are one-hot encoded
##      - Date columns (if provided with date_cols) are converted
##        to numeric as days from a reference_date (default 1/1/1970)
##    
##    Required options:
##      - Spark context
##      - Dataframe with data
##      - A row-unique id_col column name
##      - A column name specifying the treatment information (cohorts)
##      - A value indicating the 'treated' group (to which non-treated 
##        matching entries will be matched)
##      - At least one categorical, numeric, or date column to balance
##
##    Other options:
##      - exact_match_cols: column names to force exact matching on
##      - max_categories: raises an error if categorical columns have
##        more than this number of values to prevent one-hot 
##        blowup (default 20)
##
##    Result: A processed spark dataframe ready for matching, with column 
##            names encoding data types for internal use
## 
#######################################################################

features_df = generate_features(
    spark, 
    data_df,
    id_col="id",
    treatment_col="treat",
    treatment_value="1",
    categorical_cols=["race", "married", "nodegree"],
    numeric_cols=["age", "educ", "re74", "re75"],
)

features_df.show(5)
# +------+------------+---------------+----------------+---------------+--------------+--------------+---------------+---------------+--------+---------+---------+---------+------------------+--------------------+
# |id__id|treat__treat|race_black__cat|race_hispan__cat|race_white__cat|married_0__cat|married_1__cat|nodegree_0__cat|nodegree_1__cat|age__num|educ__num|re74__num|re75__num|exact_match__group|            features|
# +------+------------+---------------+----------------+---------------+--------------+--------------+---------------+---------------+--------+---------+---------+---------+------------------+--------------------+
# |  NSW1|           1|            1.0|             0.0|            0.0|           0.0|           1.0|            0.0|            1.0|    37.0|     11.0|      0.0|      0.0|               all|(11,[0,4,6,7,8],[...|
# |  NSW2|           1|            0.0|             1.0|            0.0|           1.0|           0.0|            0.0|            1.0|    22.0|      9.0|      0.0|      0.0|               all|(11,[1,3,6,7,8],[...|
# |  NSW3|           1|            1.0|             0.0|            0.0|           1.0|           0.0|            1.0|            0.0|    30.0|     12.0|      0.0|      0.0|               all|(11,[0,3,5,7,8],[...|
# |  NSW4|           1|            1.0|             0.0|            0.0|           1.0|           0.0|            0.0|            1.0|    27.0|     11.0|      0.0|      0.0|               all|(11,[0,3,6,7,8],[...|
# |  NSW5|           1|            1.0|             0.0|            0.0|           1.0|           0.0|            0.0|            1.0|    33.0|      8.0|      0.0|      0.0|               all|(11,[0,3,6,7,8],[...|
# +------+------------+---------------+----------------+---------------+--------------+--------------+---------------+---------------+--------+---------+---------+---------+------------------+--------------------+






#######################################################################
##
## 3. Perform Matching
##
##    Matching requires the processed result of generate_features above.
## 
##    Most Relevant Options:
##      - feature_space: Either "euclidean" or "mahalanobis" for
##        nearest-neighbor matching.
##        Default "euclidean".
##
##      - n_neighbors: Number of neighbors to consider for each
##        treated row; should be >= ratio_k and may be larger to
##        allow improved matching at some computational expense.
##        Default 5.
##
##      - bucket_length_multiplier: Patients are grouped into 
##        spatial buckets before efficiently computing within-bucket
##        nearest neighbors. If you are getting poor match rates you 
##        can scale buckets up or down by this factor (see also
##        the bucket stats result of match_summary() below), at
##        the expense of increased processing time or out-of-memory risk.
##        Default 1.0.
##
##      - num_patients_trigger_rebucket: Buckets with more than
##        this number of patients are re-bucketed using 4x smaller
##        bucket sizes, up to 3 times. You may consider increasing
##        this up to increase match rates at the expense of computation.
##        Default 10000.
##
##      - ratio_k: Number of control patients to attempt to match
##        to each treated patient. 
##        Default 1.
##
##      - with_replacement: Whether control patients may be matched
##        to multiple treated patients within a bucket.
##        Default: False.
##
##      - reuse_max: Maximum number of times a control patient may 
##        be matched to treated patients, if replacement is allowed.
##        Default: None (unlimited).
##
##      - require_k: Whether to only include treated matches with ratio_k
##        successful control matches (True), or allow partial-k matches (False)
##        Default: False.
##
##    Result: A tuple with three spark dataframes
##      - units: One row per patient, indicating their match (via subclass),
##               weight (0.0 if unmatched, less than 1.0 if more than
##               one match for ATT estimation), and is_treated information.
##
##               ** This the PRIMARY RESULT for downstream analysis - join it 
##                  to your original data, removing weight = 0.0 unmatched rows to
##                  keep only matched data.
##
##      - pairs: Individual treated/control match information; for each
##               treated/control match pair, the 'round' of matching
##               (when ratio_k > 1, as matches are found in a round-robin
##               fashion for fairness), how many treated/control patients
##               are matched to the listed treated/control patient, and weight
##               information. Possibly useful for debugging.
## 
##      - bucket_stats: Information about each bucket, indicating number of 
##                      treated/control patients, number matches, and seconds
##                      taken for processing.
## 
#######################################################################


units, pairs, bucket_stats = match(
    features_df,
    feature_space = "euclidean",
    ratio_k = 1,
    with_replacement = False
)

units.show(5)
# +-------+--------+------+----------+
# |     id|subclass|weight|is_treated|
# +-------+--------+------+----------+
# |  NSW98|   NSW98|   1.0|      true|
# |PSID395|   NSW98|   1.0|     false|
# |PSID268|   NSW97|   1.0|     false|
# |  NSW97|   NSW97|   1.0|      true|
# |  NSW96|   NSW96|   1.0|      true|
# +-------+--------+------+----------+

pairs.show(5)
# +------+--------+-----------+---------+-------------------+-----------+
# |    id|match_id|match_round|treated_k|control_usage_count|pair_weight|
# +------+--------+-----------+---------+-------------------+-----------+
# |NSW182|   PSID3|          1|        1|                  1|        1.0|
# |NSW183|   PSID6|          1|        1|                  1|        1.0|
# |NSW172|  PSID41|          1|        1|                  1|        1.0|
# |NSW184|   PSID5|          1|        1|                  1|        1.0|
# |NSW181|  PSID59|          1|        1|                  1|        1.0|
# +------+--------+-----------+---------+-------------------+-----------+

bucket_stats.show(5)
# +--------------------+------------+-----------+-----------+-----------+--------------------+
# |           bucket_id|num_patients|num_treated|num_control|num_matches|             seconds|
# +--------------------+------------+-----------+-----------+-----------+--------------------+
# |all:b1:-524220591...|           7|          3|          4|          3|0.002822708920575...|
# |all:b1:6396836968...|          57|         11|         46|         11|0.004017750034108758|
# |all:b1:-122457983...|          41|          8|         33|          8|0.004442250006832182|
# |all:b1:3852406253...|          57|          3|         54|          3|0.002934541087597...|
# |all:b1:5907845780...|         242|        100|        142|         55|  0.0154771669767797|
# +--------------------+------------+-----------+-----------+-----------+--------------------+




#######################################################################
##
## 4. Assess Balance
##
##    The match_summary() function provides information on matching
##    statistics and unadjusted vs. adjusted covariate balance.
## 
##    Required Inputs:
##      - The features dataframe from generate_features()
##      - The units dataframe from match()
##
##    Other Most Relevant Options:
##      - sample_frac: If the number of patients is large, it is infeasible
##        to compute balance statistics on the whole cohort. This parameter
##        sets a random sample percentage.
##        Default: 0.05.
##      
##      - include_ecdf: Whether to include ECDF (empirical cumulative distribution 
##        function) balance statistics in addition to SMD (standardized 
##        mean difference) and VR (variance ratio).
## 
##
##    Result: A tuple two pandas dataframes and a matplotlib figure
##      - balance_pandas_df: A Pandas dataframe with SMD statistics
##        for discrete variables, SMD and VR for continuous variables, and 
##        ECDF statistics if included.
##
##      - summary_pandas_df: Aggregate information about matching performance,
##        number of matched patients, averages, etc.
##
##      - fig: A matplotlib-based Love plot for plotting.
## 
#######################################################################


balance_pandas_df, summary_pandas_df, fig = match_summary(features_df, units, sample_frac=1.0)

print(balance_pandas_df.to_markdown(tablefmt="orgtbl", index=False))
# | display_name   |   mean_treated |   mean_control |   mean_treated_adj |   mean_control_adj |   smd_unadjusted |   smd_adjusted |   vr_unadjusted |   vr_adjusted |
# |----------------+----------------+----------------+--------------------+--------------------+------------------+----------------+-----------------+---------------|
# | race_black     |      0.843243  |       0.202797 |          0.747826  |          0.643478  |        1.66772   |      0.227256  |      nan        |    nan        |
# | race_hispan    |      0.0594595 |       0.142191 |          0.0956522 |          0.0956522 |       -0.27694   |      0         |      nan        |    nan        |
# | race_white     |      0.0972973 |       0.655012 |          0.156522  |          0.26087   |       -1.40574   |     -0.257791  |      nan        |    nan        |
# | married_0      |      0.810811  |       0.487179 |          0.730435  |          0.721739  |        0.719492  |      0.0194145 |      nan        |    nan        |
# | married_1      |      0.189189  |       0.512821 |          0.269565  |          0.278261  |       -0.719492  |     -0.0194145 |      nan        |    nan        |
# | nodegree_0     |      0.291892  |       0.403263 |          0.330435  |          0.347826  |       -0.235048  |     -0.036582  |      nan        |    nan        |
# | nodegree_1     |      0.708108  |       0.596737 |          0.669565  |          0.652174  |        0.235048  |      0.036582  |      nan        |    nan        |
# | age            |     25.8162    |      28.0303   |         25.687     |         26.0087    |       -0.241904  |     -0.0355228 |        0.439995 |      0.642552 |
# | educ           |     10.3459    |      10.2354   |         10.2783    |         10.2522    |        0.0447551 |      0.01057   |        0.495893 |      0.657878 |
# | re74           |   2095.57      |    5619.24     |       2956.27      |       3326.1       |       -0.595752  |     -0.0648136 |        0.518128 |      1.13056  |
# | re75           |   1532.06      |    2466.48     |       1828.79      |       1722.07      |       -0.287002  |      0.0301724 |        0.956293 |      1.32177  |

print(summary_pandas_df.to_markdown(tablefmt="orgtbl", index=False))
# | statistic                     |    value |
# |-------------------------------+----------|
# | sample_frac                   |   1      |
# | n_treated_total               | 185      |
# | n_control_total               | 429      |
# | n_treated_matched             | 115      |
# | n_control_matched             | 115      |
# | n_treated_unmatched           |  70      |
# | n_control_unmatched           | 314      |
# | pct_treated_matched           |  62.1622 |
# | pct_control_matched           |  26.8065 |
# | mean_controls_per_treated     |   1      |
# | min_controls_per_treated      |   1      |
# | max_controls_per_treated      |   1      |
# | effective_sample_size_treated | 115      |
# | effective_sample_size_control | 115      |

fig.savefig(os.path.join(script_location, "lalonde_balance.png"), dpi=150, bbox_inches="tight")





#######################################################################
##
## 5. Use Matched Data
##
##    The units dataframe generated by match() may be joined back
##    to the original data for use in downstream analyses, potentially
##    including the subclass strata and weight information.
##
##    NOTE: The units dataframe also includes unmatched patients.
##          To restrict to only matched patients, pre-filter
##          to select patients with weight > 0.
##
#######################################################################

# Filter to matched patients only (weight > 0)
units_matched = units.filter(units.weight > 0)

# Join back to original data
data_matched = (
    data_df
       .join(units_matched, data_df.id == units_matched.id, "inner")
       .drop(units_matched.id)
       .orderBy("subclass")
)

data_matched.show(5)
# +-------+-----+---+----+------+-------+--------+----+----+--------+--------+------+----------+
# |     id|treat|age|educ|  race|married|nodegree|re74|re75|    re78|subclass|weight|is_treated|
# +-------+-----+---+----+------+-------+--------+----+----+--------+--------+------+----------+
# |   NSW1|    1| 37|  11| black|      1|       1| 0.0| 0.0|9930.046|    NSW1|   1.0|      true|
# |PSID368|    0| 40|  11| black|      1|       1| 0.0| 0.0|     0.0|    NSW1|   1.0|     false|
# |  NSW10|    1| 33|  12| white|      1|       0| 0.0| 0.0|12418.07|   NSW10|   1.0|      true|
# |PSID393|    0| 38|  12| white|      1|       0| 0.0| 0.0|18756.78|   NSW10|   1.0|     false|
# | NSW100|    1| 31|   9|hispan|      0|       1| 0.0| 0.0| 26817.6|  NSW100|   1.0|      true|
# +-------+-----+---+----+------+-------+--------+----+----+--------+--------+------+----------+

# Collect the matched data to a local pandas dataframe
data_matched_pandas = data_matched.toPandas()

# Analyze it or save it
data_matched_pandas.to_csv(os.path.join(script_location, "lalonde_matched.csv"), index=False)
