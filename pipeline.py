

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6cc08c25-2ae6-471b-afd3-63f93423ea3e"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.9aa45559-4c58-4d62-8dc3-a610ead950b0")
)
# BRP Match Feature Generation (034f8d01-059d-4248-91d7-3c1c3fe53deb): v15
# BRP Match - Generate Features (98fe93fe-5772-40d6-bad2-c78d4d33bfda): v1
from foundry_ml import Model, Stage
from numpy.linalg import pinv
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors, VectorUDT, DenseVector, Vector
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.feature import (
    BucketedRandomProjectionLSH, Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler, VectorIndexer)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import DenseVector
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType, StructField, ArrayType
from pyspark.sql.window import Window
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors
import functools

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import time

def brp_generate_features(all_patients_summary_fact_table_de_id):
    """
    This function converts the columns of the input dataset into a feature vector for clustering purposes.
    Columns are handled as follows by datatype unless otherwise specified:
        1) Date: Converted to int, represented as the datediff from 2018-01-01
        2) String: Converted to onehot vector
        3) Numeric: Missing values imputed using mean imputation
    Users may specifiy a list of columns to be imputed using a Gradient Boosted Tree rather than mean imputation.
    After all features are pre-processed, a StandardScalar is applied.
    """
    # Input parameters
    cohort_column = 'gender_concept_name'
    source_cohort = "MALE"
    target_cohort = "FEMALE"

    # anything that's not already categorical will be converted to one (as string):
    categorical_cols = ["state","data_partner_id","OBESITY_indicator","TOBACCOSMOKER_indicator","DEMENTIA_indicator","HEARTFAILURE_indicator","confirmed_covid_patient"]
    # default is to use mean imputation:
    numeric_cols = ["age","BMI_max_observed_or_calculated","total_number_of_COVID_vaccine_doses"]
    # will be converted to numeric as num days to 2018-1-1 (possibly negative)
    # default is to use mean imputation
    date_cols = []

    # must be a subset of categorical_cols
    exact_match_cols = []

    # must be a subset of numeric and/or date cols
    gbt_imputation_cols = ["BMI_max_observed_or_calculated"]

    # first let's check to be sure each col is only listed once in categorical_cols, numeric_cols, and date_cols
    all_cols = categorical_cols + numeric_cols + date_cols
    for col in all_cols:
        if all_cols.count(col) != 1:
            raise RuntimeError("The column " + col + " used used multiple times.")

    # and then that exact_match_cols are a subset of categorical cols, and that gbt cols are a subset of numeric/date cols
    for col in exact_match_cols:
        if col not in categorical_cols:
            raise RuntimeError("The column " + col + " must be listed as a categorical column to be used for exact matching.")

    for col in gbt_imputation_cols:
        if col not in numeric_cols + date_cols:
            raise RuntimeError("The column " + col + " must be listed as a numeric or date column to be used for GBT imputation.")

    # filter to the two cohorts of interest
    df = all_patients_summary_fact_table_de_id.filter(F.col(cohort_column).isin([source_cohort, target_cohort]))

    # Convert categorical cols to strings (even if they already are)
    for c in categorical_cols:
        df = df.withColumn(c, F.col(c).cast('string'))

    # Transform date columns to int/numeric, and call them numeric
    for c in date_cols:
        df = df.withColumn(f'{c}_days_from_2018', F.datediff(c, F.lit("2018-01-01")))
        numeric_cols.append(f'{c}_days_from_2018')
    

    # Create exact_match_id col as concatenation of those col values
    # (thanks chatgpt for help auto converting NULLs to "NULL"s ;))
    # Replace NULL values with a placeholder
    if len(exact_match_cols) > 0:
        df = df.withColumn("exact_match_id", F.when(df[exact_match_cols[0]].isNull(), "NULL").otherwise(df[exact_match_cols[0]]))
        # Concatenate the remaining columns
        for col in exact_match_cols[1:]:
            df = df.withColumn("exact_match_id", F.concat_ws(":", df["exact_match_id"], F.when(df[col].isNull(), "NULL").otherwise(df[col])))
    else:
        df = df.withColumn("exact_match_id", F.lit(1))

    # remove the exact match cols from the categorical cols (they won't be used for feature-based matching)
    categorical_cols = list(filter(lambda x: x not in exact_match_cols, categorical_cols))

    # remove the gbt_imputation_cols cols from the numeric cols (so they are not imputed w/ mean imputation)
    numeric_cols = list(filter(lambda x: x not in gbt_imputation_cols, numeric_cols))

    # Dynamically append stages to pre-processing pipeline
    preprocessing_stages = []
        
    # Convert categorical variables to onehot encodings
    print('categorical_cols',categorical_cols)
    categorical_index_cols = []
    for c in categorical_cols:
        preprocessing_stages += [StringIndexer(inputCol = c, outputCol = f"{c}_index", handleInvalid = "keep")]
        preprocessing_stages += [OneHotEncoder(inputCol = f"{c}_index", outputCol = f"{c}_onehot", dropLast = True)]
        categorical_index_cols.append(f"{c}_index")

    # Impute missing numeric values (default mean imputation)
    print('numeric_cols',numeric_cols)
    for c in numeric_cols:
        preprocessing_stages += [Imputer(inputCol = c, outputCol = f"{c}_imputed", strategy = "mean")]

    # Create feature vector from onehot/imputed columns
    feature_cols = [f"{c}_onehot" for c in categorical_cols] + [f"{c}_imputed" for c in numeric_cols]
    preprocessing_stages += [VectorAssembler(inputCols = feature_cols, outputCol = "gbt_features")]

    df = Pipeline(stages=preprocessing_stages).fit(df).transform(df)

    if len(gbt_imputation_cols) > 0:
        # Use Gradient Boosted Tree for complex imputation of specified columns
        for c in gbt_imputation_cols:
            values = df.filter(F.col(c).isNotNull())
            gbt_model = Pipeline(stages=[
                GBTRegressor(featuresCol = "gbt_features", labelCol = c, predictionCol = f"{c}_imputed", seed = 42)
            ])
            
            # Predict column value, but use actual value when available
            df = (gbt_model
                    .fit(values)
                    .transform(df)
                    .withColumn(f"{c}_imputed", F.coalesce(c, f"{c}_imputed"))
            )

        # Add GBT features to features vector
        print(feature_cols)
        feature_cols = ["gbt_features"] + [f"{c}_imputed" for c in gbt_imputation_cols
]
        print(feature_cols)
        finish_features_p = Pipeline(stages=[
            VectorAssembler(inputCols = feature_cols, outputCol = "unscaled_features"),
            StandardScaler(inputCol = "unscaled_features", outputCol = "features", withStd = True)
        ])
    else:
        # No GBT imputation, just scale existing features
        finish_features_p = Pipeline(stages=[
            StandardScaler(inputCol = "gbt_features", outputCol = "features", withStd = True)
        ])

    df = finish_features_p.fit(df).transform(df)
    # this treat column is currently only used in the downstream loveplot
    df = df.withColumn("treat", F.when(F.col("gender_concept_name")==source_cohort,1).otherwise(0))

    df = (df.select(categorical_cols +
                    numeric_cols +
                    date_cols +
                    gbt_imputation_cols +
                    exact_match_cols +
                    categorical_index_cols + 
                    feature_cols +
                    ["exact_match_id"] +
                    ["features"] +
                    ["treat"] +
                    [cohort_column] +
                    ["person_id"])
    )
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.80759c5f-fe62-4024-a7d4-f942788d5d77"),
    brp_generate_features=Input(rid="ri.foundry.main.dataset.6cc08c25-2ae6-471b-afd3-63f93423ea3e")
)
# BRP Match Feature-Based Matching (660d3327-1fe0-413a-b6f6-516598f1dc25): v61
# BRP Match - Match Cohorts (5e263ac7-e853-4a47-8a3e-1705ed8f501d): v1
from numpy.linalg import pinv
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors, VectorUDT, DenseVector, Vector
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.feature import (
    BucketedRandomProjectionLSH, Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler, VectorIndexer)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import DenseVector
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType, StructField, ArrayType
from pyspark.sql.window import Window
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import time

def brp_match_features(brp_generate_features):
    persons_features = brp_generate_features.limit(100000)
    """
    This function takes in a dataframe containing features for patients belonging to two cohorts.
    The goal is to match each patient in the 'source' cohort to at most one similar patient in the 
    'target' cohort. The algorithm is as follows:
        1) Each patient is hashed in a bucket determined by their features via Locality-sensitive hashing.
        2) Within each bucket, we use a KNN model to identify the k most similar 'target' patients to each 
            'source' patient which share a bucket_id.
        3) These source/target pairs are one-to-many (k). To identify 1-to-1 matches, repeat the 
           following while there are still unmatched sources and targets in the pool:
              A) Rank matches by distance. For each source, select the closest target. Sources are now unique, but targets may be duplicated (used by multiple sources)
              B) For each target, keep the closest source. This provides a set of 1-to-1 matches, so remove them from the pool and continue
        4) The method stops there are no more sources, or no more targets. These are left unmatched.
        5) Returns the list of unique matches betwen the 'source' and 'target' cohorts.
    """
    # Input parameters
    cohort_column = "gender_concept_name"
    needs_matching_cohort = "MALE"
    match_to_cohort = "FEMALE"

    # default:
    within_bucket_distance_metric = "euclidean" # vs mahalanobis
    neighbor_cnt = 5

    num_patients_trigger_rebucket = 10000

    # was going to make this configurable, but the downstream brp_stratify_matches_for_love_plot 
    # is in SQL so it would be hard to compute the equivalent of 'match_' + id_col column name
    # so let's not parameterize this via the template at least
    id_col = "person_id"

    persons_features_cohorts = (persons_features
        .withColumn('feature_array', vector_to_array("features"))
    )

    # Logging
    print('Pre-filter')
    print(persons_features_cohorts.groupBy(cohort_column).count().toPandas())

    # Use Locality-sensitive hashing to cluster records into buckets
    feature_cnt = persons_features_cohorts.limit(1).select(F.size(F.col("feature_array"))).collect()[0][0]
    bucket_length = pow(persons_features_cohorts.count(), (-1/feature_cnt))

    bucket_stages = [
        BucketedRandomProjectionLSH(inputCol = "features", outputCol = "bucket_hashes1", seed = 42, numHashTables = 4, bucketLength = bucket_length),
        BucketedRandomProjectionLSH(inputCol = "features", outputCol = "bucket_hashes2", seed = 42, numHashTables = 4, bucketLength = bucket_length / 4),
        BucketedRandomProjectionLSH(inputCol = "features", outputCol = "bucket_hashes3", seed = 42, numHashTables = 4, bucketLength = bucket_length / 16),
        BucketedRandomProjectionLSH(inputCol = "features", outputCol = "bucket_hashes4", seed = 42, numHashTables = 4, bucketLength = bucket_length / 64)
    ]

    persons_bucketed = (Pipeline(stages = bucket_stages)
        .fit(persons_features_cohorts)
        .transform(persons_features_cohorts)
        .withColumn("bucket_id1", F.concat_ws(":", F.col("exact_match_id"), F.lit("b1"), F.xxhash64(F.col("bucket_hashes1"))))
        .withColumn("bucket_id2", F.concat_ws(":", F.col("exact_match_id"), F.lit("b2"), F.xxhash64(F.col("bucket_hashes2"))))
        .withColumn("bucket_id3", F.concat_ws(":", F.col("exact_match_id"), F.lit("b3"), F.xxhash64(F.col("bucket_hashes3"))))
        .withColumn("bucket_id4", F.concat_ws(":", F.col("exact_match_id"), F.lit("b4"), F.xxhash64(F.col("bucket_hashes4"))))
       # .drop("bucket_hashes1", "bucket_hashes2", "bucket_hashes3", "bucket_hashes4")
    )

    bucket_counts1 = (persons_bucketed
        .groupBy("bucket_id1")
        .agg(F.countDistinct(id_col).alias("num_patients1_raw"))
    )
    bucket_counts2 = (persons_bucketed
        .groupBy("bucket_id2")
        .agg(F.countDistinct(id_col).alias("num_patients2_raw"))
    )
    bucket_counts3 = (persons_bucketed
        .groupBy("bucket_id3")
        .agg(F.countDistinct(id_col).alias("num_patients3_raw"))
    )
    bucket_counts4 = (persons_bucketed
        .groupBy("bucket_id4")
        .agg(F.countDistinct(id_col).alias("num_patients4_raw"))
    )

    persons_bucketed = (persons_bucketed
        .join(bucket_counts1, "bucket_id1", how = "full")
        .join(bucket_counts2, "bucket_id2", how = "full")
        .join(bucket_counts3, "bucket_id3", how = "full")
        .join(bucket_counts4, "bucket_id4", how = "full")
    )

    # from left_anti:
    # person: 1024301102435558204
    # bucket_id1: 1:b1:425935482
    return persons_bucketed

    # grrr, how can there be nulls here anyway?
    # persons_bucketed = (persons_bucketed
    #          .fillna(0, subset=['num_patients1_raw',
    #                             'num_patients2_raw',
    #                             'num_patients3_raw',
    #                             'num_patients4_raw'])
    # )

    print("two")
    print(persons_bucketed.count())

    return persons_bucketed

    persons_bucketed = (persons_bucketed
        .withColumn("bucket_id", 
                    F.when(F.col("num_patients1_raw") < num_patients_trigger_rebucket, F.col("bucket_id1"))
                     .when(F.col("num_patients2_raw") < num_patients_trigger_rebucket, F.col("bucket_id2"))
                     .when(F.col("num_patients3_raw") < num_patients_trigger_rebucket, F.col("bucket_id3"))
                     .when(F.col("num_patients4_raw") < num_patients_trigger_rebucket, F.col("bucket_id4"))
                     .otherwise(F.lit(None))
                    )
       .withColumn("bucket_id_source", 
                    F.when(F.col("num_patients1_raw") < num_patients_trigger_rebucket, "bucket_1")
                     .when(F.col("num_patients2_raw") < num_patients_trigger_rebucket, "bucket_2")
                     .when(F.col("num_patients3_raw") < num_patients_trigger_rebucket, "bucket_3")
                     .when(F.col("num_patients4_raw") < num_patients_trigger_rebucket, "bucket_4")
                     .otherwise(F.lit(None))
                    )
        .drop("num_patients1_raw", "num_patients2_raw", "num_patients3_raw", "num_patients4_raw", "bucket_id1", "bucket_id2", "bucket_id3", "bucket_id4")
        #.filter(F.col("bucket_id").isNotNull()) 
    )

    #Identify buckets which don't have patients from both cohorts
    viable_buckets = (persons_bucketed
        .groupBy("bucket_id")
        .agg(F.countDistinct(cohort_column).alias("types"))
        .filter(F.col("types")==2)
        .select("bucket_id")
        .distinct()
    )

    all_persons_bucketed = persons_bucketed

    # Drop non-viable buckets without patients from both cohorts
    persons_bucketed = persons_bucketed.join(viable_buckets, "bucket_id")

    # print the user some info about the buckets for parameter tuning
    log_bucket_stats(persons_bucketed)

    # Define covariance matrix and distance function for computing Mahalanobis
    if within_bucket_distance_metric == "mahalanobis":
        # Compute the psuedoinverse of the covariance matrix for features
        vec_converter_udf = F.udf(lambda v: Vectors.dense(v.toArray()), VectorUDT())
        features_converted = (persons_features_cohorts
            .withColumn("features_converted", vec_converter_udf('features'))
            .select("features_converted")
        )
        inverse_covariance_mat = pinv(RowMatrix(features_converted).computeCovariance().toArray())

        def mahal_dist(vec1, vec2):
            "Custom distance function using our globally-computed inverse covariance matrix"
            return mahalanobis(vec1, vec2, inverse_covariance_mat) 

    # Schema for return from pandas udf next
    schema_potential_matches_arrays = StructType([
        StructField(id_col, StringType()),
        StructField("match_" + id_col, StringType()),
        StructField("match_distance", DoubleType()),
        #StructField("bucket_id", StringType()),
        StructField("bucket_num_input_patients", IntegerType()),
        StructField("bucket_seconds", DoubleType())
    ])

    @pandas_udf(schema_potential_matches_arrays, PandasUDFType.GROUPED_MAP)
    def find_neighbors(group_key, group_df):
        """
        Given a group of input rows (those with the same bucket_id),
        returns a data frame of source-target matches.
        (Note that this function must be locally defined since it carries
        with it the other variables e.g. needs_matching_cohort to the executor)
        """
        bucket_start_time = time.perf_counter()
        n = neighbor_cnt
        # Extract the individual cohorts
        needs_matching = group_df.loc[group_df[cohort_column] == needs_matching_cohort]
        match_to = group_df.loc[group_df[cohort_column] == match_to_cohort]

        # Return empty DF if either cohort is empty
        if needs_matching.count()[0] == 0 or match_to.count()[0] == 0:
            return pd.DataFrame(columns=schema_potential_matches_arrays.fieldNames())

        # Update n to size(match_to) if smaller than n
        n = min(match_to.count()[0], n)

        # Set distance metric for NN model
        if within_bucket_distance_metric == "euclidean":
            metric = within_bucket_distance_metric
        elif within_bucket_distance_metric == "mahalanobis":
            metric = mahal_dist
        else:
            raise ValueError("Within-bucket distance metric must be one of 'euclidean' or 'mahalanobis'. Got '{within_bucket_distance_metric}'.")
        
        # Extract input pandas dataframe columns to useful types
        person_ids_need_matching = list(needs_matching[id_col])
        features_need_matching = np.array(list(needs_matching["feature_array"])) #2D array
        person_ids_match_to = list(match_to[id_col])
        features_match_to = np.array(list(match_to["feature_array"])) #2D array

        # Find nearest neighbors for the features needing matching!
        model = NearestNeighbors(metric=metric).fit(features_match_to)
        neighbors = model.kneighbors(features_need_matching, n_neighbors = n, return_distance = True)
        
        # Lists of neighbors and distances for each person_id
        match_person_ids = list(map(lambda indices: [person_ids_match_to[i] for i in indices], neighbors[1].tolist()))
        match_distances = neighbors[0].tolist()
        
        results = None
        for person_id, match_ids, match_dists in zip(person_ids_need_matching, match_person_ids, match_distances):
            # Broadcast person_id
            person_id = [person_id] * len(match_ids)
            
            # Explode matches to one per row
            person_matches = np.stack([person_id, match_ids, match_dists], axis=1)

            # Iteratively stack results for each person_id
            results = np.vstack([results, person_matches]) if results is not None else person_matches

        # Rank candidates matches in both directions and sort
        match_candidates = pd.DataFrame(results, columns=[id_col,'match_' + id_col, 'match_distance'])
        match_candidates['match_distance'] = pd.to_numeric(match_candidates['match_distance'])
        match_candidates['source_target_rank'] = match_candidates.groupby([id_col])['match_distance'].rank()
        match_candidates['target_source_rank'] = match_candidates.groupby('match_' + id_col)['match_distance'].rank(method='first')
        match_candidates = match_candidates.sort_values(['source_target_rank','target_source_rank'])
        
        i = 0
        while len(match_candidates) > i:
            # Ordering and XOR drop ensures that the ith record always greedily chooses best source=>target match with ties broken by target=>source match
            person_id, match_person_id = match_candidates.iloc[i][[id_col,'match_' + id_col]]
            
            # Drop records matching person_id XOR match_person_id
            match_candidates = match_candidates[~((match_candidates.person_id==person_id) ^ (match_candidates.match_person_id==match_person_id))]
            i += 1
        
        match_candidates["bucket_num_input_patients"] = group_df.shape[0]

        bucket_end_time = time.perf_counter()
        match_candidates["bucket_seconds"] = bucket_end_time - bucket_start_time
        #match_candidates["bucket_id"] = group_key[0]
        return match_candidates.drop(columns=['source_target_rank', 'target_source_rank'])

    # Set number of neighbors to match
    # neighbor_cnt = 5
    
    # Find matches for each bucket
    matches = (persons_bucketed
        .select(id_col, 
                cohort_column, 
                "bucket_id", 
                "bucket_id_source", 
                "feature_array")
        .groupBy("bucket_id")
        .apply(find_neighbors)
    )

    bucket_info = persons_bucketed.select("person_id", "bucket_id", "bucket_id_source").distinct()

    # matches = matches.join(bucket_info, on = "person_id", how = "inner")
    
    return all_persons_bucketed

def log_bucket_stats(persons_bucketed):
    # there's gotta be a better way to just get group counts in pyspark
    bucket_counts = (persons_bucketed
        .groupBy("bucket_id")
        .agg(F.count(F.col("bucket_id")).alias("bucket_num_persons"))
    )
    print(f"Num buckets: {bucket_counts.count()}")
    print("Bucket stats:")
    test = bucket_counts.select(
        F.min(F.col("bucket_num_persons")).alias("min"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.05).alias("percentile_5"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.25).alias("percentil_25"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.5).alias("percentile_50"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.75).alias("percentile_75"),
        F.percentile_approx(F.col("bucket_num_persons"), 0.95).alias("percentil_95"),
        F.max(F.col("bucket_num_persons")).alias("max"),
        F.mean(F.col("bucket_num_persons")).alias("mean"),
        ).toPandas()
    print(test.to_string())

