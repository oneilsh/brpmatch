

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e6de0fd5-5a67-4182-a058-ec3b1eb12fb6"),
    brp_generate_features=Input(rid="ri.foundry.main.dataset.6cc08c25-2ae6-471b-afd3-63f93423ea3e"),
    brp_match_features=Input(rid="ri.foundry.main.dataset.80759c5f-fe62-4024-a7d4-f942788d5d77")
)
-- BRP Match Stratify Matches for Love Plot (24c46e76-7197-47d4-8dd8-de40fb58e9c1): v0
WITH with_strata AS (
    SELECT
        *,
        concat(person_id, ":", match_person_id) as strata
    FROM brp_match_features
),

-- from person_features, join in a column from match_cohorts that indicates if each patient is matched
matched_from AS (
    SELECT 
        brp_generate_features.*,
        with_strata.person_id AS matched_from,
        with_strata.strata AS strata_from
    FROM brp_generate_features
    LEFT OUTER JOIN with_strata USING (person_id)
),

-- from above (person_features), join in a column from match_cohorts that indicates if each patient is matched to
matched_to AS (
    SELECT 
        matched_from.*,
        with_strata.match_person_id AS matched_to,
        with_strata.strata AS strata_to
    FROM matched_from
    LEFT OUTER JOIN with_strata ON with_strata.match_person_id = matched_from.person_id
),

-- coalesce them down to a single column indicating if the patient is matched
is_matched AS (
    SELECT 
        matched_to.*,
        coalesce(matched_from, matched_to) as is_matched,
        coalesce(strata_from, strata_to) as strata
    FROM matched_to
)

SELECT * from is_matched

