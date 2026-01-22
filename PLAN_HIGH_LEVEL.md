# BRPMatch API Redesign - High-Level Plan

## Problem Statement

The current BRPMatch API has several issues that need to be addressed:

### 1. Incorrect Categorical Encoding

Categorical columns are being transformed into index columns (e.g., `race_index`) rather than being one-hot encoded (e.g., `race_white`, `race_black`, `race_hispanic`).

**Why this matters:**
- Index encoding treats nominal categories as ordinal, which is statistically incorrect
- Balance assessment should evaluate each category indicator independently
- The love plot currently shows `race_index` which is meaningless for assessing balance across racial groups

### 2. API Redundancy

The current API requires users to re-specify parameters across function calls:

```python
# Current API - repetitive
features_df = generate_features(spark, df, ..., id_col="id")
matched_df = match(features_df, ..., id_col="id")
summary = match_summary(features_df, matched_df, feature_cols, id_col="id")
```

The `feature_cols` list must be manually constructed and passed to `match_summary()`, even though this information is implicitly defined by the `generate_features()` call.

### 3. Missing match_data() Function

There is no equivalent to R's matchit `match.data()` function, which returns a filtered/weighted DataFrame suitable for downstream statistical analysis (e.g., logistic regression for treatment effect estimation).

---

## Goals

1. **Correct categorical handling**: One-hot encode categorical variables for proper distance matching and balance assessment

2. **Self-describing DataFrames**: Use column naming conventions so downstream functions can auto-discover feature columns, ID columns, and treatment indicators without re-specification

3. **Simplified API**: Reduce parameter redundancy across function calls

4. **New match_data() function**: Provide a way to extract matched data with appropriate weights for downstream analysis

5. **Improved love plot**: Show all covariates including exact match columns, with clean display names

---

## Design Decisions

### Column Naming Convention (Suffix-Based)

All columns created by `generate_features()` will use descriptive suffixes:

| Type | Pattern | Example |
|------|---------|---------|
| Categorical (one-hot) | `{col}_{value}__cat` | `race_white__cat`, `race_black__cat` |
| Numeric | `{col}__num` | `age__num`, `bmi__num` |
| Date | `{col}__date` | `diagnosis_date__date` |
| Exact match (one-hot) | `{col}_{value}__exact` | `gender_male__exact` |
| ID | `{col}__id` | `person_id__id` |
| Treatment | `treat__treat` | (standardized 0/1 indicator) |
| Exact match grouping | `exact_match__group` | (composite stratification ID) |

### Special Character Handling

For categorical values with spaces or special characters:
- Lowercase everything
- Replace spaces with `_`
- Replace problematic characters (`/`, `\`, `.`, `-`) with `_`
- Collapse multiple consecutive `_` into one

Example: `"African American"` → `race_african_american__cat`

**Note**: Column/value names containing `__` are not supported (documented limitation).

### No JSON Metadata Column

With the suffix convention, all necessary information is discoverable through pattern matching:
- Features: columns ending in `__cat`, `__num`, `__date`, `__exact`
- ID: column ending in `__id`
- Treatment: column ending in `__treat`
- Exact match grouping: column ending in `__group`

### Categorical Cardinality Limit

To avoid performance issues with high-cardinality categorical columns, `generate_features()` will enforce a maximum number of distinct values (default: 20) with an override parameter. This is appropriate because:
- Matching on high-cardinality categoricals (e.g., zip codes) is statistically unusual
- The one-hot encoding approach requires collecting distinct values to the driver
- Users can increase the limit if needed via `max_categories` parameter

### Standardization Stays in match()

Numeric standardization (z-scoring) and feature space transformations (Mahalanobis whitening) remain in the `match()` function, not `generate_features()`.

### match_data() Accepts Original DataFrame

The new `match_data()` function will accept the original DataFrame (not `features_df`) to keep `features_df` lean during matching computation. The join on ID is efficient in Spark.

### Love Plot Display

- Strip `__cat`, `__num`, `__date`, `__exact` suffixes for clean display
- Show exact match columns (they'll have perfect balance by construction, but showing them documents the matching strategy)
- Optionally distinguish exact match columns visually (different marker or annotation)

### ATT Weight Calculation for match_data()

The `match_data()` function computes weights for **Average Treatment Effect on the Treated (ATT)** estimation following the methodology described in [Greifer's matching weights blog](https://ngreifer.github.io/blog/matching-weights/) and [MatchIt documentation](https://kosukeimai.github.io/MatchIt/reference/matchit.html).

#### Why ATT (not ATE)?

Matching methods inherently estimate ATT, not ATE (Average Treatment Effect):

| Estimand | Question it answers | Target population |
|----------|---------------------|-------------------|
| **ATT** | "What was the effect of treatment on those who received it?" | The treated group |
| **ATE** | "What would be the effect if we treated everyone?" | Entire population |

Standard matching works by finding similar controls for each treated unit, keeping treated units as-is. This design answers the ATT question. To estimate ATE via matching, you would need bidirectional matching (match in both directions), which is more complex and less common.

**When ATT is appropriate** (most matching use cases):
- Treatment has barriers to participation (e.g., intensive intervention, surgery, voluntary program enrollment)
- You want to know if the treatment helped those who actually received it
- The treated population differs systematically from the general population

**When ATE might be preferred** (consider propensity score weighting instead):
- Treatment could realistically be given to everyone (e.g., low-cost intervention, information campaign)
- You want to generalize to the entire population

#### Weight Formula

**Treated units:** Weight = 1 (always)

**Control units:** Weight = sum of `1/k` for each match, where `k` is the number of controls matched to that treated unit.

Examples:
- 1:1 matching: Each matched control gets weight = 1
- 1:3 matching (without replacement): Each control gets weight = 1/3
- Matching with replacement: If control C is matched to T1 (with 3 controls) and T2 (with 2 controls), weight = 1/3 + 1/2 = 5/6

This approach uses the "stratum propensity score" concept where the proportion of treated units in a matched set determines the inverse probability weight for controls.

---

## Proposed API

```python
from brpmatch import generate_features, match, match_summary, match_data

# Step 1: Generate features
features_df = generate_features(
    spark, df,
    treatment_col="cohort",
    treatment_value="treated",
    categorical_cols=["race", "married", "nodegree"],
    numeric_cols=["age", "bmi"],
    date_cols=["diagnosis_date"],
    exact_match_cols=["gender"],
    id_col="person_id",
    max_categories=20,  # Optional: limit on distinct values per categorical (default: 20)
)
# Output columns:
# - person_id__id
# - treat__treat (0/1)
# - race_white__cat, race_black__cat, race_hispanic__cat
# - married_1__cat, married_0__cat
# - nodegree_1__cat, nodegree_0__cat
# - age__num, bmi__num
# - diagnosis_date__date
# - gender_male__exact, gender_female__exact
# - exact_match__group
# - features (assembled vector for LSH)

# Step 2: Match (no id_col parameter needed)
matched_df = match(
    features_df,
    feature_space="euclidean",
    n_neighbors=10,
    ratio_k=3,
    with_replacement=False
)

# Step 3: Summary (no feature_cols or id_col needed)
summary_df, fig = match_summary(
    features_df,
    matched_df,
    sample_frac=0.1,
    plot=True
)
# Auto-discovers features via suffixes
# Love plot shows: race_white, race_black, ..., age, bmi, ..., gender_male (exact)

# Step 4: Get weighted data for downstream analysis
result_df = match_data(
    df,           # original data
    matched_df,
    id_col="person_id"
)
# Returns original columns plus:
# - weights (for ATT estimation)
# - subclass (match group identifier)
```

---

## Summary of Changes

| Component | Current Behavior | New Behavior |
|-----------|-----------------|--------------|
| `generate_features()` | Index-encodes categoricals (`race_index`) | One-hot encodes with suffix (`race_white__cat`) |
| `generate_features()` | No systematic column naming | Suffix convention for all generated columns |
| `generate_features()` | No cardinality check | `max_categories` parameter (default: 20) |
| `match()` | Requires `id_col` parameter | Auto-discovers from `__id` suffix |
| `match_summary()` | Requires `feature_cols`, `id_col` | Auto-discovers from suffixes |
| `match_summary()` | Shows index columns on love plot | Shows one-hot columns with clean names |
| `match_summary()` | Doesn't show exact match columns | Shows exact match columns |
| `match_data()` | Does not exist | New function with proper ATT weights |

---

## Out of Scope

- Changes to the LSH bucketing algorithm
- Changes to the k-NN matching logic
- Changes to distance metric implementations
- Performance optimizations (beyond what's incidental to the refactor)
