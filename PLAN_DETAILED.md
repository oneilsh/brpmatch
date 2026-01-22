# BRPMatch API Redesign - Detailed Implementation Plan

This document provides implementation details sufficient for another agent to execute the changes.

---

## Overview of Files to Modify

| File | Changes |
|------|---------|
| `brpmatch/features.py` | Major rewrite of categorical encoding, column naming |
| `brpmatch/matching.py` | Add column auto-discovery from suffixes |
| `brpmatch/summary.py` | Add column auto-discovery, suffix stripping for display |
| `brpmatch/loveplot.py` | Strip suffixes for display, handle exact match columns |
| `brpmatch/stratify.py` | Update column name references |
| `brpmatch/__init__.py` | Export new `match_data()` function |
| `example/example.py` | Update to use new API |
| `tests/` | Update tests for new column naming |

---

## 1. features.py - generate_features()

**Location:** `brpmatch/features.py`, lines 20-213

### 1.1 Add Helper Function for Value Sanitization

Add at top of file (after imports, before `generate_features`):

```python
def _sanitize_value(value: str) -> str:
    """
    Sanitize a categorical value for use in column names.

    - Lowercase
    - Replace spaces and special chars with underscore
    - Collapse multiple underscores
    - Strip leading/trailing underscores
    """
    import re
    result = str(value).lower()
    # Replace spaces and problematic characters with underscore
    result = re.sub(r'[\s/\\.\-\(\)\[\]\{\}:;,\'\"]+', '_', result)
    # Collapse multiple underscores
    result = re.sub(r'_+', '_', result)
    # Strip leading/trailing underscores
    result = result.strip('_')
    return result
```

### 1.2 Replace StringIndexer/OneHotEncoder with Manual One-Hot Encoding

**Current approach (lines 163-173):**
```python
for c in categorical_cols:
    preprocessing_stages += [
        StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
    ]
    preprocessing_stages += [
        OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_onehot", dropLast=True)
    ]
    categorical_index_cols.append(f"{c}_index")
```

**New approach:** Create explicit one-hot columns with proper naming, with cardinality check.

```python
# Default maximum categories (can be overridden via parameter)
DEFAULT_MAX_CATEGORIES = 20


def _create_onehot_columns(
    df: DataFrame,
    col: str,
    suffix: str = "__cat",
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> Tuple[DataFrame, List[str]]:
    """
    Create one-hot encoded columns for a categorical column.

    Args:
        df: Input DataFrame
        col: Categorical column name
        suffix: Suffix for generated columns (__cat or __exact)
        max_categories: Maximum allowed distinct values (raises error if exceeded)

    Returns:
        Tuple of (DataFrame with new columns, list of new column names)

    Raises:
        ValueError: If the column has more distinct values than max_categories
    """
    # Get distinct values (sorted for deterministic ordering)
    distinct_values = [
        row[col] for row in df.select(col).distinct().collect()
        if row[col] is not None
    ]
    distinct_values = sorted([str(v) for v in distinct_values])

    # Check cardinality
    if len(distinct_values) > max_categories:
        raise ValueError(
            f"Column '{col}' has {len(distinct_values)} distinct values, "
            f"which exceeds max_categories={max_categories}. "
            f"High-cardinality categorical columns are not suitable for matching. "
            f"Consider binning the values, using it as a numeric column, "
            f"or increasing max_categories if you're sure this is intended."
        )

    new_cols = []
    for val in distinct_values:
        sanitized = _sanitize_value(val)
        new_col_name = f"{col}_{sanitized}{suffix}"
        df = df.withColumn(
            new_col_name,
            F.when(F.col(col) == val, 1.0).otherwise(0.0)
        )
        new_cols.append(new_col_name)

    return df, new_cols
```

### 1.2a Update generate_features() Signature

Add `max_categories` parameter:

```python
def generate_features(
    spark: SparkSession,
    df: DataFrame,
    treatment_col: str,
    treatment_value: str,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    exact_match_cols: Optional[List[str]] = None,
    date_reference: str = "1970-01-01",
    id_col: str = "person_id",
    max_categories: int = 20,  # NEW PARAMETER
    verbose: bool = True,
) -> DataFrame:
```

### 1.3 Update Column Naming Throughout

**Changes to make:**

1. **ID column** (around line 130):
   ```python
   # Current:
   # df uses id_col as-is

   # New:
   id_col_internal = f"{id_col}__id"
   df = df.withColumn(id_col_internal, F.col(id_col).cast("string"))
   ```

2. **Treatment column** (around line 140):
   ```python
   # Current:
   df = df.withColumn("treat", F.when(F.col(treatment_col) == treatment_value, 1).otherwise(0))

   # New:
   df = df.withColumn("treat__treat", F.when(F.col(treatment_col) == treatment_value, 1).otherwise(0))
   ```

3. **Categorical columns** (replace lines 163-173):
   ```python
   categorical_feature_cols = []
   for c in categorical_cols:
       # Skip if this is an exact match column (handled separately)
       if exact_match_cols and c in exact_match_cols:
           continue
       df, new_cols = _create_onehot_columns(df, c, suffix="__cat", max_categories=max_categories)
       categorical_feature_cols.extend(new_cols)
   ```

4. **Exact match columns** (new section):
   ```python
   exact_match_feature_cols = []
   if exact_match_cols:
       for c in exact_match_cols:
           df, new_cols = _create_onehot_columns(df, c, suffix="__exact", max_categories=max_categories)
           exact_match_feature_cols.extend(new_cols)

       # Create composite exact match grouping column
       df = df.withColumn(
           "exact_match__group",
           F.concat_ws("_", *[F.col(c) for c in exact_match_cols])
       )
   else:
       df = df.withColumn("exact_match__group", F.lit("all"))
   ```

5. **Numeric columns** (around line 150):
   ```python
   # Current: numeric columns used as-is

   # New: rename with __num suffix
   numeric_feature_cols = []
   for c in numeric_cols:
       new_col = f"{c}__num"
       df = df.withColumn(new_col, F.col(c).cast("double"))
       numeric_feature_cols.append(new_col)
   ```

6. **Date columns** (around line 155):
   ```python
   # Current: date_col_days_from_2018

   # New: date_col__date
   date_feature_cols = []
   for c in date_cols:
       new_col = f"{c}__date"
       df = df.withColumn(
           new_col,
           F.datediff(F.col(c), F.lit(date_reference)).cast("double")
       )
       date_feature_cols.append(new_col)
   ```

### 1.4 Update Feature Vector Assembly

**Current (lines 176-178):**
```python
assembler_cols = [f"{c}_onehot" for c in categorical_cols] + numeric_cols
```

**New:**
```python
assembler_cols = (
    categorical_feature_cols +    # race_white__cat, race_black__cat, ...
    exact_match_feature_cols +    # gender_male__exact, gender_female__exact, ...
    numeric_feature_cols +        # age__num, bmi__num, ...
    date_feature_cols             # diagnosis_date__date, ...
)
```

### 1.5 Update Output Column Selection

**Current (lines 200-211):**
```python
df.select(
    categorical_cols + numeric_cols + date_cols + exact_match_cols +
    categorical_index_cols + ["exact_match_id"] + ["features"] +
    ["treat"] + [treatment_col] + [id_col]
)
```

**New:**
```python
output_cols = (
    [id_col_internal] +           # person_id__id
    ["treat__treat"] +            # treatment indicator
    categorical_feature_cols +    # race_white__cat, ...
    exact_match_feature_cols +    # gender_male__exact, ...
    numeric_feature_cols +        # age__num, ...
    date_feature_cols +           # diagnosis_date__date
    ["exact_match__group"] +      # composite grouping
    ["features"]                  # assembled feature vector
)
df = df.select(output_cols)
```

### 1.6 Update Function Docstring

Update the docstring to document the new column naming convention:

```python
"""
Generate feature vectors for matching.

Output DataFrame columns:
- {id_col}__id: Patient identifier
- treat__treat: Treatment indicator (0/1)
- {cat_col}_{value}__cat: One-hot encoded categorical features
- {exact_col}_{value}__exact: One-hot encoded exact match features
- {num_col}__num: Numeric features
- {date_col}__date: Date features (days from reference)
- exact_match__group: Composite exact match grouping ID
- features: Assembled feature vector for LSH
"""
```

---

## 2. matching.py - match()

**Location:** `brpmatch/matching.py`, lines 31-664

### 2.1 Add Column Discovery Helper Functions

Add at top of file (after imports):

```python
def _discover_id_column(df: DataFrame) -> str:
    """Find the ID column by looking for __id suffix."""
    id_cols = [c for c in df.columns if c.endswith("__id")]
    if len(id_cols) != 1:
        raise ValueError(
            f"Expected exactly one column ending with '__id', found: {id_cols}"
        )
    return id_cols[0]


def _discover_treatment_column(df: DataFrame) -> str:
    """Find the treatment column by looking for __treat suffix."""
    treat_cols = [c for c in df.columns if c.endswith("__treat")]
    if len(treat_cols) != 1:
        raise ValueError(
            f"Expected exactly one column ending with '__treat', found: {treat_cols}"
        )
    return treat_cols[0]


def _discover_exact_match_column(df: DataFrame) -> str:
    """Find the exact match grouping column by looking for __group suffix."""
    group_cols = [c for c in df.columns if c.endswith("__group")]
    if len(group_cols) != 1:
        raise ValueError(
            f"Expected exactly one column ending with '__group', found: {group_cols}"
        )
    return group_cols[0]
```

### 2.2 Update Function Signature

**Current:**
```python
def match(
    features_df: DataFrame,
    feature_space: Literal["euclidean", "mahalanobis"] = "euclidean",
    n_neighbors: int = 5,
    bucket_length: Optional[float] = None,
    num_hash_tables: int = 4,
    num_patients_trigger_rebucket: int = 10000,
    features_col: str = "features",
    treatment_col: str = "treat",
    id_col: str = "person_id",
    exact_match_col: str = "exact_match_id",
    ...
) -> DataFrame:
```

**New:**
```python
def match(
    features_df: DataFrame,
    feature_space: Literal["euclidean", "mahalanobis"] = "euclidean",
    n_neighbors: int = 5,
    bucket_length: Optional[float] = None,
    num_hash_tables: int = 4,
    num_patients_trigger_rebucket: int = 10000,
    features_col: str = "features",
    # Remove these parameters - auto-discovered:
    # treatment_col: str = "treat",
    # id_col: str = "person_id",
    # exact_match_col: str = "exact_match_id",
    ...
) -> DataFrame:
```

### 2.3 Update Column Discovery in Function Body

**At the start of match() function (after signature):**

```python
# Auto-discover columns from naming convention
id_col = _discover_id_column(features_df)
treatment_col = _discover_treatment_column(features_df)
exact_match_col = _discover_exact_match_column(features_df)

# Extract base ID name for output columns (e.g., "person_id" from "person_id__id")
id_col_base = id_col.replace("__id", "")
```

### 2.4 Update Output Column Naming

**Current output schema (lines 407-418):**
```python
StructField(id_col, StringType()),
StructField("match_" + id_col, StringType()),
```

**New output schema:**
```python
# Use base name without __id suffix for cleaner output
StructField(id_col_base, StringType()),           # "person_id" not "person_id__id"
StructField(f"match_{id_col_base}", StringType()), # "match_person_id"
```

### 2.5 Update References Throughout

Search and replace throughout the function:
- `F.col(id_col)` stays the same (it now refers to discovered column)
- `F.col(treatment_col)` stays the same
- `F.col(exact_match_col)` stays the same
- Output columns use `id_col_base` instead of `id_col`

---

## 3. summary.py - match_summary()

**Location:** `brpmatch/summary.py`, lines 21-124

### 3.1 Add Column Discovery Helper Functions

Add at top of file:

```python
def _discover_feature_columns(df: DataFrame) -> List[str]:
    """Find all feature columns by suffix pattern."""
    suffixes = ("__cat", "__num", "__date", "__exact")
    return [c for c in df.columns if any(c.endswith(s) for s in suffixes)]


def _discover_id_column(df: DataFrame) -> str:
    """Find the ID column by looking for __id suffix."""
    id_cols = [c for c in df.columns if c.endswith("__id")]
    if len(id_cols) != 1:
        raise ValueError(
            f"Expected exactly one column ending with '__id', found: {id_cols}"
        )
    return id_cols[0]


def _strip_suffix(col_name: str) -> str:
    """Strip the __cat, __num, __date, __exact suffix for display."""
    for suffix in ("__cat", "__num", "__date", "__exact"):
        if col_name.endswith(suffix):
            return col_name[:-len(suffix)]
    return col_name


def _get_column_type(col_name: str) -> str:
    """Get the column type from its suffix."""
    if col_name.endswith("__exact"):
        return "exact"
    elif col_name.endswith("__cat"):
        return "categorical"
    elif col_name.endswith("__num"):
        return "numeric"
    elif col_name.endswith("__date"):
        return "date"
    return "unknown"
```

### 3.2 Update Function Signature

**Current:**
```python
def match_summary(
    features_df: DataFrame,
    matched_df: DataFrame,
    feature_cols: List[str],
    treatment_col: str = "treat",
    id_col: str = "person_id",
    match_id_col: Optional[str] = None,
    sample_frac: float = 0.05,
    ...
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, matplotlib.figure.Figure]]:
```

**New:**
```python
def match_summary(
    features_df: DataFrame,
    matched_df: DataFrame,
    # Remove feature_cols - auto-discovered
    # Remove treatment_col - auto-discovered
    # Remove id_col - auto-discovered
    sample_frac: float = 0.05,
    threshold_smd: float = 0.1,
    threshold_vr: Tuple[float, float] = (0.5, 2.0),
    plot: bool = False,
    figsize: Tuple[int, int] = (10, 12),
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, matplotlib.figure.Figure]]:
```

### 3.3 Update Function Body

**At the start of function:**

```python
# Auto-discover columns
feature_cols = _discover_feature_columns(features_df)
id_col = _discover_id_column(features_df)
treatment_col = "treat__treat"  # Standardized name

# Derive match_id_col from id_col
id_col_base = id_col.replace("__id", "")
match_id_col = f"match_{id_col_base}"
```

### 3.4 Update Balance DataFrame Output

Add display names to the balance DataFrame:

```python
# After computing balance_df
balance_df["display_name"] = balance_df["feature"].apply(_strip_suffix)
balance_df["feature_type"] = balance_df["feature"].apply(_get_column_type)
```

### 3.5 Update _print_balance_table()

**Current (lines 233-258):** Prints feature names as-is.

**New:** Strip suffixes for display:

```python
def _print_balance_table(balance_df: pd.DataFrame) -> None:
    """Print formatted balance table with clean display names."""
    print("\nBalance Summary:")
    print("-" * 80)

    for _, row in balance_df.iterrows():
        display_name = _strip_suffix(row["feature"])
        feature_type = _get_column_type(row["feature"])

        # Add indicator for exact match columns
        type_indicator = " (exact)" if feature_type == "exact" else ""

        print(f"  {display_name}{type_indicator}:")
        print(f"    SMD: {row['smd_unadjusted']:.3f} -> {row['smd_adjusted']:.3f}")
        if pd.notna(row.get('vr_unadjusted')):
            print(f"    VR:  {row['vr_unadjusted']:.3f} -> {row['vr_adjusted']:.3f}")
```

---

## 4. loveplot.py - love_plot()

**Location:** `brpmatch/loveplot.py`, lines 16-135

### 4.1 Add Helper Functions

Add the same suffix-stripping helpers (or import from summary.py):

```python
def _strip_suffix(col_name: str) -> str:
    """Strip the __cat, __num, __date, __exact suffix for display."""
    for suffix in ("__cat", "__num", "__date", "__exact"):
        if col_name.endswith(suffix):
            return col_name[:-len(suffix)]
    return col_name


def _is_exact_match_column(col_name: str) -> bool:
    """Check if column is an exact match column."""
    return col_name.endswith("__exact")
```

### 4.2 Update Function Signature

**Current:**
```python
def love_plot(
    stratified_df: DataFrame,
    feature_cols: List[str],
    treatment_col: str = "treat",
    strata_col: str = "strata",
    sample_frac: float = 0.05,
    figsize: Tuple[int, int] = (10, 12),
) -> matplotlib.figure.Figure:
```

**New:**
```python
def love_plot(
    stratified_df: DataFrame,
    # Remove feature_cols - auto-discovered
    # Remove treatment_col - standardized
    strata_col: str = "strata",
    sample_frac: float = 0.05,
    figsize: Tuple[int, int] = (10, 12),
) -> matplotlib.figure.Figure:
```

### 4.3 Update Function Body

**At start of function:**

```python
# Auto-discover feature columns
suffixes = ("__cat", "__num", "__date", "__exact")
feature_cols = [c for c in stratified_df.columns if any(c.endswith(s) for s in suffixes)]
treatment_col = "treat__treat"
```

### 4.4 Update Plot Labels

**In the plotting section (around lines 95-133):**

```python
# Create display names for y-axis labels
display_names = []
for col in feature_cols:
    name = _strip_suffix(col)
    if _is_exact_match_column(col):
        name = f"{name} (exact)"
    display_names.append(name)

# Use display_names for y-axis tick labels
ax1.set_yticklabels(display_names)
ax2.set_yticklabels(display_names)
```

### 4.5 Optional: Visual Distinction for Exact Match Columns

Add different marker style for exact match columns:

```python
# When plotting points
for i, col in enumerate(feature_cols):
    marker = 's' if _is_exact_match_column(col) else 'o'
    # ... plot with marker=marker
```

---

## 5. stratify.py - stratify_for_plot()

**Location:** `brpmatch/stratify.py`, lines 12-76

### 5.1 Update Function Signature

**Current:**
```python
def stratify_for_plot(
    features_df: DataFrame,
    matched_df: DataFrame,
    id_col: str = "person_id",
    match_id_col: str = "match_person_id",
) -> DataFrame:
```

**New:**
```python
def stratify_for_plot(
    features_df: DataFrame,
    matched_df: DataFrame,
    # Remove id_col and match_id_col - auto-discovered
) -> DataFrame:
```

### 5.2 Update Function Body

**At start of function:**

```python
# Auto-discover ID column
id_cols = [c for c in features_df.columns if c.endswith("__id")]
if len(id_cols) != 1:
    raise ValueError(f"Expected one __id column, found: {id_cols}")
id_col = id_cols[0]

# Derive base name and match column name
id_col_base = id_col.replace("__id", "")
match_id_col = f"match_{id_col_base}"
```

---

## 6. New Function: match_data()

**Location:** Add to `brpmatch/summary.py` (or create new file `brpmatch/data.py`)

### 6.1 ATT Weight Calculation Background

The weight calculation follows the methodology for **Average Treatment Effect on the Treated (ATT)**
as described in:
- [Greifer: Matching Weights are Propensity Score Weights](https://ngreifer.github.io/blog/matching-weights/)
- [MatchIt Documentation](https://kosukeimai.github.io/MatchIt/reference/matchit.html)

#### Why ATT (not ATE)?

Matching methods inherently estimate ATT, not ATE (Average Treatment Effect). This is because
standard matching keeps treated units as-is (weight = 1) and finds similar controls to match them.
This design answers the question: "What would have happened to the treated group if they hadn't
been treated?"

To estimate ATE via matching, you would need bidirectional matching (controls to treated AND
treated to controls), which is more complex. For ATE estimation, propensity score weighting
(IPTW) is typically preferred over matching.

**ATT is appropriate when:**
- Treatment has barriers to participation (surgery, intensive programs, etc.)
- You want to know if treatment helped those who actually received it
- The treated population differs systematically from the general population

**Key formula:**
- Treated units: weight = 1 (always)
- Control units: weight = sum of `1/k` for each match, where `k` = number of controls matched to that treated

This uses the "stratum propensity score" concept: in a matched set with 1 treated and k controls,
the stratum propensity score is `1/(k+1)`, giving an ATT weight of `(1/(k+1)) / (k/(k+1)) = 1/k`.

**Examples:**
- 1:1 matching: control weight = 1/1 = 1
- 1:3 matching: each control weight = 1/3
- With replacement (control matched to T1 with 3 controls, T2 with 2 controls): weight = 1/3 + 1/2 = 5/6

### 6.2 Implementation

```python
def match_data(
    original_df: DataFrame,
    matched_df: DataFrame,
    id_col: str,
) -> DataFrame:
    """
    Create a matched dataset with weights for downstream analysis.

    Analogous to R matchit's match.data() function. Computes ATT (Average Treatment
    Effect on the Treated) weights following the methodology in Greifer (2021) and
    the MatchIt package.

    Note: These weights estimate ATT, not ATE (Average Treatment Effect). This is
    inherent to matching: we keep treated units as-is and find similar controls,
    answering "what would have happened to the treated if they hadn't been treated?"
    ATT is appropriate when treatment has barriers to participation or when you want
    to evaluate the effect on those who actually received treatment. For ATE
    estimation, consider propensity score weighting (IPTW) instead of matching.

    Weight calculation:
    - Treated units: weight = 1
    - Control units: weight = sum of 1/k for each match, where k is the number
      of controls matched to that treated unit (treated_k in matched_df)

    This implements the "stratum propensity score" weighting approach where
    matched sets define strata, and inverse probability weights are computed
    based on the proportion of treated units in each stratum.

    References:
    - https://ngreifer.github.io/blog/matching-weights/
    - https://kosukeimai.github.io/MatchIt/reference/matchit.html

    Args:
        original_df: Original input DataFrame (before generate_features)
        matched_df: Output from match()
        id_col: Name of the ID column in original_df

    Returns:
        DataFrame with original columns plus:
        - weights: ATT estimation weights (see formula above; 0 for unmatched)
        - subclass: Match group identifier (treated_id for matched pairs)
        - matched: Boolean indicating if row was matched

    Example:
        >>> result = match_data(df, matched_df, id_col="person_id")
        >>> # Use for weighted regression:
        >>> result.filter(F.col("matched")).select("outcome", "treatment", "weights", ...)
    """
    # Get the match ID column name from matched_df
    match_id_col = None
    for c in matched_df.columns:
        if c.startswith("match_") and c != "match_round":
            match_id_col = c
            break

    if match_id_col is None:
        raise ValueError("Could not find match ID column in matched_df")

    # Extract base ID column name (e.g., "person_id" from "match_person_id")
    id_col_base = match_id_col.replace("match_", "")

    # Create subclass (match group) identifier
    # Each treated patient defines a subclass
    matched_with_subclass = matched_df.withColumn(
        "subclass", F.col(id_col_base)
    )

    # ----- Treated patient weights -----
    # Treated units always receive weight = 1
    treated_weights = matched_with_subclass.select(
        F.col(id_col_base).alias("_join_id"),
        F.lit(1.0).alias("weights"),
        F.col("subclass"),
        F.lit(True).alias("matched")
    ).distinct()

    # ----- Control patient weights -----
    # Weight = sum of 1/treated_k for each match this control appears in
    #
    # For each row in matched_df:
    #   - treated_k = number of controls matched to that treated unit
    #   - This control's contribution from this match = 1/treated_k
    #
    # If matching with replacement, a control may appear in multiple rows,
    # so we sum across all matches.
    #
    # Example: control C matched to T1 (treated_k=3) and T2 (treated_k=2)
    #   weight = 1/3 + 1/2 = 5/6
    control_weights = matched_with_subclass.withColumn(
        "_weight_contribution", 1.0 / F.col("treated_k")
    ).groupBy(match_id_col).agg(
        F.sum("_weight_contribution").alias("weights"),
        F.first("subclass").alias("subclass"),  # Use first subclass if matched to multiple treated
        F.lit(True).alias("matched")
    ).withColumnRenamed(match_id_col, "_join_id")

    # Combine treated and control weights
    all_weights = treated_weights.unionByName(control_weights)

    # Join with original data
    result = original_df.join(
        all_weights,
        original_df[id_col] == all_weights["_join_id"],
        "left"
    ).drop("_join_id")

    # Fill unmatched rows
    result = result.withColumn(
        "weights", F.coalesce(F.col("weights"), F.lit(0.0))
    ).withColumn(
        "matched", F.coalesce(F.col("matched"), F.lit(False))
    ).withColumn(
        "subclass", F.coalesce(F.col("subclass"), F.lit(None))
    )

    return result
```

---

## 7. __init__.py - Update Exports

**Location:** `brpmatch/__init__.py`

**Current:**
```python
from brpmatch.features import generate_features
from brpmatch.matching import match
from brpmatch.summary import match_summary
from brpmatch.loveplot import love_plot
from brpmatch.stratify import stratify_for_plot
```

**New:**
```python
from brpmatch.features import generate_features
from brpmatch.matching import match
from brpmatch.summary import match_summary, match_data
from brpmatch.loveplot import love_plot
from brpmatch.stratify import stratify_for_plot

__all__ = [
    "generate_features",
    "match",
    "match_summary",
    "match_data",
    "love_plot",
    "stratify_for_plot",
]
```

---

## 8. example/example.py - Update Example

**Key changes:**

1. Remove explicit `feature_cols` list (lines 226, etc.)
2. Remove `id_col` parameter from `match_summary()` calls
3. Add demonstration of `match_data()`

**Updated example excerpt:**

```python
# Old:
feature_cols = ["race_index", "married_index", "nodegree_index", "age", "educ", "re74", "re75"]
summary_euc, fig_euc = match_summary(
    features_df,
    matched_1to1_euc,
    feature_cols,
    id_col="id",
    sample_frac=1.0,
    plot=True,
)

# New:
summary_euc, fig_euc = match_summary(
    features_df,
    matched_1to1_euc,
    sample_frac=1.0,
    plot=True,
)

# Demonstrate match_data()
print("\nStep 10: Creating Matched Dataset for Analysis")
matched_data_df = match_data(df, matched_1to1_euc, id_col="id")
print(f"Matched dataset has {matched_data_df.count()} rows")
print(f"Matched rows: {matched_data_df.filter(F.col('matched')).count()}")
matched_data_df.select("id", "treat", "weights", "subclass", "matched").show(10)
```

---

## 9. Tests - Update for New API

**Files to update:**
- `tests/test_features.py` - Update expected column names
- `tests/test_matching.py` - Remove explicit column parameters
- `tests/test_summary.py` - Remove feature_cols parameter
- `tests/test_integration.py` - Full pipeline test

**Key test assertions to update:**

```python
# Old assertion:
assert "race_index" in features_df.columns

# New assertion:
assert any(c.endswith("__cat") and c.startswith("race_") for c in features_df.columns)

# Or more specifically:
assert "race_white__cat" in features_df.columns
assert "race_black__cat" in features_df.columns
```

---

## 10. README.md - Update Documentation

Update all code examples to reflect new API:

1. Remove `feature_cols` from `love_plot()` and `match_summary()` examples
2. Update column name references (e.g., `state_index` → `state_ca__cat`)
3. Add `match_data()` documentation section
4. Document the new naming convention

---

## Implementation Order

Recommended order to minimize breaking changes during development:

1. **features.py** - Core change, everything depends on this
2. **matching.py** - Update column discovery
3. **stratify.py** - Update column discovery
4. **summary.py** - Update column discovery, add `match_data()`
5. **loveplot.py** - Update display names
6. **__init__.py** - Export new function
7. **tests/** - Update all tests
8. **example/example.py** - Update example
9. **README.md** - Update documentation

---

## Backward Compatibility Notes

This is a **breaking change** to the API. Users will need to:

1. Remove `feature_cols` parameter from `match_summary()` and `love_plot()` calls
2. Remove `id_col` parameter from `match()` and `match_summary()` calls
3. Update any code that references old column names (e.g., `race_index` → `race_white__cat`)

Consider adding a deprecation warning period if needed, but given this is pre-1.0, a clean break is likely acceptable.
