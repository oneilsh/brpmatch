# High-Level Refactor Plan: Match Output Restructuring

## Problem Statement

The current `match()` function returns a single DataFrame with one row per (treated, control) pair. While this format captures the match structure, it has several usability issues:

### Current Output Format (pairs)
```
id,match_id,match_round,treated_k,control_usage_count,pair_weight,bucket_num_input_patients,bucket_seconds
NSW184,PSID5,1,3,4,0.083,41,0.005
NSW184,PSID6,2,3,5,0.067,41,0.005
NSW184,PSID27,3,3,5,0.067,41,0.005
```

### Issues

1. **Not analysis-ready**: Users must transform this to get per-unit weights for outcome analysis. The `match_data()` function exists for this, but requires passing the original DataFrame and manually specifying the ID column.

2. **Confusing strata assignment**: The `stratify_for_plot()` function creates strata identifiers like `"NSW184:PSID5"`, fragmenting each matched set into multiple strata. For 1:k matching, all k controls matched to a treated unit should share the same stratum.

3. **Poor naming**: `stratify_for_plot()` doesn't just prepare data for plotting - it's the key function for joining match info back to features for any downstream analysis.

4. **Bucket stats buried in pair rows**: Processing statistics (bucket size, timing) are repeated on every pair row rather than being available as a separate diagnostic output.

5. **Users need to call multiple functions**: Current workflow requires `match()` → `stratify_for_plot()` or `match_data()` → outcome analysis. The intermediate step is confusing.

## Solution: Return Three DataFrames as a Tuple

Change `match()` to return `Tuple[DataFrame, DataFrame, DataFrame]`:

```python
units, pairs, bucket_stats = match(features_df, ...)
```

### 1. Units DataFrame (Primary Output)

One row per patient (treated and control, matched and unmatched), ready for joining to outcome data:

```
id          subclass    weight    is_treated
NSW184      NSW184      1.0       True       # treated, matched
PSID5       NSW184      0.25      False      # control, matched to NSW184
PSID6       NSW184      0.20      False      # control, matched to NSW184
PSID27      NSW184      0.20      False      # control, matched to NSW184
PSID999     None        0.0       False      # control, unmatched
NSW999      None        0.0       True       # treated, unmatched
```

- **subclass**: The treated unit's ID (MatchIt convention). All units in a matched set share the same subclass.
- **weight**: ATT estimation weight. Treated = 1.0, Controls = sum(1/k) across matches, Unmatched = 0.0
- **is_treated**: Boolean treatment indicator

This is the main output users will work with for outcome analysis.

### 2. Pairs DataFrame (Secondary Output)

The current format, preserving detailed match structure for inspection/debugging:

```
id        match_id    match_round    treated_k    control_usage_count    pair_weight
NSW184    PSID5       1              3            4                      0.083
NSW184    PSID6       2              3            5                      0.067
NSW184    PSID27      3              3            5                      0.067
```

Useful for:
- Inspecting specific matches manually
- Debugging matching issues
- Understanding control reuse patterns (with replacement)
- Computing custom weights

### 3. Bucket Stats DataFrame (Diagnostic Output)

One row per LSH bucket with processing statistics (computed during the bucketing phase):

```
bucket_id    num_patients    num_treated    num_control    num_matches    seconds
0            41              8              33             24             0.005
1            188             52             136            156            0.014
```

Columns:
- **bucket_id**: LSH bucket identifier
- **num_patients**: Total patients in bucket (num_treated + num_control)
- **num_treated**: Treated patients in bucket
- **num_control**: Control patients in bucket
- **num_matches**: Match pairs produced from this bucket
- **seconds**: Processing time for this bucket

Useful for:
- Performance tuning (identifying slow buckets)
- Diagnosing coverage issues (buckets with few treated or controls)
- Understanding the LSH bucketing behavior
- Verifying treated/control distribution across buckets

## Key Design Decisions

### Decision 1: Include all units (matched + unmatched)

The `units` DataFrame includes all patients from the input `features_df`, not just matched ones. Unmatched units have `subclass=None` and `weight=0.0`.

**Rationale**:
- Simplifies joining back to original data
- Enables analyses that compare matched vs unmatched
- Follows MatchIt's `match.data()` convention

### Decision 2: Use treated ID as subclass

For 1:k matching, the subclass is the treated patient's ID. All k controls matched to that treated unit share the same subclass.

**Rationale**:
- Follows MatchIt convention
- Simple and intuitive
- Enables clustering standard errors on subclass
- Works correctly for both 1:1 and 1:k matching

### Decision 3: Remove `stratify_for_plot()` from public API

Since `match()` now returns the `units` DataFrame directly, `stratify_for_plot()` is no longer needed as a public function.

**Rationale**:
- Eliminates confusing intermediate step
- Reduces API surface area
- The functionality is now built into `match()`

### Decision 4: Keep `match_data()` but simplify it

`match_data()` becomes a convenience function for joining `units` back to the original (pre-features) DataFrame.

**Rationale**:
- Users may want to analyze with their original column names
- Still useful for joining match info to outcome data

### Decision 5: Keep `match_summary()` unchanged (mostly)

`match_summary()` continues to provide balance statistics. It will need minor updates to work with the new `units` DataFrame format instead of calling `stratify_for_plot()`.

**Rationale**:
- Summary/diagnostic functionality is separate from the match output format
- Users still need balance statistics and love plots

## Implications

### What Changes

1. **`match()` return type**: `DataFrame` → `Tuple[DataFrame, DataFrame, DataFrame]`
2. **`stratify_for_plot()`**: Remove from public API (keep internally if needed)
3. **`match_summary()`**: Update to use `units` DataFrame directly
4. **`match_data()`**: Simplify - now just joins `units` to original data
5. **All examples**: Update to unpack tuple return
6. **All tests**: Update assertions for new return type

### What Stays the Same

1. **`generate_features()`**: No changes
2. **`love_plot()`**: No changes (works on stratified data)
3. **Matching algorithm**: No changes to core matching logic
4. **`match_summary()` output format**: Same balance statistics

## Migration Path

The return type change is breaking. Users will need to update their code:

```python
# Before
matched = match(features_df, ...)
stratified = stratify_for_plot(features_df, matched)

# After
units, pairs, bucket_stats = match(features_df, ...)
# units is ready to use directly - no stratify_for_plot() needed
```

For users who only want the pairs (backward compatibility concern), they can simply ignore the other outputs:

```python
_, pairs, _ = match(features_df, ...)
```
