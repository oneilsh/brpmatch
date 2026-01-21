# High-Level Plan: 1-to-k Matching Implementation

## Overview

Extend the current 1-to-1 nearest neighbor matching in `brpmatch/matching.py` to support 1-to-k matching with configurable replacement options, inspired by R's MatchIt package.

## Current Algorithm Summary

The existing `match()` function implements 1-to-1 nearest neighbor matching:

1. **Feature preprocessing**: Optional Mahalanobis whitening transform, L2 normalization
2. **Multi-level LSH bucketing**: 4 bucket levels with progressively finer granularity; each patient assigned to finest level below size threshold
3. **Within-bucket matching** (`find_neighbors` pandas UDF):
   - sklearn `NearestNeighbors` finds k-nearest controls for each treated
   - Bidirectional ranking (treated→control and control→treated)
   - Greedy 1-to-1 matching: iteratively select best match, remove both patients from candidate pool
4. **Output**: Matched pairs with bucket metadata

## New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ratio_k` | `int` | `1` | Number of controls to match per treated patient (k in k:1 matching) |
| `with_replacement` | `bool` | `False` | Whether controls can be reused across different treated patients |
| `reuse_max` | `Optional[int]` | `None` | Maximum times a control can be reused (only applies when `with_replacement=True`). `None` means unlimited. |
| `require_k` | `bool` | `True` | If `True`, treated patients who cannot get exactly k matches are excluded. If `False`, treated patients may receive fewer than k matches. |

## Algorithm Design

### Without Replacement (`with_replacement=False`)

**Round-robin approach** ensures fairness—all treated patients get their 1st match before any get their 2nd:

```
Round 1: Each treated patient gets their best available match
         → Remove matched controls from pool
Round 2: Each treated patient gets their best available match from remaining
         → Remove matched controls from pool
...
Round k: Each treated patient gets their kth match
```

**Rationale**: A naive "greedy per-treated" approach (each treated grabs k controls, then next treated picks from remainder) would allow early patients to hoard the best controls, leaving later patients with poor matches or none.

**Implementation** (within `find_neighbors`):
```python
all_matches = []
available_controls = set(person_ids_match_to)

for round_num in range(1, ratio_k + 1):
    if not available_controls:
        break

    # Filter candidates to available controls only
    round_candidates = match_candidates[
        match_candidates["match_" + id_col].isin(available_controls)
    ]

    # Re-rank within available controls
    # Run standard 1-to-1 greedy matching
    round_matches = greedy_1to1_match(round_candidates)

    # Record round number and accumulate
    round_matches["match_round"] = round_num
    all_matches.append(round_matches)

    # Remove matched controls from pool
    available_controls -= set(round_matches["match_" + id_col])
```

### With Replacement (`with_replacement=True`)

**Independent matching**—each treated patient gets their k nearest controls regardless of other treated patients:

```python
all_matches = []
control_usage = defaultdict(int)  # Track usage for reuse_max

for treated_id in person_ids_need_matching:
    matches_for_treated = []

    for candidate in sorted_candidates_for_treated:
        control_id = candidate["match_" + id_col]

        # Check reuse_max constraint
        if reuse_max is not None and control_usage[control_id] >= reuse_max:
            continue

        matches_for_treated.append(candidate)
        control_usage[control_id] += 1

        if len(matches_for_treated) == ratio_k:
            break

    all_matches.extend(matches_for_treated)
```

**Rationale**: With replacement, there's no scarcity problem—each treated patient can always access their nearest controls. This maximizes balance (match quality) but reduces effective sample size.

### Handling `require_k=False`

When `require_k=False`, treated patients may receive fewer than k matches if:
- Without replacement: control pool exhausted before k rounds complete
- With replacement + `reuse_max`: all nearby controls hit their reuse limit

The output will include `treated_k` column showing actual matches per treated.

When `require_k=True` (default), treated patients with fewer than k matches are excluded from the final output.

## Output Schema Changes

### New Columns

| Column | Type | Description |
|--------|------|-------------|
| `match_round` | `Integer` | Which round this match came from (1 = best match, 2 = second best, etc.) |
| `treated_k` | `Integer` | Total number of controls matched to this treated patient (may be < ratio_k if `require_k=False`) |
| `control_usage_count` | `Integer` | Number of times this control was matched globally (always 1 if `with_replacement=False`) |
| `pair_weight` | `Double` | Precomputed weight: `1.0 / (treated_k × control_usage_count)` |

### Weighting Rationale

For ATT (Average Treatment Effect on the Treated) estimation:

1. **k:1 adjustment**: When a treated patient has k controls, each control contributes 1/k to that patient's effect estimate
2. **Replacement adjustment**: A control matched to multiple treated patients shouldn't be over-counted; weight by 1/(times used)

Combined: `pair_weight = 1 / (treated_k × control_usage_count)`

**Example**: Control C is matched to:
- Treated A (who has 2 total matches): pair_weight = 1/(2×2) = 0.25
- Treated B (who has 3 total matches): pair_weight = 1/(3×2) = 0.167

Users can apply these weights in downstream analyses or ignore them if using other weighting schemes.

### Implementation Note

Since each patient is assigned to exactly one bucket (lines 280-327 in current implementation), both `control_usage_count` and `treated_k` are **bucket-local** properties. They can be computed directly within the `find_neighbors` pandas UDF:

```python
# Inside find_neighbors, after building all_matches list
result_df = pd.DataFrame(all_matches)

# Both are counts within this bucket's matches
result_df["treated_k"] = result_df.groupby(id_col)[id_col].transform("count")
result_df["control_usage_count"] = result_df.groupby("match_" + id_col)["match_" + id_col].transform("count")
result_df["pair_weight"] = 1.0 / (result_df["treated_k"] * result_df["control_usage_count"])
```

No post-processing step required.

## Parameter Interactions and Validation

| `with_replacement` | `reuse_max` | Behavior |
|--------------------|-------------|----------|
| `False` | any | `reuse_max` ignored (always 1 by definition) |
| `True` | `None` | Unlimited reuse |
| `True` | `n` | Each control used at most n times |

**Validation rules**:
- `ratio_k >= 1`
- `reuse_max >= 1` if provided
- Warn if `ratio_k > n_neighbors` (not enough candidates)

## Summary Output Updates

Current output (line 498):
```
BRPMatch: 1:1 nearest neighbor matching via LSH
```

New output:
```
BRPMatch: 1:k nearest neighbor matching via LSH
 - ratio: 1:3 (with replacement, reuse_max=5)
 - ...
 - matched: 1500 pairs across 500 treated (mean 3.0 controls/treated)
 - control pool: 2000 unique controls used (of 5000 available)
```

## Tradeoffs to Document

| Setting | Balance Quality | Precision | Complexity | Use Case |
|---------|-----------------|-----------|------------|----------|
| k=1, no replacement | Baseline | Baseline | Simple | Standard matching |
| k>1, no replacement | May degrade in later rounds | Improved | Moderate | Limited control pool |
| k>1, with replacement | Best (always nearest) | Reduced (effective n↓) | Simple | Large control pool, prioritize balance |
| k>1, replacement + reuse_max | Middle ground | Middle ground | Higher | Balance vs. over-reliance on few controls |

MatchIt documentation notes that precision gains diminish rapidly after k=4.

## Testing Considerations

1. **k=1 backward compatibility**: Ensure `ratio_k=1, with_replacement=False` produces identical results to current implementation
2. **Round-robin fairness**: Verify all treated get round 1 matches before round 2
3. **reuse_max enforcement**: Confirm controls don't exceed limit
4. **require_k behavior**: Test both True and False with insufficient controls
5. **Weight correctness**: Verify `pair_weight` computation
6. **Edge cases**: Empty buckets, k > available controls, single treated/control

## Next Steps

1. Create PLAN_DETAILED.md with line-by-line implementation guidance
2. Implement changes to `match()` function
3. Update tests in `tests/test_matching.py`
4. Update any documentation/docstrings
