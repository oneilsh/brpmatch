# Documentation Style Guide

A reusable guide for writing clear, example-driven README.md files. The examples here
are based on a Python library, but the same principles apply for most other programming
projects.


## Core Principles

### 1. Examples First, API Reference Second

Lead with working examples that users can copy and run. API reference sections come after users understand the workflow through examples.

### 2. Progressive Complexity

Structure documentation from simple to complex:

1. **Minimal working example** - The simplest possible usage
2. **Common use cases** - Typical configurations with more options
3. **Advanced features** - Complex scenarios, edge cases, customization
4. **API reference** - Complete parameter documentation

Each section builds on the previous, introducing one or two new concepts at a time.

### 3. Inline Comments Over Prose

Explain code within the code block itself:

```python
# Good: Comment explains the non-obvious
features_df = generate_features(
    spark,
    df,
    treatment_col="cohort",
    treatment_value="treated",
    exact_match_cols=["gender"],  # Match within same gender only
)

# Avoid: Obvious comments that add no value
features_df = generate_features(
    spark,  # The spark session
    df,     # The dataframe
    ...
)
```

Reserve prose for concepts that can't be explained in a single line.

### 4. Show What Changes

When introducing features, show before/after or contrast examples:

```python
# Basic matching (1-to-1)
matched_df = match(features_df)

# Ratio matching (1-to-3)
matched_df = match(features_df, ratio_k=3, with_replacement=True)
```

### 5. Known Limitations Upfront

Be transparent about constraints early in the documentation rather than burying them in footnotes.

---

## Section Organization

A typical README should follow this structure. Not all sections may be applicable, depending on the project.

```
# Project Name
One-line description.

## Features (or Overview)
Bullet points of key capabilities.

## Installation
How to install (pip, poetry, from source).

## Quick Start
Minimal working example - the fastest path to success.

## Usage Examples
Progressive examples showing common patterns.

## [Feature-Specific Sections]
Dedicated sections for complex features that merit explanation.

## API Summary
A compact, comprehensive list of major API elements.

## Development
How to contribute, run tests, build.

## Requirements
Dependencies, version constraints.

## Changelog
A simple list of version numbers and changes.
```

---

## Code Example Guidelines

### Structure
This may vary depending on the API.

```python
# 1. Imports first, grouped logically
from library import func1, func2

# 2. Setup/configuration
config = create_config(...)

# 3. Main workflow with inline comments
result = func1(
    data,
    option="value",      # Explain non-obvious options
)

# 4. Output/result handling
print(result)
```

### Length
General guidelines on examples, which may be broken when doing so would improve clarity.

- Quick Start examples: 10-20 lines
- Feature examples: 20-40 lines
- Complete examples: Can be longer, but chunk into logical sections

### Completeness

Examples should be copy-pasteable and runnable. Include:
- Necessary imports
- Sample data creation (or a description of input data if this is non-trivial)
- The API calls
- What to do with results

### Parameter Descriptions

- State the type implicitly through examples, not explicitly (avoid "str", "int" clutter)
- Mention options and defaults as inline comments: ` # "euclidean" or "mahalanobis"; default "euclidean"`
- Group related parameters together
- Call out important constraints: "must not contain nulls"

### Auto-Discovered vs Explicit Parameters

When a library uses naming conventions for auto-discovery, document the convention and provide illustrative examples.

---

## Writing Style

### Voice

- Use imperative mood for instructions: "Now we run the matching algorithm" not "You can run..."
- Be concise - every word should earn its place

### Formatting

- Use code formatting for: function names, parameter names, file names, values
- Use bold for: key terms on first use
- Use bullet points for: lists of 3+ items

### Emojis

Use emojis sparingly, unless required for examples or outputs that include them.

### Callouts

Use sparingly for critical information:

```markdown
**Note:** Important but not critical information.

**Warning:** Something that could cause problems if ignored.

**Tip:** Helpful suggestion that improves the experience.
```

---

## Maintenance

### Keep Examples in Sync

Examples in documentation should match the actual API. When the API changes, so should the documentation.

### Version Documentation

The Changelog section should contain a bullet list with short summaries of all user-facing changes, bugfixes, or new features, organized by version number using semantic versioning. 
This should be based on the version listed in the pyproject.toml file (or similar in other types of projects) when adding content. The user will change the version number manually to manage build and publish processes. 