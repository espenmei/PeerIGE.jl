# Roadmap

## Immediate Tracks

### 1. Modularization

- extract data loading and model construction from the script tail into named functions
- isolate preconditioner construction behind a smaller public API
- separate threaded and non-threaded solve setup more clearly

### 2. Correctness And Validation

- add dimension and input checks before model construction
- add focused tests for `apply_A!`, preconditioners, and `get_Vi_u!`
- compare key outputs between `emai_direct.jl` and simplified settings in `emai.jl`

### 3. Experiment Workflow

- make run parameters configurable without editing the script body each time
- standardize output naming for logs and result files
- document a baseline run configuration for regression checking

### 4. Documentation

- keep `computational_approach.md` aligned with implementation details
- document Norwegian domain terms that appear in comments and variable names
- add short rationale notes where numerical safeguards are easy to misread as arbitrary

## Recommended Order

1. Extract script-tail setup into functions
2. Add dimension and file validation
3. Add a baseline verification run workflow
4. Add small tests for the safest operator blocks
5. Only then consider larger structural refactors

## Session Rule

Each work session should end with one of these outcomes:

- one verified code change
- one verified explanation of a mathematical block
- one new validation artifact
- one documentation improvement that reduces future ambiguity