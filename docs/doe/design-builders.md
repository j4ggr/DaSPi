# Design Builders

The BaseDesignBuilder is an abstract class that provides the foundation for all DOE builders in DaSPi. It defines the common interface and functionality shared by all design types.

## Common Features

All design builders inherit these features from BaseDesignBuilder:

### Factors

- **Factor Definition**: Define experimental factors with their levels
- **Mixed Types**: Support for both numerical and categorical factors
- **Validation**: Automatic validation of factor names and properties

### Replication

- **Multiple Runs**: Repeat each factor combination multiple times
- **Error Estimation**: Replication helps estimate experimental error
- **Precision**: More replicates increase precision of effect estimates

### Blocking

Blocking helps control for nuisance variables that might affect your response:

- **Statistical Blocking**: Confound blocks with high-order interactions
- **Simple Blocking**: Divide runs evenly across blocks
- **Replica Blocking**: Each replicate becomes a block

### Central Points

- **Curvature Detection**: Central points help detect non-linear effects
- **Pure Error**: Provide an estimate of pure experimental error
- **Model Adequacy**: Help assess whether a linear model is adequate

### Randomization

- **Run Order**: Randomize the order of experimental runs
- **Block-wise**: Randomization within each block
- **Bias Reduction**: Helps reduce the effect of time trends and other nuisance factors

## Design Matrix Output

All builders produce standardized design matrices with these columns:

- **std_order**: Original order before randomization
- **run_order**: Randomized execution order
- **central_point**: 1 for factor points, 0 for central points
- **replica**: Replicate number for each factor combination
- **block**: Block assignment
- **Factor columns**: One column per factor with levels

## Usage Pattern

All design builders follow the same basic pattern:

```python
import daspi as dsp

# 1. Define factors
factor_a = dsp.Factor('A', (1, 2))
factor_b = dsp.Factor('B', (10, 20))

# 2. Create builder with options
builder = dsp.SomeDesignBuilder(
    factor_a, factor_b,
    replicates=2,
    central_points=3,
    blocks='highest',
    shuffle=True
)

# 3. Generate design matrix
design = builder.build_design(corrected=False)

# 4. Use the design for your experiments
print(design)
```

## Coded vs Original Values

Builders can return designs in two formats:

```python
# Original factor values (for experimental use)
design_original = builder.build_design(corrected=False)

# Coded values (for statistical analysis)
design_coded = builder.build_design(corrected=True)
```

Coded values use standardized scales (typically -1, 0, +1) that make statistical analysis easier and more interpretable.

## Validation and Error Checking

All builders perform extensive validation:

- Factor names must be unique
- Factor names cannot conflict with standard column names
- Positive number of replicates
- Valid block specifications
- Proper generator syntax (for fractional factorials)

## API Reference

::: daspi.doe.BaseDesignBuilder
    options:
      show_source: false
      heading_level: 3