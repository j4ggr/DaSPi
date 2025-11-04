# Full Factorial Designs

Full factorial designs include every possible combination of factor levels, providing complete information about main effects and all interactions between factors.

## When to Use Full Factorial Designs

- **Small number of factors**: Typically 2-4 factors with 2-3 levels each
- **Complete understanding needed**: When you need to understand all interactions
- **Sufficient resources**: When you can afford the number of experimental runs
- **Baseline studies**: When establishing a foundation for future experiments

## Available Builders

### FullFactorialDesignBuilder

The general-purpose builder for full factorial designs with factors having any number of levels.

```python
import daspi as dsp

# Mixed level design: 2×3×2 = 12 runs
temperature = dsp.Factor('Temperature', (150, 200))
pressure = dsp.Factor('Pressure', (10, 12, 15))  # 3 levels
catalyst = dsp.Factor('Catalyst', ('A', 'B'), is_categorical=True)

builder = dsp.FullFactorialDesignBuilder(
    temperature, pressure, catalyst,
    replicates=2,
    central_points=3,
    blocks='highest',
    shuffle=True
)

design = builder.build_design(corrected=False)
```

### FullFactorial2kDesignBuilder

Specialized builder for 2-level factorial designs (2^k designs). These are the most common experimental designs.

```python
import daspi as dsp

# 2^3 design: 2×2×2 = 8 runs
temperature = dsp.Factor('Temperature', (150, 200))
pressure = dsp.Factor('Pressure', (10, 15))
time = dsp.Factor('Time', (30, 60))

builder = dsp.FullFactorial2kDesignBuilder(
    temperature, pressure, time,
    replicates=2,
    central_points=4,
    shuffle=True
)

design = builder.build_design(corrected=False)
```

## Design Features

### Replication

Replication helps estimate experimental error and increases precision:

```python
# Each factor combination run 3 times
builder = dsp.FullFactorialDesignBuilder(
    factor_a, factor_b,
    replicates=3
)
```

### Central Points

Central points help detect curvature in the response and estimate pure error:

```python
# Add 5 center point runs to each block
builder = dsp.FullFactorial2kDesignBuilder(
    factor_a, factor_b,
    central_points=5
)
```

### Blocking

Blocking helps control for nuisance variables:

```python
# Block by highest-order interaction (recommended)
builder = dsp.FullFactorialDesignBuilder(
    factor_a, factor_b, factor_c,
    blocks='highest'
)

# Block by specific interaction
builder = dsp.FullFactorialDesignBuilder(
    factor_a, factor_b, factor_c,
    blocks=['A', 'B']  # Block by A×B interaction
)

# Simple blocking (not confounded)
builder = dsp.FullFactorialDesignBuilder(
    factor_a, factor_b, factor_c,
    blocks=4  # Split into 4 blocks evenly
)
```

## Design Size Considerations

| Factors | Levels | Runs | Suitable For |
|---------|--------|------|--------------|
| 2       | 2×2    | 4    | Quick screening |
| 3       | 2×2×2  | 8    | Small studies |
| 4       | 2×2×2×2| 16   | Standard studies |
| 5       | 2×2×2×2×2 | 32 | Large studies |
| 3       | 3×3×3  | 27   | Response surfaces |

## API Reference

::: daspi.doe.FullFactorialDesignBuilder
    options:
      show_source: false
      heading_level: 3

::: daspi.doe.FullFactorial2kDesignBuilder
    options:
      show_source: false
      heading_level: 3
