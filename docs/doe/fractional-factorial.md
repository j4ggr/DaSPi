# Fractional Factorial Designs

Fractional factorial designs run only a carefully selected subset of the full factorial design, reducing experimental cost while still providing valuable information about main effects and important interactions.

## When to Use Fractional Factorial Designs

- **Many factors to screen**: 5 or more factors where full factorial is too expensive
- **Limited resources**: When experimental runs are costly or time-consuming
- **Screening studies**: To identify the most important factors before detailed study
- **Sequential experimentation**: As a first step before running a full factorial on important factors

## Key Concepts

### Aliasing (Confounding)

In fractional factorial designs, some effects are aliased (confounded) with others. This means you cannot distinguish between aliased effects from the experimental data alone.

### Resolution

Resolution describes the quality of a fractional factorial design:

- **Resolution III**: Main effects are not aliased with each other (but may be aliased with 2-factor interactions)
- **Resolution IV**: Main effects are not aliased with each other or 2-factor interactions
- **Resolution V**: Main effects and 2-factor interactions are not aliased with each other

### Generators

Generators define how the fractional design is constructed. For example, in a 2^(4-1) design with generator `D=ABC`, factor D is set equal to the product of factors A, B, and C.

## Basic Usage

### Simple Fractional Factorial

```python
import daspi as dsp

# 2^(3-1) design: 4 runs instead of 8
A = dsp.Factor('A', (-1, 1))
B = dsp.Factor('B', (-1, 1))
C = dsp.Factor('C', (-1, 1))

builder = dsp.FractionalFactorialDesignBuilder(
    A, B, C,
    generators=['C=AB']  # C is determined by A×B
)

design = builder.build_design(corrected=False)
print(design)
```

### Using Default Generators

For common designs, you can use standard generators:

```python
# Get standard generators for 2^(5-2) design
generators = dsp.get_default_generators(k=5, p=2)
print(generators)  # ['D=AB', 'E=AC']

# Create the design
A = dsp.Factor('A', (-1, 1))
B = dsp.Factor('B', (-1, 1))
C = dsp.Factor('C', (-1, 1))
D = dsp.Factor('D', (-1, 1))
E = dsp.Factor('E', (-1, 1))

builder = dsp.FractionalFactorialDesignBuilder(
    A, B, C, D, E,
    generators=generators
)

design = builder.build_design(corrected=False)
```

## Foldover Designs

Foldover adds a second fraction where the signs of one or more factors are reversed, helping to resolve aliasing between main effects and 2-factor interactions.

### Complete Foldover

```python
# Foldover all factors
builder = dsp.FractionalFactorialDesignBuilder(
    A, B, C,
    generators=['C=AB'],
    fold=True  # Reverses all factors
)

design = builder.build_design(corrected=False)
# This creates 8 runs total (4 original + 4 foldover)
```

### Partial Foldover

```python
# Foldover only factor A
builder = dsp.FractionalFactorialDesignBuilder(
    A, B, C,
    generators=['C=AB'],
    fold='A'  # Reverses only factor A
)

design = builder.build_design(corrected=False)
```

## Common Fractional Factorial Designs

| Design | Basic Factors | Generated Factors | Generators | Resolution |
|--------|---------------|-------------------|------------|------------|
| 2^(3-1) | A, B | C | C=AB | III |
| 2^(4-1) | A, B, C | D | D=ABC | IV |
| 2^(5-2) | A, B, C | D, E | D=AB, E=AC | III |
| 2^(6-3) | A, B, C | D, E, F | D=AB, E=AC, F=BC | III |
| 2^(7-3) | A, B, C, D | E, F, G | E=ABC, F=ABD, G=BCD | IV |

## Design Strategy

1. **Start small**: Use a fractional factorial to screen many factors
2. **Analyze results**: Identify the most important factors
3. **Follow up**: Run a full factorial or higher-resolution fractional factorial on important factors
4. **Use foldover**: If main effects are confounded with 2-factor interactions

## Advanced Features

### With Blocking and Replication

```python
builder = dsp.FractionalFactorialDesignBuilder(
    A, B, C, D, E,
    generators=['D=AB', 'E=AC'],
    replicates=2,
    central_points=3,
    blocks=2,  # Split into 2 blocks
    fold=True,
    shuffle=True
)

design = builder.build_design(corrected=False)
```

## API Reference

::: daspi.doe.FractionalFactorialDesignBuilder
    options:
      show_source: false
      heading_level: 3

::: daspi.doe.get_default_generators
    options:
      show_source: false
      heading_level: 3
