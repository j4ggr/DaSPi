# Design of Experiments Guide

Design of Experiments (DOE) is a systematic approach to understanding how different factors affect a response variable. Instead of changing one factor at a time, DOE allows you to study multiple factors simultaneously, revealing important interactions while minimizing experimental effort.

## When Should You Use DOE?

- **Process Optimization**: Finding the best settings for your manufacturing process
- **Product Development**: Understanding how design parameters affect performance
- **Quality Improvement**: Identifying factors that impact product quality
- **Screening Studies**: Determining which factors are most important among many candidates
- **Cost Reduction**: Getting maximum information from minimum experimental effort

## Basic DOE Workflow

### 1. Problem Definition

Start by clearly defining:

- **Response variable**: What you want to improve or understand
- **Factors**: Variables you can control that might affect the response
- **Objectives**: Are you screening, optimizing, or characterizing?
- **Resources**: How many experimental runs can you afford?

### 2. Factor Selection and Levels

```python
import daspi as dsp

# Continuous factors (temperature, pressure, concentration)
temperature = dsp.Factor('Temperature', (150, 200))
pressure = dsp.Factor('Pressure', (10, 15))
time = dsp.Factor('Time', (30, 60))

# Categorical factors (machine, operator, material)
machine = dsp.Factor('Machine', ('A', 'B', 'C'), is_categorical=True)
operator = dsp.Factor('Operator', ('Day', 'Night'), is_categorical=True)
```

### 3. Choose Design Type

The choice depends on your objectives and resources:

```python
# Full factorial: Complete information, higher cost
builder = dsp.FullFactorialDesignBuilder(temperature, pressure, time)

# Fractional factorial: Screening many factors, lower cost
builder = dsp.FractionalFactorialDesignBuilder(
    temp, pressure, time, concentration, catalyst,
    generators=['D=AB', 'E=AC']  # 5 factors in 8 runs
)

# 2^k specialized: Most common for industrial experiments
builder = dsp.FullFactorial2kDesignBuilder(temperature, pressure, time)
```

### 4. Add Design Features

```python
# Enhance your design with:
builder = dsp.FullFactorial2kDesignBuilder(
    temperature, pressure, time,
    replicates=3,        # Repeat for error estimation
    central_points=5,    # Detect curvature
    blocks='highest',    # Control for time trends
    shuffle=True         # Randomize run order
)
```

### 5. Generate and Execute

```python
# Generate the design matrix
design = builder.build_design(corrected=False)
print(design)

# Execute experiments according to run_order
# Record your response data
```

## Practical Example: Chemical Process Optimization

Let's walk through a complete example optimizing a chemical reaction:

### Problem Setup

You want to maximize yield in a chemical reaction and suspect that temperature, pressure, and catalyst type affect the outcome.

```python
import daspi as dsp
import numpy as np

# Define factors
temperature = dsp.Factor('Temperature', (150, 200))  # °C
pressure = dsp.Factor('Pressure', (1.0, 1.5))       # bar
catalyst = dsp.Factor('Catalyst', ('A', 'B'), is_categorical=True)

# Create a 2^3 design with center points
builder = dsp.FullFactorial2kDesignBuilder(
    temperature, pressure, catalyst,
    replicates=2,        # Run each combination twice
    central_points=4,    # 4 center points to check curvature
    shuffle=True         # Randomize to avoid time effects
)

design = builder.build_design(corrected=False)
print(f"Total experimental runs: {len(design)}")
print(design.head(10))
```

### Analyzing Results

After running your experiments, you can analyze the results:

```python
# Add your response data (simulated here)
np.random.seed(42)
design['Yield'] = (
    75 +  # baseline
    5 * (design['Temperature'] == 200) +     # temperature effect
    3 * (design['Pressure'] == 1.5) +       # pressure effect
    -2 * (design['Catalyst'] == 'B') +      # catalyst effect
    2 * ((design['Temperature'] == 200) & (design['Pressure'] == 1.5)) +  # interaction
    np.random.normal(0, 1.5, len(design))   # experimental error
)

# Calculate factor effects
high_temp_yield = design[design['Temperature'] == 200]['Yield'].mean()
low_temp_yield = design[design['Temperature'] == 150]['Yield'].mean()
temperature_effect = high_temp_yield - low_temp_yield

print(f"Temperature effect: {temperature_effect:.2f} percentage points")
```

## Design Selection Guide

### Number of Factors vs. Design Choice

| Factors | Recommended Design | Runs | Purpose |
|---------|-------------------|------|---------|
| 2-4     | Full Factorial    | 4-16 | Complete understanding |
| 5-8     | Fractional Factorial | 8-16 | Factor screening |
| 9+      | Sequential DOE    | Variable | Multi-stage approach |

### Resource Constraints

**Limited time/money**: Start with fractional factorial for screening

```python
# Screen 6 factors in just 8 runs
builder = dsp.FractionalFactorialDesignBuilder(
    *six_factors,
    generators=['D=AB', 'E=AC', 'F=BC']
)
```

**Need complete picture**: Use full factorial

```python
# All interactions for 3 factors
builder = dsp.FullFactorialDesignBuilder(*three_factors)
```

## Advanced Techniques

### Sequential Experimentation

1. **Screen** with fractional factorial to identify important factors
2. **Optimize** with full factorial or response surface methods on important factors
3. **Confirm** with verification runs at optimal conditions

### Blocking Strategy

Control for known nuisance variables:

```python
# Block by time periods
builder = dsp.FullFactorialDesignBuilder(
    temp, pressure, catalyst,
    blocks=4,  # Split across 4 time periods
    replicates=2
)

# Block by highest-order interaction (recommended)
builder = dsp.FullFactorialDesignBuilder(
    temp, pressure, catalyst,
    blocks='highest',  # Confound with 3-way interaction
)
```

### Foldover for Fractional Factorials

Resolve confounding between main effects and interactions:

```python
# Initial fractional factorial
builder = dsp.FractionalFactorialDesignBuilder(
    A, B, C,
    generators=['C=AB']
)

# Add foldover to resolve confounding
builder = dsp.FractionalFactorialDesignBuilder(
    A, B, C,
    generators=['C=AB'],
    fold=True  # Doubles the runs, resolves main effect confounding
)
```

## Common Pitfalls and Solutions

### Too Many Factors Initially

**Problem**: Trying to study 10+ factors in one experiment

**Solution**: Use sequential approach - screen first, then optimize

### Ignoring Practical Constraints

**Problem**: Factors levels that are impractical or unsafe

**Solution**: Choose levels based on practical operating ranges

### Confusing Correlation with Causation

**Problem**: Assuming all significant effects are causal

**Solution**: Use engineering knowledge to interpret statistical results

### Not Randomizing

**Problem**: Running experiments in standard order

**Solution**: Always randomize unless there are strong practical constraints

## Best Practices

1. **Start Simple**: Begin with screening designs for many factors
2. **Use Replication**: Include replicates to estimate error
3. **Add Center Points**: Help detect curvature in responses
4. **Block When Possible**: Control for known nuisance variables
5. **Randomize Run Order**: Reduce bias from time trends
6. **Document Everything**: Record all conditions and observations
7. **Validate Results**: Confirm findings with follow-up experiments

## Next Steps

After mastering basic DOE:

- Learn response surface methodology for optimization
- Explore mixture designs for formulation problems
- Study robust design for reducing variation
- Consider computer experiments for simulation studies

The key is to start with simple designs and build complexity as you gain experience and understanding of your system.

