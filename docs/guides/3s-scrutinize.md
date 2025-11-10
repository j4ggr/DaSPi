# Scrutinize Phase: Investigate & Analyze

The **Scrutinize** phase is the analytical heart of the 3S methodology, focusing on systematic investigation and root cause identification. This phase employs rigorous data analysis and statistical methods to uncover the true causes behind the problem.

## Overview

![Scrutinize Phase Overview](../img/3s-scrutinize-overview.svg)

The Scrutinize phase corresponds to the fourth step of the 8D approach (D4) and aligns with the Analyze phase of Six Sigma DMAIC. It emphasizes thorough investigation and evidence-based decision making.

## Phase Objectives

### Primary Goals

- **Identify the most likely root causes** using structured methodologies
- **Weight and prioritize causes** based on impact and likelihood
- **Conduct statistical experiments** to validate hypotheses
- **Develop evidence-based conclusions** through significance testing

### Success Criteria

✅ **Root causes identified** and validated through data analysis  
✅ **Statistical evidence** supports cause-and-effect relationships  
✅ **Interaction effects** are understood and quantified  
✅ **Hypotheses tested** with appropriate confidence levels  

## Key Activities

### 1. Root Cause Identification

**Objective:** Generate and organize potential causes systematically.

#### Brainstorming Techniques

**Mind Mapping**
- Central problem in the center
- Branch out to major categories
- Sub-branches for specific causes
- Visual representation of relationships

**Ishikawa (Fishbone) Diagram**
- Categories: Man, Machine, Method, Material, Environment, Measurement
- Systematic exploration of each category
- Team-based cause identification

**Process Mapping**
- Step-by-step process documentation
- Identify potential failure points
- Map inputs, outputs, and controls

```python
import daspi as dsp
import pandas as pd

# Example: Organize root cause data
causes_data = {
    'Category': ['Machine', 'Machine', 'Material', 'Method', 'Environment'],
    'Potential_Cause': ['Calibration drift', 'Wear', 'Contamination', 'Procedure gap', 'Temperature'],
    'Likelihood': [0.8, 0.6, 0.7, 0.9, 0.5],
    'Impact': [0.9, 0.7, 0.8, 0.8, 0.6]
}

causes_df = pd.DataFrame(causes_data)
causes_df['Priority_Score'] = causes_df['Likelihood'] * causes_df['Impact']
print(causes_df.sort_values('Priority_Score', ascending=False))
```

#### FMEA (Failure Mode and Effects Analysis)

Systematic analysis of potential failures:

| Failure Mode | Effects | Causes | Severity | Occurrence | Detection | RPN |
|--------------|---------|--------|----------|------------|-----------|-----|
| Calibration drift | Out-of-spec products | Temperature variation | 8 | 6 | 4 | 192 |
| Tool wear | Surface defects | Usage cycles | 7 | 7 | 3 | 147 |

### 2. Cause Prioritization

**Objective:** Weight and rank potential causes based on data and expert judgment.

#### Pairwise Comparison Method

```python
import numpy as np
from scipy import stats

# Example: Pairwise comparison matrix
causes = ['Calibration', 'Temperature', 'Material_quality', 'Operator_skill']
n_causes = len(causes)

# Create comparison matrix (example values)
comparison_matrix = np.array([
    [1, 3, 2, 4],    # Calibration vs others
    [1/3, 1, 1/2, 2], # Temperature vs others  
    [1/2, 2, 1, 3],   # Material vs others
    [1/4, 1/2, 1/3, 1] # Operator vs others
])

# Calculate priority weights (simplified eigenvalue method)
weights = np.mean(comparison_matrix / np.sum(comparison_matrix, axis=0), axis=1)
priority_df = pd.DataFrame({
    'Cause': causes,
    'Weight': weights,
    'Rank': stats.rankdata(-weights)
})

print(priority_df.sort_values('Rank'))
```

#### Cause & Effect Matrix

Link causes to customer requirements:

```python
# Example: C&E Matrix analysis
requirements = ['Quality', 'Cost', 'Delivery']
requirement_weights = [9, 7, 5]  # Importance ratings

ce_matrix = pd.DataFrame({
    'Cause': causes,
    'Quality_Impact': [9, 7, 8, 6],
    'Cost_Impact': [6, 4, 7, 5],
    'Delivery_Impact': [3, 2, 4, 8]
})

# Calculate weighted scores
for i, req in enumerate(requirements):
    col_name = f'{req}_Impact'
    ce_matrix[f'{req}_Weighted'] = ce_matrix[col_name] * requirement_weights[i]

ce_matrix['Total_Score'] = (ce_matrix['Quality_Weighted'] + 
                           ce_matrix['Cost_Weighted'] + 
                           ce_matrix['Delivery_Weighted'])

print(ce_matrix[['Cause', 'Total_Score']].sort_values('Total_Score', ascending=False))
```

#### Pareto Analysis

Visualize the vital few causes:

```python
import matplotlib.pyplot as plt

# Create Pareto chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pareto chart of causes
sorted_data = ce_matrix.sort_values('Total_Score', ascending=False)
cumulative_pct = np.cumsum(sorted_data['Total_Score']) / sorted_data['Total_Score'].sum() * 100

ax1.bar(sorted_data['Cause'], sorted_data['Total_Score'], alpha=0.7)
ax1.set_ylabel('Impact Score')
ax1.set_title('Cause Impact Analysis')
ax1.tick_params(axis='x', rotation=45)

# Cumulative percentage line
ax1_twin = ax1.twinx()
ax1_twin.plot(sorted_data['Cause'], cumulative_pct, 'ro-', color='red')
ax1_twin.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Line')
ax1_twin.set_ylabel('Cumulative %')
ax1_twin.set_ylim(0, 100)

plt.tight_layout()
plt.show()
```

### 3. Statistical Experimentation

**Objective:** Design and conduct experiments to validate cause-and-effect relationships.

#### Design of Experiments (DOE)

**Full Factorial Design**

```python
from itertools import product

# Example: 2^3 factorial design
factors = {
    'Temperature': [80, 120],
    'Pressure': [2, 4], 
    'Time': [30, 60]
}

# Generate all combinations
design_points = list(product(*factors.values()))
factor_names = list(factors.keys())

design_df = pd.DataFrame(design_points, columns=factor_names)
design_df['Run_Order'] = np.random.permutation(len(design_df))
design_df = design_df.sort_values('Run_Order').reset_index(drop=True)

print("Experimental Design:")
print(design_df)
```

**Response Surface Methodology (RSM)**

For optimization studies when approaching optimal conditions:

```python
# Example: Central Composite Design
from scipy.stats import norm

# Define factor levels for CCD
alpha = np.sqrt(len(factor_names))  # Rotatable design
center_points = 3

# Generate design matrix (simplified)
print(f"Alpha value for rotatable design: {alpha:.2f}")
print(f"Recommended center points: {center_points}")
```

#### EVOP (Evolutionary Operation)

For ongoing process improvement during production:

```python
# Example: Simple EVOP cycle
current_conditions = {'Temperature': 100, 'Pressure': 3}
evop_changes = {'Temperature': ±2, 'Pressure': ±0.2}

print("EVOP Experimental Conditions:")
for factor, base_value in current_conditions.items():
    change = evop_changes[factor]
    print(f"{factor}: {base_value-change}, {base_value}, {base_value+change}")
```

### 4. Hypothesis Testing and Validation

**Objective:** Use statistical methods to prove cause-and-effect relationships.

![Hypothesis Testing Framework](../img/hypothesis-testing.png)

#### ANOVA (Analysis of Variance)

Test for significant differences between factor levels:

```python
import daspi as dsp
from scipy import stats

# Example: One-way ANOVA
group1 = np.random.normal(100, 5, 30)  # Control
group2 = np.random.normal(105, 5, 30)  # Treatment A
group3 = np.random.normal(98, 5, 30)   # Treatment B

# Perform ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_value:.3f}")

if p_value < 0.05:
    print("Significant difference detected between groups")
    
    # Post-hoc analysis
    from scipy.stats import tukey_hsd
    result = tukey_hsd(group1, group2, group3)
    print(result)
```

#### Correlation Analysis

Identify linear relationships between variables:

```python
# Example: Correlation analysis
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 2 * x + np.random.normal(0, 0.5, 100)  # Strong correlation
z = np.random.normal(0, 1, 100)             # No correlation

correlation_data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
correlation_matrix = correlation_data.corr()

print("Correlation Matrix:")
print(correlation_matrix.round(3))

# Test significance
from scipy.stats import pearsonr
r_xy, p_xy = pearsonr(x, y)
print(f"\nX-Y Correlation: r={r_xy:.3f}, p={p_xy:.3f}")
```

#### Regression Analysis

Quantify relationships and predict outcomes:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Multiple regression example
X = correlation_data[['X', 'Z']]
y = correlation_data['Y']

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"R-squared: {r2:.3f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.3f}")
```

#### Multivariate Analysis

For complex interactions and multiple responses:

```python
# Example: Principal Component Analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Simulate multivariate data
process_data = pd.DataFrame({
    'Temperature': np.random.normal(100, 10, 200),
    'Pressure': np.random.normal(50, 5, 200),
    'Flow_Rate': np.random.normal(25, 3, 200),
    'Humidity': np.random.normal(60, 8, 200)
})

# Standardize and perform PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(process_data)

pca = PCA()
pca_result = pca.fit_transform(scaled_data)

print("Explained Variance Ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.3f}")
```

## Decision Point: Can Problem Be Demonstrably Solved?

At the end of the Scrutinize phase, evaluate whether sufficient evidence exists:

### ✅ **Yes** → Proceed to Stabilize Phase

- Root causes are identified and validated
- Statistical evidence supports cause-and-effect relationships
- Confidence in solution approach is high

### ❌ **No** → Return to Experimentation

- Conduct additional experiments
- Explore alternative hypotheses
- Consider different experimental designs

## Tools and Techniques Summary

### Essential Tools for Scrutinize Phase

| Tool | Purpose | Best Used When |
|------|---------|----------------|
| **Ishikawa Diagram** | Systematic cause identification | Starting root cause analysis |
| **Pairwise Comparison** | Objective cause prioritization | Multiple competing causes |
| **DOE** | Validate cause-and-effect | Controllable factors exist |
| **ANOVA** | Test statistical significance | Comparing multiple groups |
| **Regression** | Quantify relationships | Continuous variables |
| **Multivariate Analysis** | Complex interactions | Many variables involved |

### DaSPi Library Support

The DaSPi library provides comprehensive tools for the Scrutinize phase:

```python
import daspi as dsp

# Statistical testing
result = dsp.hypothesis_test(data1, data2, test_type='t_test')

# ANOVA analysis  
anova_result = dsp.anova_analysis(data, factors=['A', 'B', 'C'])

# Process capability
capability = dsp.ProcessEstimator(samples=data, spec_limits=limits)

# Correlation analysis
correlations = dsp.correlation_matrix(dataframe)

# Visualization
dsp.pareto_chart(causes, impacts)
dsp.scatter_plot_matrix(variables)
```

## Common Challenges and Solutions

### Challenge 1: Too Many Potential Causes

**Solution:**

- Use structured prioritization methods
- Focus on causes with highest impact and likelihood
- Apply the 80/20 rule to identify vital few causes

### Challenge 2: Limited Data for Analysis

**Solution:**

- Design efficient experiments with maximum information
- Use historical data and process knowledge
- Consider observational studies when experiments are not feasible

### Challenge 3: Complex Interactions

**Solution:**

- Use factorial designs to study interactions
- Apply multivariate analysis techniques
- Break complex problems into smaller, manageable parts

## Next Steps

Once root causes are validated and evidence is compelling:

🔄 **[Proceed to Stabilize Phase](3s-stabilize.md)** to develop and implement solutions

The thorough investigation completed in the Scrutinize phase provides the foundation for effective solution development and implementation.

---

**Phase Complete!** You have successfully identified and validated the root causes of the problem. The team now has solid evidence to guide solution development in the Stabilize phase.
 
 