# Stabilize Phase: Implement & Control

The **Stabilize** phase is the execution and sustainability phase of the 3S methodology, focusing on solution implementation and long-term control. This phase ensures that improvements are not only effective but also sustainable over time.

## Overview

![Stabilize Phase Overview](../img/3s-stabilize-overview.svg)

The Stabilize phase corresponds to steps D5-D8 of the 8D approach and combines the Improve and Control phases of Six Sigma DMAIC. It emphasizes sustainable implementation and knowledge transfer.

## Phase Objectives

### Primary Goals

- **Select the most economical solution** while considering customer needs
- **Implement and validate corrective measures** with statistical confidence
- **Establish control systems** to ensure long-term sustainability
- **Document and transfer knowledge** for future applications

### Success Criteria

✅ **Solution implemented** and validated as effective  
✅ **Control systems** are operational and monitored  
✅ **Immediate containment** measures are removed safely  
✅ **Knowledge documented** and shared with stakeholders  

## Key Activities

### 1. Solution Selection (D5)

**Objective:** Choose the most effective and economical solution while considering customer impact.

![Solution Selection Process](../img/solution-selection.png)

#### Target Optimization

Optimize solutions for multiple objectives:

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Example: Multi-objective optimization
def objective_function(x):
    # x = [temperature, pressure, speed]
    quality = -(x[0] - 100)**2/100 - (x[1] - 50)**2/25  # Maximize quality
    cost = 0.1*x[0] + 0.05*x[1] + 0.2*x[2]              # Minimize cost
    time = 60/x[2]                                        # Minimize cycle time
    
    # Weighted objective (maximize)
    return -(0.5*quality - 0.3*cost - 0.2*time)

# Constraints
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 80},    # Temp >= 80
    {'type': 'ineq', 'fun': lambda x: 120 - x[0]},   # Temp <= 120
    {'type': 'ineq', 'fun': lambda x: x[1] - 40},    # Pressure >= 40
    {'type': 'ineq', 'fun': lambda x: 60 - x[1]},    # Pressure <= 60
    {'type': 'ineq', 'fun': lambda x: x[2] - 10},    # Speed >= 10
    {'type': 'ineq', 'fun': lambda x: 30 - x[2]}     # Speed <= 30
]

# Optimize
result = minimize(objective_function, [100, 50, 20], constraints=constraints)
print(f"Optimal settings: Temperature={result.x[0]:.1f}, Pressure={result.x[1]:.1f}, Speed={result.x[2]:.1f}")
```

#### Solution Selection Matrix

Systematic evaluation of alternative solutions:

```python
# Example: Solution evaluation matrix
solutions = ['Process_Redesign', 'Equipment_Upgrade', 'Training_Program', 'Inspection_Increase']
criteria = ['Cost', 'Effectiveness', 'Implementation_Time', 'Risk', 'Customer_Impact']
weights = [0.25, 0.30, 0.15, 0.15, 0.15]  # Criterion weights

# Scores (1-10 scale, higher is better)
scores = {
    'Process_Redesign': [6, 9, 4, 7, 8],
    'Equipment_Upgrade': [3, 8, 2, 8, 9],
    'Training_Program': [9, 6, 8, 9, 7],
    'Inspection_Increase': [8, 5, 9, 8, 6]
}

# Calculate weighted scores
evaluation_df = pd.DataFrame(scores, index=criteria).T
weighted_scores = {}

for solution in solutions:
    weighted_score = sum(evaluation_df.loc[solution, criterion] * weight 
                        for criterion, weight in zip(criteria, weights))
    weighted_scores[solution] = weighted_score

# Rank solutions
ranking = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
print("Solution Ranking:")
for i, (solution, score) in enumerate(ranking, 1):
    print(f"{i}. {solution}: {score:.2f}")
```

#### Importance-Urgency Matrix (Eisenhower Matrix)

Prioritize implementation activities:

```python
# Example: Eisenhower Matrix for implementation tasks
tasks = [
    {'Task': 'Update work instructions', 'Importance': 9, 'Urgency': 8},
    {'Task': 'Train operators', 'Importance': 8, 'Urgency': 9},
    {'Task': 'Install new equipment', 'Importance': 7, 'Urgency': 6},
    {'Task': 'Revise procedures', 'Importance': 9, 'Urgency': 7},
    {'Task': 'Update documentation', 'Importance': 6, 'Urgency': 5}
]

task_df = pd.DataFrame(tasks)

# Categorize tasks
def categorize_task(importance, urgency):
    if importance >= 7 and urgency >= 7:
        return 'Do First (Urgent & Important)'
    elif importance >= 7 and urgency < 7:
        return 'Schedule (Important, Not Urgent)'
    elif importance < 7 and urgency >= 7:
        return 'Delegate (Urgent, Not Important)'
    else:
        return 'Eliminate (Neither)'

task_df['Category'] = task_df.apply(lambda x: categorize_task(x['Importance'], x['Urgency']), axis=1)
print(task_df[['Task', 'Category']].sort_values(['Importance', 'Urgency'], ascending=False))
```

### 2. Implementation and Validation (D6)

**Objective:** Implement corrective measures and validate their effectiveness statistically.

![Implementation Validation](../img/implementation-validation.png)

#### Hypothesis Testing for Validation

Compare before and after performance:

```python
import daspi as dsp
from scipy import stats

# Example: Validation study
np.random.seed(42)
before_implementation = np.random.normal(100, 15, 50)  # Historical data
after_implementation = np.random.normal(110, 12, 50)   # Post-implementation

# Perform t-test for improvement
t_stat, p_value = stats.ttest_ind(before_implementation, after_implementation)

print(f"Before Implementation - Mean: {np.mean(before_implementation):.2f}, Std: {np.std(before_implementation):.2f}")
print(f"After Implementation - Mean: {np.mean(after_implementation):.2f}, Std: {np.std(after_implementation):.2f}")
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")

if p_value < 0.05:
    improvement = np.mean(after_implementation) - np.mean(before_implementation)
    print(f"Statistically significant improvement: {improvement:.2f}")
```

#### Process Capability Assessment

Evaluate process capability with new conditions:

```python
import daspi as dsp

# Example: Process capability analysis
spec_limits = dsp.SpecLimits(lower=85, upper=115)

# Before implementation
estimator_before = dsp.ProcessEstimator(
    samples=before_implementation,
    spec_limits=spec_limits
)

# After implementation  
estimator_after = dsp.ProcessEstimator(
    samples=after_implementation,
    spec_limits=spec_limits
)

print("Process Capability Comparison:")
print(f"Before - Cp: {estimator_before.cp:.3f}, CpK: {estimator_before.cpk:.3f}")
print(f"After  - Cp: {estimator_after.cp:.3f}, CpK: {estimator_after.cpk:.3f}")
print(f"Sigma Level Before: {estimator_before.sigma_level:.2f}")
print(f"Sigma Level After: {estimator_after.sigma_level:.2f}")
```

#### Charts for Location, Variation and Proportions

Create comprehensive visualization of improvements:

```python
import matplotlib.pyplot as plt

# Create comparison charts
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Histogram comparison
ax1.hist(before_implementation, alpha=0.7, label='Before', bins=15)
ax1.hist(after_implementation, alpha=0.7, label='After', bins=15)
ax1.set_title('Distribution Comparison')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.axvline(spec_limits.lower, color='red', linestyle='--', label='LSL')
ax1.axvline(spec_limits.upper, color='red', linestyle='--', label='USL')

# Box plot comparison
ax2.boxplot([before_implementation, after_implementation], 
           labels=['Before', 'After'])
ax2.set_title('Variation Comparison')
ax2.set_ylabel('Value')
ax2.axhline(spec_limits.lower, color='red', linestyle='--')
ax2.axhline(spec_limits.upper, color='red', linestyle='--')

# Process capability bars
categories = ['Before', 'After']
cp_values = [estimator_before.cp, estimator_after.cp]
cpk_values = [estimator_before.cpk, estimator_after.cpk]

x_pos = np.arange(len(categories))
width = 0.35

ax3.bar(x_pos - width/2, cp_values, width, label='Cp')
ax3.bar(x_pos + width/2, cpk_values, width, label='CpK')
ax3.set_title('Process Capability Comparison')
ax3.set_ylabel('Capability Index')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.axhline(1.33, color='green', linestyle='--', label='Target')

# Sigma level comparison
sigma_values = [estimator_before.sigma_level, estimator_after.sigma_level]
ax4.bar(categories, sigma_values, color=['orange', 'green'], alpha=0.7)
ax4.set_title('Sigma Level Improvement')
ax4.set_ylabel('Sigma Level')
ax4.axhline(6.0, color='red', linestyle='--', label='Six Sigma Target')
ax4.legend()

plt.tight_layout()
plt.show()
```

### 3. Control Systems Implementation (D7)

**Objective:** Establish sustainable control mechanisms to maintain improvements.

![Control Systems](../img/control-systems.png)

#### Statistical Process Control (SPC)

Implement ongoing monitoring:

```python
import daspi as dsp

# Example: Control chart setup
control_data = np.random.normal(110, 5, 100)  # Ongoing process data

# Calculate control limits
mean = np.mean(control_data)
std = np.std(control_data)
ucl = mean + 3 * std
lcl = mean - 3 * std

# Create control chart
plt.figure(figsize=(12, 6))
plt.plot(control_data, 'bo-', markersize=4)
plt.axhline(mean, color='green', label=f'Mean = {mean:.2f}')
plt.axhline(ucl, color='red', linestyle='--', label=f'UCL = {ucl:.2f}')
plt.axhline(lcl, color='red', linestyle='--', label=f'LCL = {lcl:.2f}')
plt.fill_between(range(len(control_data)), lcl, ucl, alpha=0.2, color='green')
plt.title('Statistical Process Control Chart')
plt.xlabel('Sample Number')
plt.ylabel('Measurement Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check for out-of-control conditions
out_of_control = (control_data > ucl) | (control_data < lcl)
if np.any(out_of_control):
    print(f"Out-of-control points detected at samples: {np.where(out_of_control)[0]}")
```

#### Control Plans and OCAP

Develop Out-of-Control Action Plans:

```python
# Example: Control plan structure
control_plan = pd.DataFrame({
    'Process_Step': ['Mixing', 'Heating', 'Cooling', 'Packaging'],
    'Characteristic': ['Viscosity', 'Temperature', 'Time', 'Weight'],
    'Specification': ['100±10', '120±5', '30±2', '500±5'],
    'Measurement_Method': ['Viscometer', 'Thermocouple', 'Timer', 'Scale'],
    'Sample_Size': [3, 5, 1, 10],
    'Frequency': ['Hourly', 'Continuous', 'Each batch', 'Every 10 units'],
    'Control_Method': ['SPC Chart', 'Alarm', 'Check sheet', 'SPC Chart'],
    'Reaction_Plan': ['Stop & adjust', 'Auto control', 'Manual check', 'Investigate']
})

print("Control Plan:")
print(control_plan)
```

#### Time Series Analysis

Monitor trends and patterns:

```python
from sklearn.linear_model import LinearRegression

# Example: Trend analysis
time_points = np.arange(len(control_data))
trend_model = LinearRegression().fit(time_points.reshape(-1, 1), control_data)
trend_line = trend_model.predict(time_points.reshape(-1, 1))

# Plot with trend
plt.figure(figsize=(12, 6))
plt.plot(time_points, control_data, 'bo-', markersize=4, label='Data')
plt.plot(time_points, trend_line, 'r-', linewidth=2, label=f'Trend (slope={trend_model.coef_[0]:.4f})')
plt.title('Process Trend Analysis')
plt.xlabel('Time')
plt.ylabel('Measurement Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Trend significance test
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(time_points, control_data)
print(f"Trend Analysis: Slope={slope:.4f}, R²={r_value**2:.3f}, p-value={p_value:.3f}")
```

### 4. Knowledge Transfer and Closure (D8)

**Objective:** Document lessons learned and apply knowledge to similar projects.

![Knowledge Transfer](../img/knowledge-transfer.png)

#### Lessons Learned Documentation

```python
# Example: Lessons learned template
lessons_learned = {
    'Project': '3S Quality Improvement',
    'Duration': '3 months',
    'Team_Size': 6,
    'Problem_Category': 'Manufacturing Defect',
    'Root_Causes': ['Equipment calibration drift', 'Operator procedure gaps'],
    'Solutions_Implemented': ['Automated calibration system', 'Enhanced training program'],
    'Key_Learnings': [
        'Early containment prevented customer impact',
        'Statistical validation was crucial for buy-in',
        'Automated solutions more sustainable than manual procedures'
    ],
    'Recommendations': [
        'Implement similar automated systems in other areas',
        'Standardize statistical validation approach',
        'Create reusable training materials'
    ],
    'Metrics_Improved': {
        'Defect_Rate': {'Before': 0.05, 'After': 0.01, 'Improvement': '80%'},
        'CpK': {'Before': 1.1, 'After': 1.8, 'Improvement': '64%'},
        'Customer_Complaints': {'Before': 12, 'After': 2, 'Improvement': '83%'}
    }
}

print("PROJECT SUMMARY")
print("="*50)
for key, value in lessons_learned.items():
    if key != 'Metrics_Improved':
        print(f"{key.replace('_', ' ').title()}: {value}")

print("\nKEY METRICS IMPROVEMENT")
print("="*30)
for metric, values in lessons_learned['Metrics_Improved'].items():
    print(f"{metric}: {values['Before']} → {values['After']} ({values['Improvement']} improvement)")
```

#### Final Presentation Template

Key elements for project closure:

1. **Executive Summary**
   - Problem statement and business impact
   - Solution overview and results
   - Return on investment

2. **Technical Details**
   - Root cause analysis findings
   - Statistical validation results
   - Implementation approach

3. **Results and Benefits**
   - Performance improvements
   - Cost savings
   - Customer impact

4. **Sustainability Plan**
   - Control systems implemented
   - Training completed
   - Monitoring procedures

5. **Recommendations**
   - Applications to other areas
   - Additional improvement opportunities
   - Process standardization needs

## Removal of Immediate Containment

**Critical Step:** Once permanent solutions are validated, safely remove temporary containment:

```python
# Example: Containment removal validation
def validate_containment_removal(performance_data, threshold=0.01):
    """Validate that containment can be safely removed"""
    recent_performance = performance_data[-20:]  # Last 20 data points
    defect_rate = np.sum(recent_performance > threshold) / len(recent_performance)
    
    if defect_rate < 0.05:  # Less than 5% above threshold
        return True, f"Safe to remove containment. Recent defect rate: {defect_rate:.3f}"
    else:
        return False, f"Keep containment. Recent defect rate: {defect_rate:.3f} too high"

# Test containment removal
safe_to_remove, message = validate_containment_removal(control_data, threshold=120)
print(message)
```

## Success Metrics and KPIs

Track these metrics throughout the Stabilize phase:

### Financial Impact

```python
# Example: ROI calculation
implementation_cost = 50000  # Total cost of solution
annual_savings = 120000      # Annual savings from improvement

# Simple ROI
roi = (annual_savings - implementation_cost) / implementation_cost * 100
payback_period = implementation_cost / annual_savings * 12  # months

print(f"Return on Investment: {roi:.1f}%")
print(f"Payback Period: {payback_period:.1f} months")
```

### Quality Metrics

- **Process Capability:** CpK improvement
- **Defect Reduction:** DPU, FPY improvements  
- **Customer Satisfaction:** Complaint reduction
- **Process Stability:** Control chart performance

### Operational Metrics

- **Implementation Timeline:** On schedule completion
- **Training Effectiveness:** Competency assessments
- **Control System Performance:** Monitoring compliance
- **Knowledge Transfer:** Documentation completion

## Common Challenges and Solutions

### Challenge 1: Solution Implementation Resistance

**Solution:**

- Involve stakeholders in solution design
- Provide clear communication of benefits
- Start with pilot implementation to prove effectiveness

### Challenge 2: Inadequate Control Systems

**Solution:**

- Design robust control plans with clear responsibilities
- Implement multiple layers of control (prevention, detection, response)
- Provide training and support for control system operation

### Challenge 3: Loss of Improvements Over Time

**Solution:**

- Establish regular review and audit cycles
- Create clear escalation procedures for out-of-control conditions
- Maintain active engagement from management and stakeholders

## Project Completion Checklist

- [ ] **Solution implemented** and statistically validated
- [ ] **Control systems** operational with trained personnel
- [ ] **Containment measures** safely removed
- [ ] **Documentation** complete and accessible
- [ ] **Training** delivered and competency verified
- [ ] **Lessons learned** documented and shared
- [ ] **Final presentation** delivered to stakeholders
- [ ] **Project closure** completed with stakeholder sign-off

---

**Congratulations!** You have successfully completed the 3S methodology. The problem has been systematically resolved with sustainable solutions, and valuable knowledge has been captured for future improvements.
