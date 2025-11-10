# Specify Phase: Define & Contain

The **Specify** phase is the foundation of the 3S methodology, focusing on clear problem definition and immediate damage control. This phase ensures that improvement efforts are properly scoped, resourced, and contained before moving into detailed analysis.

## Overview

![Specify Phase Overview](../img/3s-specify-overview.png)

The Specify phase corresponds to the first three steps of the traditional 8D approach (D1-D3) and combines elements of Six Sigma's Define and Measure phases.

## Phase Objectives

### Primary Goals

- **Form an effective problem-solving team** with appropriate skills and authority
- **Define the problem precisely** with clear scope and measurable impact
- **Implement immediate containment** to prevent further damage or defects
- **Establish baseline measurements** and data collection systems

### Success Criteria

✅ **Team formed** with clear roles and responsibilities  
✅ **Problem statement** is specific, measurable, and time-bound  
✅ **Containment measures** are effective and validated  
✅ **Data collection** systems are operational  

## Key Activities

### 1. Team Formation (D1)

**Objective:** Assemble a cross-functional team with the right skills and authority.

![Team Formation Process](../img/team-formation.png)

#### Essential Team Roles

| Role | Responsibility | Skills Required |
|------|----------------|----------------|
| **Project Champion** | Overall leadership, resource allocation | Leadership, decision-making authority |
| **Technical Expert** | Process knowledge, technical analysis | Domain expertise, problem-solving |
| **Data Analyst** | Statistical analysis, measurement | Analytics, statistical tools |
| **Implementation Lead** | Solution deployment, change management | Project management, communication |

#### Team Formation Checklist

- [ ] Project champion identified and committed
- [ ] Team members have appropriate expertise
- [ ] Team has decision-making authority
- [ ] Roles and responsibilities defined
- [ ] Communication plan established
- [ ] Meeting schedule set

### 2. Problem Definition (D2)

**Objective:** Create a precise, quantified problem statement with clear scope.

![Problem Definition Framework](../img/problem-definition.png)

#### SIPOC Analysis

Use SIPOC (Suppliers, Inputs, Process, Outputs, Customers) to understand the problem context:

```python
import daspi as dsp

# Example: Creating a SIPOC analysis visualization
sipoc_data = {
    'Suppliers': ['Material suppliers', 'Equipment vendors'],
    'Inputs': ['Raw materials', 'Energy', 'Information'],
    'Process': ['Manufacturing', 'Quality control', 'Packaging'],
    'Outputs': ['Finished products', 'Waste', 'Reports'],
    'Customers': ['End users', 'Distributors', 'Retailers']
}

# Visualize SIPOC (placeholder for future implementation)
```

#### Problem Statement Template

**Problem:** [What is going wrong?]  
**Impact:** [How much? How often? Since when?]  
**Scope:** [Where does it occur? Where doesn't it occur?]  
**Goal:** [What improvement is targeted?]  
**Timeline:** [When will this be resolved?]  

#### Quantification Techniques

Use these metrics to quantify the problem:

- **DPU (Defects Per Unit):** Number of defects divided by number of units
- **FPY (First Pass Yield):** Percentage of units passing without rework
- **CpK (Process Capability):** Measure of process capability relative to specifications

```python
import daspi as dsp
import numpy as np

# Example: Calculate process metrics
data = np.random.normal(100, 5, 1000)  # Simulated process data
spec_limits = dsp.SpecLimits(lower=85, upper=115)

# Calculate process capability
estimator = dsp.ProcessEstimator(
    samples=data, 
    spec_limits=spec_limits
)

print(f"Process Mean: {estimator.mean:.2f}")
print(f"CpK: {estimator.cpk:.2f}")
print(f"DPU: {estimator.dpu:.3f}")
```

### 3. Immediate Containment (D3)

**Objective:** Implement immediate measures to prevent further defects or damage.

![Containment Strategy](../img/containment-strategy.png)

#### Containment Action Categories

1. **Detection Enhancement**
   - Increased inspection frequency
   - Additional checkpoints
   - Improved testing methods

2. **Process Modification**
   - Temporary parameter changes
   - Additional process steps
   - Enhanced monitoring

3. **Material/Product Segregation**
   - Quarantine suspect materials
   - Enhanced sorting procedures
   - Special handling protocols

#### Containment Effectiveness Validation

Containment measures must be validated through:

- **Statistical testing** of effectiveness
- **Process capability** analysis
- **Cost-benefit** evaluation

```python
# Example: Validate containment effectiveness
before_containment = np.array([...])  # Pre-containment data
after_containment = np.array([...])   # Post-containment data

# Perform hypothesis test for improvement
test_result = dsp.hypothesis_test(
    before_containment, 
    after_containment,
    test_type='two_sample'
)

if test_result.p_value < 0.05:
    print("Containment is statistically effective")
```

## Tools and Techniques

### Essential Tools for Specify Phase

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **Project Charter** | Define project scope and objectives | Start of project |
| **SIPOC Diagram** | Understand process boundaries | Problem scoping |
| **Action Plan** | Track containment activities | Implementation |
| **Eisenhower Matrix** | Prioritize urgent actions | Resource allocation |

### Data Collection Setup

Establish measurement systems early:

```python
import daspi as dsp
import pandas as pd

# Example: Set up data collection framework
measurement_plan = {
    'metric': ['defect_rate', 'cycle_time', 'customer_complaints'],
    'frequency': ['hourly', 'daily', 'weekly'],
    'target': [0.01, 120, 5],
    'responsibility': ['QC', 'Production', 'Customer Service']
}

tracking_df = pd.DataFrame(measurement_plan)
print(tracking_df)
```

## Decision Point: Is Immediate Measure Effective?

At the end of the Specify phase, evaluate containment effectiveness:

### ✅ **Yes** → Proceed to Scrutinize Phase

- Containment is validated and effective
- Problem is contained and data collection is active
- Team is ready for root cause investigation

### ❌ **No** → Revise Containment Strategy

- Review and enhance containment measures
- Consider additional actions or different approaches
- Re-validate effectiveness before proceeding

## Common Challenges and Solutions

### Challenge 1: Vague Problem Definition

**Symptoms:** Problem statement lacks specificity or measurable impact

**Solution:**

- Use the 5W2H method (Who, What, When, Where, Why, How, How Much)
- Quantify with specific metrics and timeframes
- Validate problem scope with stakeholders

### Challenge 2: Ineffective Team Composition

**Symptoms:** Missing expertise, lack of authority, or poor engagement

**Solution:**

- Review team skills matrix against problem requirements
- Ensure management support and clear authority levels
- Establish clear communication protocols and meeting rhythms

### Challenge 3: Weak Containment Actions

**Symptoms:** Continued defect occurrence, ineffective temporary measures

**Solution:**

- Implement multiple containment layers
- Use statistical validation of effectiveness
- Consider more aggressive temporary measures

## Deliverables and Outputs

### Phase Completion Checklist

- [ ] **Team Charter** - Roles, responsibilities, and communication plan
- [ ] **Problem Statement** - Quantified and scoped problem definition
- [ ] **SIPOC Analysis** - Process understanding and boundaries
- [ ] **Containment Plan** - Immediate measures with validation
- [ ] **Data Collection System** - Baseline measurements and ongoing tracking
- [ ] **Project Timeline** - Milestones and resource allocation

### Key Metrics and KPIs

Track these metrics throughout the Specify phase:

- **Containment Effectiveness:** Reduction in defect rate or problem impact
- **Team Readiness:** Completion of required training and role clarity
- **Data Quality:** Completeness and accuracy of baseline measurements
- **Stakeholder Alignment:** Agreement on problem definition and scope

## Next Steps

Once the Specify phase is complete and containment is validated as effective:

🔄 **[Proceed to Scrutinize Phase](3s-scrutinize.md)** to begin systematic root cause analysis

The solid foundation established in the Specify phase enables effective investigation and analysis in the next phase. Ensure all deliverables are complete before transitioning.

---

**Phase Complete!** You have successfully contained the immediate problem and established a strong foundation for systematic improvement. The team is ready to dive deep into root cause analysis in the Scrutinize phase.
 
 