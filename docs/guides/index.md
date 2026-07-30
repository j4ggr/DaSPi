# User Guide

This guide shows you how to use DaSPi's **three flagship workflows** for process analysis, from installation through real-world applications.

---

## 🎯 Start Here: Three Essential Workflows

These workflows cover 90% of process analysis tasks. Each produces **visual + interpretation** in under 20 lines.

### 📊 1. Process Capability Analysis

**When to use:** Evaluate if your process meets specifications (Cp, Cpk analysis).

```python
import daspi as dsp

df = dsp.load_dataset("drop_card")
spec_limits = dsp.SpecLimits(0, float(df.loc[0, "usl"]))

chart = dsp.ProcessCapabilityAnalysisCharts(
    source=df,
    target="distance",
    spec_limits=spec_limits,
    hue="method"
).plot().stripes().label(
    fig_title="Process Capability Analysis of Drop Card Data",
    sub_title="Comparison of two methods",
    info=True)
```

**Output:** Distribution analysis, Cp/Cpk/Pp/Ppk indices, capability interpretation.

**[📖 Complete Capability Guide →](workflow-capability.md)**

---

### 🔍 2. Root Cause Analysis

**When to use:** Identify which factors significantly impact your process.

```python
import daspi as dsp

df = dsp.load_dataset("painkillers-dissolution")

model = dsp.LinearModel(
    source=df,
    target="dissolution",
    factors=["employee", "brand", "catalyst"],
    covariates=["temperature"]
)
model.recursive_elimination()

dsp.ResidualsCharts(model).plot().stripes().label(info=True)
dsp.ParameterRelevanceCharts(model).plot().stripes().label(info=True)
```

**Output:** ANOVA tables, parameter effects, residual diagnostics, significance tests.

**[📖 Complete Root Cause Guide →](workflow-root-cause.md)**

---

### 📈 3. Statistical Process Control (SPC)

**When to use:** Monitor process stability and detect out-of-control conditions.

```python
import daspi as dsp

df = dsp.load_dataset("grnr_spc")

chart = dsp.SingleChart(
    source=df,
    target="result",
    feature="measurement_order"
).plot(
    dsp.Scatter
).stripes(
    mean=True,
    control_limits=True,
    spec_limits=dsp.SpecLimits(lower=2.0, upper=4.5),
    agreement=3
).label(
    fig_title="SPC Chart: Process Monitoring",
    info=True
)
```

**Output:** Control chart with mean, UCL/LCL, specification limits, trend analysis.

**[📖 Complete SPC Guide →](workflow-spc.md)**

---

## 🚀 Getting Started

### Step 1: Install DaSPi
```bash
pip install daspi
```

### Step 2: Choose Your Workflow
- **Need to verify process capability?** → Start with [Workflow 1](workflow-capability.md)
- **Need to find root causes?** → Start with [Workflow 2](workflow-root-cause.md)
- **Need to monitor stability?** → Start with [Workflow 3](workflow-spc.md)

### Step 3: Explore Advanced Topics
Once you master the three workflows, explore:
- [Design of Experiments (DOE)](doe.md) for systematic testing
- [Gage R&R Analysis](gage_analysis.md) for measurement system validation
- [3S Methodology](3s-methodology.md) for structured problem-solving

---

## 🚀 Getting Started

| Guide | What You'll Learn |
| ----- | ----------------- |
| [Installing](installing.md) | Install DaSPi from PyPI and verify your setup |
| [Configuration](configuration.md) | Configure language, username, and plotting styles |
| [Plotting](plotting.md) | Create professional charts for process data visualization |

---

## 📊 Process Analysis Workflows

| Guide | What You'll Learn |
| ----- | ----------------- |
| [ANOVA](anova.md) | Fit linear models, run ANOVA, and identify key factors automatically |
| [DOE](doe.md) | Design efficient experiments (full & fractional factorial) |
| [Hypothesis Testing](hypothesis-testing.md) | Test normality, variance, location, and proportions |
| [Gage Analysis](gage_analysis.md) | Evaluate measurement systems (MSA Type 1, Gage R&R) |

---

## 🧭 3S Methodology

The **3S Methodology** is a streamlined, three-phase problem-solving
framework for process improvement that combines best practices from
8D and Six Sigma DMAIC.

| Phase | Focus |
| ----- | ----- |
| [Overview](3s-methodology.md) | Introduction, comparison with DMAIC / 8D |
| **Specify** | Define & contain — team, charter, SIPOC, containment |
| **Scrutinize** | Investigate & analyze — root cause, DOE, hypothesis tests |
| **Stabilize** | Implement & control — solution validation, SPC, knowledge transfer |

!!! note "Phase guides coming soon"
    Detailed step-by-step guidance for the Specify, Scrutinize, and
    Stabilize phases is in preparation.

---

## 🔄 Typical Analysis Workflow

A process analysis with DaSPi typically follows these steps:

### 1. Load Data
```python
df = dsp.load_dataset("drop_card")  # or read your own CSV/Excel
```

### 2. Explore Visually
Use `SingleChart` or `MultivariateChart` with plotters like `Scatter`, 
`GaussianKDE`, or `QuantileBoxes` to understand your data.

### 3. Check Assumptions
Run `anderson_darling_test` and `variance_test` to verify statistical
prerequisites.

### 4. Test Hypotheses
Apply `position_test` or `proportions_test` to compare groups or conditions.

### 5. Fit a Model
Build a `LinearModel` with optional backward elimination to identify
significant factors.

### 6. Validate Residuals
Use `ResidualsCharts(model).plot()` to check model assumptions.

### 7. Interpret Results
Analyze with `ParameterRelevanceCharts`, `model.anova()`, and 
`model.gof_metrics()`.

### 8. Assess Capability
Evaluate process performance with `ProcessCapabilityAnalysisCharts` 
and `SpecLimits`.

---

## 🎨 Data Visualization Architecture

DaSPi's plotting system is built in layers for maximum flexibility:

```
AxesFacets          ← subplot grid (rows × cols or mosaic layout)
  └─ Chart          ← data wiring (source, target, hue, shape, size)
       ├─ Plotter   ← mark drawing (Scatter, Line, GaussianKDE, …)
       └─ Facets    ← labels, legend, reference stripes
```

This layered approach lets you create simple charts quickly while
maintaining the flexibility to customize every detail when needed.

See the [Plotting Guide](plotting.md) for examples at every layer.
