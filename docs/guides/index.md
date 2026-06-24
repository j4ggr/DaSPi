# User Guide

This section covers hands-on usage of DaSPi — from installation through
process analysis workflows, statistical methods, and measurement system
evaluation.

---

## 🚀 Getting Started

| Guide | What You'll Learn |
| ----- | ----------------- |
| [Installing](installing.md) | Install DaSPi from PyPI and verify your setup |
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
