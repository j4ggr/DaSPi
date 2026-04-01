# User Guide

This section covers hands-on usage of DaSPi — from installation through
data visualization, statistical analysis, and measurement system
evaluation.

---

## Getting Started

| Guide | Description |
| ----- | ----------- |
| [Installing](installing.md) | Install DaSPi from PyPI and verify the setup |
| [Plotting](plotting.md) | Build charts with `SingleChart`, `MultivariateChart`, and the `Facets` helpers |
| [ANOVA](anova.md) | Fit linear models, run ANOVA, and automate backward elimination |
| [DOE](doe.md) | Design full- and fractional-factorial experiments |
| [Hypothesis Testing](hypothesis-testing.md) | Normality, variance, location, and proportion tests |
| [Gage Analysis](gage_analysis.md) | MSA Type 1 gage studies and Gage R&R |

---

## 3S Methodology

The **3S Methodology** is a streamlined, three-phase problem-solving
framework that combines the best elements from 8D and Six Sigma DMAIC.

| Phase | Focus |
| ----- | ----- |
| [Overview](3s-methodology.md) | Introduction, comparison with DMAIC / 8D |
| Specify | Define & contain — team, charter, SIPOC, containment |
| Scrutinize | Investigate & analyse — root cause, DOE, hypothesis tests |
| Stabilize | Implement & control — solution validation, SPC, knowledge transfer |

!!! note "Phase guides coming soon"
    Detailed step-by-step guidance for the Specify, Scrutinize, and
    Stabilize phases is in preparation.

---

## Statistical Analysis Workflow

A typical DaSPi analysis follows these steps:

1. **Load data** — `dsp.load_dataset(name)` or read your own CSV/Excel.
2. **Explore visually** — use `SingleChart` or `MultivariateChart` with
   `Scatter`, `GaussianKDE`, or `QuantileBoxes`.
3. **Check assumptions** — `anderson_darling_test`, `variance_test`.
4. **Test hypotheses** — `position_test`, `proportions_test`.
5. **Fit a model** — `LinearModel` with optional backward elimination.
6. **Validate residuals** — `ResidualsCharts(model).plot()`.
7. **Interpret** — `ParameterRelevanceCharts`, `model.anova()`,
   `model.gof_metrics()`.
8. **Assess capability** — `ProcessCapabilityAnalysisCharts` with
   `SpecLimits`.

---

## Data Visualization Overview

DaSPi's plotting system is built in layers:

```
AxesFacets          ← subplot grid (rows × cols or mosaic)
  └─ Chart          ← data wiring (source, target, hue, shape, size)
       ├─ Plotter   ← mark drawing (Scatter, Line, GaussianKDE, …)
       └─ Facets    ← labels, legend, reference stripes
```

See the [Plotting Guide](plotting.md) for examples at every layer.
