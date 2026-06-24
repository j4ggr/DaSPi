# DaSPi — Process Analytics & Six Sigma in Python

**DaSPi helps engineers analyze and improve processes using statistical workflows.**

---

## 🎯 The Problem

Process analysis in practice is fragmented:

- **Excel** is error-prone and hard to scale  
- **Minitab / JMP** are expensive and closed  
- **Python tools** (pandas, scipy, statsmodels) are powerful but disconnected  

👉 Engineers spend more time combining tools than improving processes.

---

## ✅ The Solution

DaSPi provides **integrated workflows for process analytics**:

- ✅ **Capability analysis** (Cp, Cpk)  
- ✅ **Root cause analysis** (ANOVA, regression)  
- ✅ **Statistical process control** (SPC)  
- ✅ **Professional visualization**  

All in one consistent and intuitive interface.

---

## 🚀 Three Flagship Workflows

DaSPi provides **three ready-to-use workflows** that cover the most common process analysis tasks. Each workflow produces **visual output + interpretation** in under 20 lines of code.

---

### 📊 Workflow 1: Process Capability Analysis

**Evaluate if your process meets specifications.**

```python
import daspi as dsp

# Load data
df = dsp.load_dataset("drop_card")
spec_limits = dsp.SpecLimits(0, float(df.loc[0, "usl"]))

# Analyze capability
chart = dsp.ProcessCapabilityAnalysisCharts(
    source=df,
    target="distance",
    spec_limits=spec_limits,
    hue="method"
).plot().stripes().label(info=True)

chart.show()
```

**Output:** 5-panel analysis with Cp, Cpk, Pp, Ppk, distribution plots, and statistical interpretation.

---

### 🔍 Workflow 2: Root Cause Analysis

**Identify which factors significantly impact your process.**

```python
import daspi as dsp

# Load data
df = dsp.load_dataset("painkillers-dissolution")

# Fit model with automatic factor selection
model = dsp.LinearModel(
    source=df,
    target="dissolution",
    factors=["employee", "brand", "catalyst"],
    covariates=["temperature"]
)
model.recursive_elimination()

# Visualize results
dsp.ResidualsCharts(model).plot().stripes().label(info=True)
dsp.ParameterRelevanceCharts(model).plot().stripes().label(info=True)
```

**Output:** Residual diagnostics + parameter effects with ANOVA tables and significance tests.

---

### 📈 Workflow 3: Statistical Process Control (SPC)

**Monitor process stability and detect out-of-control conditions.**

```python
import daspi as dsp

# Load process data
df = dsp.load_dataset("grnr_spc")

# Create control chart
chart = dsp.SingleChart(
    source=df,
    target="layer_thickness",
    feature="sample"
).plot(
    dsp.Scatter
).stripes(
    mean=True,
    control_limits=True,  # UCL/LCL at 3-sigma
    spec_limits=dsp.SpecLimits(lower=45, upper=55),
    agreement=3
).label(
    fig_title="SPC Chart: Layer Thickness",
    sub_title="Control limits at ±3σ",
    info=True
)

chart.show()
```

**Output:** Control chart with mean, control limits (UCL/LCL), specification limits, and trend analysis.

---

## 🏭 Use Cases

- **Manufacturing:** Monitor tolerances and reduce defects  
- **Quality Engineering:** Automate Six Sigma workflows (DMAIC)  
- **Process Optimization:** Identify key drivers of variation  
- **Data Analysts:** Unify statistics and visualization in one tool  

---

## 🚀 Getting Started

### Installation

```bash
pip install daspi
```

### Next Steps

- 📖 [User Guide](guides/index.md) — Complete tutorials for each workflow  
- 🔧 [API Reference](anova/index.md) — Detailed documentation  
- 📊 [3S Methodology](guides/3s-methodology.md) — Structured problem-solving framework  

---

## 💡 Why DaSPi?

### Integrated Workflows
No more juggling multiple tools. Each workflow combines data loading, analysis, visualization, and interpretation in one seamless interface.

### Professional Output
Publication-ready charts with automatic formatting, legends, and statistical annotations.

### Six Sigma Ready
Built-in support for Cp/Cpk, ANOVA, DOE, Gage R&R, and control charts — everything you need for DMAIC projects

---

## 🔧 Technical Features

- **Multivariate visualization** — Explore complex relationships  
- **Linear models & ANOVA** — Statistical inference made simple  
- **Hypothesis testing** — Confidence intervals and p-values  
- **Monte Carlo simulation** — Assess uncertainty  
- **Process capability** — Cp, Cpk, Pp, Ppk calculations

---

## ⚙️ Built on Proven Libraries

DaSPi leverages the Python scientific stack:

- **pandas** — Data manipulation  
- **numpy** — Numerical computing  
- **matplotlib** — Visualization  
- **scipy** — Statistical functions  
- **statsmodels** — Advanced statistics  

---

## 🤝 Feedback & Contributions

**If you use DaSPi in real-world process analysis:**  
👉 We would love to hear your use case.

Feedback, ideas, and contributions are very welcome.

---

::: daspi
    options:
        members: no
