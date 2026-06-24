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

## ⚡ Quick Example: Process Capability in 5 Lines

```python
import daspi as dsp

df = dsp.load_dataset("drop_card")
spec_limits = dsp.SpecLimits(0, float(df.loc[0, "usl"]))

chart = dsp.ProcessCapabilityAnalysisCharts(
    source=df,
    target="distance",
    spec_limits=spec_limits,
    hue="method"
).plot().label(info=True)

chart.show()
```

**Output includes:**

- ✅ Cp / Cpk metrics  
- ✅ Distribution plots  
- ✅ Interpretation-ready visuals

---

## 🏭 Use Cases

- **Manufacturing:** Monitor tolerances and reduce defects  
- **Quality Engineering:** Automate Six Sigma workflows  
- **Process Optimization:** Identify key drivers of variation  
- **Data Analysts:** Unify statistics and visualization  

---

## 🚀 Getting Started

### Installation

```bash
pip install daspi
```

### Next Steps

- 📖 [User Guide](guides/index.md) — Hands-on tutorials and workflows  
- 🔧 [API Reference](anova/index.md) — Complete module documentation  
- 📊 [3S Methodology](guides/3s-methodology.md) — Structured problem-solving framework  

---

## 🧭 Core Workflows

### 📊 Process Capability Analysis
Evaluate variation and performance vs specification limits

### 🔍 Root Cause Analysis
Identify influencing factors using regression and ANOVA

### 📈 Statistical Process Control (SPC)
Monitor process stability with control charts

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
