![pypi](https://badge.fury.io/py/daspi.svg)
![licence](https://img.shields.io/github/license/j4ggr/daspi.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/j4ggr/daspi)
![downloads](https://img.shields.io/pypi/dm/daspi)

![logo](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/logo.svg)

# DaSPi — Process Analytics & Six Sigma in Python

DaSPi helps engineers analyze and improve processes using statistical workflows.

## 🎯 The Problem

Process analysis in practice is fragmented:

- Excel is error-prone and hard to scale  
- Minitab / JMP are expensive and closed  
- Python tools (pandas, scipy, statsmodels) are powerful but disconnected  

👉 Engineers spend more time combining tools than improving processes.

## ✅ The Solution

DaSPi provides **integrated workflows for process analytics**:

- Capability analysis (Cp, Cpk)  
- Root cause analysis (ANOVA, regression)  
- Statistical process control (SPC)  
- Professional visualization  

All in one consistent and intuitive interface.

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

## 🏭 Use Cases

- **Manufacturing:** Monitor tolerances and reduce defects  
- **Quality Engineering:** Automate Six Sigma workflows  
- **Process Optimization:** Identify key drivers of variation  
- **Data Analysts:** Unify statistics and visualization  

### 🚀 What You Can Do with DaSPi

- ✅ Run process capability analysis in seconds  
- ✅ Identify root causes using ANOVA and regression  
- ✅ Generate publication-ready charts automatically  
- ✅ Combine statistics and visualization seamlessly

### 📊 Example Outputs

#### Visualization

[![Visualization](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/iris_contour_size-leaf-species.png)](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/iris_contour_size-leaf-species.png)

#### ANOVA Analysis
[![ANOVA Analysis 1](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/anova_dissolution_residues.png)](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/anova_dissolution_residues.png)
[![ANOVA Analysis 2](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/anova_dissolution_params.png)](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/anova_dissolution_params.png)

#### Process Capability

[![Process Capability](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/cpk-analysis_drop-card.png)](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/cpk-analysis_drop-card.png)

## 🚀 Installation

```bash
pip install daspi
```

## 🧭 Core Workflows

### 📊 Process Capability Analysis
Evaluate variation and performance vs specification limits

### 🔍 Root Cause Analysis
Identify influencing factors using regression and ANOVA

### 📈 Statistical Process Control (SPC)
Monitor process stability with control charts

## 🔧 Technical Features

- **Multivariate visualization** — Explore complex relationships  
- **Linear models & ANOVA** — Statistical inference made simple  
- **Hypothesis testing** — Confidence intervals and p-values  
- **Monte Carlo simulation** — Assess uncertainty  
- **Process capability** — Cp, Cpk, Pp, Ppk calculations

## ⚙️ Built on Proven Libraries

DaSPi leverages the Python scientific stack:

- **pandas** — Data manipulation  
- **numpy** — Numerical computing  
- **matplotlib** — Visualization  
- **scipy** — Statistical functions  
- **statsmodels** — Advanced statistics  

## 👤 About

DaSPi is created and maintained by **Reto Jäggli**, Data Scientist at Festo Microtechnology AG.

The project is driven by a passion to make **process analytics and Six Sigma workflows more accessible in Python**.

## ⚠️ Disclaimer

DaSPi is under active development and may contain bugs.  
Results should be validated with trusted statistical tools when required.

## 🤝 Feedback & Contributions

**If you use DaSPi in real-world process analysis:**  
👉 I would love to hear your use case.

Feedback, ideas, and contributions are very welcome.

---
