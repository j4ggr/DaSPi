"""Generate images for the three flagship workflow guides."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # DaSPi Workflow Image Generator

    This notebook generates example images for the three flagship workflows:
    1. **Process Capability Analysis** (Cp/Cpk)
    2. **Root Cause Analysis** (ANOVA/Regression)
    3. **Statistical Process Control** (SPC Charts)

    All images are saved to `docs/img/` directory.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import os

    from pathlib import Path
    if not str(Path.cwd()).endswith('DaSPi'):
        os.chdir(Path.cwd().parent.resolve())
    os.sys.path.append(str(Path.cwd().resolve()))

    import daspi as dsp
    # Create output directory if needed
    os.makedirs("../docs/img", exist_ok=True)

    dsp.CONFIG.username = 'j4ggr'
    return dsp, mo, plt


@app.cell
def _(mo):
    mo.md("""
    ## Workflow 1: Process Capability Analysis
    """)
    return


@app.cell
def _(dsp, plt):
    # Load data
    df_capability = dsp.load_dataset("drop_card")
    spec_limits_capability = dsp.SpecLimits(0, float(df_capability.loc[0, "usl"]))

    # Generate capability analysis chart
    chart_capability = dsp.ProcessCapabilityAnalysisCharts(
        source=df_capability,
        target="distance",
        spec_limits=spec_limits_capability,
        hue="method"
    ).plot().stripes().label(
        fig_title="Process Capability Analysis Example",
        sub_title="Drop Card Distance by Method",
        info=True
    )

    # Save image
    chart_capability.save("../docs/img/workflow-capability-example.png", dpi=150)
    plt.close('all')

    chart_capability
    return


@app.cell
def _(mo):
    mo.md("""
    ✓ Saved: `docs/img/workflow-capability-example.png`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Workflow 2: Root Cause Analysis
    """)
    return


@app.cell
def _(dsp):
    # Load data and fit model
    df_root_cause = dsp.load_dataset("painkillers-dissolution")

    model = dsp.LinearModel(
        source=df_root_cause,
        target="dissolution",
        factors=["employee", "brand", "catalyst"],
        covariates=["temperature"]
    )
    model.recursive_elimination()
    return (model,)


@app.cell
def _(dsp, model, plt):
    # Generate residuals diagnostic chart
    residuals = dsp.ResidualsCharts(model).plot().stripes().label(
        fig_title="Root Cause Analysis: Residual Diagnostics",
        sub_title="Painkillers Dissolution Time",
        info=True
    )
    residuals.save("../docs/img/workflow-root-cause-residuals.png", dpi=150)
    plt.close('all')

    residuals
    return


@app.cell
def _(mo):
    mo.md("""
    ✓ Saved: `docs/img/workflow-root-cause-residuals.png`
    """)
    return


@app.cell
def _(dsp, model, plt):
    # Generate parameter relevance chart
    parameters = dsp.ParameterRelevanceCharts(model).plot().stripes().label(
        fig_title="Root Cause Analysis: Parameter Effects",
        sub_title="Painkillers Dissolution Time",
        info=True
    )
    parameters.save("../docs/img/workflow-root-cause-parameters.png", dpi=150)
    plt.close('all')

    parameters
    return


@app.cell
def _(mo):
    mo.md("""
    ✓ Saved: `docs/img/workflow-root-cause-parameters.png`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Workflow 3: Statistical Process Control (SPC)
    """)
    return


@app.cell
def _(dsp, plt):
    # Load SPC data
    df_spc = dsp.load_dataset("grnr_spc")

    # Generate SPC control chart
    chart_spc = dsp.SingleChart(
        source=df_spc,
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
        fig_title="Statistical Process Control Example",
        sub_title="Process Monitoring (3σ control limits)",
        feature_label="Measurement Order",
        target_label="Measurement Result",
        info=True
    )

    chart_spc.save("../docs/img/workflow-spc-example.png", dpi=150)
    plt.close('all')

    chart_spc
    return


@app.cell
def _(mo):
    mo.md("""
    ✓ Saved: `docs/img/workflow-spc-example.png`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ✅ Summary

    All workflow images generated successfully!

    **Generated files:**
    - `docs/img/workflow-capability-example.png`
    - `docs/img/workflow-root-cause-residuals.png`
    - `docs/img/workflow-root-cause-parameters.png`
    - `docs/img/workflow-spc-example.png`
    """)
    return


if __name__ == "__main__":
    app.run()
