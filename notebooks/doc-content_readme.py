import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import os
    import sys
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    if not str(Path.cwd()).endswith('DaSPi'):
        os.chdir(Path.cwd().parent.resolve())
    sys.path.insert(0, Path.cwd().as_posix())

    import daspi as dsp
    DPI = 120
    dsp.CONFIG.username = 'j4ggr'
    dsp.__version__
    return DPI, dsp, np, pd, plt


@app.cell
def _(dsp):
    df_iris = dsp.load_dataset('iris')
    df_dc = dsp.load_dataset('drop_card')
    df_pkd = dsp.load_dataset('painkillers-dissolution')
    df_grnr = dsp.load_dataset('grnr_layer_thickness')
    df_spc = dsp.load_dataset('grnr_spc')
    return df_dc, df_grnr, df_pkd, df_spc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 📏 Workflow 1: Gage R&R Analysis
    """)
    return


@app.cell
def _(DPI, df_grnr, dsp):
    # Step 1: Evaluate the gage itself (MSA Type 1)
    gage = dsp.GageStudyModel(
        source=df_grnr,
        target='result_gage',
        reference='reference',
        u_cal=df_grnr['U_cal'][0],
        tolerance=df_grnr['tolerance'][0],
        resolution=df_grnr['resolution'][0]
    )

    # Step 2: Evaluate repeatability & reproducibility (MSA Type 2)
    _model = dsp.GageRnRModel(
        source=df_grnr,
        target='result_rnr',
        part='part',
        gage=gage,
        u_av='operator'  # Operator variation
    )

    chart_gage = dsp.GageStudyCharts(
            gage,
            stretch_figsize=1.5
        ).plot(
        ).stripes(
        ).label(
            fig_title='Gage Study Layer Thickness Measurement System',
            sub_title='Measurement System Analysis (MSA Type 1)',
            info=True
        ).save(
            './docs/img/gage_study_single_reference.png',
            dpi=DPI)

    # Visualize complete analysis
    chart_rnr = dsp.GageRnRCharts(
            _model,
            spread_accepted_limit=0.1,   # 10% threshold for acceptance
            spread_rejected_limit=0.3,   # 30% threshold for rejection
            u_accepted_limit=0.15,       # 15% uncertainty threshold
            stretch_figsize=1.5
        ).plot(
        ).stripes(  # Adds acceptance zones
        ).label(
            fig_title='Gage R&R Layer Thickness Measurement System',
            sub_title='Measurement System Analysis (MSA Type 2)',
            info=True
        ).save('docs/img/gage_rnr_layer_thickness.png')

    return chart_gage, chart_rnr


@app.cell
def _(chart_gage):
    chart_gage
    return


@app.cell
def _(chart_rnr):
    chart_rnr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 📊 Workflow 2: Process Capability Analysis
    """)
    return


@app.cell
def _(DPI, df_dc, dsp):
    _spec_limits = dsp.SpecLimits(0, float(df_dc.loc[0, 'usl']))

    # Analyze capability
    _chart = dsp.ProcessCapabilityAnalysisCharts(
        source=df_dc,
        target='distance',
        spec_limits=_spec_limits,
        hue='method'
    ).plot().stripes().label(
        fig_title='Process Capability Analysis of Drop Card Data',
        sub_title='Drop Card Distance Comparison between Two Methods',
        target_label='Distance s (cm)',
        info=True)

    _chart.save('./docs/img/cpk-analysis_drop-card.png', dpi=DPI)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🔍 Workflow 3: Root Cause Analysis
    """)
    return


@app.cell
def _(DPI, df_pkd, dsp, pd):
    model = dsp.LinearModel(
        source=df_pkd,
        target='dissolution',
        factors=['employee', 'brand', 'water', 'catalyst'],
        covariates=['temperature'],
        order=2,
    )
    df_gof = pd.concat(model.recursive_elimination())
    chart_res = dsp.ResidualsCharts(model).plot().stripes().label(info=True)
    chart_param = dsp.ParameterRelevanceCharts(model).plot().stripes().label(info=True)

    chart_res.save('./docs/img/anova_dissolution_residues.png', dpi=DPI)
    chart_param.save('./docs/img/anova_dissolution_params.png', dpi=DPI)
    model
    return chart_param, chart_res


@app.cell
def _(chart_res):
    chart_res
    return


@app.cell
def _(chart_param):
    chart_param
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 📈 Workflow 4: Statistical Process Control (SPC)
    """)
    return


@app.cell
def _(DPI, df_spc, dsp):
    _chart = dsp.JointChart(
            source=df_spc,
            target='result',
            feature=('measurement_order', ''),
            nrows=1,
            ncols=2,
            width_ratios=[4, 1],
            sharey=True,
        ).plot(
            dsp.Scatter
        ).plot(
            dsp.Line,
            on_last_axes=True,
        ).plot(
            dsp.GaussianKDE,
            hide_axis='feature',
            visible_spines='target',
        ).stripes(
            mean=True,
            control_limits=True,    # UCL/LCL at 3-sigma
            agreement=3,            # 3-sigma agreement lines
            strategy='norm',        # Use normal distribution for control limits
        ).label(
            fig_title='SPC Chart: Layer Thickness',
            sub_title='Control limits at ±3σ',
            target_label='Layer Thickness (µm)',
            feature_label=('Measurement Order', ''),
            info=True
        ).save(
            './docs/img/spc_chart_layer_thickness.png',
            dpi=DPI
        )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logo
    """)
    return


@app.cell
def _(dsp, np, plt):
    plt.rcParams.update({
        'figure.facecolor': (1., 1., 1., 1.),
        'axes.facecolor': (1., 1., 1., 1.),
        'savefig.facecolor': (1., 1., 1., 1.),
    })

    PHI = (1 + 5**0.5) / 2
    SHORT = 1 / (1 + PHI)
    LONG = PHI / (1 + PHI)

    edges = np.array([
        (SHORT, 1+SHORT),
        (0, SHORT),
        (1, SHORT),
        (SHORT, 1),
        (1+SHORT, 1),
        (1, 0)])

    color = dsp.COLOR.PALETTE[47]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(
        edges[:, 0],
        edges[:, 1],
        linewidth=25,
        color=color,
        solid_capstyle='butt',
        solid_joinstyle='bevel')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.margins(0.1, 0.1)
    fig.savefig(f'./docs/img/logo_base.svg', transparent=True)
    print(dsp.COLOR.PALETTE[47], dsp.COLOR.PALETTE[43])
    return


@app.cell
def _(dsp, pd):
    df = dsp.load_dataset('drop_card')
    spec_limits = dsp.SpecLimits(0, float(df.loc[0, 'usl']))
    target = 'distance'

    chart = dsp.ProcessCapabilityAnalysisCharts(
            source=df,
            target=target,
            spec_limits=spec_limits,
            hue='method'
        ).plot(
        ).stripes(
        ).label(
            fig_title='Process Capability Analysis',
            sub_title='Drop Card Experiment',
            target_label='Distance (cm)',
            info=True
        )

    samples_parallel = df[df['method']=='parallel'][target]
    samples_series = df[df['method']=='perpendicular'][target]
    df_e = pd.concat([
        dsp.ProcessEstimator(samples_parallel, spec_limits).describe(),
        dsp.ProcessEstimator(samples_series, spec_limits).describe()],
        axis=1,
        ignore_index=True,
    ).rename(
        columns={0: 'parallel', 1: 'perpendicular'}
    )
    print(df_e)
    return


if __name__ == "__main__":
    app.run()
