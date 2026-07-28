import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import os
    import sys
    from pathlib import Path
    if not str(Path.cwd()).endswith('DaSPi'):
        os.chdir(Path.cwd().parent.resolve())
    sys.path.insert(0, Path.cwd().as_posix())

    import daspi as dsp
    DPI = 120
    dsp.CONFIG.username = 'j4ggr'
    dsp.__version__
    return (DPI,)


@app.cell
def _(DPI):
    import daspi as dsp
    df = dsp.load_dataset('iris')

    chart = dsp.MultivariateChart(
            source=df,
            target='length',
            feature='width',
            hue='species',
            col='leaf',
            markers=('x',)
        ).plot(
            dsp.GaussianKDEContour
        ).plot(
            dsp.Scatter
        ).label(
            fig_title='Iris dataset',
            sub_title='Contours',
            feature_label='leaf width (cm)',
            target_label='leaf length (cm)',
            info=True
        )

    chart.save('./docs/img/iris_contour_size-leaf-species.png', dpi=DPI)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logo
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import daspi as dsp

    plt.rcParams.update({
        "figure.facecolor": (1., 1., 1., 1.),
        "axes.facecolor": (1., 1., 1., 1.),
        "savefig.facecolor": (1., 1., 1., 1.),
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
def _():
    import daspi as dsp
    import pandas as pd

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
