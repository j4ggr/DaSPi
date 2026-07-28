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
    dsp.STR.language = 'de'
    dsp.STR.username = 'j4ggr'
    dsp.__version__
    return (dsp,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Line, StripeSpan, SpecLimits, COLOR

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(x = np.random.weibull(a=1.5, size=100)))
    line = Line(source=df, target='x', ax=ax)
    line(marker='o')
    spec_limits=SpecLimits(lower=0, upper=2.5)
    ok_area = StripeSpan(
        label=r'OK',
        lower_position=spec_limits.lower,
        upper_position=spec_limits.upper,
        color=COLOR.GOOD,
        orientation='horizontal')
    ok_area(ax=ax)

    ax.legend([ok_area.handle], [ok_area.label])
    return df, pd


@app.cell
def _():
    import daspi as dsp

    chart = dsp.SingleChart(
            source=dsp.load_dataset('shoe-sole'),
            target='wear',
            feature='status',
        ).plot(
            dsp.BlandAltman,
            identity='tester',
            feature_axis='mean',
            reverse=True)
    return (dsp,)


@app.cell
def _():
    import pandas as pd
    import daspi as dsp
    df = dsp.load_dataset('painkillers-dissolution')
    df.info()
    return df, dsp, pd


@app.cell
def _(df, dsp):
    # dsp.style.use('seaborn')
    chart = dsp.MultivariateChart(
            source=df,
            target='dissolution',
            feature='temperature',
            hue='employee',
            shape='water',
            col='brand',
            row='catalyst',
            stretch_figsize=False
        ).plot(
            dsp.Scatter
        ).stripes(
            mean=True,
            confidence=0.95
        ).label(
            feature_label=True,
            target_label=True
        )
    return


@app.cell
def _(df, dsp, pd):
    ALPHA = 0.05
    model = dsp.LinearModel(
        source=df,
        target='dissolution',
        features=[
            'employee',
            'stirrer',
            'brand',
            'catalyst',
            'water'],
        covariates=['temperature'],
        alpha=ALPHA,
        order=2)

    df_gof = pd.concat(model.recursive_elimination())

    # rename columns for mathematical notation
    columns_map = {
        'p_least': '$p_{least}$',
        'r2': '$r^2$',
        'r2_adj': '$r^2_{adj}$',
        'r2_pred': '$r^2_{pred}$'}
    data = df_gof.rename(columns=columns_map)

    # reshape into long format
    data = pd.melt(
        data, value_vars=columns_map.values(), var_name='metric')

    chart = dsp.SingleChart(
            source=data,
            target='value',
            hue='metric'
        ).plot(
            dsp.Line
        ).stripes(
            stripes=[dsp.StripeLine(r'\alpha', ALPHA, color='red')]
        ).label(
            fig_title='Goodness of fit',
            sub_title='Different metrics during recursive feature elimination',
            target_label=True,
            feature_label='elimination step'
        )
    _ = chart.axes[0, 0].set(xlim=(1, len(df_gof)), ylim=(0, 1))
    return (model,)


@app.cell
def _(dsp, model):
    dsp.ResidualsCharts(model).plot().stripes().label()
    dsp.ParameterRelevanceCharts(model).plot().label()
    model
    return


@app.cell
def _(df, dsp):
    chart = dsp.MultivariateChart(
            source=df,
            target='dissolution',
            feature='water',
            col='brand',
            hue='employee',
            dodge=True,
        ).plot(
            dsp.Jitter
        ).plot(
            dsp.SpreadWidth,
            agreement=6,
            strategy='norm',
            bars_same_color=True
        ).stripes(
            mean=True,
            confidence=0.95
        ).label(
            fig_title='Painkillers Dissolution Process Analysis',
            sub_title='Expected spread (6σ) based on the normal distribution',
            feature_label=True,
            target_label='Dissolution time (s)'
        )
    return


@app.cell
def _(df, dsp):
    chart = dsp.JointChart(
            source=df,
            target='dissolution',
            feature=('water', 'employee'),
            hue=('brand', ''),
            ncols=2,
            nrows=1,
            sharey=True,
            dodge=True,
        )
    for i in range(chart.n_axes):
        chart.plot(
            dsp.CenterLocation,
            show_center=False,
        ).plot(
            dsp.MeanTest,
            n_groups=1,
            show_center=False,
            on_last_axes=True
        ).plot(
            dsp.Beeswarm,
            on_last_axes=True
        )

    chart.label(
        fig_title='Painkillers Dissolution Process Analysis',
        sub_title='Expected mean dissolution time for each combination of water and employee',
        feature_label=('water type', 'employee'),
        target_label='Dissolution time (s)',
        info=True
    )

    return


if __name__ == "__main__":
    app.run()
