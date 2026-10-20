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
    dsp.CONFIG.configure(language='de', username='j4ggr')
    dsp.__version__
    return (dsp,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Line, StripeSpan, SpecLimits, COLOR

    _fig, _ax = plt.subplots()
    _df = pd.DataFrame(dict(x = np.random.weibull(a=1.5, size=100)))
    _line = Line(source=_df, target='x', ax=_ax)
    _line(marker='o')
    _spec_limits=SpecLimits(lower=0, upper=2.5)
    _ok_area = StripeSpan(
        label=r'OK',
        lower_position=_spec_limits.lower,
        upper_position=_spec_limits.upper,
        color=COLOR.GOOD,
        orientation='horizontal')
    _ok_area(ax=_ax)

    _ax.legend([_ok_area.handle], [_ok_area.label])
    return (pd,)


@app.cell
def _(dsp):
    _chart = dsp.SingleChart(
            source=dsp.load_dataset('shoe-sole'),
            target='wear',
            feature='status',
        ).plot(
            dsp.BlandAltman,
            identity='tester',
            feature_axis='mean',
            reverse=True)
    return


@app.cell
def _(dsp, pd):
    df = dsp.load_dataset('painkillers-dissolution')
    df.info()
    return (df,)


@app.cell
def _(df, dsp):
    # dsp.style.use('seaborn')
    _chart = dsp.MultivariateChart(
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
        factors=[
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

    _chart = dsp.SingleChart(
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
    _ = _chart.axes[0, 0].set(xlim=(1, len(df_gof)), ylim=(0, 1))
    return (model,)


@app.cell
def _(dsp, model):
    dsp.ResidualsCharts(model).plot().stripes().label()
    dsp.ParameterRelevanceCharts(model).plot().label()
    model
    return


@app.cell
def _(df, dsp):
    _chart = dsp.MultivariateChart(
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
    _chart = dsp.JointChart(
            source=df,
            target='dissolution',
            feature=('water', 'employee'),
            hue=('brand', ''),
            ncols=2,
            nrows=1,
            sharey=True,
            dodge=True,
        )
    for _i in range(_chart.n_axes):
        _chart.plot(
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

    _chart.label(
        fig_title='Painkillers Dissolution Process Analysis',
        sub_title='Expected mean dissolution time for each combination of water and employee',
        feature_label=('water type', 'employee'),
        target_label='Dissolution time (s)',
        info=True
    )

    return


if __name__ == "__main__":
    app.run()
