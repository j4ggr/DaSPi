import marimo

__generated_with = "0.23.9"
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
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from math import ceil
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    DPI = 120
    KW_PLOT = dict(visible_spines='none', hide_axis='both')
    BASE_FIGSIZE = 2
    N_COLS = 3
    INFO = f'DaSPi v{dsp.__version__}'
    img_dir = './docs/img/'
    dsp.CONFIG.username = 'j4ggr'
    INFO
    return (
        BASE_FIGSIZE,
        DPI,
        KW_PLOT,
        Line2D,
        N_COLS,
        Patch,
        ceil,
        dsp,
        img_dir,
        pd,
        plt,
    )


@app.cell
def _(dsp):
    df_pk = dsp.load_dataset('painkillers-dissolution')
    df_iris = dsp.load_dataset('iris')
    df_card = dsp.load_dataset('drop_card')
    return df_card, df_iris, df_pk


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AxesFacets
    """)
    return


@app.cell
def _(DPI, dsp, img_dir):
    _axes = dsp.AxesFacets(mosaic=[
        'aaa.',
        'bbbc',
        'bbbc',
        'bbbc'])
    _axes.figure.savefig(img_dir+'facets_axes-mosaic.png', bbox_inches='tight', dpi=DPI)
    _axes.figure
    return


@app.cell
def _(dsp):
    _axes = dsp.AxesFacets(
        mosaic=['a.', 'bc'], width_ratios=[3, 1], height_ratios=[1, 3])
    _axes.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### StripesFacets
    """)
    return


@app.cell
def _(DPI, df_pk, img_dir, plt):
    _fig, _axes = plt.subplots(
        nrows=1, ncols=df_pk['employee'].nunique(), sharex=True, sharey=True)

    for _ax, (_name, _group) in zip(_axes, df_pk.groupby('employee')):
        _ax.scatter(_group['temperature'], _group['dissolution'])
        _ax.set_title(str(_name))

    _fig.savefig(img_dir+'facets_stripes-missing.png', bbox_inches='tight', dpi=DPI)
    _fig
    return


@app.cell
def _(DPI, df_pk, dsp, img_dir, plt):
    _fig, _axes = plt.subplots(
        nrows=1, ncols=df_pk['employee'].nunique(), sharex=True, sharey=True)

    for _ax, (_name, _group) in zip(_axes, df_pk.groupby('employee')):
        _stripes = dsp.StripesFacets(
            _group['dissolution'],
            target_on_y=True,
            single_axes=False,
            mean=True,
            confidence=0.95,
            spec_limits=dsp.SpecLimits(upper=25))
        _ax.scatter(_group['temperature'], _group['dissolution'])
        _ax.set_title(str(_name))
        _stripes.draw(_ax)

    _fig.savefig(img_dir+'facets_stripes-drawn.png', bbox_inches='tight', dpi=DPI)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LabelFacets
    """)
    return


@app.cell
def _(DPI, Line2D, Patch, dsp, img_dir):
    _axes = dsp.AxesFacets(nrows=3, ncols=2, sharey=True)

    green = dsp.COLOR.GOOD
    red = dsp.COLOR.BAD

    _legend_data={
        'Lines': (
            (Line2D([0], [0], c=red), Line2D([0], [0], c=green)),
            ('red line', 'green line')),
        'Patches': (
            (Patch(color=red), Patch(color=green)), 
            ('red patch', 'green patch'))}

    _labels = dsp.LabelFacets(
        axes=_axes,
        fig_title='Title',
        sub_title='Subtitle',
        xlabel=('xlabel tl', 'xlabel tr', 'xlabel cl', 'xlabel cr', 'xlabel bl', 'xlabel br'),
        ylabel='single ylabel at center',
        info='Info goes here',
        cols=('col 1', 'col 2'),
        col_title='Column title',
        rows=('row 1', 'row 2', 'row3'),
        row_title='Row title',
        legend_data=_legend_data)
    _labels.draw()
    _axes.figure.savefig(img_dir+'facets_labels.png', bbox_inches='tight', dpi=DPI)
    _axes.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Combination
    """)
    return


@app.cell
def _(DPI, df_pk, dsp, img_dir):
    _axes = dsp.AxesFacets(
        nrows=1, ncols=df_pk['employee'].nunique(), sharex=True, sharey=True)

    for _ax, (_name, _group) in zip(_axes, df_pk.groupby('employee')):
        _stripes = dsp.StripesFacets(
            _group['dissolution'],
            target_on_y=True,
            single_axes=False,
            mean=True,
            confidence=0.95,
            spec_limits=dsp.SpecLimits(upper=25))
        _ax.scatter(_group['temperature'], _group['dissolution'])
        _stripes.draw(_ax)

    _legend_data = {'Lines': _stripes.handles_labels()}

    _labels = dsp.LabelFacets(
        axes=_axes,
        fig_title='Painkillers Dissolution Analysis',
        sub_title='Dissolution time ~ temperature + employee',
        xlabel='Temperature (°C)',
        ylabel='Dissolution time (s)',
        info='Mini-project from the Six Sigma Black Belt training',
        cols=tuple(df_pk['employee'].unique()),
        col_title='Employee',
        legend_data=_legend_data)
    _labels.draw()

    _axes.figure.savefig(img_dir+'facets_combined.png', bbox_inches='tight', dpi=DPI)
    _axes.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotters
    """)
    return


@app.cell
def _(DPI, df_iris, dsp, img_dir):
    dsp.style.use('daspi')

    _axes = dsp.AxesFacets(nrows=1, ncols=1)
    _kwds = dict(
        source=df_iris,
        target='length',
        feature='width',
        color=dsp.DEFAULT.PLOTTING_COLOR,
        ax=_axes[0])

    _loess_plot = dsp.LoessLine(show_ci=True, **_kwds)
    _loess_plot()
    _scatter_plot = dsp.Scatter(**_kwds)
    _scatter_plot()

    _axes.figure.savefig(img_dir+'plotters_xy-example.png', bbox_inches='tight', dpi=DPI)
    _axes.figure
    return


@app.cell
def _(
    BASE_FIGSIZE,
    DPI,
    KW_PLOT,
    N_COLS,
    ceil,
    df_card,
    df_iris,
    dsp,
    img_dir,
):
    dsp.style.use('daspi-dark')
    _target_c='distance'
    _feature_c='observation'
    _target_i='width'
    _feature_i='length'

    _df_card = df_card.reset_index().rename(columns={'index': _feature_c})
    _df_card[_feature_c] = _df_card[_feature_c] + 1

    _plots = (
        'Line', 'Scatter', 'Stem', 'LinearRegressionLine + Scatter', 'LoessLine + Scatter', 
        'GaussianKDEContour')
    _n_rows = ceil(len(_plots) / N_COLS)

    _chart = dsp.JointChart(
            source=(_df_card,)*3 + (df_iris,)*3,
            target=(_target_c,)*3 + (_target_i,)*3,
            feature=(_feature_c,)*3 + (_feature_i,)*3,
            nrows=_n_rows,
            ncols=N_COLS,
            sharex='row',
            sharey='row',
            figsize=(BASE_FIGSIZE*N_COLS, BASE_FIGSIZE*_n_rows)
        ).plot(
            dsp.Line,
            **KW_PLOT
        ).plot(
            dsp.Scatter,
            **KW_PLOT
        ).plot(
            dsp.Stem,
            **KW_PLOT
        ).plot(
            dsp.LinearRegressionLine,
            show_fit_ci=True,
            show_scatter=True,
            kw_call=dict(kw_scatter=dict(marker='x', alpha=0.2)),
            **KW_PLOT
        ).plot(
            dsp.LoessLine,
            show_fit_ci=True,
            **KW_PLOT
        ).plot(
            dsp.Scatter,
            marker='x',
            on_last_axes=True,
            kw_call=dict(alpha=0.2)
        ).plot(
            dsp.GaussianKDEContour,
            **KW_PLOT
        ).label(
            axes_titles=_plots,
        )
    _chart.axes[0, 0].set(xlim=(-1, 42), ylim=(-5, 100))
    _chart.axes[1, 0].set(xlim=(0.5, 8.5), ylim=(-0.5, 5))
    _chart.save(img_dir+'plotters_xy.png', dpi=DPI)
    return


@app.cell
def _(BASE_FIGSIZE, DPI, KW_PLOT, N_COLS, ceil, df_iris, dsp, img_dir):
    dsp.style.use('daspi-dark')
    _target='width'
    _feature='species'

    _plots = (
        'Jitter', 'Beeswarm', 'GaussianKDEContourUnivariate', 'Violin', 'Box', 
        'QuantileBoxes', 'GaussianKDE', 'SpreadWidth', 'Probability (Q-Q)')
    _n_rows = ceil(len(_plots) / N_COLS)

    _chart = dsp.JointChart(
            source=df_iris,
            target=_target,
            feature=_feature,
            nrows=_n_rows,
            ncols=N_COLS,
            sharey=True,
            target_on_y=True,
            categorical_feature=True,
            figsize=(BASE_FIGSIZE*N_COLS, BASE_FIGSIZE*_n_rows)
        ).plot(
            dsp.Jitter,
            **KW_PLOT
        ).plot(
            dsp.Beeswarm,
            **KW_PLOT
        ).plot(
            dsp.GaussianKDEContourUnivariate,
            fill=True,
            fade_outers=True,
            **KW_PLOT
        ).plot(
            dsp.Violin,
            agreements=(),
            **KW_PLOT
        ).plot(
            dsp.Box,
            **KW_PLOT
        ).plot(
            dsp.QuantileBoxes,
            **KW_PLOT
        ).plot(
            dsp.GaussianKDE,
            agreements=(),
            ignore_feature=False,
            **KW_PLOT
        ).plot(
            dsp.SpreadWidth,
            strategy='data',
            **KW_PLOT
        ).plot(
            dsp.Probability,
            kind='sq',
            **KW_PLOT
        ).label(
            axes_titles=_plots,
        )
    _chart.axes[6].set(xlim=(-0.1, 2.9))
    _chart.axes[0].set(ylim=(-0.5, 5.5))
    _chart.save(img_dir+'plotters_univariate.png', dpi=DPI)
    return


@app.cell
def _(BASE_FIGSIZE, DPI, KW_PLOT, N_COLS, ceil, df_card, dsp, img_dir):
    dsp.style.use('daspi-dark')
    _target='distance'
    _feature='method'
    _osg=50
    df_card['identity'] = list(range(len(df_card)//2)) * 2
    df_card['observations'] = 1
    df_card ['events'] = list(map(lambda x: 1 if x > _osg else 0, df_card[_target]))
    df_card['proportion'] = list(map(lambda x: 2/len(df_card) if x else 0, df_card['events']))

    _plots = (
        'ParallelCoordinate', 'CenterLocation + Scatter', 
        'StandardErrorMean + Scatter', 'MeanTest', 'VariationTest', 
        'CapabilityConfidenceInterval', 'ProportionTest + Bar')
    _n_plots = len(_plots)
    _n_rows = ceil(_n_plots / N_COLS)

    _chart = dsp.JointChart(
            source=df_card,
            target=(_target,)*(_n_plots -1) + ('proportion', '', ''),
            feature=_feature,
            nrows=_n_rows,
            ncols=N_COLS,
            target_on_y=True,
            categorical_feature=True,
            figsize=(BASE_FIGSIZE*N_COLS, BASE_FIGSIZE*_n_rows)
        ).plot(
            dsp.ParallelCoordinate,
            identity='identity',
            **KW_PLOT
        ).plot(
            dsp.CenterLocation,
            **KW_PLOT
        ).plot(
            dsp.Scatter,
            marker='x',
            on_last_axes=True,
            **KW_PLOT
        ).plot(
            dsp.StandardErrorMean,
            **KW_PLOT
        ).plot(
            dsp.Scatter,
            marker='x',
            on_last_axes=True,
            **KW_PLOT
        ).plot(
            dsp.MeanTest,
            n_groups=1,
            **KW_PLOT
        ).plot(
            dsp.VariationTest,
            n_groups=1,
            **KW_PLOT
        ).plot(
            dsp.CapabilityConfidenceInterval,
            n_groups=1,
            kind='cpk',
            spec_limits=dsp.SpecLimits(upper=_osg),
            **KW_PLOT
        ).plot(
            dsp.ProportionTest,
            n_groups=1,
            observations='observations',
            events='events',
            **KW_PLOT
        ).plot(
            dsp.Bar,
            method='sum',
            on_last_axes=True,
            **KW_PLOT
        ).plot(
            dsp.HideSubplot
        ).plot(
            dsp.HideSubplot
        ).label(
            axes_titles=_plots,
        )
    # _chart.axes[0, 3].set(xlim=(-0.1, 2.9))
    # _chart.axes[0, 6].set(ylim=(-0.5, 5.5))
    _chart.save(img_dir+'plotters_differences.png', dpi=DPI)
    return


@app.cell
def _(BASE_FIGSIZE, DPI, df_card, df_pk, dsp, img_dir, pd):
    dsp.style.use('daspi-dark')
    _plots = (
        'Pareto', 'BlandAltman')
    _n_plots = len(_plots)

    _axes = dsp.AxesFacets(
        nrows=1,
        ncols=_n_plots,
        figsize=(BASE_FIGSIZE*_n_plots, BASE_FIGSIZE)
    )

    _model = dsp.LinearModel(
        source=df_pk,
        target='dissolution',
        factors=['employee', 'stirrer', 'brand', 'catalyst', 'water'],
        covariates=['temperature', 'preparation'])
    _effects =  _model.effects()
    _data = (pd
        .concat([_model.anova(typ='I'), _effects], axis=1)
        .reset_index(drop=False)
        .rename(columns={'index': dsp.ANOVA.SOURCE}))

    _pareto = dsp.Pareto(
        source=_data,
        target=dsp.ANOVA.TABLE_COLNAMES[1],
        feature=dsp.ANOVA.SOURCE,
        visible_spines ='none',
        hide_axis= 'both',
        color=dsp.DEFAULT.PLOTTING_COLOR,
        ax=_axes[0, 0])
    _pareto()

    _data = df_card.copy()
    _data['identity'] = list(range(len(_data)//2)) * 2
    _blantaltman = dsp.BlandAltman(
        source=_data,
        target='distance',
        feature='method',
        identity='identity',
        visible_spines ='none',
        hide_axis= 'both',
        color=dsp.DEFAULT.PLOTTING_COLOR,
        ax=_axes[0, 1])
    _blantaltman()

    _labels = dsp.LabelFacets(
        _axes,
        axes_titles=_plots).draw()

    _axes.figure.savefig(img_dir+'plotters_special.png', bbox_inches='tight', dpi=DPI)
    _axes.figure
    return


@app.cell
def _(df_iris, dsp, plt):
    dsp.style.use('seaborn')

    exogenous ='length'
    endogenous = 'width'
    _data = df_iris.sort_values(endogenous)
    _fig, _axs = plt.subplots(
        2, 2, sharex='col', sharey='row', width_ratios=[3, 1], height_ratios=[1, 3])

    tl_kde = dsp.GaussianKDE(
        source=_data,
        target=endogenous,
        target_on_y=False,
        hide_axis='feature',
        visible_spines='target',
        ax=_axs[0, 0])
    tl_kde()

    tr_hide = dsp.HideSubplot(_axs[0, 1])
    tr_hide()

    bl_linreg = dsp.LinearRegressionLine(
        source=_data,
        target=exogenous,
        feature=endogenous,
        show_fit_ci=True,
        ax=_axs[1, 0])
    bl_linreg()

    br_kde = dsp.GaussianKDE(
        source=_data,
        target=exogenous,
        target_on_y=True,
        hide_axis='feature',
        visible_spines='target',
        ax=_axs[1, 1])
    br_kde()
    _fig
    return


@app.cell
def _(DPI, df_pk, dsp, img_dir):
    dsp.style.use('daspi')

    _chart = dsp.SingleChart(
            source=df_pk,
            target='dissolution',
            feature='employee',
            hue='brand',
            dodge=True,
        ).plot(
            dsp.Beeswarm
        ).plot(
            dsp.CenterLocation,
            show_line=True,
            show_center=False,
        ).plot(
            dsp.MeanTest,
            n_groups=1,
            marker='_',
            kw_center={'size': 100}
        ).stripes(
            mean=True,
            confidence=0.95
        ).label(
            fig_title='Painkillers Dissolution Analysis',
            sub_title='Dissolution time vs. Employee, Brand, and Stirrer',
            target_label='Dissolution time (s)',
            feature_label='Employee',
            info='Mini-project from the Six Sigma Black Belt training'
        )

    _chart.save(img_dir+'plotters_single-chart_example.png', dpi=DPI)
    return


if __name__ == "__main__":
    app.run()
