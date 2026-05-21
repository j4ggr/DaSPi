import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import os
    if not str(Path.cwd()).endswith('DaSPi'):
        os.chdir(Path.cwd().parent.resolve())
    Path.cwd()
    return


@app.cell
def _():
    import daspi as dsp
    DPI = 120
    dsp.STR._username_ = 'j4ggr'
    dsp.__version__
    return DPI, dsp


@app.cell
def _():
    import daspi as dsp
    fA = dsp.Factor('A', (-1, 1))
    fB = dsp.Factor('B', (-1, 1))
    fC = dsp.Factor('C', (-1, 1))
    builder = dsp.FractionalFactorialDesignBuilder(
        fA, fB, fC, generators=['C=AB'], fold='A', shuffle=False)

    df = builder.build_design(corrected=False)
    return df, dsp


@app.cell
def _():
    import daspi as dsp
    import numpy as np

    np.random.seed(42) # optional for reproducibility

    factor_a = dsp.Factor('A', (0, 1))
    factor_b = dsp.Factor('B', (0, 1))
    builder = dsp.FullFactorial2kDesignBuilder(
        factor_a, factor_b, replicates=3, central_points=2,
        blocks='highest', shuffle=True)
    df = builder.build_design(corrected=True)
    print(df)
    return df, dsp


@app.cell
def _():
    import daspi as dsp
    df = dsp.load_dataset('grnr_layer_thickness')
    gage = dsp.GageStudyModel(
        source=df,
        target='result_gage',
        reference='reference',
        u_cal=df['U_cal'][0],
        tolerance=df['tolerance'][0],
        resolution=df['resolution'][0],)
    chart = dsp.GageStudyCharts(
            gage, stretch_figsize=1.3
        ).plot(
        ).stripes(
        ).label(
        ) # .save('path/to/file.png')
    gage # or print(repr(gage))
    return df, dsp


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import daspi as dsp

    from pathlib import Path

    valid_data_dir = Path.cwd()/'tests'/'data'
    df_lin = pd.read_csv(
        valid_data_dir/'gage_study.csv', skiprows=54, nrows=30, sep=';')
    df_single = pd.read_csv(
        valid_data_dir/'gage_study.csv', skiprows=1, nrows=50, sep=';')

    gage_single = dsp.GageStudyModel(
        source=df_single,
        target='result',
        reference='reference',
        u_cal=df_single['U_cal'][0],
        tolerance=df_single['tolerance'][0],
        resolution=df_single['resolution'][0],
        k=2)

    gage_lin = dsp.GageStudyModel(
        source=df_lin,
        target='result',
        reference='reference',
        u_cal=df_lin['U_cal'][0],
        tolerance=df_lin['tolerance'][0],
        resolution=df_lin['resolution'][0],
        bias_corrected=True,
        k=2)
    gage_lin
    return (dsp,)


@app.cell
def _():
    import daspi as dsp
    df = dsp.load_dataset('grnr_layer_thickness')
    gage = dsp.GageStudyModel(
        source=df,
        target='result_gage',
        reference='reference',
        u_cal=df['U_cal'][0],
        tolerance=df['tolerance'][0],
        resolution=df['resolution'][0],)

    rnr_model = dsp.GageRnRModel(
        source=df,
        target='result_rnr',
        part='part',
        u_av='operator',
        gage=gage)
    chart = dsp.GageRnRCharts(rnr_model).plot().stripes().label()
    rnr_model
    return df, dsp


@app.cell
def _(DPI):
    import daspi as dsp

    df = dsp.load_dataset('anova')
    chart = dsp.SingleChart(
            source=df,
            target='Pain threshold',
            feature='Hair color',
            categorical_feature=True,
        ).plot(
            dsp.Jitter
        ).label(
            fig_title='Pain threshold by hair color',
            sub_title='Jittered',
            feature_label=True,
            target_label=True,
            info=True
        )

    chart.save('./docs/img/anova_jitter_pain-color.png', dpi=DPI)
    return df, dsp


@app.cell
def _(df, dsp):
    model = dsp.LinearModel(df, 'Pain threshold', ['Hair color'])
    model
    return (model,)


@app.cell
def _(df):
    x_bar_group = df.groupby('Hair color')['Pain threshold'].mean()
    x_bar = df['Pain threshold'].mean()
    (x_bar_group - x_bar).abs()
    return


@app.cell
def _(DPI, df, dsp):
    LEVEL = 0.95
    n_groups = df.groupby(['Hair color']).ngroups
    chart = dsp.SingleChart(
            source=df,
            target='Pain threshold',
            feature='Hair color',
            categorical_feature=True,
        ).plot(
            dsp.MeanTest,
            n_groups=n_groups,
            confidence_level=LEVEL,
            show_center=False
        ).plot(
            dsp.CenterLocation
        ).stripes(
            mean=True
        ).label(
            fig_title='Pain threshold by hair color',
            sub_title=f'Mean and {int(100*LEVEL)} % confidence interval',
            feature_label=True,
            target_label=True,
        )

    chart.save('./docs/img/anova_mean-ci_pain-color.png', dpi=DPI)
    return


@app.cell
def _(model):
    model.uncertainty
    return


@app.cell
def _(DPI, dsp, model):
    chart_r = dsp.ResidualsCharts(model).plot().stripes().label()
    chart_p = dsp.ParameterRelevanceCharts(model).plot().stripes().label()

    chart_r.save('./docs/img/anova_residues_pain-color.png', dpi=DPI)
    chart_p.save('./docs/img/anova_parameter-relevance_pain-color.png', dpi=DPI)
    return


if __name__ == "__main__":
    app.run()
