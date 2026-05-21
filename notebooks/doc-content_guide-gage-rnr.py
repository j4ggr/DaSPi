import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import daspi as dsp
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon
    import numpy as np
    import pandas as pd
    from daspi.constants import COLOR

    IMAGE_DIR = Path.cwd().parent / 'docs' / 'img'
    assert IMAGE_DIR.is_dir(), f'Image path {IMAGE_DIR} does not exist.'
    DPI = 120
    return COLOR, DPI, FancyBboxPatch, IMAGE_DIR, Polygon, dsp, pd, plt


@app.cell
def _(dsp):
    df = dsp.load_dataset('grnr_adjustment')
    gage = dsp.GageEstimator(
        samples=df['result_gage'],
        reference=df['reference'][0],
        u_cal=df['U_cal'][0],
        tolerance=df['tolerance'][0],
        resolution=df['resolution'][0]
    )

    # Get metrics from describe
    desc = gage.describe()
    desc
    return (desc,)


@app.cell
def _(DPI, IMAGE_DIR, desc, dsp, pd):
    # Create DataFrame for plotting
    df_cap = pd.DataFrame({
        'Metric': ['Cg', 'Cgk', 'Resolution\nRatio (%)'],
        'Value': [
            float(desc.loc['cg'].iloc[0]),
            float(desc.loc['cgk'].iloc[0]),
            float(desc.loc['resolution_ratio'].iloc[0]) * 100
        ]
    })

    # Create chart using native DaSPi plotting with stripes
    acceptance_line = dsp.StripeLine(
        label='Minimum Acceptance (1.33)',
        position=1.33,
        orientation='horizontal',
        color=dsp.COLOR.SPECIAL_LINE,
        linestyle=dsp.LINE.DASHED,
        # width=2
    )

    chart = dsp.SingleChart(
            source=df_cap,
            target='Value',
            feature='Metric',
            categorical_feature=True,
        ).plot(
            dsp.Bar
        ).stripes(
            stripes=[acceptance_line]
        ).label(
            fig_title='Gage Capability Metrics',
            sub_title='Adjustment Measurement System',
            feature_label='Metric',
            target_label='Value'
        ).save(
            IMAGE_DIR / 'gage_capability_metrics.png',
            dpi=DPI,)
    return


@app.cell
def _(dsp):
    df = dsp.load_dataset('grnr_layer_thickness')
    gage_study = dsp.GageStudyModel(
        source=df,
        target='result_gage',
        reference='reference',
        u_cal=df['U_cal'][0],
        tolerance=df['tolerance'][0],
        resolution=df['resolution'][0]
    )

    model = dsp.GageRnRModel(
        source=df,
        target='result_rnr',
        part='part',
        gage=gage_study,
        u_av='operator'
    )
    model
    return (model,)


@app.cell
def _(DPI, IMAGE_DIR, dsp, model, pd):
    df_u = model.df_u

    # Prepare data for MS components
    ms_components = ['CAL', 'RE', 'BI', 'LIN', 'EVR']
    df_ms = pd.DataFrame({
        'Component': ms_components,
        'Q': (df_u.loc[ms_components, 'Q'] * 100).values
    })

    # Prepare data for MP components
    mp_components = ['EVO', 'AV', 'GV', 'IA']
    df_mp = pd.DataFrame({
        'Component': mp_components,
        'Q': (df_u.loc[mp_components, 'Q'] * 100).values
    })

    # Create JointChart with two subplots and custom stripes
    limit_line = dsp.StripeLine(
        label=r'15\% Limit',  # Use raw string with LaTeX escape
        position=15,
        orientation='vertical',
        color=dsp.COLOR.SPECIAL_LINE,
        linestyle=dsp.LINE.DASHED,
    )

    chart = dsp.JointChart(
            source=(df_ms, df_mp),
            target='Q',
            feature='Component',
            categorical_feature=True,
            ncols=2,
            nrows=1,
            target_on_y=False
        ).plot(
            dsp.Bar
        ).plot(
            dsp.Bar
        ).stripes(
            stripes=[limit_line]
        ).label(
            fig_title='Uncertainty Budget Breakdown',
            sub_title='Measurement System (MS) and Measurement Process (MP)',
            target_label='Proportion of Tolerance (%)',
            feature_label=('MS Component', 'MP Component')
        ).save(
            IMAGE_DIR / 'uncertainty_budget_breakdown.png',
            dpi=DPI,)
    return


@app.cell
def _(COLOR, DPI, FancyBboxPatch, IMAGE_DIR, Polygon, plt):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    # Get native DaSPi colors
    good_color = COLOR.GOOD
    bad_color = COLOR.BAD
    process_color = COLOR.ANOMALY
    decision_color = COLOR.TRANSPARENT

    # Define boxes with positions and text
    boxes = {
        'start': (0.5, 0.95, 'Start:\nMeasurement System\nAnalysis', 'process'),
        'type1': (0.5, 0.85, 'Type 1: Gage Study\nSingle/Multiple References', 'process'),
        'collect1': (0.5, 0.75, 'Collect ≥25-50 measurements\non reference standard(s)', 'process'),
        'analyze1': (0.5, 0.65, 'GageStudyModel\nCalculate Cg, Cgk, u_MS', 'process'),
        'decision1': (0.5, 0.55, 'Cg, Cgk ≥ 1.33?\nQ_MS < 0.15?', 'decision'),
        'type23': (0.5, 0.40, 'Type 2/3: Gage R&R\nMultiple Parts + Operators/Conditions', 'process'),
        'collect2': (0.5, 0.30, 'Collect measurements:\n≥10 parts × ≥2 operators × ≥2 trials', 'process'),
        'analyze2': (0.5, 0.20, 'GageRnRModel\nCalculate %Spread, uncertainties', 'process'),
        'decision2': (0.5, 0.10, '%Spread < 30%?\nQ_total < 0.15?', 'decision'),
        'accept': (0.25, 0.02, 'ACCEPT\nUse system', 'accept'),
        'reject': (0.75, 0.02, 'REJECT\nImprove system', 'reject'),
    }

    # Draw boxes
    width, height = 0.18, 0.06
    for key, (x, y, text, box_type) in boxes.items():
        if box_type == 'accept':
            color = good_color
            # Rounded rectangle
            patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                                  boxstyle="round,pad=0.01", 
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(patch)
        elif box_type == 'reject':
            color = bad_color
            # Rounded rectangle
            patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(patch)
        elif box_type == 'decision':
            # Diamond/rhombus shape for decisions
            color = decision_color
            diamond = Polygon([
                (x, y + height/2),           # top
                (x + width/2, y),            # right
                (x, y - height/2),           # bottom
                (x - width/2, y)             # left
            ], facecolor=color, edgecolor='black', linewidth=2, closed=True)
            ax.add_patch(diamond)
        else:
            # Process steps - rounded rectangles
            color = process_color
            patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(patch)
    
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                fontweight='bold', wrap=True)

    # Draw arrows
    arrows = [
        ((0.5, 0.91), (0.5, 0.87)),  # start -> type1
        ((0.5, 0.81), (0.5, 0.77)),  # type1 -> collect1
        ((0.5, 0.71), (0.5, 0.67)),  # collect1 -> analyze1
        ((0.5, 0.61), (0.5, 0.57)),  # analyze1 -> decision1
        ((0.5, 0.51), (0.5, 0.42)),  # decision1 -> type23 (YES)
        ((0.5, 0.36), (0.5, 0.32)),  # type23 -> collect2
        ((0.5, 0.26), (0.5, 0.22)),  # collect2 -> analyze2
        ((0.5, 0.16), (0.5, 0.12)),  # analyze2 -> decision2
        ((0.42, 0.10), (0.32, 0.05)), # decision2 -> accept (YES)
        ((0.58, 0.10), (0.68, 0.05)), # decision2 -> reject (NO)
        ((0.62, 0.55), (0.75, 0.45)), # decision1 -> improve (NO)
        ((0.75, 0.45), (0.75, 0.08)), # improve -> reject
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Add YES/NO labels
    ax.text(0.52, 0.47, 'YES', fontsize=8, color=good_color, fontweight='bold')
    ax.text(0.65, 0.50, 'NO', fontsize=8, color=bad_color, fontweight='bold')
    ax.text(0.35, 0.08, 'YES', fontsize=8, color=good_color, fontweight='bold')
    ax.text(0.60, 0.08, 'NO', fontsize=8, color=bad_color, fontweight='bold')

    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(-0.05, 1)
    ax.set_title('Measurement System Analysis Decision Flow', 
                 fontsize=16, fontweight='bold', pad=20)

    plt.savefig(IMAGE_DIR / 'msa_decision_flow.png', dpi=DPI,)
    return


if __name__ == "__main__":
    app.run()
