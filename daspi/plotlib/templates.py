from typing import Self

from .chart import JointChart
from .plotter import Line
from .plotter import Scatter
from .plotter import Probability
from .plotter import GaussianKDE

from ..anova import LinearModel
from ..strings import STR


class ResiduesCharts:

    def __init__(self, linear_model: LinearModel) -> None:
        self.lm = linear_model
        self.chart = JointChart(
            source=self.lm.residual_data(),
            target='Residues',
            feature=('', '', 'Prediction', 'Observation'),
            nrows=2,
            ncols=2,
            sharey=True,
            stretch_figsize=False)
        
    def plot(self) -> Self:
        self.chart.plot(Probability, show_fit_ci=True
            ).plot(GaussianKDE
            ).plot(Scatter
            ).plot(Line, {'marker': 'o'})
        return self

    def label(self, info: bool | str = True) -> Self:
        sub_title = f'{self.lm.target} ~ {" + ".join(self.lm.effects().index)}'
        self.chart.label(
            fig_title = STR['residcharts_fig_title'], # type: ignore
            sub_title = sub_title,
            target_label = 'Residuen',
            feature_label = STR['residcharts_feature_labels'],
            info = info)
        return self

__all__ = ['ResiduesCharts']