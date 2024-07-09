import pandas as pd

from typing import Any
from typing import Self
from typing import Tuple
from pathlib import Path

from .chart import JointChart
from .plotter import Line
from .plotter import Pareto
from .plotter import Scatter
from .plotter import Probability
from .plotter import GaussianKDE

from ..anova import LinearModel
from ..strings import STR
from ..constants import KW
from ..constants import ANOVA


class BaseTemplate:
    """Base class for all templates."""
    chart: JointChart
    
    def stripes(self,
            mean: bool = False,
            median: bool = False,
            control_limits: bool = False,
            spec_limits: Tuple[float | None, float | None] = (None, None),
            confidence: float | None = None,
            **kwds) -> Self:
        """Plot location and spread width lines, specification limits 
        and/or confidence interval areas as stripes on each Axes. The 
        location and spread (and their confidence bands) represent the 
        data per axes.

        Parameters
        mean : bool, optional
            Whether to plot the mean value of the plotted data on the 
            axes, by default False.
        median : bool, optional
            Whether to plot the median value of the plotted data on the 
            axes, by default False.
        control_limits : bool, optional
            Whether to plot control limits representing the process 
            spread, by default False.
        spec_limits : Tuple[float], optional
            If provided, specifies the specification limits. The tuple 
            must contain two values for the lower and upper limits. If a 
            limit is not given, use None, by default ().
        confidence : float, optional
            The confidence level between 0 and 1, by default None.
        **kwds:
            Additional keyword arguments for configuring StripesFacets.

        Returns
        -------
        Self
            The `ParameterRelevanceCharts` instance, for method chaining.
        """
        self.chart.stripes(
            mean=mean, median=median, control_limits=control_limits,
            spec_limits=spec_limits, confidence=confidence, **kwds)
        return self
    
    def save(self, file_name: str | Path, **kwds) -> None:
        """Save the chart to a file.

        Parameters
        ----------
        file_name : str | Path
            The file name or path to save the chart to.
            The file format is inferred from the file name extension.
            Supported formats include: 'png', 'jpg', 'jpeg', 'pdf',
            'eps', 'ps'.
        **kwds:
            Additional keyword arguments to be passed to the `save()`
            method of the underlying chart.

        Returns
        -------
        Self
            The instance with the saved chart.
        """
        self.chart.save(file_name, **kwds)


class ParameterRelevanceCharts(BaseTemplate):
    """
    Provides a set of charts for visualizing the relevance of a linear
    regression model's parameters.

    The `ParameterRelevanceCharts` class takes a `LinearModel` instance 
    and generates a set of two charts for visualizing the relevance
    of the model's parameters:
    - Pareto chart of the parameter standardized effects
    - Pareto chart of the Sum of Squares for each parameter

    On the first chart, the standardized effect of each parameter is 
    visualized as a Pareto chart. The red line indicates the alpha
    level of significance.

    Parameters
    ----------
    linear_model : LinearModel
        The linear regression model whose parameters will be visualized.
    drop_intercept : bool, optional
        Whether to drop the intercept from the model, by default True.
    """
    def __init__(
            self, linear_model: LinearModel, drop_intercept: bool = True
            ) -> None:
        self.lm = linear_model
        effects =  self.lm.effects()
        if ANOVA.INTERCEPT in effects and drop_intercept:
            effects = effects.drop(ANOVA.INTERCEPT)
        data = (pd
            .concat([self.lm.anova('I'), effects], axis=1)
            .reset_index(drop=False)
            .rename(columns={'index': ANOVA.SOURCE}))

        self.chart = JointChart(
                source=data,
                target=(ANOVA.EFFECTS, ANOVA.TABLE_COLNAMES[1]),
                feature=ANOVA.SOURCE,
                target_on_y=False,
                ncols=2,
                nrows=1,
                stretch_figsize=True)
        
    def plot(self) -> Self:
        """Generates a set of two charts for visualizing the relevance 
        of the model's parameters:
        - Pareto chart of the parameter standardized effects
        - Pareto chart of the Sum of Squares for each parameter
        
        Returns
        -------
        Self: 
            The `ParameterRelevanceCharts` instance, for method chaining.

        Note
        ----
        The `plot()` method will generate two charts. The first chart
        will contain the Pareto chart of the parameter standardized
        effects. The second chart will contain the Pareto chart of the
        Sum of Squares for each parameter.

        The red line in the Pareto charts represents the threshold
        for the effect of a parameter. The threshold is calculated as
        the alpha risk where the parameter is not relevant.
        """
        self.chart.plot(
                Pareto, no_percentage_line=True, skip_na='all'
            ).plot(
                Pareto, highlight=ANOVA.RESIDUAL, skip_na='all')
        
        self.chart.axes[0][0].axvline(
            self.lm.effect_threshold, **KW.SPECIFICATION_LINE)
        return self

    def label(self, info: bool | str = True, **kwds) -> Self:
        """Adds titles and labels to the charts generated by the 
        `plot()` method.
        
        Parameters
        ----------
        info : bool | str, optional
            If `True`, the method will add an informative subtitle to 
            the chart. If a string is provided, it will be used as the 
            subtitle, by default True.
        **kwds
            Additional keyword arguments to be passed to the `label()`
            method of the `JointChart` instance.
        
        Returns
        -------
        Self
            The `ParameterRelevanceCharts` instance, for method chaining.
        """      
        labels = dict(
                fig_title=f'{STR["paramcharts_fig_title"]}',
                sub_title=f'{STR["paramcharts_sub_title"]}',
                target_label=(f'{STR["effects_label"]}', f'{STR["ss_label"]}'),
                feature_label=STR["paramcharts_feature_label"],
                info = info
            ) | kwds
        self.chart.label(**labels) # type: ignore
        return self


class ResiduesCharts(BaseTemplate):
    """
    Provides a set of charts for visualizing the residuals of a linear 
    regression model.

    The `ResiduesCharts` class takes a `LinearModel` instance and 
    generates a set of four charts:
    - Probability plot of the residuals
    - Gaussian kernel density estimate of the residuals
    - Scatter plot of the predicted values vs. the observed values
    - Line plot of the predicted values vs. the observed values

    The `plot()` method generates the charts, and the `label()` method 
    adds titles and labels to the charts.

    Parameters
    ----------
    linear_model : LinearModel
        The linear regression model whose residuals will be visualized.
    """
    def __init__(self, linear_model: LinearModel) -> None:
        self.lm = linear_model
        self.chart = JointChart(
            source=self.lm.residual_data(),
            target=ANOVA.RESIDUAL,
            feature=('', '', ANOVA.PREDICTION, ANOVA.OBSERVATION),
            nrows=2,
            ncols=2,
            sharey=True,
            stretch_figsize=False)
        
    def plot(self) -> Self:
        """Generates a set of four charts for visualizing the residuals 
        of a linear regression model:
        - Probability plot of the residuals
        - Gaussian kernel density estimate of the residuals
        - Scatter plot of the predicted values vs. the observed values
        - Line plot of the predicted values vs. the observed values
        
        Returns
        -------
        Self: 
            The `ResiduesCharts` instance, for method chaining.
        """
        self.chart.plot(Probability, show_fit_ci=True
            ).plot(GaussianKDE
            ).plot(Scatter
            ).plot(Line, {'marker': 'o'})
        return self

    def label(self, info: bool | str = True, **kwds) -> Self:
        """Adds titles and labels to the charts generated by the 
        `plot()` method.
        
        Parameters
        ----------
        info : bool | str, optional
            If `True`, the method will add an informative subtitle to 
            the chart. If a string is provided, it will be used as the 
            subtitle, by default True.
        **kwds
            Additional keyword arguments to be passed to the `label()`
            method of the `JointChart` instance.
        
        Returns
        -------
        Self
            The `ResiduesCharts` instance, for method chaining.
        """
        sub_title = f'{self.lm.target} ~ {" + ".join(self.lm.effects().index)}'
        feature_label = list(STR["residcharts_feature_label"])
        feature_label[2] = f'{feature_label[2]} {self.lm.target}'
        labels = dict(
                fig_title=f'{STR["residcharts_fig_title"]}',
                sub_title=sub_title,
                target_label=f'{STR["resid_name"]}',
                feature_label=tuple(feature_label),
                info = info
            ) | kwds
        self.chart.label(**labels) # type: ignore
        return self

__all__ = [
    'ResiduesCharts',
    'ParameterRelevanceCharts',
    ]
