import numpy as np
import pandas as pd

from typing import Any
from typing import List
from typing import Self
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Literal
from typing import Generator
from matplotlib.axes import Axes
from pandas.core.frame import DataFrame
from scipy.stats._distn_infrastructure import rv_continuous

from .chart import JointChart
from .plotter import Line
from .plotter import Pareto
from .plotter import Plotter
from .plotter import Scatter
from .plotter import MeanTest
from .plotter import StripeLine
from .plotter import SkipSubplot
from .plotter import HideSubplot
from .plotter import Probability
from .plotter import GaussianKDE
from .plotter import BlandAltman
from .plotter import QuantileBoxes
from .plotter import ParallelCoordinate
from .plotter import CapabilityConfidenceInterval

from ..anova import LinearModel
from ..strings import STR
from ..constants import DIST
from ..constants import COLOR
from ..constants import ANOVA
from ..constants import CATEGORY

from ..statistics import SpecLimits
from ..statistics import ProcessEstimator


class ParameterRelevanceCharts(JointChart):
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
    stretch_figsize : bool, optional
        If True, stretch the figure height and width based on the number of
        rows and columns, by default False.
    
    Examples
    --------

    ``` python
    import daspi as dsp
    import pandas as pd

    df = dsp.load_dataset('aspirin-dissolution')
    model = dsp.LinearModel(
        source=df,
        target='dissolution',
        features=['employee', 'stirrer', 'brand', 'catalyst', 'water'],
        disturbances=['temperature', 'preparation'],
        order=2)
    df_gof = pd.DataFrame()
    for data_gof in model.recursive_elimination():
        df_gof = pd.concat([df_gof, data_gof])
    dsp.ParameterRelevanceCharts(model).plot().label(info=True)
    ```
    """
    __slots__ = ('lm')

    lm: LinearModel
    """The linear regression model whose parameters are visualized."""

    def __init__(
            self,
            linear_model: LinearModel,
            drop_intercept: bool = True,
            stretch_figsize: bool = False
            ) -> None:
        self.lm = linear_model
        effects =  self.lm.effects()
        if ANOVA.INTERCEPT in effects and drop_intercept:
            effects = effects.drop(ANOVA.INTERCEPT)
        data = (pd
            .concat([self.lm.anova('I'), effects], axis=1)
            .reset_index(drop=False)
            .rename(columns={'index': ANOVA.SOURCE}))

        super().__init__(
            source=data,
            target=(ANOVA.EFFECTS, ANOVA.TABLE_COLNAMES[1]),
            feature=ANOVA.SOURCE,
            target_on_y=False,
            ncols=2,
            nrows=1,
            stretch_figsize=stretch_figsize)
        
    def plot(self) -> Self: # type: ignore
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
        super().plot(
            Pareto, no_percentage_line=True, skip_na='all')
        super().plot(
            Pareto, highlight=ANOVA.RESIDUAL, skip_na='all')
        return self
    
    def stripes(self) -> Self: # type: ignore
        """Adds a line at position 0 for each subplot except for the 
        probability plot. This line represents the fit of the model."""
        threshold = StripeLine(
            label=STR['charts_label_alpha_th'].format(alpha=self.lm.alpha),
            position=self.lm.effect_threshold,
            orientation='vertical',
            color=COLOR.PALETTE[0])
        for chart in self.itercharts():
            if chart == self.charts[0]:
                chart.stripes([threshold])
                break
        return self

    def label(self, info: bool | str = False, **kwds) -> Self: # type: ignore
        """Adds titles and labels to the charts generated by the 
        `plot()` method.
        
        Parameters
        ----------
        info : bool | str, optional
            If `True`, the method will add an informative subtitle to 
            the chart. If a string is provided, it will be used as the 
            subtitle, by default False.
        **kwds
            Additional keyword arguments to be passed to the `label()`
            method of the `JointChart` instance.
        
        Returns
        -------
        Self
            The `ParameterRelevanceCharts` instance, for method chaining.
        """      
        labels = dict(
                fig_title=STR['paramcharts_fig_title'],
                sub_title=STR['paramcharts_sub_title'],
                target_label=(STR['effects_label'], STR['ss_label']),
                feature_label=(STR['paramcharts_feature_label'], ''),
                info = info
            ) | kwds
        super().label(**labels) # type: ignore
        return self


class ResidualsCharts(JointChart):
    """
    Provides a set of charts for visualizing the residuals of a linear 
    regression model.

    The `ResidualsCharts` class takes a `LinearModel` instance and 
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
    stretch_figsize : bool, optional
        If True, stretch the figure height and width based on the number of
        rows and columns, by default False.
    
    Examples
    --------

    ``` python
    import daspi as dsp
    import pandas as pd

    df = dsp.load_dataset('aspirin-dissolution')
    model = dsp.LinearModel(
        source=df,
        target='dissolution',
        features=['employee', 'stirrer', 'brand', 'catalyst', 'water'],
        disturbances=['temperature', 'preparation'],
        order=2)
    df_gof = pd.DataFrame()
    for data_gof in model.recursive_elimination():
        df_gof = pd.concat([df_gof, data_gof])
    dsp.ResidualsCharts(model).plot().stripes().label(info=True)
    ```
    """
    __slots__ = ('lm')

    lm: LinearModel
    """The linear regression model whose residuals are visualized."""

    def __init__(
            self,
            linear_model: LinearModel,
            stretch_figsize: bool = False
            ) -> None:
        self.lm = linear_model
        super().__init__(
            source=self.lm.residual_data(),
            target=ANOVA.RESIDUAL,
            feature=('', '', ANOVA.PREDICTION, ANOVA.OBSERVATION),
            nrows=2,
            ncols=2,
            sharey=True,
            stretch_figsize=stretch_figsize)
        
    def plot(self) -> Self: # type: ignore
        """Generates a set of four charts for visualizing the residuals 
        of a linear regression model:
        - Probability plot of the residuals
        - Gaussian kernel density estimate of the residuals
        - Scatter plot of the predicted values vs. the observed values
        - Line plot of the predicted values vs. the observed values
        
        Returns
        -------
        Self: 
            The `ResidualsCharts` instance, for method chaining.
        """
        super().plot(Probability, show_fit_ci=True)
        super().plot(GaussianKDE)
        super().plot(Scatter)
        super().plot(Line, kw_call={'marker': 'o'})
        return self
    
    def stripes(self) -> Self: # type: ignore
        """Adds a line at position 0 for each subplot except for the 
        probability plot. This line represents the fit of the model."""
        fit = StripeLine(
            label=STR['fit'],
            position=0,
            orientation='horizontal',
            color=COLOR.PALETTE[0])
        for chart in self.itercharts():
            if chart == self.charts[0]:
                continue
            chart.stripes([fit])
        return self

    def label(self, info: bool | str = False, **kwds) -> Self: # type: ignore
        """Adds titles and labels to the charts generated by the 
        `plot()` method.
        
        Parameters
        ----------
        info : bool | str, optional
            If `True`, the method will add an informative subtitle to 
            the chart. If a string is provided, it will be used as the 
            subtitle, by default False.
        **kwds
            Additional keyword arguments to be passed to the `label()`
            method of the `JointChart` instance.
        
        Returns
        -------
        Self
            The `ResidualsCharts` instance, for method chaining.
        """
        sub_title = f'{self.lm.target} ~ {" + ".join(self.lm.effects().index)}'
        labels = dict(
                fig_title=STR['residcharts_fig_title'],
                sub_title=sub_title,
                target_label=STR['resid_name'],
                feature_label=(
                    STR['charts_flabel_quantiles'],
                    STR['charts_flabel_density'],
                     f'{STR["charts_flabel_predicted"]} {self.lm.target}',
                    STR['charts_flabel_observed']),
                info = info
            ) | kwds
        super().label(**labels) # type: ignore
        return self


class PairComparisonCharts(JointChart):
    """Provides a set of charts for visualizing the pairwise 
    comparison of two variables.
    
    Parameters
    ----------
    source : DataFrame
        The source data.
    target : str or Tuple[str]
        The target variable(s).
    feature : str or Tuple[str]
        The feature variable(s).
    identity : str
        Column name containing identities of each sample, must occur 
        once for each measurement.
    stretch_figsize : bool, optional
        If True, stretch the figure height and width based on the number of
        rows and columns, by default False.
    
    Examples
    --------

    ``` python
    import daspi as dsp

    df = dsp.load_dataset('shoe-sole')
    chart = dsp.PairComparisonCharts(
            source=df,
            target='wear',
            feature='status',
            identity='tester'
        ).plot(
        ).label(
            info=True
        )
    ```
    """

    __slots__ = ('identity')

    identity: str
    """Column name containing identities of each sample."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            identity: str,
            stretch_figsize: bool = False
            ) -> None:
        self.identity = identity
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            nrows=1,
            ncols=2,
            width_ratios=[4, 1],
            categorical_feature = (False, True),
            stretch_figsize=stretch_figsize)
    
    def plot(self) -> Self: # type: ignore
        """Generates a set of two charts for visualizing the difference
        between each pair.
        - BlandAltman where the difference is shown on the y-axis
        - ParallelCoordinate including a MeanTest and Violine plot, to 
        reflect the difference within the absolute values.
        
        Returns
        -------
        Self: 
            The `PairComparisonCharts` instance, for method chaining.
        """
        super().plot(BlandAltman, identity=self.identity, feature_axis='data')
        super().plot(
            ParallelCoordinate, identity=self.identity, show_points=False)
        super().plot(MeanTest, n_groups=2, on_last_axes=True)
        super().plot(QuantileBoxes, strategy='fit', on_last_axes=True)
        return self

    def label(self, info: bool | str = False, **kwds) -> Self: # type: ignore
        """Adds titles and labels to the charts generated by the 
        `plot()` method.
        
        Parameters
        ----------
        info : bool | str, optional
            If `True`, the method will add an informative subtitle to 
            the chart. If a string is provided, it will be used as the 
            subtitle, by default False.
        **kwds
            Additional keyword arguments to be passed to the `label()`
            method of the `JointChart` instance.
        
        Returns
        -------
        Self
            The `PairComparisonCharts` instance, for method chaining.
        """
        labels = dict(
                fig_title=STR['paircharts_fig_title'],
                sub_title=STR['paircharts_sub_title'],
                target_label=(True, True),
                feature_label=(True, True),
                info = info
            ) | kwds
        super().label(**labels) # type: ignore
        return self


class BivariateUnivariateCharts(JointChart):
    """Provides a set of charts for visualizing the relationship between
    a target variable and a feature variable.

    This class prepares a grid for drawing a bivariate plot with 
    marginal univariate plots:
    - Bottom left there is a large square axis for the bivariate
      diagram. For example, use a `LinearRegressionLine` plotter to show 
      the dependence of the target variable on the feature variable.
    - Top left there is a small square axis for the univariate
      plot of the feature variable.
    - Bottom right there is a small square axis for the univariate
      plot of the target variable.
    - Top right there is a small square axis hidden as soon a plot is 
      done.

    Parameters
    ----------
    source : DataFrame
        The source data.
    target : str
        The target (dependant) variable.
    feature : str
        The feature (independant) variable.
    hue : str, optional
        The hue variable.
    dodge_univariates : bool, optional
        Whether to dodge the univariate plots, by default False.
    categorical_feature_univariates : bool, optional
        Whether to treat the feature variable for univariate charts as
        categorical, by default False.
    ratios : List[float], optional
        The ratios of the bivariate axes height to the univariate axes 
        height, by default [4, 1].
    stretch_figsize : bool, optional
        If True, stretch the figure height and width based on the number of
        rows and columns, by default False.
    colors: Tuple[str, ...], optional
        Tuple of unique colors used for hue categories as hex or str. If
        not provided, the default colors will be used, by default ().
    
    Examples
    --------

    ``` python
    import daspi as dsp

    df = dsp.load_dataset('aspirin-dissolution')
    hue = 'brand'
    n_groups = df.groupby(hue).ngroups
    chart = dsp.BivariateUnivariateCharts(
            source=df,
            target='dissolution',
            feature='temperature',
            hue=hue,
            dodge_univariates=True,
        ).plot_univariates(
            dsp.MeanTest, n_groups=n_groups
        ).plot_univariates(
            dsp.QuantileBoxes, strategy='fit'
        ).plot_bivariate(
            dsp.LinearRegressionLine, show_fit_ci=True
        ).label(
            fig_title='Regression and distribution analysis',
            sub_title='Aspirin dissolution time vs. temperature',
            feature_label='Water temperature (°C)',
            target_label='Dissolution time (s)',
            axes_titles=('95 % Bonferroni confidence interval of mean', '', '', ''),
            info=True
        )
    ```
    
    Information about which distribution is used to calculate the
    quantile boxes can be obtained as follows:

    ``` python
    brands = (brand for brand in df[hue].unique().tolist()*2)
    for plot in chart.plots:
        if isinstance(plot, dsp.QuantileBoxes):
            print(f'{next(brands)} {plot.target}: {plot.estimation.dist.name}')
    ```
    """
    __slots__ = ('_top_right_hidden')

    _top_right_hidden: bool
    """Whether the top right axis is hidden."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            hue: str = '',
            dodge_univariates: bool = False,
            categorical_feature_univariates: bool = False,
            ratios: List[float] = [3, 1],
            stretch_figsize: bool = False,
            colors: Tuple[str, ...] = ()
            ) -> None:
        assert len(ratios) == 2, ('ratios must be a list of two floats')
        self._top_right_hidden = False

        super().__init__(
            source=source,
            target=(
                feature, '',
                target, target),
            feature=(
                '', '',
                feature, ''),
            hue=hue,
            dodge=(
                dodge_univariates, False,
                False, dodge_univariates),
            categorical_feature=(
                categorical_feature_univariates, False,
                False, categorical_feature_univariates),
            target_on_y=(
                False, True,
                True, True),
            nrows=2,
            ncols=2,
            sharex='col',
            sharey='row',
            width_ratios=ratios,
            height_ratios=ratios[::-1],
            stretch_figsize=stretch_figsize,
            colors=colors or CATEGORY.PALETTE)
    
    @property
    def hidden_ax(self) -> Axes:
        """Get the top right axis (read-only)."""
        return self.axes[0, 1]
    
    @property
    def univariate_axs(self) -> List[Axes]:
        """Get the univariate axes (read-only)."""
        return [self.axes[0], self.axes[-1]]
    
    @property
    def bivariate_ax(self) -> Axes:
        """Get the bivariate axis (read-only)."""
        return self.axes[1, 0]
    
    def hide_top_right_ax(self) -> None:
        """Hides the top right axis if it is not already hidden."""
        if self._top_right_hidden:
            return

        HideSubplot(self.hidden_ax)()
        self._top_right_hidden = True
    
    def plot( # type: ignore
            self,
            plotter: Type[Plotter],
            kind: Literal['univariate', 'bivariate'],
            *,
            kw_call: Dict[str, Any] = {},
            kw_where: Dict[str, Any] = {},
            **kwds
            ) -> Self:
        if kind == 'bivariate':
            self.plot_bivariate(
                plotter,
                kw_call=kw_call,
                kw_where=kw_where,
                **kwds)
        elif kind == 'univariate':
            self.plot_univariates(
                plotter,
                kw_call=kw_call,
                kw_where=kw_where,
                **kwds)
        else:
            raise ValueError('kind must be either "bivariate" or "univariate"')
        return self
    
    def plot_univariates(
            self,
            plotter: Type[Plotter],
            *,
            kw_call: Dict[str, Any] = {},
            kw_where: Dict[str, Any] = {},
            **kwds
            ) -> Self:
        """Plots the univariate plots on the top left and bottom right
        axes.

        Parameters
        ----------
        plotter : Type[Plotter]
            The plotter to use for the univariate plots.
        kw_call : Dict[str, Any]
            Additional keyword arguments for the plotter call method.
        kw_where : Dict[str, Any]
            Additional keyword arguments for the where method used to
            filter the data.
        **kwds
            Additional keyword arguments to be passed to the super plot
            method.

        Returns
        -------
        Self
            The `BivariateUnivariateCharts` instance, for method 
            chaining.
        """
        self.hide_top_right_ax()
        for ax in self.axes:
            if ax in self.univariate_axs:
                super().plot(
                    plotter,
                    kw_call=kw_call,
                    kw_where=kw_where,
                    on_last_axes=False,
                    **kwds)
            else:
                super().plot(SkipSubplot)
        return self

    def plot_bivariate(
            self,
            plotter: Type[Plotter],
            *,
            kw_call: Dict[str, Any] = {},
            kw_where: Dict[str, Any] = {},
            **kwds) -> Self:
        """Plots the bivariate plot on the bottom left axis.

        Parameters
        ----------
        plotter : Type[Plotter]
            The plotter to use for the univariate plots.
        kw_call : Dict[str, Any]
            Additional keyword arguments for the plotter call method.
        kw_where : Dict[str, Any]
            Additional keyword arguments for the where method used to
            filter the data.
        **kwds
            Additional keyword arguments to be passed to the super plot
            method.

        Returns
        -------
        Self
            The `BivariateUnivariateCharts` instance, for method 
            chaining.
        """
        self.hide_top_right_ax()
        for ax in self.axes:
            if ax == self.bivariate_ax:
                super().plot(
                    plotter,
                    kw_call=kw_call,
                    kw_where=kw_where,
                    on_last_axes=False,
                    **kwds)
            else:
                super().plot(SkipSubplot)
        return self
    
    def label(
            self,
            *,
            fig_title: str = '',
            sub_title: str = '',
            feature_label: str | bool | Tuple = '', 
            target_label: str | bool | Tuple = '', 
            info: bool | str = False,
            axes_titles: Tuple[str, ...] = (),
            rows: Tuple[str, ...] = (),
            cols: Tuple[str, ...] = (),
            row_title: str = '',
            col_title: str = '') -> Self:
        """Add labels and titles to the chart.

        This method sets various labels and titles for the chart,
        including figure title, subplot title, axis labels, row and
        column titles, and additional information.

        Parameters
        ----------
        fig_title : str, optional
            The main title for the entire figure, by default ''.
        sub_title : str, optional
            The subtitle for the entire figure, by default ''.
        feature_label : str | bool | None, optional
            The label for the feature variable (x-axis), by default ''.
            If set to True, the feature variable name will be used.
            If set to False or None, no label will be added.
        target_label : str | bool | None, optional
            The label for the target variable (y-axis), by default ''.
            If set to True, the target variable name will be used.
            If set to False or None, no label will be added.
        info : bool | str, optional
            Additional information to display on the chart. If True,
            the date and user information will be automatically added at
            the lower left corner of the figure. If a string is
            provided, it will be shown next to the date and user,
            separated by a comma. By default, no additional information
            is displayed.
        axes_titles : Tuple[str, ...]
            Title for each Axes, by default ()
        rows: Tuple[str, ...], optional
            The row labels of the figure, by default ().
        cols: Tuple[str, ...], optional
            The column labels of the figure, by default ().
        row_title : str, optional
            The title of the rows, by default ''.
        col_title : str, optional
            The title of the columns, by default ''.

        Returns
        -------
        Self
            The `BivariateUnivariateCharts` instance, for method
            chaining.

        Notes
        -----
        This method allows customization of chart labels and titles to
        enhance readability and provide context for the visualized data.
        """
        super().label(
            fig_title=fig_title,
            sub_title=sub_title,
            feature_label=(
                '', '',
                feature_label, ''),
            target_label=(
                '', '',
                target_label, ''),
            info=info,
            axes_titles=axes_titles,
            rows=rows,
            cols=cols,
            row_title=row_title,
            col_title=col_title)
        return self


class ProcessCapabilityAnalysisCharts(JointChart):
    """A class for creating process capability analysis charts.

    This class extends the `JointChart` class and provides methods for
    creating process capability analysis charts. It allows you to
    visualize process capability analysis data and perform various
    analysis tasks.


    Parameters
    ----------
    source : DataFrame
        The source data.
    target : str
        The target (dependant) variable.
    spec_limits : SpecLimits
        The specification limits for the process capability analysis.
    hue : str, optional
        The hue variable for the chart, by default ''.
    dist : scipy stats rv_continuous, optional
        The probability distribution use for creating feature data
        (the theoretical values). Default is 'norm'.
    error_values : tuple of float, optional
        If the process data may contain coded values for measurement 
        errors or similar, they can be specified here, 
        by default [].
    strategy : {'eval', 'fit', 'norm', 'data'}, optional
        Which strategy should be used to determine the control 
        limits (process spread):
        - `eval`: The strategy is determined according to the given 
        evaluate function. If none is given, the internal `evaluate`
        method is used.
        - `fit`: First, the distribution that best represents the 
        process data is searched for and then the agreed process 
        spread is calculated
        - norm: it is assumed that the data is subject to normal 
        distribution. The variation tolerance is then calculated as 
        agreement * standard deviation
        - data: The quantiles for the process variation tolerance 
        are read directly from the data.
        by default 'norm'
    agreement : float or int, optional
        Specify the tolerated process variation for which the 
        control limits are to be calculated. 
        - If int, the spread is determined using the normal 
        distribution agreement*σ, 
        e.g. agreement = 6 -> 6*σ ~ covers 99.75 % of the data. 
        The upper and lower permissible quantiles are then 
        calculated from this.
        - If float, the value must be between 0 and 1.This value is
        then interpreted as the acceptable proportion for the 
        spread, e.g. 0.9973 (which corresponds to ~ 6 σ)
        by default 6
    possible_dists : tuple of strings or rv_continous, optional
        Distributions to which the data may be subject. Only 
        continuous distributions of scipy.stats are allowed,
        by default DIST.COMMON
    stretch_figsize : bool, optional
        Flag indicating whether figure size should be stretched to fill
        the grid, by default False.
    
    Examples
    --------
    
    ``` python
    import daspi as dsp

    df = dsp.load_dataset('drop_card')
    spec_limits = 0, float(df.loc[0, 'usl'])
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
            info=True)
    ``` 
    """
    __slots__ = ('spec_limits', 'dist', 'kw_estim')

    spec_limits: SpecLimits
    """The specification limits for the process capability analysis."""

    dist: rv_continuous | str
    """The probability distribution use for creating feature data
    (the theoretical values)."""

    kw_estim: Dict[str, Any]
    """Keyword arguments for the ProcessEstimator instances used for 
    calculating the capability indices."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            *,
            spec_limits: SpecLimits,
            hue: str = '',
            dist: rv_continuous | str = 'norm',
            error_values: Tuple[float, ...] = (),
            strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
            agreement: float | int = 6, 
            possible_dists: Tuple[str | rv_continuous, ...] = DIST.COMMON,
            stretch_figsize: bool = False,
            ) -> None:
        assert not spec_limits.both_unbounded, (
            'At least one specification limit must not be None')
        self.spec_limits = spec_limits
        self.dist = dist
        self.kw_estim = dict(
            error_values=error_values,
            strategy=strategy,
            agreement=agreement,
            possible_dists=possible_dists)
        super().__init__(
            source=source,
            target=target,
            feature='',
            hue=hue,
            mosaic=('rr', 'pd', 'ck',),
            height_ratios=[3, 3, 1],
            sharex='row',
            dodge=(False, False, False, True, True),
            target_on_y=(True, False, False, False, False),
            stretch_figsize=stretch_figsize)
    
    @property
    def processes(self) -> List[ProcessEstimator]:
        """Returns a list of ProcessEstimator instances used for 
        calculating the capability indices (read-only).
        """
        for plot in self.plots:
            if isinstance(plot, CapabilityConfidenceInterval):
                return plot.processes
        return []

    def plot(self, **kwds_cpi) -> Self: # type: ignore
        """Plot the process capability analysis charts.
        
        This method plots the process capability analysis charts, including
        scatter plots, probability density functions, Gaussian kernel
        density estimation, and capability confidence intervals.

        Parameters
        ----------
        kwds_cpi : dict
            Additional keyword arguments for the capability confidence
            interval plot.

        Returns
        -------
        Self
            The `ProcessCapabilityAnalysisCharts` instance, for method
            chaining.
        """
        hue = self.hues[0]
        n_groups = self.source.groupby(hue).ngroups if hue else 1
        _kwds_cpi: Dict[str, Any] = dict(
            n_groups=n_groups,
            spec_limits=self.spec_limits,
            show_center=True,
            bars_same_color=True,
            hide_axis='feature',
            visible_spines='target',
            kw_estim=self.kw_estim) | kwds_cpi
        super().plot(Scatter)
        super().plot(Probability, dist=self.dist)
        super().plot(GaussianKDE, hide_axis='feature', visible_spines='target')
        super().plot(CapabilityConfidenceInterval, kind='cpk', **_kwds_cpi)
        if self.spec_limits.is_unbounded:
            super().plot(HideSubplot)
        else:
            super().plot(CapabilityConfidenceInterval, kind='cp', **_kwds_cpi)

        return self
    
    def stripes(
            self,
            *,
            mean: bool = True,
            median: bool = True,
            control_limits: bool = True
            ) -> Self: # type: ignore
        """This method adds stripes to the process capability analysis 
        charts, including mean, median, control limits and specification 
        limits.

        Parameters
        ----------
        mean : bool, optional
            Whether to add a line for the mean, by default True.
        median : bool, optional
            Whether to add a line for the median
            (if applicable), by default True.
        control_limits : bool, optional
            Whether to add lines for the control limits
            (if applicable), by default True.
        
        Returns
        -------
        Self
            The `ProcessCapabilityAnalysisCharts` instance, for method
            chaining.
        """
        def flags(x: bool) -> Generator[bool, Any, None]:
            for flag in (x, False, x, False, False):
                yield flag
        
        super().stripes(
            mean=tuple(flags(mean)),
            median=tuple(flags(median)),
            control_limits=tuple(flags(control_limits)),
            spec_limits=tuple(
                self.spec_limits if f else SpecLimits() for f in flags(True)))
        return self

    def label( # type: ignore
            self,
            *,
            fig_title: str = '',
            sub_title: str = '',
            feature_label: Tuple[str, ...] = (),
            target_label: str = '',
            info: bool | str = False,
            ) -> Self:
        """Label the process capability analysis charts.
        
        This method labels the process capability analysis charts with
        custom labels and titles.

        Parameters
        ----------
        fig_title : str, optional
            The title of the figure, by default ''.
        sub_title : str, optional
            The subtitle of the figure, by default ''.
        feature_label : Tuple[str, ...], optional
            The labels for the features. If not provided, default labels
            will be used. The tuple should contain five elements. By
            default, the labels are:
            ('Observation order',
            '', 'Quantiles of standard normal distribution', 
            '', '')
        target_label : str, optional
            The label for the target variable, by default ''.
        info : bool | str, optional
            Whether to display information about the chart. If True, the
            information will be displayed. If a string is provided, it
            will be used as the title of the information box. By default,
            the information is not displayed.

        Returns
        -------
        Self
            The `ProcessCapabilityAnalysisCharts` instance, for method
            chaining.
        """
        if not target_label:
            target_label = self.targets[0]
        _target_label = 3 * (target_label, ) + (STR['cpk'], STR['cp'])
        if not feature_label:
            feature_label = (
                STR['charts_flabel_observed'], 
                STR['charts_flabel_quantiles'], '', '', '')
        super().label(
            fig_title=fig_title,
            sub_title=sub_title,
            target_label=_target_label,
            feature_label=feature_label,
            info=info)
        return self


__all__ = [
    'ResidualsCharts',
    'ParameterRelevanceCharts',
    'PairComparisonCharts',
    'BivariateUnivariateCharts',
    'ProcessCapabilityAnalysisCharts',
    ]
