"""
The plotter module provides classes for creating and customizing various
types of plots. It also provides a collection of classes for statistical
plotting and analysis.

This module contains several classes that are useful for visualizing and
analyzing statistical data. The classes provide functionality for 
creating various types of plots, including Gaussian KDE plots,
violine plots, error bar plots, and confidence interval plots. These
plots can be used to visually assess statistical differences,
estimate kernel density, display error bars, and evaluate confidence
intervals.

Classes
-------
Plotter:
    An abstract base class for creating plotters.
Scatter:
    A scatter plotter that extends the `Plotter` base class.
Line:
    A line plotter that extends the `Plotter` base class.
LinearRegression:
    A linear regression plotter that extends the `Plotter` base class.
Probability:
    A probability plotter that extends the `LinearRegression` class.
TransformPlotter:
    A base class for creating plotter classes that perform
    transformations on data.
CenterLocation:
    A CenterLocation plotter that extends the TransformPlotter class.
GaussianKDE:
    A class for creating Gaussian Kernel Density Estimation (KDE)
    plotters.
Violine:
    A class for creating violine plotters.
Errorbar:
    A class for creating error bar plotters.
StandardErrorMean:
    A class for creating plotters with error bars representing the
    standard error of the mean.
SpreadWidth:
    A class for creating plotters with error bars representing the 
    spread width.
DistinctionTest:
    A class for creating plotters with error bars representing 
    distinction tests.
MeanTest:
    A class for creating plotters with error bars representing 
    confidence intervals for the mean.
VariationTest:
    A class for creating plotters with error bars representing
    confidence intervals for variation measures.

    
These classes are designed to provide a convenient and intuitive way to
visualize and analyze statistical data. They can be used in a variety of
applications, including data exploration, hypothesis testing, and data
presentation.

Usage
-----
Import the module and use the classes to create and customize plots.

Example
-------
from plotter import Plotter
from plotter import Scatter
from plotter import LinearRegression
from plotter import Probability
from plotter import TransformPlotter
from plotter import CenterLocation

# Create a scatter plot
data = {...}  # Data source for the plot
scatter_plot = Scatter(data, 'target_variable')
scatter_plot()

# Create a linear regression plot
data = {...}  # Data source for the plot
linear_regression_plot = LinearRegression(data, 'target_variable', 'feature_variable')
linear_regression_plot()

# Create a probability plot
data = {...}  # Data source for the plot
probability_plot = Probability(data, 'target_variable', dist='norm', kind='qq')
probability_plot()

# Create a CenterLocation plot
data = {...}  # Data source for the plot
center_plot = CenterLocation(data, 'target_variable', 'feature_variable', kind='mean')
center_plot()
"""#TODO write module docstring

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod

from typing import Any
from typing import Self
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from typing import Hashable
from typing import Callable
from typing import Iterable
from typing import Generator

from numpy.typing import NDArray
from numpy.typing import ArrayLike

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from matplotlib.container import BarContainer

from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous

from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.regression.linear_model import OLSResults

from pandas.api.types import is_scalar
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from .utils import shared_axes

from ..constants import KW
from ..constants import DIST
from ..constants import COLOR
from ..constants import DEFAULT
from ..constants import PLOTTER
from ..constants import CATEGORY
from ..statistics.confidence import fit_ci
from ..statistics.confidence import mean_ci
from ..statistics.confidence import stdev_ci
from ..statistics.estimation import Estimator
from ..statistics.confidence import variance_ci
from ..statistics.confidence import prediction_ci
from ..statistics.estimation import estimate_kernel_density


class Plotter(ABC):
    """Abstract base class for creating plotters.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on the
        y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    """
    __slots__ = (
        'source', 'target', 'feature', '_color', 'target_on_y', 'fig', 'ax')
    source: DataFrame
    target: str
    feature: str
    _color: str | None
    target_on_y: bool
    fig: Figure
    ax: Axes

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            target_on_y : bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            ) -> None:
        self.target_on_y = target_on_y
        self.source = source
        if not feature:
            feature = PLOTTER.FEATURE
            self.source[feature] = np.arange(len(source[target]))
        self.feature = feature
        self.target = target
        
        self.fig, self.ax = plt.subplots(1, 1) if ax is None else ax.figure, ax # type: ignore
        self._color = color
    
    @property
    def x_column(self) -> str:
        """Get column name used to access data for x-axis (read-only)."""
        return self.feature if self.target_on_y else self.target
    
    @property
    def y_column(self) -> str:
        """Get column name used to access data for y-axis (read-only)."""
        return self.target if self.target_on_y else self.feature
        
    @property
    def x(self) -> ArrayLike:
        """Get values used for x-axis (read-only)."""
        return self.source[self.x_column]
    
    @property
    def y(self) -> ArrayLike:
        """Get values used for y-axis (read-only)"""
        return self.source[self.y_column]
    
    @property
    def color(self) -> str | None:
        """Get color of drawn artist"""
        if self._color is None:
            self._color = COLOR.PALETTE[0]
        return self._color

    @abstractmethod
    def __call__(self):
        """
        Perform the plotting operation.

        This method should be overridden by subclasses to provide the specific plotting functionality.
        """

class Scatter(Plotter):
    """A scatter plotter that extends the Plotter base class.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str
        Column name of the feature variable for the plot
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on the
        y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the scatter plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    size : Iterable[int] | None, optional
        The size of the markers in the scatter plot, by default None
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    """
    __slots__ = ('marker', 'size')
    marker: str | None
    size: Iterable[int] | None
    
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str, 
            target_on_y: bool = True, 
            color: str | None = None,
            marker: str | None = None,
            size: Iterable[int] | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)
        self.marker = marker
        self.size = size
    
    def __call__(self, **kwds) -> None:
        """Perform the scatter plot operation.

        Parameters
        ----------
        **kwds : 
            Additional keyword arguments to be passed to the Axes 
            `scatter` method.
        """
        _kwds: Dict[str, Any] = dict(
            c=self.color, marker=self.marker, s=self.size,
            alpha=COLOR.MARKER_ALPHA) | kwds
        self.ax.scatter(self.x, self.y, **_kwds)


class Line(Plotter):
    """A line plotter that extends the Plotter base class.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on the
        y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    """
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            target_on_y: bool = True, 
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)
    
    def __call__(self, marker=None, **kwds) -> None:
        """Perform the scatter plot operation.

        Parameters
        ----------
        marker : str | None, optional
            The marker style for the scatter plot. Available markers see:
            https://matplotlib.org/stable/api/markers_api.html, 
            by default None
        **kwds : 
            Additional keyword arguments to be passed to the Axes `plot` 
            method.
        """
        alpha = None if marker is None else COLOR.MARKER_ALPHA
        _kwds: Dict[str, Any] = dict(
            c=self.color, marker=marker, alpha=alpha) | kwds
        self.ax.plot(self.x, self.y, **_kwds)
            

class LinearRegression(Plotter):
    """A linear regression plotter that extends the Plotter base class.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str
        Column name of the feature variable for the plot,
        by default ''
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on the
        y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    show_points : bool, optional
        Flag indicating whether to show the individual points, 
        by default True.
    show_fit_ci : bool, optional
        Flag indicating whether to show the confidence interval for
        the fitted line as filled area, by default False.
    show_pred_ci : bool, optional
        Flag indicating whether to show the confidence interval for 
        predictions as additional lines, by default False
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    """
    __slots__ = (
        'model', 'target_fit', 'show_points', 'show_fit_ci', 'show_pred_ci')
    model: OLSResults
    target_fit: str
    show_points: bool
    show_fit_ci: bool
    show_pred_ci: bool

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            show_points: bool = True,
            show_fit_ci: bool = False,
            show_pred_ci: bool = False,
            **kwds) -> None:
        self.target_fit = PLOTTER.FITTED_VALUES_NAME
        self.show_points = show_points
        self.show_fit_ci = show_fit_ci
        self.show_pred_ci = show_pred_ci
        df = source if isinstance(source, DataFrame) else pd.DataFrame(source)
        df = (df
            .sort_values(feature)
            [[feature, target]]
            .dropna(axis=0, how='any')
            .reset_index(drop=True))
        self.model = sm.OLS(df[target], sm.add_constant(df[feature])).fit() # type: ignore
        df[self.target_fit] = self.model.fittedvalues
        df = pd.concat([df, self.ci_data()], axis=1)
        super().__init__(
            source=df, target=target, feature=feature, target_on_y=target_on_y,
            color=color, ax=ax)
        
    @property
    def x_fit(self) -> ArrayLike:
        """Get values used for x-axis for fitted line (read-only)."""
        name = self.feature if self.target_on_y else self.target_fit
        return self.source[name]
    
    @property
    def y_fit(self) -> ArrayLike:
        """Get values used for y-axis for fitted line (read-only)"""
        name = self.target_fit if self.target_on_y else self.feature
        return self.source[name]
    
    def ci_data(self) -> pd.DataFrame:
        """Get confidence interval for prediction and fitted line as 
        DataFrame."""
        data = (
            *fit_ci(self.model),
            *prediction_ci(self.model))
        ci_data = pd.DataFrame(
            data = np.array(data).T, 
            columns = PLOTTER.REGRESSION_CI_NAMES)
        return ci_data
    
    def __call__(
            self, kw_scatter: dict = {}, kw_fit_ci: dict = {},
            kw_pred_ci: dict = {}, **kwds) -> None:
        """
        Perform the linear regression plot operation.

        Parameters
        ----------
        kw_scatter : dict, optional
            Additional keyword arguments for the Axes `scatter` method,
            by default {}.
        kw_fit_ci : dict, optional
            Additional keyword arguments for the confidence interval of 
            the fitted line (Axes `fill_between` method), by default {}.
        kw_pred_ci : dict, optional
            Additional keyword arguments for the confidence interval of
            the predictions (Axes plot method), by default {}
        **kwds:
            Additional keyword arguments to be passed to the fit line
            plot (Axes `plot` method).
        """
        color = dict(color=self.color)
        _kwds: Dict[str, Any] = KW.FIT_LINE | color | kwds
        self.ax.plot(self.x_fit, self.y_fit, **_kwds)
        
        if self.show_points:
            kw_scatter = color | kw_scatter
            self.ax.scatter(self.x, self.y, **kw_scatter)
        if self.show_fit_ci:
            kw_fit_ci = KW.FIT_CI | color | kw_fit_ci
            lower = self.source[PLOTTER.FIT_CI_LOW]
            upper = self.source[PLOTTER.FIT_CI_UPP]
            if self.target_on_y:
                self.ax.fill_between(self.x, lower, upper, **kw_fit_ci)
            else:
                self.ax.fill_betweenx(self.y, lower, upper, **kw_fit_ci)
        if self.show_pred_ci:
            kw_pred_ci = KW.PRED_CI | color | kw_pred_ci
            lower = self.source[PLOTTER.PRED_CI_LOW]
            upper = self.source[PLOTTER.PRED_CI_UPP]
            if self.target_on_y:
                self.ax.plot(self.x, lower, self.x, upper, **kw_pred_ci)
            else:
                self.ax.plot(lower, self.y, upper, self.y, **kw_pred_ci)


class Probability(LinearRegression):
    """A Q-Q and P-P probability plotter that extends the 
    LinearRegression class.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    dist : scipy stats rv_continuous
        The probability distribution use for creating feature data
        (the theoretical values).
    kind : Literal['qq', 'pp', 'sq', 'sp']:
        The type of probability plot to create. The first letter
        corresponds to the target, the second to the feature.
        - qq: target = sample quantile; feature = theoretical
            quantile
        - pp: target = sample percentile; feature = theoretical 
            percentile
        - sq: target = sample data; feature = theoretical quantiles
        - sp: target = sample data, feature = theoretical
            percentiles
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    show_points : bool, optional
        Flag indicating whether to show the individual points, 
        by default True.
    show_fit_ci : bool, optional
        Flag indicating whether to show the confidence interval for
        the fitted line as filled area, by default False.
    show_pred_ci : bool, optional
        Flag indicating whether to show the confidence interval for 
        predictions as additional lines, by default False.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Raises
    ------
    AssertionError
        If given kind is not one of 'qq', 'pp', 'sq' or 'sp'
    """
    __slots__ = ('dist', 'kind', 'prob_fit')
    dist: rv_continuous
    kind: Literal['qq', 'pp', 'sq', 'sp']
    prob_fit: ProbPlot

    def __init__(
            self,
            source: DataFrame,
            target: str,
            dist: str | rv_continuous = 'norm',
            kind: Literal['qq', 'pp', 'sq', 'sp'] = 'sq',
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            show_points: bool = True,
            show_fit_ci: bool = True,
            show_pred_ci: bool = True,
            **kwds) -> None:
        assert kind in ('qq', 'pp', 'sq', 'sp'), (
            f'kind must be one of {"qq", "pp", "sq", "sp"}, got {kind}')

        self.kind = kind
        self.dist = dist if not isinstance(dist, str) else getattr(stats, dist)
        self.prob_fit = ProbPlot(source[target], self.dist, fit=True) # type: ignore
        
        feature_kind = 'quantiles' if self.kind[1] == "q" else 'percentiles'
        feature = f'{self.dist.name}_{feature_kind}'
        df = pd.DataFrame({
            target: self.sample_data,
            feature: self.theoretical_data})

        super().__init__(
            source=df, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax, 
            show_points=show_points, show_fit_ci=show_fit_ci,
            show_pred_ci=show_pred_ci)
    
    def _xy_scale_(self) -> Tuple[str, str]:
        """If given distribution is exponential or logaritmic, change
        axis scale for samples or quantiles (not for percentiles) from
        'linear' to 'log'."""
        xscale = yscale = 'linear'
        if self.dist.name in ('expon', 'log', 'lognorm'):
            if self.kind[1] == 'q': 
                xscale = 'log'
            if self.kind[0] == 'q': 
                yscale = 'log'
        if not self.target_on_y:
            xscale, yscale = yscale, xscale
        return xscale, yscale
        
    def format_axis(self) -> None:
        """Format the x-axis and y-axis based on the probability plot 
        type."""
        xscale, yscale = self._xy_scale_()
        self.ax.set(xscale=xscale, yscale=yscale)
        if self.kind[0] == 'p':
            axis = self.ax.yaxis if self.target_on_y else self.ax.xaxis
            axis.set_major_formatter(PercentFormatter(xmax=1.0, symbol='%'))
        if self.kind[1] == 'p':
            axis = self.ax.xaxis if self.target_on_y else self.ax.yaxis
            axis.set_major_formatter(PercentFormatter(xmax=1.0, symbol='%'))
    
    @property
    def sample_data(self) -> NDArray:
        """Get fitted samples (target data) according to given kind"""
        match self.kind[0]:
            case 'q':
                data = self.prob_fit.sample_percentiles
            case 'p':
                data = self.prob_fit.sample_percentiles
            case 's' | _:
                data = self.prob_fit.sorted_data
        return data

    @property
    def theoretical_data(self) -> NDArray:
        """Get theoretical data (quantiles or percentiles) according to 
        the given kind."""
        match self.kind[1]:
            case 'q':
                data = self.prob_fit.theoretical_quantiles
            case 'p' | _:
                data = self.prob_fit.theoretical_percentiles
        return data
    
    def __call__(
            self, kw_scatter: dict = {}, kw_fit_ci: dict = {},
            kw_pred_ci: dict = {}, **kwds) -> None:
        """Perform the probability plot operation.

        Parameters
        ----------
        kw_scatter : dict, optional
            Additional keyword arguments for the Axes `scatter` method,
            by default {}.
        kw_fit_ci : dict, optional
            Additional keyword arguments for the confidence interval of 
            the fitted line (Axes `fill_between` method), by default {}.
        kw_pred_ci : dict, optional
            Additional keyword arguments for the confidence interval of
            the predictions (Axes plot method), by default {}
        **kwds:
            Additional keyword arguments to be passed to the fit line
            plot (Axes `plot` method).
        """
        super().__call__(kw_scatter, kw_fit_ci, kw_pred_ci, **kwds)
        self.format_axis()


class ParallelCoordinate(Plotter):
    """This plotter allows to compare the feature of several individual 
    observations on a set of numeric variables.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str
        Column name of the categorical feature variable (coordinates).
    identities : str
        Column name of identities of each sample, must occur once for 
        each coordinate.
    show_points : bool, optional
        Flag indicating whether to show the individual points, 
        by default True.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    """

    __slots__ = ('identities', 'show_points')
    identities: str
    show_points: bool

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            identities: str,
            show_points: bool = True,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.identities = identities
        self.show_points = show_points
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)

    def __call__(self, **kwds):
        """Perform the parallel coordinate plot

        Parameters
        ----------
        **kwds:
            Additional keyword arguments to be passed to the fit line
            plot (Axes `plot` method)."""
        marker = kwds.pop('marker', plt.rcParams['lines.marker'])
        if not self.show_points:
            marker = None
        _kwds: Dict[str, Any] = dict(color=self.color, marker=marker) | kwds
        for identity, group in self.source.groupby(self.identities):
            self.ax.plot(group[self.x_column], group[self.y_column], **_kwds)


class BlandAltman(Plotter):
    """Generate a Bland-Altman plot to compare two sets of measurements.

    Bland-Altman plots [1]_ are extensively used to evaluate the 
    agreement among two different instruments or two measurements 
    techniques. They allow identification of any systematic difference 
    between the measurements (i.e., fixed bias) or possible outliers.

    The mean difference (= second - first) is the estimated bias, and 
    the SD of the differences measures the random fluctuations around 
    this mean. If the mean value of the difference differs significantly 
    from 0 on the basis of a 1-sample t-test, this indicates the 
    presence of fixed bias. If there is a consistent bias, it can be 
    adjusted for by subtracting the mean difference from the new method.

    It is common to compute 95% limits of agreement for each comparison 
    (average difference ± 1.96 standard deviation of the difference), 
    which tells us how far apart measurements by 2 methods were more 
    likely to be for most individuals. If the differences within 
    mean ± 1.96 SD are not clinically important, the two methods may be 
    used interchangeably. The 95% limits of agreement can be unreliable 
    estimates of the population parameters especially for small sample 
    sizes so, when comparing methods or assessing repeatability, it is 
    important to calculate confidence intervals for the 95% limits of 
    agreement.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot. 
    target : str
        Column name of the target variable (the second measurement).
    feature : str
        Column name of the feature variable (the first or reference
        measurement).
    agreement : float, optional
        Multiple of the standard deviation to plot agreement limits
        (in both direction). The defaults is 3.92 (± 1.96), which 
        corresponds to 95 % confidence interval if the differences
        are normally distributed.
    confidence : float or None, optional
        If not None, plot the specified percentage confidence
        interval of the mean and limits of agreement. The CIs of the
        mean difference and agreement limits describe a possible
        error in the estimate due to a sampling error. The greater
        the sample size, the narrower the CIs will be,
        by default 0.95
    feature_axis : Literal['mean', 'data']
        Definition of data used as feature axis (reference axis). 
        - If 'mean' the mean for each measurement is calculated
        as (first + second)/2.
        - If 'data' the feature values are used for feature axis.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).

    Notes
    -----
    The code is an adaptation of the Pingouin package.
    https://pingouin-stats.org/generated/pingouin.plot_blandaltman.html
    
    The pingouin implementation is also a simplified version of the 
    PyCombare package:
    https://github.com/jaketmp/pyCompare
    
    References
    ----------
    .. [1] Bland, J. M., & Altman, D. (1986). Statistical methods for 
           assessing agreement between two methods of clinical
           measurement. The lancet, 327(8476), 307-310.
    """
    __slots__ = ('confidence', 'estimation')
    confidence: float
    estimation: Estimator

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            agreement: float = 3.92,
            confidence: float = 0.95, 
            feature_axis: Literal['mean', 'data'] = 'mean',
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        df = pd.DataFrame()
        _target = f'{target} - {feature}'
        df[_target] = source[target] - source[feature]
        if feature_axis == 'mean':
            _feature = feature_axis
            df[_feature] = (np
                .vstack((source[target].values, source[feature].values)) # type: ignore
                .mean(axis=0))
        else:
            _feature = feature
            df[_feature] = source[feature]
        super().__init__(
            source=df, target=_target, feature=_feature,
            target_on_y=target_on_y, color=color, ax=ax)
        self.confidence = confidence
        self.estimation = Estimator(
            samples=df[_target], strategy='norm', agreement=agreement)

    def __call__(self, **kwds):
        """Perform the Bland-Altman plot operation.

        Parameters
        ----------
        **kwds : 
            Additional keyword arguments to be passed to the Axes
            `scatter` method.
        """
        _kwds: Dict[str, Any] = dict(color=self.color) | kwds
        self.ax.scatter(self.x, self.y, **_kwds)
        
        kws = (KW.CONTROL_LINE, KW.CONTROL_LINE, KW.MEAN_LINE)
        attrs = ('lcl', 'ucl', 'mean')
        ci_funs = ('stdev_ci', 'stdev_ci', 'mean_ci')
        for kw, attr, ci_fun in zip(kws, attrs, ci_funs):
            value = getattr(self.estimation, attr)
            _low, _upp = getattr(self.estimation, ci_fun)()
            span = _upp - _low
            low = value - span/2
            upp = value + span/2
            if self.target_on_y:
                self.ax.axhline(value, **kw)
                if self.confidence is not None:
                    self.ax.axhspan(low, upp, **KW.STRIPES_CONFIDENCE)
            else:
                self.ax.axvline(value, **kw)
                if self.confidence is not None:
                    self.ax.axvspan(low, upp, **KW.STRIPES_CONFIDENCE)


class TransformPlotter(Plotter):
    """Abstract base class for creating plotters.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''
    f_base : int | float, optional
        Value that serves as the base location (offset) of the 
        feature values. Only taken into account if feature is not 
        given, by default `DEFAULT.FEATURE_BASE
`.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    """
    __slots__ = ('_f_base', '_original_f_values')
    _f_base: int | float
    source: DataFrame
    _original_f_values: tuple
    
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            f_base: int | float = DEFAULT.FEATURE_BASE
,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self._f_base = f_base
        self.target = target
        self.feature = feature
        self._original_f_values = ()
        
        df = pd.DataFrame()
        for _feature, _target in self.feature_grouped(source):
            _data = self.transform(_feature, _target)
            df = pd.concat([df, _data], axis=0)
        df.reset_index(drop=True, inplace=True)
        super().__init__(
            source=df, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)
    
    def feature_grouped(
            self, source: DataFrame) -> Generator[Tuple, Self, None]:
        """Group the data by the feature variable and yield the 
        transformed data for each group.

        Parameters
        ----------
        source : pandas DataFrame
            Pandas long format DataFrame containing the data source for
            the plot.

        Yields
        ------
        feature_data : int | float
            Base location (offset) of feature axis.
        target_data : pandas Series
            feature grouped target data used for transformation.
        """
        if self.feature and self.feature != PLOTTER.TRANSFORMED_FEATURE:
            for i, (f_value, group) in enumerate(
                    source.groupby(self.feature, sort=True),
                    start=DEFAULT.FEATURE_BASE
):
                self._original_f_values = self._original_f_values + (f_value, )
                if isinstance(f_value, (float, int)):
                    feature_data = f_value
                else:
                    feature_data = i
                target_data = group[self.target]
                yield feature_data, target_data
        else:
            self.feature = PLOTTER.TRANSFORMED_FEATURE
            self._original_f_values = (self._f_base, )
            feature_data = self._f_base
            target_data = source[self.target]
            yield feature_data, target_data
    
    @abstractmethod
    def transform(
        self, feature_data: float | int, target_data: Series) -> DataFrame:
         """Perform the transformation on the target data and return the
        transformed data.

        This method should be overridden by subclasses to provide the
        specific plotting functionality.

        Parameters
        ----------
        feature_data : int | float
            Base location (offset) of feature axis coming from 
            `feature_grouped' generator.
        target_data : pandas Series
            feature grouped target data used for transformation, coming
            from `feature_grouped' generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        """
    
    @abstractmethod
    def __call__(self) -> None:
        """Perform the plotting operation.

        This method should be overridden by subclasses to provide the
        specific plotting functionality.
        """


class CenterLocation(TransformPlotter):
    """A center location (mean or median) plotter that extends the 
    TransformPlotter class.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''
    kind : Literal['mean', 'median'], optional
        The type of center to plot ('mean' or 'median'),
        by default 'mean'.
    marker : str | None, optional
        The marker style for the scatter plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    show_line : bool
        Flag indicating whether to draw a line between the calculated 
        mean or median points.
    show_points : bool, optional
        Flag indicating whether to show the individual points, 
        by default True.
    f_base : int | float, optional
        Value that serves as the base location (offset) of the 
        feature values. Only taken into account if feature is not 
        given, by default `DEFAULT.FEATURE_BASE
`.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    """
    __slots__ = ('_kind', 'show_line', 'show_points', 'marker')
    _kind: Literal['mean', 'median']
    points: bool
    show_line: bool
    show_points: bool
    marker: str | None
    source: DataFrame

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            kind: Literal['mean', 'median'] = 'mean',
            marker: str | None = None,
            show_line: bool = True,
            show_points: bool = True, #FIXME does not work as individual points only for center points: change to center
            f_base: int | float = DEFAULT.FEATURE_BASE
,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self._kind = 'mean'
        self.kind = kind
        self.show_line = show_line
        self.show_points = show_points
        self.marker = marker if marker else plt.rcParams['scatter.marker']
        super().__init__(
            source=source, target=target, feature=feature, f_base=f_base,
            target_on_y=target_on_y, color=color, ax=ax)

    @property
    def kind(self)-> Literal['mean', 'median']:
        """Get and set the type of location to plot ('mean' or 'median')
        
        Raises
        ------
        AssertionError
            If neither 'mean' or 'median' is given when setting `kind`.
        """
        return self._kind
    @kind.setter
    def kind(self, kind: Literal['mean', 'median']):
        assert kind in ('mean', 'median')
        self._kind = kind
    
    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        """Calculate the mean or median of the target data and return the
        transformed data.

        Parameters
        ----------
        feature_data : int | float
            Base location (offset) of feature axis coming from 
            `feature_grouped' generator.
        target_data : pandas Series
            feature grouped target data used for transformation, coming
            from `feature_grouped' generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        """
        t_value = getattr(target_data, self.kind)()
        data = pd.DataFrame({
            self.target: [t_value],
            self.feature: [feature_data]})
        return data
    
    def __call__(self, **kwds) -> None:
        """
        Perform the center plot operation.

        Parameters
        ----------
        **kwds
            Additional keyword arguments to be passed to the Axes
            `plot` method.
        """
        marker = self.marker if self.show_points else ''
        linestyle = plt.rcParams['lines.linestyle'] if self.show_line else ''
        alpha = None if marker is None else COLOR.MARKER_ALPHA
        _kwds: Dict[str, Any] = dict(
            c=self.color, marker=marker, linestyle=linestyle, alpha=alpha
            ) | kwds
        self.ax.plot(self.x, self.y, **_kwds)


class Bar(TransformPlotter):
    """A bar plotter that extends the TransformPlotter class.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str
        Column name of the feature variable for the plot.
    stack : bool, optional
        Flag indicating whether to stack the bars. It is checked 
        whether there are already bars at each feature position, if
        so the new ones are stacked on top of the existing ones,
        by default True
    width: float, optional
        Width of the bars, by default `CATEGORY.FEATURE_SPACE`
    kw_method : dict, optional
        Additional keyword arguments to be passed to the method,
        by default {}
    method : str, optional
        A pandas Series method to use for aggregating target values 
        within each feature level. Like 'sum', 'count' or similar
        that returns a scalar, by default None
    f_base : int | float, optional
        Value that serves as the base location (offset) of the 
        feature values. Only taken into account if feature is not 
        given, by default `DEFAULT.FEATURE_BASE
`.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    """
    __slots__ = ('method', 'kw_method', 'stack', 'width')
    method: str | None
    kw_method: dict
    stack: bool
    width: float

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            stack: bool = True,
            width: float = CATEGORY.FEATURE_SPACE,
            method: str | None = None,
            kw_method: dict = {},
            f_base: int | float = DEFAULT.FEATURE_BASE
,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.stack = stack
        self.width = width
        self.method = method
        self.kw_method = kw_method
        super().__init__(
            source=source, target=target, feature=feature, f_base=f_base,
            target_on_y=target_on_y, color=color, ax=ax, **kwds)

        if self.method is not None:
            target = f'{self.target} {self.method}'
            self.source = self.source.rename(columns={self.target: target})
            self.target = target

    @property
    def bars(self) -> List[BarContainer]:
        """Get a list of BarContainer objects representing the bars in
        the plot."""
        return [c for c in self.ax.containers if isinstance(c, BarContainer)]
    
    @property
    def t_base(self) -> NDArray:
        """Get the base values for the bars (target), contains zeros 
        when not stacked."""
        feature_ticks = self.source[self.feature]
        t_base: NDArray = np.zeros(len(feature_ticks))
        if not self.stack: 
            return t_base

        for bar in self.bars:
            boxs = [p.get_bbox() for p in bar.patches]
            if self.target_on_y:
                low, upp = map(tuple, zip(*[(b.x0, b.x1) for b in boxs]))
            else:
                low, upp = map(tuple, zip(*[(b.y0, b.y1) for b in boxs]))
            data_values: NDArray = np.array(bar.datavalues)
            if (all(np.greater(feature_ticks, low))
                and all(np.less(feature_ticks, upp))
                and any(np.greater(data_values, t_base))):
                t_base = data_values
        return t_base
    
    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        """Perform the given Series method on the target data if given 
        and return the transformed data. If no method is given, the 
        target data is adopted directly.

        Parameters
        ----------
        feature_data : int | float
            Base location (offset) of feature axis coming from 
            `feature_grouped' generator.
        target_data : pandas Series
            feature grouped target data used for transformation, coming
            from `feature_grouped' generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        
        Raises
        ------
        AssertionError
            When transformed target data (or original data if no method
            is specified) is non-scalar.
        """
        if self.method is not None:
            t_value = getattr(target_data, self.method)(**self.kw_method)
            assert is_scalar(t_value), (
                f'{self.method} does not return a scalar')
            t_value = [t_value]
        else:
            t_value = target_data
            assert len(t_value) <= 1, (
                'Each feature level must contain only one target value, '
                'as the length of the bar')
        
        data = pd.DataFrame({
            self.target: t_value,
            self.feature: [feature_data]})
        return data

    def __call__(self, **kwds) -> None:
        """
        Perform the bar plot operation.

        Parameters
        ----------
        **kwds
            Additional keyword arguments to be passed to the Axes `bar`
            (or 'barh' if target_on_y is False) method.
        """
        if self.target_on_y:
            self.ax.bar(
                self.x, self.y, width=self.width, bottom=self.t_base, **kwds)
        else:
            self.ax.barh(
                self.y, self.x, height=self.width, left=self.t_base, **kwds)


class Pareto(Bar):
    """A plotter to perform a pareto chart that extends the Bar plotter
    class.

    A Pareto chart is a type of chart that combines a bar graph and a 
    line graph. It is used to display and analyze data in order to 
    prioritize and identify the most significant factors contributing 
    to a particular phenomenon or problem. The line graph in a Pareto 
    chart shows the cumulative percentage of the total, which helps 
    identify the point at which a significant portion of the cumulative 
    total is reached.

    Pareto charts are commonly used in quality control, process 
    improvement, and decision-making processes. They allow users to 
    visually identify and focus on the most significant factors that 
    contribute to a problem or outcome, enabling them to allocate 
    resources and address the most critical issues first.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str
        Column name of the feature variable for the plot.
    highlight : Any, optional
        The feature value whose bar should be highlighted in the 
        diagram, by default None.
    highlight_color : str, optional
        The color to use for highlighting, by default `COLOR.BAD`.
    highlighted_as_last: bool, optional
        Whether the highlighted bar should be at the end, by default
        True.
    width: float, optional
        Width of the bars, by default `CATEGORY.FEATURE_SPACE`.
    method : str, optional
        A pandas Series method to use for aggregating target values 
        within each feature level. Like 'sum', 'count' or similar that
        returns a scalar, by default None.
    kw_method : dict, optional
        Additional keyword arguments to be passed to the method,
        by default {}.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on the
        y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first color
        is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with one
        Axes is created, by default None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is used
        within chart objects).
    
    Raises
    ------
    AssertionError
        If 'categorical_feature' is True, coming from Chart objects.
    AssertionError
        If an other Axes object in this Figure instance shares the
        feature axis.
    """
    __slots__ = ('highlight', 'highlight_color', 'highlighted_as_last')
    highlight: Any
    highlight_color: str
    highlighted_as_last: bool

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            highlight: Any = None,
            highlight_color: str = COLOR.BAD,
            highlighted_as_last: bool = True,
            width: float = CATEGORY.FEATURE_SPACE,
            method: str | None = None,
            kw_method: dict = {},
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        assert not (kwds.get('categorical_feature', False)), (
            "Don't set categorical_feature to True for Pareto charts, "
            'it would mess up the axis tick labels')
        self.highlight = highlight
        self.highlight_color = highlight_color
        self.highlighted_as_last = highlighted_as_last
        
        super().__init__(
            source=source, target=target, feature=feature, stack=False,
            width=width, method=method, kw_method=kw_method,
            target_on_y=target_on_y, color=color, ax=ax, **kwds)
        self.source[self.feature] = self._original_f_values
        assert not self.shared_feature_axes, (
            'Do not use Pareto plotter in an chart where the feature axis '
            'is shared with other axes. Pareto sorts the feature axis, '
            'which can mess up the other axes')
    
    @property
    def shared_feature_axes(self) -> bool:
        """True if any other ax in this figure shares the feature axes."""
        return any(shared_axes(self.ax, 'x' if self.target_on_y else 'y', True))

    @property
    def indices(self) -> List[int] | List[Hashable]:
        """Get arranged index values to access the target data (from 
        source data) in the order to be plotted."""
        indices = (self.source
            .sort_values(self.target, ascending = not self.target_on_y)
            .index.to_list())
        if self.highlighted_as_last and self.highlight is not None:
            items = self.source[self.feature].items()
            idx_last = [i for i, v in items if v == self.highlight]
            if self.target_on_y:
                indices = [i for i in indices if i not in idx_last] + idx_last
            else:
                indices = idx_last + [i for i in indices if i not in idx_last]
        return indices
    
    @property
    def x(self) -> Any:
        """Get the values used for the x-axis so that the target is 
        displayed in descending order and the highlighted bar is at the
        end (if so specified)."""
        if self.target_on_y:
            return self.source.loc[self.indices, self.feature] # type: ignore
        else:
            return self.source.loc[self.indices, self.target] # type: ignore
    
    @property
    def y(self) -> Any:
        """Get the values used for the y-axis so that the target is 
        displayed in descending order and the highlighted bar is at the
        end (if so specified)."""
        if not self.target_on_y:
            return self.source.loc[self.indices, self.feature] # type: ignore
        else:
            return self.source.loc[self.indices, self.target] # type: ignore

    def _highlight_bar_(self, bars: BarContainer) -> None:
        """Highlight the specified bar if `highlight` is set True."""
        feature_values = self.x if self.target_on_y else self.y
        for i, f_value in enumerate(feature_values):
            if f_value == self.highlight:
                bars[i].set_color(self.highlight_color)
    
    def _set_margin_(self) -> None:
        """Set margin in feature axis direction to ensure space for 
        percentage values."""
        if self.target_on_y:
            self.ax.set_xmargin(PLOTTER.PARETO_F_MARGIN)
        else:
            self.ax.set_ymargin(PLOTTER.PARETO_F_MARGIN)
    
    def _remove_feature_grid_(self) -> None:
        """Remove grid lines of feature axis"""
        if self.target_on_y:
            self.ax.xaxis.grid(False)
        else:
            self.ax.yaxis.grid(False)

    def add_percentage_texts(self) -> None:
        """Add percentage texts on top of major grids"""
        if hasattr(self.ax, 'has_pc_texts'):
            return
        
        n_texts = PLOTTER.PARETO_N_TICKS-1
        max_value = self.source[self.target].sum()
        ticks = np.linspace(0, max_value, PLOTTER.PARETO_N_TICKS)
        positions = np.linspace(0.1, 1, n_texts)/PLOTTER.PARETO_AXLIM_FACTOR
        texts = [f'{int(pc)} %' for pc in np.linspace(10, 100, n_texts)]
        if self.target_on_y:
            self.ax.set_yticks(ticks)
            self.ax.set_ylim(top=max_value*PLOTTER.PARETO_AXLIM_FACTOR)
            for pos, text in zip(positions, texts):
                self.ax.text(
                    y=pos, s=text, transform=self.ax.transAxes, **KW.PARETO_V)
        else:
            self.ax.set_xticks(ticks)
            self.ax.set_xlim(right=max_value*PLOTTER.PARETO_AXLIM_FACTOR)
            for pos, text in zip(positions, texts):
                self.ax.text(
                    x=pos, s=text, transform=self.ax.transAxes, **KW.PARETO_H)
        self.ax.has_pc_texts = True # type: ignore

    def __call__(self, **kwds) -> None:
        """Perform the pareto plot operation.

        Parameters
        ----------
        **kwds
            Additional keyword arguments to be passed to the Axes `bar`
            (or 'barh' if target_on_y is False) method.
        """
        if self.target_on_y:
            bars = self.ax.bar(self.x, self.y, width=self.width, **kwds)
            self.ax.plot(
                self.x, self.y.cumsum(), color=self.color,
                **KW.PARETO_LINE)
        else:
            bars = self.ax.barh(self.y, self.x, height=self.width, **kwds)
            self.ax.plot(
                self.x[::-1].cumsum(), self.y[::-1], color=self.color,
                **KW.PARETO_LINE)
        self._highlight_bar_(bars)
        self._set_margin_()
        self._remove_feature_grid_()
        self.add_percentage_texts()


class Jitter(TransformPlotter):
    """A class for creating jitter plotters.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''
    width : float
        The width of the jitter, by default `CATEGORY.FEATURE_SPACE`.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure ovject with
        one Axes is created, by default None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    """
    __slots__ = ('width')
    width: float

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            width: float = CATEGORY.FEATURE_SPACE,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.width = width
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax, **kwds)
        
    def jitter(self, loc: float, size: int) -> NDArray:
        """Generates normally distributed jitter values. The standard 
        deviation is selected so that +- 6 sigma corresponds to the 
        permissible width. To ensure the width, values that lie outside 
        this range are restricted to the limits.
        
        Parameters
        ----------
        loc : float
            Center position (feature axis) of the jitted values.
        size : int
            Amount of valaues to generate
        
        Returns
        -------
        jitter : 1D array
            Normally distributed values, but not wider than the given 
            width
        """
        jiiter = np.clip(
            np.random.normal(loc=loc, scale=self.width/6, size=size),
            a_min = loc - self.width/2,
            a_max = loc + self.width/2)
        return jiiter
        
    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        """Normally randomize the target data for each feature value in 
        the feature axis direction.

        Parameters
        ----------
        feature_data : int | float
            Base location (offset) of feature axis coming from 
            `feature_grouped' generator.
        target_data : pandas Series
            feature grouped target data used for transformation, coming
            from `feature_grouped' generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        """
        data = pd.DataFrame({
            self.target: target_data,
            self.feature: self.jitter(feature_data, target_data.size)})
        return data

    def __call__(self, **kwds) -> None:
        """Perform the jitter plot operation.

        Parameters
        ----------
        **kwds
            Additional keyword arguments to be passed to the Axes 
            `scatter` method.
        """
        _kwds: Dict[str, Any] = dict(color=self.color) | kwds
        self.ax.scatter(self.x, self.y, **_kwds)


class GaussianKDE(TransformPlotter):
    """Class for creating Gaussian Kernel Density Estimation (KDE) 
    plotters.

    Kernel density estimation is a way to estimate the probability 
    density function (PDF) of a random variable in a non-parametric way.
    The used gaussian_kde function of scipy.stats works for both
    uni-variate and multi-variate data. It includes automatic bandwidth
    determination. The estimation works best for a unimodal 
    distribution; bimodal or multi-modal distributions tend to be 
    oversmoothed.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    stretch : float, optional
        Factor by which the curve was stretched in height,
        by default 1.
    height : float | None, optional
        Height of the KDE curve at its maximum, by default None.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure object with 
        one Axes is created, by default None.
    show_density_axis : bool, optional
        Flag indicating whether to show the density axis,
        by default True.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    """
    __slots__ = ('_height', '_stretch', 'show_density_axis')
    _height: float | None
    _stretch: float
    show_density_axis: bool

    def __init__(
            self,
            source: DataFrame,
            target: str,
            stretch: float = 1,
            height: float | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            show_density_axis: bool = True,
            **kwds) -> None:
        self._height = height
        self._stretch = stretch
        self.show_density_axis = show_density_axis
        feature = PLOTTER.TRANSFORMED_FEATURE
        _feature = kwds.pop('feature', '')
        if type(self) != GaussianKDE and _feature:
            feature = _feature
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax, **kwds)
        
    @property
    def height(self) -> float | None:
        """Height of kde curve at its maximum."""
        return self._height
    
    @property
    def stretch(self) -> float:
        """Factor by which the curve was stretched in height"""
        return self._stretch
        
    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        """Perform the transformation on the target data by estimating 
        its kernel density. To obtain a uniform curve, a sequence 
        is generated with a specific number of points in the same range 
        (min to max) as the target data.

        Parameters
        ----------
        feature_data : float | int
            Base location (offset) of feature axis coming from 
            `feature_grouped` generator.
        target_data : pandas Series
            Feature grouped target data used for transformation,
            coming from `feature_grouped` generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot. Contains the 
            generated sequence as target data and the estimation as
            feature data.
        """
        sequence, estimation = estimate_kernel_density(
            data=target_data, stretch=self.stretch, height=self.height,
            base=feature_data)
        data = pd.DataFrame({
            self.target: sequence,
            self.feature: estimation,
            PLOTTER.F_BASE_NAME: feature_data * np.ones(len(sequence))})
        return data
    
    def hide_density_axis(self) -> None:
        """Hide the density axis (spine, ticks and labels)."""
        axis = 'xaxis' if self.target_on_y else 'yaxis'
        spine = 'bottom' if self.target_on_y else 'left'
        getattr(self.ax, axis).set_visible(False)
        self.ax.spines[spine].set_visible(False)
        
    def __call__(self, kw_line: dict = {}, **kw_fill) -> None:
        """Perform the plotting operation.

        Parameters
        ----------
        kw_line : dict, optional
            Additional keyword arguments for the axes `plot` method,
            by default {}.
        **kw_fill : dict, optional
            Additional keyword arguments for the axes `fill_between`
            method, by default {}.
        """
        self.ax.plot(self.x, self.y, **kw_line)
        _kw_fill: Dict[str, Any] = dict(alpha=COLOR.FILL_ALPHA) | kw_fill
        if self.target_on_y:
            self.ax.fill_betweenx(self.y, self._f_base, self.x, **_kw_fill)
        else:
            self.ax.fill_between(self.x, self._f_base, self.y, **_kw_fill)
        if not self.show_density_axis:
            self.hide_density_axis()


class Violine(GaussianKDE):
    """Class for creating violine plotters.

    This violin plot is composed of a double-sided Gaussian kernel
    density estimate. The width of the violin is stretched to fill the
    available width.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''.
    width : float, optional
        Width of the violine, by default CATEGORY.FEATURE_SPACE.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure object with
        one Axes is created, by default None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    """
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            width: float = CATEGORY.FEATURE_SPACE,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            height=width/2, target_on_y=target_on_y, color=color, ax=ax,
            show_density_axis=True, **kwds)

    def __call__(self, **kwds) -> None:
        """
        Perform the plotting operation.

        Parameters
        ----------
        **kwds : dict, optional
            Additional keyword arguments for the fill plot, by default {}.
        """
        _kwds: Dict[str, Any] = dict(
            color=self.color, alpha=COLOR.FILL_ALPHA) | kwds
        for f_base, group in self.source.groupby(PLOTTER.F_BASE_NAME):
            estim_upp = group[self.feature]
            estim_low = 2*f_base - estim_upp # type: ignore
            sequence = group[self.target]
            if self.target_on_y:
                self.ax.fill_betweenx(sequence, estim_low, estim_upp, **_kwds)
            else:
                self.ax.fill_between(sequence, estim_low, estim_upp, **_kwds)


class Errorbar(TransformPlotter):
    """Class for creating error bar plotters.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    lower : str
        Column name of the lower error values.
    upper : str
        Column name of the upper error values.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''.
    show_center : bool, optional
        Flag indicating whether to show the center points,
        by default True.
    bars_same_color : bool, optional
        Flag indicating whether to use same color for error bars as 
        markers for center. If False, the error bars are black,
        by default False
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure object with
        one Axes is created, by default None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    """
    __slots__ = ('lower', 'upper', 'show_center', 'bars_same_color')
    lower: str
    upper: str
    show_center: bool
    bars_same_color: bool

    def __init__(
            self,
            source: DataFrame,
            target: str,
            lower: str,
            upper: str,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.lower = lower
        self.upper = upper
        self.show_center = show_center
        self.bars_same_color = bars_same_color
        if feature not in source:
            feature = PLOTTER.FEATURE
            source[feature] = np.arange(len(source[target]))

        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, color=color, ax=ax)
        
    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        """Perform the transformation on the target data and return the
        transformed data.

        Parameters
        ----------
        feature_data : float | int
            Base location (offset) of feature axis coming from
            `feature_grouped` generator.
        target_data : pandas Series
            Feature grouped target data used for transformation, coming
            from `feature_grouped` generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        """
        data = pd.DataFrame({
            self.target: target_data,
            self.feature: [feature_data]})
        return data
    
    @property
    def err(self) -> NDArray:
        """Get separated error lengths as 2D array. 
        First row contains the lower errors, the second row contains the 
        upper errors."""
        err = np.array([
            self.source[self.target] - self.source[self.lower],
            self.source[self.upper] - self.source[self.target]])
        return err
    
    def __call__(self, kw_points: dict = {}, **kwds):
        """Perform the plotting operation.

        Parameters
        ----------
        kw_points : dict, optional
            Additional keyword arguments for the axes `scatter` method,
            by default {}.
        **kwds :
            Additional keyword arguments for the axes `errorbar` method.
        """
        if self.show_center:
            kw_points = dict(color=self.color) | kw_points
            self.ax.scatter(self.x, self.y, **kw_points)
        _color = dict(color=self.color) if self.bars_same_color else {}
        _kwds: Dict[str, Any] = KW.ERROR_BAR | _color | kwds
        if self.target_on_y:
            self.ax.errorbar(self.x, self.y, yerr=self.err, **_kwds)
        else:
            self.ax.errorbar(self.x, self.y, xerr=self.err, **_kwds)


class StandardErrorMean(Errorbar):
    """Class for creating plotters with error bars representing the
    standard error of the mean.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''.
    show_center : bool, optional
        Flag indicating whether to show the center points,
        by default True.
    bars_same_color : bool, optional
        Flag indicating whether to use same color for error bars as 
        markers for center. If False, the error bars are black,
        by default False
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure object with
        one Axes is created, by default None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    """
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, lower=PLOTTER.ERR_LOW,
            upper=PLOTTER.ERR_UPP, feature=feature, show_center=show_center,
            bars_same_color=bars_same_color, target_on_y=target_on_y,
            color=color, ax=ax)

    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        """Perform the transformation on the target data using the 
        `Estimator` class and return the transformed data.

        Parameters
        ----------
        feature_data : float | int
            Base location (offset) of feature axis coming from
            `feature_grouped` generator.
        target_data : pandas Series
            Feature grouped target data used for transformation, coming
            from `feature_grouped` generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        """
        estimation = Estimator(target_data)
        data = pd.DataFrame({
            self.target: [estimation.mean],
            self.feature: [feature_data],
            self.lower: [estimation.mean - estimation.sem],
            self.upper: [estimation.mean + estimation.sem]})
        return data


class SpreadWidth(Errorbar):
    """Class for creating plotters with error bars representing the 
    spread width.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''.
    strategy : Literal['eval', 'fit', 'norm', 'data'], optional
        Strategy for estimating the spread width, by default 'norm'.
    agreement : float | int, optional
        Agreement value for the spread width estimation,
        by default 6.
    possible_dists : Tuple[str | rv_continuous], optional
        Tuple of possible distributions for the spread width
        estimation, by default DIST.COMMON.
    show_center : bool, optional
        Flag indicating whether to show the center points,
        by default True.
    bars_same_color : bool, optional
        Flag indicating whether to use same color for error bars as 
        markers for center. If False, the error bars are black,
        by default False
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure object with
        one Axes is created, by default None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    """#TODO copy docstring from Estimator
    __slots__ = ('strategy', 'agreement', 'possible_dists')
    strategy: Literal['eval', 'fit', 'norm', 'data']
    agreement: float | int
    possible_dists: Tuple[str | rv_continuous]

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
            agreement: float | int = 6,
            possible_dists: Tuple[str | rv_continuous] = DIST.COMMON,
            show_center: bool = True,
            bars_same_color: bool = False,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.strategy = strategy
        self.agreement = agreement
        self.possible_dists = possible_dists
        
        super().__init__(
            source=source, target=target, lower=PLOTTER.ERR_LOW,
            upper=PLOTTER.ERR_UPP, feature=feature, show_center=show_center,
            bars_same_color=bars_same_color, target_on_y=target_on_y,
            color=color, ax=ax)

    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        """Perform the transformation on the target data using the 
        `Estimator` class and return the transformed data.

        Parameters
        ----------
        feature_data : float | int
            Base location (offset) of feature axis coming from
            `feature_grouped` generator.
        target_data : pandas Series
            Feature grouped target data used for transformation,
            coming from `feature_grouped` generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        """
        estimation = Estimator(
            samples=target_data, strategy=self.strategy, agreement=self.agreement,
            possible_dists=self.possible_dists)
        data = pd.DataFrame({
            self.target: [estimation.median],
            self.feature: [feature_data],
            self.lower: [estimation.lcl],
            self.upper: [estimation.ucl]})
        return data
    
    def __call__(self, kw_points: dict = {}, **kwds) -> None:
        """Perform the plotting operation.

        Parameters
        ----------
        kw_points : dict, optional
            Additional keyword arguments for the axes `scatter` method,
            by default {}.
        **kwds :
            Additional keyword arguments for the axes `errorbar` method.
        """
        kw_points = dict(marker= '_' if self.target_on_y else '|') | kw_points
        return super().__call__(kw_points, **kwds)


class ConfidenceInterval(Errorbar):
    """Class for creating plotters with error bars representing optical
    distinction tests.

    This class is useful for visually testing whether there is a 
    statistically significant difference between groups or conditions.
    By plotting confidence intervals around the actual value, it 
    provides a visual representation of the uncertainty in the estimate 
    and allows for a quick assessment of whether the intervals overlap 
    or not.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''.
    show_center : bool, optional
        Flag indicating whether to show the center points,
        by default True.
    bars_same_color : bool, optional
        Flag indicating whether to use same color for error bars as 
        markers for center. If False, the error bars are black,
        by default False
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    confidence_level : float, optional
        Confidence level for the confidence intervals,
        by default 0.95.
    ci_func : Callable, optional
        Function for calculating the confidence intervals,
        by default mean_ci.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure object with
        one Axes is created, by default None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    """
    __slots__ = ('confidence_level', 'ci_func', 'n_groups')
    confidence_level: float
    ci_func: Callable
    n_groups: int

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            ci_func: Callable = mean_ci,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        self.confidence_level = confidence_level
        self.ci_func = ci_func
        self.n_groups = pd.Series(source[feature]).nunique() if feature else 1
        
        super().__init__(
            source=source, target=target, lower=PLOTTER.ERR_LOW,
            upper=PLOTTER.ERR_UPP, feature=feature, show_center=show_center,
            bars_same_color=bars_same_color, target_on_y=target_on_y,
            color=color, ax=ax)
    
    def transform(
            self, feature_data: float | int, target_data: Series
            ) -> pd.DataFrame:
        """Perform the transformation on the target data by using the
        given function `ci_func' and return the transformed data.

        Parameters
        ----------
        feature_data : float | int
            Base location (offset) of feature axis coming from
            `feature_grouped` generator.
        target_data : pandas Series
            Feature grouped target data used for transformation, coming
            from `feature_grouped` generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        """
        center, lower, upper = self.ci_func(
            target_data, self.confidence_level, self.n_groups)
        data = pd.DataFrame({
            self.target: [center],
            self.feature: [feature_data],
            self.lower: lower,
            self.upper: upper})
        return data


class MeanTest(ConfidenceInterval):
    """Class for creating plotters with error bars representing
    confidence intervals for the mean.

    This class is specifically designed for testing the statistical
    significance of the mean difference between groups or conditions.
    It uses confidence intervals to visually represent the uncertainty
    in the mean estimates and allows for a quick assessment of whether
    the intervals overlap or not.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''.
    show_center : bool, optional
        Flag indicating whether to show the center points,
        by default True.
    bars_same_color : bool, optional
        Flag indicating whether to use same color for error bars as 
        markers for center. If False, the error bars are black,
        by default False
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    confidence_level : float, optional
        Confidence level for the confidence intervals,
        by default 0.95.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure object with
        one Axes is created, by default None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    """
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        super().__init__(
            source=source, target=target, feature=feature,
            show_center=show_center, bars_same_color=bars_same_color,
            target_on_y=target_on_y, confidence_level=confidence_level,
            ci_func=mean_ci, color=color, ax=ax)


class VariationTest(ConfidenceInterval):
    """Class for creating plotters with error bars representing 
    confidence intervals for variation measures.

    This class is specifically designed for testing the statistical
    significance of variation measures, such as standard deviation or
    variance, between groups or conditions. It uses confidence intervals
    to visually represent the uncertainty in the variation estimates and
    allows for a quick assessment of whether the intervals overlap or
    not.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''.
    show_center : bool, optional
        Flag indicating whether to show the center points,
        by default True.
    bars_same_color : bool, optional
        Flag indicating whether to use same color for error bars as 
        markers for center. If False, the error bars are black,
        by default False
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    confidence_level : float, optional
        Confidence level for the confidence intervals,
        by default 0.95.
    kind : Literal['stdev', 'variance'], optional
        Type of variation measure to use, by default 'stdev'.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    ax : Axes | None, optional
        The axes object for the plot. If None, a Figure object with
        one Axes is created, by default None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    """
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            kind: Literal['stdev', 'variance'] = 'stdev',
            color: str | None = None,
            ax: Axes | None = None,
            **kwds) -> None:
        ci_func = stdev_ci if kind == 'stdev' else variance_ci
        super().__init__(
            source=source, target=target, feature=feature,
            show_center=show_center, bars_same_color=bars_same_color,
            target_on_y=target_on_y, confidence_level=confidence_level,
            ci_func=ci_func, color=color, ax=ax)

                
__all__ = [
    'Plotter',
    'Scatter',
    'Line',
    'LinearRegression',
    'Probability',
    'BlandAltman',
    'TransformPlotter',
    'CenterLocation',
    'Bar',
    'Pareto',
    'Jitter',
    'GaussianKDE',
    'Violine',
    'Errorbar',
    'StandardErrorMean',
    'SpreadWidth',
    'ConfidenceInterval',
    'MeanTest',
    'VariationTest',
    ]
