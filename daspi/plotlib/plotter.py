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

The following classes are designed to provide a convenient and intuitive
way to visualize and analyze statistical data. They can be used in a
variety of applications, including data exploration, hypothesis testing,
and data presentation.

Classes
-------
Plotter:
    An abstract base class for creating plotters.
Scatter:
    A scatter plotter that extends the `Plotter` base class.
Line:
    A line plotter that extends the `Plotter` base class.
LinearRegressionLine:
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

Usage
-----
Import the module and use the classes to create and customize plots.

Example
-------
Create a scatter plot
```python
from plotter import Plotter
from plotter import Scatter
from plotter import LinearRegressionLine
from plotter import Probability
from plotter import TransformPlotter
from plotter import CenterLocation

data = {...}  # Data source for the plot
scatter_plot = Scatter(data, 'target_variable')
scatter_plot()
```	

Create a linear regression plot
```python
data = {...}  # Data source for the plot
linear_regression_plot = LinearRegressionLine(
    data, 'target_variable', 'feature_variable')
linear_regression_plot()
```

Create a probability plot
```python
data = {...}  # Data source for the plot
probability_plot = Probability(data, 'target_variable', dist='norm', kind='qq')
probability_plot()
```

Create a CenterLocation plot
```python
data = {...}  # Data source for the plot
center_plot = CenterLocation(data, 'target_variable', 'feature_variable', kind='mean')
center_plot()
```
"""
import warnings
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

from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.container import BarContainer

from scipy.stats._distn_infrastructure import rv_continuous

from itertools import cycle

from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.regression.linear_model import OLSResults

from pandas.api.types import is_scalar
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from .._typing import LineStyle

from ..strings import STR

from ..constants import KW
from ..constants import LINE
from ..constants import DIST
from ..constants import COLOR
from ..constants import DEFAULT
from ..constants import PLOTTER
from ..constants import CATEGORY

from ..statistics import Loess
from ..statistics import Lowess
from ..statistics import fit_ci
from ..statistics import mean_ci
from ..statistics import stdev_ci
from ..statistics import Estimator
from ..statistics import SpecLimits
from ..statistics import variance_ci
from ..statistics import prediction_ci
from ..statistics import proportion_ci
from ..statistics import ensure_generic
from ..statistics import ProcessEstimator
from ..statistics import estimate_kernel_density
from ..statistics import estimate_kernel_density_2d
from ..statistics import estimate_capability_confidence


class SpreadOpacity:
    """A class for controlling the opacity of plot elements based on 
    quantile agreements to enhance data visualization.

    This class is designed to adjust the opacity of filled areas in 
    plots, allowing for better visual emphasis on quantiles derived from 
    a given dataset. By using different strategies for quantile 
    estimation, it provides flexibility in how the data's spread is 
    represented. The opacity is determined based on the specified 
    agreements, which can be either integer or float values, allowing 
    the user to define the tolerance of process variation.

    Parameters
    ----------
    strategy : {'eval', 'fit', 'norm', 'data'}, optional
        Which strategy should be used to determine the quantiles:
        - `eval`: The strategy is determined according to the given 
          evaluate function. If none is given, the internal `evaluate`
          method is used.
        - `fit`: First, the distribution that best represents the 
          process data is searched for and then the agreed process 
          spread is calculated
        - `norm`: it is assumed that the data is subject to normal 
          distribution. The variation tolerance is then calculated as 
          agreement * standard deviation
        - `data`: The quantiles for the process variation tolerance 
          are read directly from the data.
        
        Default is 'data'.

    agreements : Tuple[float, ...] or Tuple[int, ...], optional
        Specifies the tolerated process variation for calculating 
        quantiles. These quantiles are used to represent the filled area 
        with different opacity, thus highlighting the quantiles in the
        plot. The agreements can be either integers or floats,
        determining the process variation tolerance in the following
        ways:
        - If integers, the quantiles are determined using the normal 
          distribution (agreement * σ), e.g., agreement = 6 covers 
          ~99.75% of the data.
        - If floats, values must be between 0 and 1, interpreted as 
          acceptable proportions for the quantiles, e.g., 0.9973 
          corresponds to ~6σ.
        
        Default is `DEFAULT.AGREEMENTS` = (2, 4, 6), corresponding to 
        (±1σ, ±2σ, ±3σ).
    possible_dists : tuple of strings or rv_continous, optional
        Distributions to which the data may be subject. Only 
        continuous distributions of scipy.stats are allowed,
        by default `DIST.COMMON`
    """
    strategy: Literal['eval', 'fit', 'norm', 'data']
    """Strategy for estimating the spread width."""
    _agreements: Tuple[float, ...] | Tuple[int, ...]
    """The agreement values used to calculate the quantiles."""
    possible_dists: Tuple[str | rv_continuous, ...]
    """Tuple of possible distributions for the spread width
    estimation."""
    estimation: Estimator
    """The estimator used to calculate the quantiles."""
    
    @property
    def agreements(self) -> Tuple[float, ...] | Tuple[int, ...]:
        """Get a tuple containing unique agreement values, which can be 
        either floats or integers. The values are sorted in ascending 
        order.
        
        Set the agreements by passing a tuple of floats or integers.
        It ensures that the agreements are unique by using `np.unique`, 
        and sorts them in ascending order before storing them in the 
        instance."""
        return self._agreements
    @agreements.setter
    def agreements(
            self, agreements: Tuple[float, ...] | Tuple[int, ...]) -> None:
        self._agreements = tuple(sorted(np.unique(agreements), reverse=False))

    @property
    def _alphas(self) -> Dict[float | int, float]:
        """Get color transparency for each quantile range."""
        n = len(self.agreements)
        alphas = ((i + 1) * COLOR.FILL_ALPHA / n for i in range(n))
        return {q: a for q, a in zip(self.agreements[::-1], alphas)}
    
    def _kw_fill(
            self,
            color: str | Tuple,
            agreement: float,
            data: DataFrame
            ) -> Dict[str, Any]:
        """"""
        alpha = self._alphas.get(agreement, COLOR.FILL_ALPHA)
        kw = dict(
            color=color,
            alpha=alpha,
            edgecolor=mcolors.to_rgb(color) + (alpha/2,),
            linewidth=1,
            where=data[PLOTTER.SUBGROUP]==agreement,
            interpolate=False)
        return kw

    def quantiles(self, target_data: Series) -> List[float]:
        """Calculate the quantiles in ascending order:
        
        Parameters
        ----------
        target_data : pandas Series
            feature grouped target data used for transformation, coming
            from `feature_grouped' generator.
        
        Returns
        -------
        List[float]
            quantiles in ascending order
        """
        quantiles = []
        
        self.estimation = Estimator(
            samples=target_data,
            strategy=self.strategy,
            possible_dists=self.possible_dists)
        for agreement in self.agreements:
            self.estimation.agreement = agreement
            k = 1 if agreement == max(self.agreements) else 2
            quantiles.extend([self.estimation.lcl, self.estimation.ucl] * k)
        return sorted(quantiles)
    
    def subgroup_values(
            self,
            sequence: NDArray | int,
            quantiles: List[float]) -> List[float | int]:
        """Generate a list of subgroup values based on the agreements.

        This method creates a list of subgroup values derived from the 
        instance's agreements. It iterates through the agreements and 
        appends each agreement to the subgroup list. If the current 
        agreement is the same as the first agreement, the last 
        subgroup value is appended instead.

        Returns
        -------
        List[float | int]:
            A list of subgroup values, which can include both floats 
            and integers.
        """
        subgroup = []
        names = self.agreements[::-1] + self.agreements[1:]
        if isinstance(sequence, int):
            for name in names:
                subgroup.extend([name] * sequence)
        else:
            _quantiles = np.unique(quantiles)
            for name, beg, end in zip(names, _quantiles[:-1], _quantiles[1:]):
                if not subgroup:
                    n_values = np.sum((sequence <= end))
                elif name == names[-1]:
                    n_values = np.sum((sequence > beg))
                else:
                    n_values = np.sum((sequence > beg) & (sequence <= end))
                subgroup.extend([name] * n_values)
            if difference := len(sequence) - len(subgroup):
                if difference < 0:
                    subgroup = subgroup[:difference]
                else:
                    subgroup.extend([names[-1]] * difference)
        return subgroup


class Plotter(ABC):
    """Abstract base class for creating plotters.

    This class serves as a blueprint for plotter implementations, 
    facilitating the visualization of data from a given source.

    Parameters
    ----------
    source : pandas.DataFrame
        A long format DataFrame containing the data source for the plot.
    target : str
        The column name of the target variable to be plotted.
    feature : str, optional
        The column name of the feature variable to be plotted. 
        Defaults to an empty string ('').
    target_on_y : bool, optional
        A flag indicating whether the target variable is plotted on the 
        y-axis. Defaults to True.
    color : str or None, optional
        The color used for drawing the plot elements. If None, the first 
        color from the color cycle is used. Defaults to None.
    marker : str or None, optional
        The marker style for scatter plots. For available markers, see:
        https://matplotlib.org/stable/api/markers_api.html. 
        Defaults to None.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawnaccording to the stylesheet.
        Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, no changes are 
        made, and the axes are displayed according to the stylesheet.
        Defaults to None.
    **kwds :
        Additional keyword arguments that are ignored in this class but 
        may be used for compatibility with chart objects.

    Notes
    -----
    This class is intended to be subclassed, and specific plotting 
    functionality should be implemented in derived classes.
    """
    __slots__ = (
        'source', 'target', 'feature', 'target_on_y', '_color', '_marker',
        'fig', 'ax', '_visible_spines', '_hide_axis')
    source: DataFrame
    """The data source for the plot"""
    target: str
    """The column name of the target variable."""
    feature: str
    """The column name of the feature variable."""
    _color: str | None
    """Provided color for the artist"""
    _marker: str | None
    """Provided marker for the artist"""
    target_on_y: bool
    """Flag indicating whether the target variable is plotted on the
    y-axis."""
    fig: Figure
    """The top level container for all the plot elements."""
    ax: Axes
    """The axes object for the plot."""
    _visible_spines: Literal['target', 'feature', 'none'] | None
    """Which spine is visible."""
    _hide_axis: Literal['target', 'feature', 'both'] | None
    """Which axis is hidden if specified."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            target_on_y : bool = True,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.target_on_y = target_on_y
        self._visible_spines = None
        self._hide_axis = None
        self.source = source
        if not feature:
            feature = PLOTTER.FEATURE
            self.source[feature] = np.arange(len(source[target])) + 1
        self.feature = feature
        self.target = target
        
        self.fig, self.ax = self.figure_axes(ax)
        self._color = color
        self._marker = marker
        self.visible_spines = visible_spines
        self.hide_axis = hide_axis
    
    @property
    def visible_spines(self) -> Literal['target', 'feature', 'none'] | None:
        """Get which spines are visible if specified.

        Set which spines should be visible, the others are hidden.
        If 'none', no spines are visible. If None, no changes are made 
        and the spines are drawn according to the stylesheet.
        """
        return self._visible_spines
    @visible_spines.setter
    def visible_spines(
            self,
            visible_spines: Literal['target', 'feature', 'none'] | None
            ) -> None:
        if self._visible_spines == visible_spines:
            return

        assert visible_spines in ['target', 'feature', 'none', None], (
            'visible_spines must be one of {"target", "feature", "none", None}'
            f', got {visible_spines}')
        self._visible_spines = visible_spines
        positions = ('top', 'bottom', 'left', 'right')
        ys = positions[:2]
        xs = positions[2:]
        for pos in positions:
            self.ax.spines[pos].set_visible(plt.rcParams[f'axes.spines.{pos}'])

        if visible_spines == 'target':
            hidden_positions = ys if self.target_on_y else xs
        elif visible_spines == 'feature':
            hidden_positions = xs if self.target_on_y else ys
        elif visible_spines == 'none':
            hidden_positions = positions
        else:
            hidden_positions = ()

        for pos in hidden_positions:
            self.ax.spines[pos].set_visible(False)
    
    @property
    def hide_axis(self) -> Literal['target', 'feature', 'both'] | None:
        return self._hide_axis
    @hide_axis.setter
    def hide_axis(
            self,
            hide_axis: Literal['target', 'feature', 'both'] | None
            ) -> None:
        if self._hide_axis == hide_axis:
            return

        assert hide_axis in ['target', 'feature', 'both', None], (
            'hide_axis must be one of {"target", "feature", "both", None}'
            f', got {hide_axis}')
        self._hide_axis = hide_axis
        xaxis = self.ax.xaxis
        yaxis = self.ax.yaxis

        if hide_axis == 'target':
            if self.target_on_y:
                xaxis.set_visible(True)
                yaxis.set_visible(False)
            else:
                xaxis.set_visible(False)
                yaxis.set_visible(True)
        elif hide_axis == 'feature':
            if self.target_on_y:
                xaxis.set_visible(False)
                yaxis.set_visible(True)
            else:
                xaxis.set_visible(True)
                yaxis.set_visible(False)
        elif hide_axis == 'both':
            xaxis.set_visible(False)
            yaxis.set_visible(False)

    @property
    @abstractmethod
    def kw_default(self) -> Dict[str, Any]:
        """Override this property to provide the default keyword 
        arguments for plotting as read-only."""
        raise NotImplementedError
    
    @property
    def x_column(self) -> str:
        """Get column name used to access data for x-axis (read-only)."""
        return self.feature if self.target_on_y else self.target
    
    @property
    def y_column(self) -> str:
        """Get column name used to access data for y-axis (read-only)."""
        return self.target if self.target_on_y else self.feature
        
    @property
    def x(self) -> Series:
        """Get values used for x-axis (read-only)."""
        return self.source[self.x_column]
    
    @property
    def y(self) -> Series:
        """Get values used for y-axis (read-only)"""
        return self.source[self.y_column]
    
    @property
    def color(self) -> str:
        """Get color of drawn artist"""
        if self._color is None:
            self._color = COLOR.PALETTE[0]
        return self._color

    @property
    def marker(self) -> str | None:
        """Get marker of drawn artist"""
        return self._marker
    
    @staticmethod
    def figure_axes(ax: Axes | None) -> Tuple[Figure, Axes]:
        """Create a figure and axes object if not provided."""
        ax = plt.gca() if ax is None else ax
        fig: Figure = ax.get_figure() # type: ignore
        return fig, ax
    
    @staticmethod
    def shared_axes(
            ax: Axes, which: Literal['x', 'y'], exclude: bool = True
            ) -> List[bool]:
        """Get all the axes from the figure of the given `ax` and 
        compare whether the `ax` share the given axis. 
        Get a map of boolean values as a list where all are `True` when 
        the axis is shared.
        
        Parameters
        ----------
        ax : Axes
            Base axes object to add second axis
        which : {'x', 'y'}
            From which axis a second one should be added
        exclude : bool, optional
            If True excludes the given `ax` in the returned map
        
        Returns
        -------
        List[bool]
            Flat map for axes that shares same axis
        """
        assert which in ('x', 'y')
        view = getattr(ax, f'get_shared_{which}_axes')()
        axes = [_ax for _ax in ax.figure.axes] if exclude else ax.figure.axes  # type: ignore
        return [view.joined(ax, _ax) for _ax in axes]

    @abstractmethod
    def __call__(self) -> None:
        """Perform the plotting operation.

        This method should be overridden by subclasses to provide the 
        specific plotting functionality.
        """
    
    def label_feature_ticks(self) -> None:
        """Label the feature ticks with categorical feature names.

        This method gets the original feature values from the source
        DataFrame and labels the feature ticks with the feature names.
        Call this method if feature values are categorical.
        """
        ax = self.ax
        tick_labels = getattr(self, '_original_f_values', None) # TransformPlotters
        if tick_labels is None:
            tick_labels = self.source[self.feature].unique()
        n_labels = len(tick_labels)
        lower = DEFAULT.FEATURE_BASE - 0.5
        xy = 'x' if self.target_on_y else 'y'
        settings = {
            f'{xy}ticks': list(range(n_labels)),
            f'{xy}ticklabels': tick_labels,
            f'{xy}lim': (lower, lower + n_labels)}
        ax.set(**settings)
    
    def target_as_percentage(
            self,
            xmax: float = 1.0,
            decimals: int | None = None,
            symbol: str | None = '%'
            ) -> None:
        """Format the numbers on target axis to percentage.

        Parameters
        ----------
        xmax : float, optional
            Determines how the number is converted into a percentage. 
            xmax is the data value that corresponds to 100%. Percentages 
            are computed as x / xmax * 100. So if the data is already 
            scaled to be percentages, xmax will be 100. Another common 
            situation is where xmax is 1.0.
        decimals : int | None, optional
            The number of decimal places to place after the point. If 
            None (the default), the number will be computed 
            automatically.
        symbol : str | None, optional
            A string that will be appended to the label. It may be None 
            or empty to indicate that no symbol should be used. LaTeX 
            special characters are escaped in symbol whenever latex mode 
            is enabled, unless is_latex is True.
        """
        axis = self.ax.yaxis if self.target_on_y else self.ax.xaxis
        axis.set_major_formatter(PercentFormatter(xmax=xmax, symbol='%'))


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
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Scatter

    fig, ax = plt.subplots()
    df = pd.DataFrame({'x': list(x*np.pi/50 for x in range(100))})
    df['y'] = np.cos(df['x'])
    scatter = Scatter(source=df, target='y', feature='x', ax=ax)
    scatter(color='red', s=20, marker='s', alpha=0.6)
    ```

    Apply using the plot method of a DaSPi Chart object

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame({'x': list(x*np.pi/50 for x in range(100))})
    df['y'] = np.cos(df['x'])

    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x'
        ).plot(
            dsp.Scatter,
            kw_call=dict(color='red', s=20, marker='s', alpha=0.6))
    ```
    
    """
    __slots__ = ('size')
    size: Iterable[int] | None
    """The sizes of the markers in the scatter plot."""

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
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
        self.size = size
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(
            color=self.color,
            marker=self.marker,
            s=self.size,
            alpha=COLOR.MARKER_ALPHA)
        return kwds
    
    def __call__(self, **kwds) -> None:
        """Perform the scatter plot operation.

        Parameters
        ----------
        **kwds : 
            Additional keyword arguments to be passed to the Axes 
            `scatter` method.
        """
        _kwds = self.kw_default | kwds
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
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Line

    fig, ax = plt.subplots()
    df = pd.DataFrame({'x': list(x*np.pi/50 for x in range(100))})
    df['y'] = np.cos(df['x'])
    line = Line(source=df, target='y', feature='x', ax=ax)
    line(color='red', lw=2, alpha=0.6)
    ```

    Apply using the plot method of a DaSPi Chart object

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame({'x': list(x*np.pi/50 for x in range(100))})
    df['y'] = np.cos(df['x'])

    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x'
        ).plot(
            dsp.Line,
            kw_call=dict(color='red', lw=2, alpha=0.6))
    ```
    """
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            target_on_y: bool = True, 
            color: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            target_on_y=target_on_y,
            color=color,
            marker=None,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
        
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(color=self._color)
        return kwds
    
    def __call__(self, marker: str | None = None, **kwds) -> None:
        """Perform the line plot operation.

        Parameters
        ----------
        marker : str | None, optional
            The marker style for the individuals. Available markers see:
            https://matplotlib.org/stable/api/markers_api.html, 
            by default None
        **kwds : 
            Additional keyword arguments to be passed to the Axes `plot` 
            method.
        """
        alpha = COLOR.MARKER_ALPHA if marker is not None else None
        _kwds = self.kw_default | dict(marker=marker, alpha=alpha) | kwds
        self.ax.plot(self.x, self.y, **_kwds)


class Stem(Plotter):
    """A stem plotter that extends the Plotter base class.
    
    This class is designed to create a stem plot using the data provided 
    in the source DataFrame.

    Parameters
    ----------
    source : pandas DataFrame
        A long format DataFrame containing the data source for the plot.
    target : str
        Column name of the target variable for the plot.
    feature : str
        Column name of the feature variable for the plot.
    bottom : float, optional
        The Y/X position of the baseline (depending on the orientation 
        or `target_on_y`). Default is 0.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on the 
        y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the markers. If None, the first color 
        is taken from the color cycle, by default None.
    line_color: str | None, optional
        Color to be used to draw the vertical lines. If None, the same
        color is taken as the color of the markers. Default is None
    base_color : str | None, optional
        Color to be used to draw the base line. If None, the same color
        is taken as the color of the markers. Default is None
    marker : str | None, optional
        The marker style for the stem plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn 
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, no changes are 
        made, and the axes are displayed according to the stylesheet.
        Defaults to None.
    **kwds:
        Additional keyword arguments that may be used for compatibility 
        with other chart objects.

    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Stem

    fig, ax = plt.subplots()
    df = pd.DataFrame({'x': list(x*np.pi/25 for x in range(50))})
    df['y'] = np.cos(df['x'])
    stem = Stem(
        source=df, target='y', feature='x', target_on_y=False, bottom=1,
        color='deepskyblue', line_color='steelblue', base_color='skyblue',
        ax=ax)
    stem(alpha=0.2, line_style='--', base_line_style='-.')
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame({'x': list(x*np.pi/25 for x in range(50))})
    df['y'] = np.cos(df['x'])

    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            target_on_y=False
        ).plot(
            dsp.Stem,
            bottom=1,
            color='deepskyblue',
            line_color='steelblue',
            base_color='skyblue',
            kw_call=dict(alpha=0.2, line_style='--', base_line_style='-.'))
    ```

    Notes
    -----
    To change the markers, vertical lines and base line when drawing
    use the keword arguments `markerfmt`, `linefmt` and `basefmt`. For
    further information see:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stem.html
    """
    __slots__ = ('bottom', 'line_color', 'base_color')

    bottom: float
    """The Y/X position of the baseline (depending on the orientation or 
    `target_on_y`)."""

    line_color: str
    """Color to be used to draw the vertical lines. If None, the same
    color is taken as the color of the markers."""

    base_color: str
    """Color to be used to draw the base line. If None, the same color
    is taken as the color of the markers."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            bottom: float = 0,
            target_on_y: bool = True,
            color: str | None = None,
            line_color: str | None = None,
            base_color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
        self.bottom = bottom
        self.line_color = line_color if line_color is not None else self.color
        self.base_color = base_color if base_color is not None else self.color
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kw_default = dict(
            markerfmt=f'{self.marker}',
            linefmt=f'{self.line_color}',
            basefmt=f'{self.base_color}',
            orientation='vertical' if self.target_on_y else 'horizontal',
            bottom=self.bottom)
        return kw_default

    def __call__(
            self,
            alpha=COLOR.MARKER_ALPHA,
            line_style: str | None = None,
            base_line_style: str | None = None,
            **kwds) -> None:
        """Perform the stem plot operation.

        Parameters
        ----------
        alpha : float, optional
            Transparency of the marker. Default is 
            `COLOR.MARKER_ALPHA`.
        line_style : str, optional
            Style of the line connecting the marker to the baseline.
            This option is ignored if it is None or `linefmt` is in 
            `kwds`. Default is `None`.
        base_line_style : str, optional
            Style of the baseline. This option is ignored if it is 
            None or `basefmt` is in `kwds`. Default is `None`.
        **kwds : dict, optional
            Additional keyword arguments to be passed to the Axes 
            `stem` method.
        """
        _kwds = self.kw_default | kwds

        # self.x and self.y do not work here.
        # The stem function handles this internally with the `orientation` option.
        x = self.source[self.feature]
        y = self.source[self.target]
        markerline, stemlines, baseline = self.ax.stem(x, y, **_kwds)

        markerline.set(alpha=alpha)
        if 'markerfmt' not in kwds:
            markerline.set(markerfacecolor=self.color)
        
        if line_style is not None and 'linefmt' not in kwds:
            stemlines.set_linestyle(line_style)
        
        if base_line_style is not None and 'basefmt' not in kwds:
            baseline.set_linestyle(base_line_style)


class LinearRegressionLine(Plotter):
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
    marker : str | None, optional
        The marker style for the scatter plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    show_scatter : bool, optional
        Flag indicating whether to show the individual points, 
        by default False.
    show_fit_ci : bool, optional
        Flag indicating whether to show the confidence interval for
        the fitted line as filled area, by default False.
    show_pred_ci : bool, optional
        Flag indicating whether to show the confidence interval for 
        predictions as additional lines, by default False
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import LinearRegressionLine

    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    df = pd.DataFrame(dict(
        x=x,
        y=x + np.random.randn(100) - 0.05 * np.square(x)))
    reg_line = LinearRegressionLine(
        source=df, target='y', feature='x', ax=ax,
        show_scatter=True, show_fit_ci=True, show_pred_ci=True)
    reg_line(
        kw_scatter=dict(color='black', s=10, alpha=0.5),
        kw_fit_ci=dict(color='skyblue'),
        kw_pred_ci=dict(color='steelblue', alpha=1),
        color='deepskyblue')
    ```

    Apply using the plot method of a DaSPi Chart object

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    x = np.linspace(0, 10, 100)
    df = pd.DataFrame(dict(
        x=x,
        y=x + np.random.randn(100) - 0.05 * np.square(x)))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x'
        ).plot(
            dsp.LinearRegressionLine,
            show_scatter=True,
            show_fit_ci=True,
            show_pred_ci=True,
            kw_call=dict(
                kw_scatter=dict(color='black', s=10, alpha=0.5),
                kw_fit_ci=dict(color='skyblue'),
                kw_pred_ci=dict(color='steelblue', alpha=1),
                color='deepskyblue'))
    ```
    """
    __slots__ = (
        'model', 'fit', 'show_scatter', 'show_fit_ci', 'show_pred_ci')
    
    model: OLSResults
    """The fitted results of the linear regression model."""
    fit: Literal['_fitted_values_']
    """The name of the column containing the fitted values as defined 
    in the PLOTTER.FITTED_VALUES."""
    show_scatter: bool
    """Whether to show the individual points in the plot."""
    show_fit_ci: bool
    """Whether to show the confidence interval for the fitted line."""
    show_pred_ci: bool
    """Whether to show the confidence interval for the predictions."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            show_scatter: bool = False,
            show_fit_ci: bool = False,
            show_pred_ci: bool = False,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.fit = PLOTTER.FITTED_VALUES
        self.show_scatter = show_scatter
        self.show_fit_ci = show_fit_ci
        self.show_pred_ci = show_pred_ci
        df = source if isinstance(source, DataFrame) else pd.DataFrame(source)
        df = (df
            .sort_values(feature)
            [[feature, target]]
            .dropna(axis=0, how='any')
            .reset_index(drop=True))
        self.model = sm.OLS(df[target], sm.add_constant(df[feature])).fit() # type: ignore
        df[self.fit] = self.model.fittedvalues
        df = pd.concat([df, self.ci_data()], axis=1)
        super().__init__(
            source=df,
            target=target,
            feature=feature,
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = KW.FIT_LINE | dict(color=self.color)
        return kwds
    
    @property
    def x_fit(self) -> ArrayLike:
        """Get values used for x-axis for fitted line (read-only)."""
        name = self.feature if self.target_on_y else self.fit
        return self.source[name]
    
    @property
    def y_fit(self) -> ArrayLike:
        """Get values used for y-axis for fitted line (read-only)"""
        name = self.fit if self.target_on_y else self.feature
        return self.source[name]
    
    def ci_data(self) -> pd.DataFrame:
        """Get confidence interval for prediction and fitted line as 
        DataFrame."""
        data = (
            *fit_ci(self.model)[-2:],
            *prediction_ci(self.model)[-2:])
        ci_data = pd.DataFrame(
            data = np.array(data, dtype=float).T, 
            columns = PLOTTER.REGRESSION_CI_NAMES)
        return ci_data
    
    def __call__(
            self,
            kw_scatter: dict = {},
            kw_fit_ci: dict = {},
            kw_pred_ci: dict = {},
            **kwds) -> None:
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
        _kwds = self.kw_default | kwds
        self.ax.plot(self.x_fit, self.y_fit, **_kwds)
        
        if self.show_scatter:
            kw_scatter = color | dict(marker=self.marker) | kw_scatter
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
            

class LoessLine(Plotter):
    """A class for plotting a loess line.
    
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
    marker : str | None, optional
        The marker style for the scatter plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    show_scatter : bool, optional
        Flag indicating whether to show the individual points, 
        by default False.
    show_fit_ci : bool, optional
        Flag indicating whether to show the confidence interval for
        the lowess line as filled area, by default False.
    confidence_level: float, optional
        Calculate confidence bands for the lowess line at this level 
        (0 to 1). Defaults to 0.95.
    fraction : float | None, optional
        The fraction of the data used for each local regression. A good 
        value to start with is 2/3 (default value of statsmodels). 
        Reduce the value to avoid underfitting. A value below 0.2 
        usually leads to overfitting, except for gaussian weights. 
        Defaults to 0.2 because of default gaussian kernel.
    order : Literal[0, 1, 2, 3], optional
        The order of the local regression to be fitted. This determines 
        the degree of the polynomial used in the local regression:

        0: No smoothing (interpolation)
        1: Linear regression
        2: Quadratic regression
        3: Cubic regression Default is 3.

    kernel : Literal['tricube', 'gaussian', 'epanechnikov'], optional
        The kernel function used to calculate the weights. Available kernels are:
        'tricube': Tricube kernel function
        'gaussian': Gaussian kernel function
        'epanechnikov': Epanechnikov kernel function.
        Default is 'gaussian', because it will not run in a Singular
        Matrix Error.
    n_points : int, optional
        Number of points the smoothed line and its sequence 
        should have, by default LOWESS_SEQUENCE_LEN 
        (defined in constants.py).
    kind : Literal['LOESS', 'LOWESS'], optional
        The type of lowess line to be plotted. Available options are:
        'LOESS': Lowess line using the LOESS algorithm
        'LOWESS': Lowess line using the LOWESS algorithm
        Default is 'LOESS' because it needs less computational
        power.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import LoessLine

    fig, ax = plt.subplots()
    x = 5*np.random.random(100)
    df = pd.DataFrame(dict(
        x = x,
        y = np.sin(x) * 3*np.exp(-x) + np.random.normal(0, 0.2, 100)))
    loess_line = LoessLine(
        source=df, target='y', feature='x',
        show_scatter=True, show_fit_ci=True, ax=ax)
    loess_line(
        kw_scatter=dict(color='black', s=10, alpha=0.5),
        kw_fit_ci=dict(color='skyblue'),
        color='deepskyblue')
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    x = 5*np.random.random(100)
    df = pd.DataFrame(dict(
        x = x,
        y = np.sin(x) * 3*np.exp(-x) + np.random.normal(0, 0.2, 100)))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x'
        ).plot(
            dsp.LoessLine,
            show_scatter=True,
            show_fit_ci=True,
            kw_call=dict(
                kw_scatter=dict(color='black', s=10, alpha=0.5),
                kw_fit_ci=dict(color='skyblue'),
                color='deepskyblue'))
    ```
    """
    __slots__ = (
        'model', 'show_scatter', 'show_fit_ci')
    
    model: Loess | Lowess
    """The fitted results of the linear regression model."""
    show_scatter: bool
    """Whether to show the individual points in the plot."""
    show_fit_ci: bool
    """Whether to show the confidence interval for the fitted lowess 
    line."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            show_scatter: bool = False,
            show_fit_ci: bool = False,
            confidence_level: float = 0.95,
            fraction: float = 0.2,
            order: Literal[0, 1, 2, 3] = 3,
            kernel: Literal['tricube', 'gaussian', 'epanechnikov'] = 'gaussian',
            n_points: int = DEFAULT.LOWESS_SEQUENCE_LEN,
            kind: Literal['LOESS', 'LOWESS'] = 'LOESS',
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.show_scatter = show_scatter
        self.show_fit_ci = show_fit_ci
        source = source[[target, feature]].copy()
        Model_ = Loess if kind.upper() == 'LOESS' else Lowess
        self.model = Model_(source=source, target=target, feature=feature)
        self.model.fit(fraction=fraction, order=order, kernel=kernel)
        sequence, prediction, lower, upper = self.model.fitted_line(
            confidence_level=confidence_level,
            n_points=n_points)
        df = pd.DataFrame({
            PLOTTER.LOWESS_TARGET: prediction,
            PLOTTER.LOWESS_FEATURE: sequence,
            PLOTTER.LOWESS_LOW: lower,
            PLOTTER.LOWESS_UPP: upper,})
        super().__init__(
            source=df,
            target=PLOTTER.LOWESS_TARGET,
            feature=PLOTTER.LOWESS_FEATURE,
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = KW.FIT_LINE | dict(color=self.color)
        return kwds
    
    def __call__(
            self,
            kw_scatter: dict = {},
            kw_fit_ci: dict = {},
            **kwds) -> None:
        """
        Perform the linear regression plot operation.

        Parameters
        ----------
        kw_scatter : dict, optional
            Additional keyword arguments for the Axes `scatter` method,
            by default {}.
        kw_fit_ci : dict, optional
            Additional keyword arguments for the confidence interval of 
            the lowess line (Axes `fill_between` method), by default {}.
        **kwds:
            Additional keyword arguments to be passed to the fit line
            plot (Axes `plot` method).
        """
        color = dict(color=self.color)
        _kwds = self.kw_default | kwds
        self.ax.plot(self.x, self.y, **_kwds)
        
        if self.show_scatter:
            kw_scatter = color | dict(marker=self.marker) | kw_scatter
            self.ax.scatter(self.x, self.y, **kw_scatter)
        if self.show_fit_ci:
            _kw_fit_ci = KW.FIT_CI | color | kw_fit_ci
            lower = self.source[PLOTTER.LOWESS_LOW]
            upper = self.source[PLOTTER.LOWESS_UPP]
            if self.target_on_y:
                self.ax.fill_between(self.x, lower, upper, **_kw_fit_ci)
            else:
                self.ax.fill_betweenx(self.y, lower, upper, **_kw_fit_ci)


class Probability(LinearRegressionLine):
    """A Q-Q and P-P probability plotter that extends the 
    LinearRegressionLine class.
    
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
    kind : Literal['qq', 'pp', 'sq', 'sp'], optional
        The type of probability plot to create. The first letter
        corresponds to the target, the second to the feature.
        Defaults to 'sq':
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
    marker : str | None, optional
        The marker style for the scatter plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    show_scatter : bool, optional
        Flag indicating whether to show the individual points, 
        by default True.
    show_fit_ci : bool, optional
        Flag indicating whether to show the confidence interval for
        the fitted line as filled area, by default False.
    show_pred_ci : bool, optional
        Flag indicating whether to show the confidence interval for 
        predictions as additional lines, by default False.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Probability

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        y = np.random.weibull(a=1, size=100)))
    prob_line = Probability(
        source=df, target='y', kind='pp', show_scatter=True, show_fit_ci=True,
        ax=ax)
    prob_line(
        kw_scatter=dict(color='black', s=10, alpha=0.5),
        kw_fit_ci=dict(color='skyblue'),
        color='deepskyblue')
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        y = np.random.weibull(a=1, size=100)))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x'
        ).plot(
            dsp.Probability,
            kind='pp',
            show_scatter=True,
            show_fit_ci=True,
            kw_call=dict(
                kw_scatter=dict(color='black', s=10, alpha=0.5),
                kw_fit_ci=dict(color='skyblue'),
                color='deepskyblue')
        )
    ```
    
    Raises
    ------
    AssertionError
        If given kind is not one of 'qq', 'pp', 'sq' or 'sp'
    """
    __slots__ = ('dist', 'kind', 'prob_fit')
    
    dist: rv_continuous
    """The probability distribution use for creating feature data."""
    kind: Literal['qq', 'pp', 'sq', 'sp']
    """The type of probability plot to create."""
    prob_fit: ProbPlot
    """The probability fit for the given distribution."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            dist: str | rv_continuous = 'norm',
            kind: Literal['qq', 'pp', 'sq', 'sp'] = 'sq',
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            show_scatter: bool = True,
            show_fit_ci: bool = True,
            show_pred_ci: bool = True,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        assert kind in ('qq', 'pp', 'sq', 'sp'), (
            f'kind must be one of {"qq", "pp", "sq", "sp"}, got {kind}')

        self.kind = kind
        self.dist = ensure_generic(dist)
        self.prob_fit = ProbPlot(source[target], self.dist, fit=True) # type: ignore
        
        feature_kind = 'quantiles' if self.kind[1] == "q" else 'percentiles'
        feature = f'{self.dist.name}_{feature_kind}'
        df = pd.DataFrame({
            target: self.sample_data,
            feature: self.theoretical_data})

        super().__init__(
            source=df,
            target=target,
            feature=feature,
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            show_scatter=show_scatter,
            show_fit_ci=show_fit_ci,
            show_pred_ci=show_pred_ci,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
    
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
            self,
            kw_scatter: dict = {},
            kw_fit_ci: dict = {},
            kw_pred_ci: dict = {},
            **kwds) -> None:
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
    identity : str
        Column name containing identities of each sample, must occur 
        once for each coordinate.
    show_scatter : bool, optional
        Flag indicating whether to show the individual points, 
        by default True.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the scatter plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import matplotlib.pyplot as plt
    from daspi import ParallelCoordinate, load_dataset

    fig, ax = plt.subplots()
    df = load_dataset('shoe-sole')
    parallel = ParallelCoordinate(
        source=df, target='wear', feature='status', identity='tester',
        show_scatter=True, ax=ax)
    parallel()
    parallel.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = dsp.load_dataset('shoe-sole')
    chart = dsp.SingleChart(
            source=df,
            target='wear',
            feature='status',
            categorical_feature=True,
        ).plot(
            dsp.ParallelCoordinate,
            identity='tester',
            show_scatter=True,
        ).label() # neded to label feature ticks
    ```
    """

    __slots__ = ('identity', 'show_scatter')

    identity: str
    """Column name containing identities of each sample.""" 
    show_scatter: bool
    """Whether to show the individual points or not."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            identity: str,
            show_scatter: bool = True,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.identity = identity
        self.show_scatter = show_scatter
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(color=self.color)
        return kwds

    def __call__(self, **kwds) -> None:
        """Perform the parallel coordinate plot

        Parameters
        ----------
        **kwds:
            Additional keyword arguments to be passed to the fit line
            plot (Axes `plot` method)."""
        marker = kwds.pop('marker', self.marker) or DEFAULT.MARKER
        if not self.show_scatter:
            marker = None
        _kwds = self.kw_default | dict(marker=marker) | kwds
        for i, group in self.source.groupby(self.identity, observed=True):
            self.ax.plot(group[self.x_column], group[self.y_column], **_kwds)


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
        given, by default `DEFAULT.FEATURE_BASE`.
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    """
    __slots__ = ('_f_base', '_original_f_values', 'skip_na')

    _f_base: int | float
    """Value that serves as the base location (offset) of the feature 
    values."""
    _original_f_values: tuple
    """Original values of the feature values."""

    skip_na: Literal['all', 'any'] | None
    """Flag indicating whether to skip missing values in the feature 
    grouped data.
    - `None`: no missing values are skipped
    - `'all'`: grouped data is skipped if all values are missing
    - `'any'`: grouped data is skipped if any value is missing
    """
    
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            f_base: int | float = DEFAULT.FEATURE_BASE,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self._f_base = f_base
        self.target = target
        self.feature = feature
        self._original_f_values = ()
        self.skip_na = skip_na
        
        df =  pd.concat(
            [self.transform(f, t) for f, t in self.feature_grouped(source)],
            axis=0,
            ignore_index=True)

        super().__init__(
            source=df,
            target=target,
            feature=feature,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
    
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
                    source.groupby(self.feature, sort=True, observed=True),
                    start=DEFAULT.FEATURE_BASE):
                target_data = group[self.target]
                if self.skip_na and getattr(target_data.isna(), self.skip_na)():
                    continue
                
                self._original_f_values = self._original_f_values + (f_value, )
                if isinstance(f_value, (float, int, pd.Timestamp)):
                    feature_data = f_value
                else:
                    feature_data = i
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
    show_line : bool, optional
        Flag indicating whether to draw a line between the calculated 
        mean or median points, by default True
    show_center : bool, optional
        Flag indicating whether to show the center points, 
        by default True.
    f_base : int | float, optional
        Value that serves as the base location (offset) of the 
        feature values. Only taken into account if feature is not 
        given, by default `DEFAULT.FEATURE_BASE`.
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the center points. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import CenterLocation, Scatter

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    scatter=Scatter(source=df, target='y', feature='x', ax=ax)
    center = CenterLocation(
        source=df, target='y', feature='x', kind='median',
        show_center=True, show_line=True, ax= ax)
    center(marker='_', markersize=10)
    center.label_feature_ticks()
    scatter(color='black')
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True,
        ).plot(
            dsp.CenterLocation,
            kind='median',
            show_line=True,
            show_center=True,
            kw_call=dict(marker='_', markersize=30)
        ).plot(
            dsp.Scatter,
        ).label() # neded to label feature ticks
    ```
    
    Note
    ----
    Be careful with the `show_center` and `show_line` flags. Either 
    `show_center` or `show_line` must be True, otherwise this plot makes 
    no sense. If both are False, nothing will be drawn.
    """
    __slots__ = ('_kind', 'show_line', 'show_center')

    _kind: Literal['mean', 'median']
    """The type of center to plot ('mean' or'median')."""
    show_line: bool
    """Whether to draw a line between the calculated means or medians."""
    show_center: bool
    """Whether to draw the center points."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            kind: Literal['mean', 'median'] = 'mean',
            show_line: bool = True,
            show_center: bool = True,
            f_base: int | float = DEFAULT.FEATURE_BASE,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        assert any((show_line, show_center)), (
            'Either show_line or show_center must be True.')
        self._kind = 'mean'
        self.kind = kind
        self.show_line = show_line
        self.show_center = show_center
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            f_base=f_base,
            skip_na=skip_na,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
        self._marker = marker

    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(
            color=self.color,
            marker=self.marker,
            linestyle=self.linestyle)
        if self.marker:
            kwds = kwds | dict(alpha=COLOR.MARKER_ALPHA)
        return kwds
    
    @property
    def kind(self)-> Literal['mean', 'median']:
        """Get and set the type of location ('mean' or 'median') to 
        plot.
        
        Raises
        ------
        AssertionError
            If neither 'mean' or 'median' is given when setting `kind`.
        """
        return self._kind
    @kind.setter
    def kind(self, kind: Literal['mean', 'median']) -> None:
        assert kind in ('mean', 'median')
        self._kind = kind
    
    @property
    def marker(self) -> str:
        """Get the marker style for the center points if show_center
        is True, otherwise '' is returned (read-only)."""
        if self._marker is None:
            self._marker = DEFAULT.MARKER
        return self._marker if self.show_center else ''
    
    @property
    def linestyle(self) -> str:
        """Get rcParams line style if show_line is True, otherwise '' is
        returned (read-only)."""
        return plt.rcParams['lines.linestyle'] if self.show_line else ''
    
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
        
        t_value = getattr(Estimator(target_data), self.kind)
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
        _kwds = self.kw_default | kwds
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
        given, by default `DEFAULT.FEATURE_BASE`.
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Bar

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = list('abcdefghijklmno'),
        y = list(100/x for x in range(1, 16))))
    bar = Bar(
        source=df, target='y', feature='x')
    bar()
    bar.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = list('abcdefghijklmno'),
        y = list(100/x for x in range(1, 16))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True,
        ).plot(
            dsp.Bar,
        ).label() # neded to label feature ticks
    ```

    You can also aggregate a function on the target column
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Bar

    limit = 3.5
    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50) > limit)
            + list(np.random.normal(loc=4, scale=1, size=50) > limit)
            + list(np.random.normal(loc=2, scale=1, size=50) > limit))))
    bar = Bar(
        source=df, target='y', feature='x', method='sum', ax=ax)
    bar()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    limit = 3.5
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50) > limit)
            + list(np.random.normal(loc=4, scale=1, size=50) > limit)
            + list(np.random.normal(loc=2, scale=1, size=50) > limit))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True,
        ).plot(
            dsp.Bar,
            method='sum',
        ).label() # neded to label feature ticks
    ```
    """
    __slots__ = ('method', 'kw_method', 'stack', 'width')

    method: str | None
    """The provided pandas Series method to use for aggregating target
    values."""
    kw_method: dict
    """The provided keyword arguments to be passed to the method."""
    stack: bool
    """Whether to stack the bars."""
    width: float
    """Width of the bars."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            stack: bool = True,
            width: float = CATEGORY.FEATURE_SPACE,
            method: str | None = None,
            kw_method: dict = {},
            f_base: int | float = DEFAULT.FEATURE_BASE,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.stack = stack
        self.width = width
        self.method = method
        self.kw_method = kw_method
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            f_base=f_base,
            skip_na=skip_na,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis,
            **kwds)

        if self.method is not None:
            target = f'{self.target} {self.method}'
            self.source = self.source.rename(columns={self.target: target})
            self.target = target

    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        facecolor = mcolors.to_rgba(self.color, alpha=COLOR.FILL_ALPHA) # type: ignore[attr-defined]
        kwds = dict(
            facecolor=facecolor,
            edgecolor=self.color,
            linewidth=plt.rcParams['lines.linewidth'])
        return kwds
    
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
            f_low, f_upp = map(np.array, zip(*[(b.x0, b.x1) for b in boxs]))
            t_low, t_upp = map(np.array, zip(*[(b.y0, b.y1) for b in boxs]))
            if not self.target_on_y:
                f_low, f_upp, t_low, t_upp = t_low, t_upp, f_low, f_upp
            n = min(len(feature_ticks), len(f_low))
            if (all(feature_ticks[:n] > f_low[:n])
                and all(feature_ticks[:n] < f_upp[:n])
                and any(t_upp[:n] > t_base[:n])):
                t_base[:n] = t_upp[:n]
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
        _kwds = self.kw_default | kwds
        if self.target_on_y:
            self.ax.bar(
                self.x, self.y, width=self.width, bottom=self.t_base, **_kwds)
        else:
            self.ax.barh(
                self.y, self.x, height=self.width, left=self.t_base, **_kwds)


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
    no_percentage_line : bool, optional
        Whether to draw a line as cumulative percentage, by default True
    width: float, optional
        Width of the bars, by default `CATEGORY.FEATURE_SPACE`.
    method : str, optional
        A pandas Series method to use for aggregating target values 
        within each feature level. Like 'sum', 'count' or similar that
        returns a scalar, by default None.
    kw_method : dict, optional
        Additional keyword arguments to be passed to the method,
        by default {}.
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on the
        y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first color
        is taken from the color cycle, by default None.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is used
        within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Pareto

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = list('abcdefghijklmno'),
        y = list(100/x for x in range(1, 16))))
    pareto = Pareto(
        source=df, target='y', feature='x', ax=ax)
    pareto()
    ```

    You can also combine and highlight small frequencies:

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Pareto

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = list('abcdefghijklmno'),
        y = list(100/x for x in range(1, 16))))
    low_values = df.y <= 10
    df2 = df[~low_values].copy()
    df2.loc[len(df)-sum(low_values)] = ('rest', df[low_values].y.sum())
    pareto = Pareto(
        source=df2, target='y', feature='x', highlight='rest', 
        highlight_color='#ff000090', highlighted_as_last=True)
    pareto()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = list('abcdefghijklmno'),
        y = list(100/x for x in range(1, 16))))
    low_values = df.y <= 10
    df2 = df[~low_values].copy()
    df2.loc[len(df)-sum(low_values)] = ('rest', df[low_values].y.sum())
    chart = dsp.SingleChart(
            source=df2,
            target='y',
            feature='x',
        ).plot(
            dsp.Pareto,
            highlight='rest',
            highlight_color=dsp.COLOR.BAD,
            no_percentage_line=False
        )
    ```
    
    Raises
    ------
    AssertionError
        If 'categorical_feature' is True, coming from Chart objects.
    AssertionError
        If an other Axes object in this Figure instance shares the
        feature axis.
    """
    __slots__ = (
        'highlight', 'highlight_color', 'highlighted_as_last',
        'no_percentage_line')

    highlight: Any
    """The feature value whose bar should be highlighted in the chart."""
    highlight_color: str
    """The color to use for highlighting."""
    highlighted_as_last: bool
    """Whether the highlighted bar should be at the end."""
    no_percentage_line: bool
    """Whether to draw the percentage line and the percentage text."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            highlight: Any = None,
            highlight_color: str = COLOR.BAD,
            highlighted_as_last: bool = True,
            no_percentage_line: bool = False,
            width: float = CATEGORY.FEATURE_SPACE,
            method: str | None = None,
            kw_method: dict = {},
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        assert not (kwds.get('categorical_feature', False)), (
            "Don't set categorical_feature to True for Pareto charts, "
            'it would mess up the axis tick labels')
        self.no_percentage_line = no_percentage_line
        self.highlight = highlight
        self.highlight_color = highlight_color
        self.highlighted_as_last = highlighted_as_last
        
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            stack=False,
            width=width,
            method=method,
            kw_method=kw_method,
            skip_na=skip_na,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis,
            **kwds)
        self.source[self.feature] = self._original_f_values
        assert not self.shared_feature_axes, (
            'Do not use Pareto plotter in an chart where the feature axis '
            'is shared with other axes. Pareto sorts the feature axis, '
            'which can mess up the other axes')
    
    @property
    def shared_feature_axes(self) -> bool:
        """True if any other ax in this figure shares the feature axes."""
        which = 'x' if self.target_on_y else 'y'
        return any(self.shared_axes(self.ax, which, True))

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
        if hasattr(self.ax, 'has_pc_texts') or self.no_percentage_line:
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
        _kwds = self.kw_default | kwds
        if self.target_on_y:
            bars = self.ax.bar(self.x, self.y, width=self.width, **_kwds)
            if not self.no_percentage_line:
                self.ax.plot(
                    self.x, self.y.cumsum(), color=self.color,
                    **KW.PARETO_LINE)
        else:
            bars = self.ax.barh(self.y, self.x, height=self.width, **_kwds)
            if not self.no_percentage_line:
                self.ax.plot(
                    self.x[::-1].cumsum(), self.y[::-1], color=self.color,
                    **KW.PARETO_LINE)
        self._highlight_bar_(bars)
        self._set_margin_()
        self._remove_feature_grid_()
        self.add_percentage_texts()
    
    def label_feature_ticks(self) -> None:
        warnings.warn(
            'Calling this method is unnecessary, Pareto plotter already does '
            'this during plotting',
            UserWarning)


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
    width : float, optional
        The width of the jitter, by default `CATEGORY.FEATURE_SPACE`.
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the scatter plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Jitter

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    jitter = Jitter(
        source=df, target='y', feature='x', ax=ax)
    jitter()
    jitter.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True # neded to label feature ticks
        ).plot(
            dsp.Jitter,
        ).label() # neded to label feature ticks
    ```
    """
    __slots__ = ('width')

    width: float
    """The width of the jitter."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            width: float = CATEGORY.FEATURE_SPACE,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.width = width
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            skip_na=skip_na,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis,
            **kwds)
        self._marker = marker
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(color=self.color, marker=self.marker)
        return kwds
        
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
            a=np.random.normal(loc=loc, scale=self.width/6, size=size),
            a_min=loc - self.width/2,
            a_max=loc + self.width/2,)
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
        _kwds = self.kw_default | kwds
        self.ax.scatter(self.x, self.y, **_kwds)


class Beeswarm(TransformPlotter):
    """A class to create and display a basic bee swarm plot.

    This class includes methods that organize the input data into bins 
    according to a specified number of bins (or a default value if none 
    is provided). It calculates the upper limits for each bin and 
    positions the data points within these bins to achieve a horizontal
    distribution in the plot, ensuring as little overlap as possible 
    among the points.

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
    n_bins : int | None, optional
        The number of bins to divide the data into. If not specified, 
        it is calculated as the length of the data divided by 6. 
        Defaults to None
    width : float, optional
        The width of the beeswarm, by default `CATEGORY.FEATURE_SPACE`.
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the scatter plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Beeswarm

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    beeswarm = Beeswarm(
        source=df, target='y', feature='x', ax=ax)
    beeswarm()
    beeswarm.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True # neded to label feature ticks
        ).plot(
            dsp.Beeswarm,
        ).label() # neded to label feature ticks
    ```
    
    Source
    ------
    This code is based on the following source: 
    https://python-graph-gallery.com/509-introduction-to-swarm-plot-in-matplotlib/
    """
    __slots__ = ('width', 'n_bins')

    width: float
    """The maximum width of the beeswarm."""
    n_bins: int
    """The number of bins to divide the data into"""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            n_bins: int | None = None,
            width: float = CATEGORY.FEATURE_SPACE,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        if n_bins is None:
            n_bins = len(source[target]) // 6
        self.n_bins = n_bins
        self.width = width
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            skip_na=skip_na,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis,
            **kwds)
        self._marker = marker
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(color=self.color, marker=self.marker)
        return kwds
        
    def _spread_(
            self,
            n_values: int,
            delta: float
            ) -> Generator[float, Any, None]:
        """Generates the spread values for the beeswarm within the 
        current bin.

        This method calculates the positions of the data points in the 
        beeswarm plot based on the number of values in the current bin 
        and a specified delta value. It determines the direction of 
        spreading based on whether the number of values is odd or even, 
        ensuring a balanced distribution of points.

        Parameters
        ----------
        n_values : int
            The number of values in the current bin for which spread 
            values are to be generated.

        delta : float
            The distance between adjacent spread values, which 
            determines the overall spread of the points in the beeswarm 
            plot.

        Yields
        ------
        float
            The calculated spread values for each point in the current 
            bin, allowing for an alternating distribution to minimize 
            overlap.
        """
        odd = bool(n_values % 2)
        direction = cycle([-1, 1]) if odd else cycle([1, -1])
        offset = 1 if odd else 2
        for i in range(n_values):
            if odd:
                pos = delta * ((i + offset) // 2)
            else:
                pos =  delta * ((i + offset) // 2) - delta/2
            yield next(direction) * pos
        
    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        """Generates the spread values for the beeswarm plot by 
        arranging the target data into bins.

        The method divides the input data into bins based on the 
        specified number of bins and calculates the spread of values 
        within each bin to create a horizontal distribution.

        Parameters
        ----------
        feature_data : float
            The center position on the feature axis where the beeswarm 
            values will be centered.
        target_data : pandas Series
            feature grouped target data, coming from `feature_grouped' 
            generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        """
        hist, _ = np.histogram(target_data, bins=self.n_bins)
        delta = self.width / hist.max()
        beeswarm = np.array(
            [pos for n in hist for pos in self._spread_(n, delta)])
        
        data = pd.DataFrame({
            self.target: sorted(target_data),
            self.feature: feature_data + beeswarm})
        return data

    def __call__(self, **kwds) -> None:
        """Perform the jitter plot operation.

        Parameters
        ----------
        **kwds
            Additional keyword arguments to be passed to the Axes 
            `scatter` method.
        """
        _kwds = self.kw_default | kwds
        self.ax.scatter(self.x, self.y, **_kwds)


class QuantileBoxes(SpreadOpacity, TransformPlotter):
    """TransformPlotter for visualizing quantiles through box plots.

    This class is designed to create box plots that represent various 
    quantiles of the data based on specified ranges. The ranges are used
    to calculate the lower and upper quantiles, which define the 
    boundaries of the boxes.

    Parameters
    ----------
    source : pandas DataFrame
        A long-format DataFrame containing the data source for the plot.
    target : str
        The column name of the target variable to be plotted.
    feature : str, optional
        The column name of the feature variable for the plot, by 
        default an empty string.
    strategy : {'eval', 'fit', 'norm', 'data'}, optional
        Which strategy should be used to determine the quantiles:
        - `eval`: The strategy is determined according to the given 
          evaluate function. If none is given, the internal `evaluate`
          method is used.
        - `fit`: First, the distribution that best represents the 
          process data is searched for and then the agreed process 
          spread is calculated
        - `norm`: it is assumed that the data is subject to normal 
          distribution. The variation tolerance is then calculated as 
          agreement * standard deviation
        - `data`: The quantiles for the process variation tolerance 
          are read directly from the data.
        
        Default is 'data'.
    agreements : Tuple[float, ...] or Tuple[int, ...], optional
        Specifies the tolerated process variation for calculating 
        quantiles. These quantiles are used to represent the filled area 
        with different opacity, thus highlighting the quantiles. The 
        agreements can be either integers or floats, determining the 
        process variation tolerance in the following ways:
        - If integers, the quantiles are determined using the normal 
          distribution (agreement * σ), e.g., agreement = 6 covers 
          ~99.75% of the data.
        - If floats, values must be between 0 and 1, interpreted as 
          acceptable proportions for the quantiles, e.g., 0.9973 
          corresponds to ~6σ.
        
        Default is `DEFAULT.AGREEMENTS` = (2, 4, 6), corresponding to 
        (±1σ, ±2σ, ±3σ).
    possible_dists : tuple of strings or rv_continous, optional
        Distributions to which the data may be subject. Only 
        continuous distributions of scipy.stats are allowed,
        by default `DIST.COMMON`
    vary_width : float, optional
        If True, the center box is the widest, while the outer boxes are 
        progressively narrower, reflecting the distribution of the data.
        Defaults to True
    width : float, optional
        The width of the boxes. If vary_width is set to True, the 
        central box has this width, all others are narrower.
        Defaults to `CATEGORY.FEATURE_SPACE`.
    skip_na : Literal['none', 'all', 'any'], optional
        A flag indicating how to handle missing values in the feature 
        grouped data:
        - 'none': No missing values are skipped.
        - 'all': Grouped data is skipped if all values are missing.
        - 'any': Grouped data is skipped if any value is missing.

    target_on_y : bool, optional
        A flag indicating whether the target variable is plotted on the 
        y-axis, by default True.
    color : str | None, optional
        The color used to draw the box plots. If None, the first color 
        from the color cycle is used, by default None.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds :
        Additional keyword arguments that are ignored in this context, 
        primarily serving to capture any extra arguments when this class 
        is used within chart objects.
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import QuantileBoxes

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    boxes = QuantileBoxes(
        source=df, target='y', feature='x', strategy='norm',
        agreements=(0.25, 0.5, 0.75, 0.95), vary_width=False, width=0.2,
        ax=ax)
    boxes()
    boxes.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True # neded to label feature ticks
        ).plot(
            dsp.QuantileBoxes,
            strategy='norm',
            agreements=(0.25, 0.5, 0.75, 0.95),
            vary_width=False,
            width=0.2
        ).label() # neded to label feature ticks
    ```
    """
    __slots__ = (
        'strategy', '_agreements', 'possible_dists', 'estimation', 
        'vary_width', 'width')
    
    vary_width: bool
    """Flag that indicates whether the width of the boxes should vary, 
    with the widest box in the middle and the narrower one towards the 
    outside. If False, all have the same width."""
    width: float
    """The maximum width of the center box in the plot."""
    
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            strategy: Literal['eval', 'fit', 'norm', 'data'] = 'data',
            agreements: Tuple[float, ...] = DEFAULT.AGREEMENTS,
            possible_dists: Tuple[str | rv_continuous, ...] = DIST.COMMON,
            vary_width: bool = True,
            width: float = CATEGORY.FEATURE_SPACE,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.vary_width = vary_width
        self.strategy = strategy
        self.agreements = agreements
        self.possible_dists = possible_dists
        self.width = width / len(agreements) if vary_width else width
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            skip_na=skip_na,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis,
            **kwds)
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(color=self.color)
        return kwds
    
    def width_values(self) -> NDArray:
        """Returns the widths of the boxes in the plot."""
        widths = []
        for i in range(len(self.agreements)):
            if i > 0:
                widths.append(widths[-1])
            widths.append((i + 1 if self.vary_width else 1) * self.width / 2)
        return np.array(widths + widths[::-1])
    
    def transform(
            self, feature_data: float | int, target_data: Series) -> DataFrame:
        """Generates the spread values for the beeswarm plot by 
        arranging the target data into bins.

        The method divides the input data into bins based on the 
        specified number of bins and calculates the spread of values 
        within each bin to create a horizontal distribution.

        Parameters
        ----------
        feature_data : float
            The center position on the feature axis where the beeswarm 
            values will be centered.
        target_data : pandas Series
            feature grouped target data, coming from `feature_grouped' 
            generator.

        Returns
        -------
        data : pandas DataFrame
            The transformed data source for the plot.
        """
        data = pd.DataFrame()
        widths = self.width_values()
        quantiles = self.quantiles(target_data)
        data = pd.DataFrame({
            self.target: quantiles,
            self.feature: feature_data,
            PLOTTER.LOWER: feature_data - widths,
            PLOTTER.UPPER: feature_data + widths,
            PLOTTER.SUBGROUP: self.subgroup_values(2, quantiles),
            PLOTTER.F_BASE_NAME: feature_data * np.ones(len(widths))})
        return data

    def __call__(self, **kwds) -> None:
        """Perform the quantiles plot operation.

        Parameters
        ----------
        **kwds
            Additional keyword arguments to be passed to the Axes 
            `fill_between` method.
        """
        for f_base, group in self.source.groupby(PLOTTER.F_BASE_NAME, observed=True):
            quantiles = group[self.target]
            lower = group[PLOTTER.LOWER]
            upper = group[PLOTTER.UPPER]
            for agreement in group[PLOTTER.SUBGROUP].unique():
                _kwds = self._kw_fill(self.color, agreement, group) | kwds
                if self.target_on_y:
                    self.ax.fill_betweenx(quantiles, lower, upper, **_kwds)
                else:
                    self.ax.fill_between(quantiles, lower, upper, **_kwds)


class GaussianKDE(SpreadOpacity, TransformPlotter):
    """Class for creating Gaussian Kernel Density Estimation (KDE) 
    plotters. Use this class for univariate plots.

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
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    ignore_feature: bool, optional
        Flag indicating whether the feature axis should be ignored. If 
        True, all curves have base 0 on the feature axis,
        by default True
    margin : float, optional
        Margin for the sequence as factor of data range (max - min ). 
        If margin is 0, The two ends of the estimated density curve then 
        show the minimum and maximum value. Default is 0.
    fill : bool, optional
        Flag whether to fill in the curves, by default True
    agreements : Tuple[float, ...] or Tuple[int, ...], optional
        Specifies the tolerated process variation for calculating 
        quantiles. These quantiles are used to represent the filled area 
        with different opacity, thus highlighting the quantiles.If you 
        want the filled area to be uniform without highlighting the 
        quantiles, provide an empty tuple. This argument is only taken 
        into account if fill is set to True. The agreements can be either 
        integers or floats, determining the process variation tolerance 
        in the following ways:
        - If integers, the quantiles are determined using the normal 
          distribution (agreement * σ), e.g., agreement = 6 covers 
          ~99.75% of the data.
        - If floats, values must be between 0 and 1, interpreted as 
          acceptable proportions for the quantiles, e.g., 0.9973 
          corresponds to ~6σ.
        - If empty tuple, the filled area is uniform without 
          highlighting the quantiles.
        
        Default is `DEFAULT.AGREEMENTS` = (2, 4, 6), corresponding to 
        (±1σ, ±2σ, ±3σ).
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    n_points : int, optional
        Number of points the kernel density estimation and sequence 
        should have, by default KD_SEQUENCE_LEN 
        (defined in constants.py).
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import GaussianKDE

    fig, ax = plt.subplots()
    colors = ('#1f77b4', '#ff7f0e', '#2ca02c')
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    for color, (name, group) in zip(colors, df.groupby('x')):
        kde = GaussianKDE(
            source=group, target='y', strategy='norm', agreements=(2, 4, 6),
            target_on_y=False, color=color, margin=0.3, ax=ax)
        kde(label=name)
    handles, labels = ax.get_legend_handles_labels()
    agreements = kde.agreements[::-1] * df.x.nunique()
    labels = [f'{l} {a}σ' for l, a in zip(labels, agreements)]
    ax.legend(handles, labels)
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            hue='x',
            target_on_y=False,
        ).plot(
            dsp.GaussianKDE,
            strategy='norm',
            agreements=(2, 4, 6), # std for area color opacities
            visible_spines='target',
            hide_axis='feature',
        ).label() # neded to label feature ticks
    ```
    """
    __slots__ = (
        'strategy', '_agreements', 'possible_dists', '_height', '_stretch', 
        'fill', 'n_points', 'margin')

    _height: float | None
    """Height of kde curve at its maximum."""
    _stretch: float
    """Factor by which the curve was stretched in height."""
    fill: bool
    """Flag whether to fill in the curves"""
    n_points: int
    """Number of points that have the kde and its sequence."""
    margin: float
    """Margin for the sequence as factor of data range (max - min ). If
    margin is 0, The two ends of the estimated density curve then show 
    the minimum and maximum value."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            stretch: float = 1,
            height: float | None = None,
            skip_na: Literal['all', 'any'] | None = None,
            ignore_feature: bool = True,
            margin: float = 0,
            fill: bool = True,
            agreements: Tuple[float, ...] | Tuple[int, ...] = DEFAULT.AGREEMENTS,
            target_on_y: bool = True,
            color: str | None = None,
            n_points: int = DEFAULT.KD_SEQUENCE_LEN,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.strategy= 'data'
        self.agreements = agreements
        self.possible_dists = DIST.COMMON
        self._stretch = stretch
        self.n_points = n_points
        self.margin = margin
        self.fill = fill
        if not ignore_feature and kwds.get('feature', False):
            height = height if height else CATEGORY.FEATURE_SPACE
        else:
            kwds['feature'] = PLOTTER.TRANSFORMED_FEATURE
        self._height = height
        f_base = kwds.pop('f_base', DEFAULT.FEATURE_BASE)
        super().__init__(
            source=source,
            target=target,
            f_base=f_base,
            skip_na=skip_na,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis,
            **kwds)
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(
            color=COLOR.DARKEN if self.fill else self.color,)
        return kwds
    
    @property
    def height(self) -> float | None:
        """Height of kde curve at its maximum."""
        return self._height
    
    @property
    def stretch(self) -> float:
        """Factor by which the curve was stretched in height"""
        return self._stretch

    @property
    def highlight_quantiles(self) -> bool:
        """Flag indicating whether the quantiles should be highlighted."""
        return bool(self.agreements) and self.fill
        
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

        first_value: float = target_data.iloc[0]
        if all(target_data == first_value):
            stretch = self.stretch if self.height is None else self.height
            estimation = [stretch + feature_data]
            sequence = [first_value]
        else:
            sequence, estimation = estimate_kernel_density(
                data=target_data,
                stretch=self.stretch,
                height=self.height,
                base=feature_data,
                n_points=self.n_points,
                margin=self.margin)
        
        ones = np.ones(len(sequence))
        if self.highlight_quantiles and not isinstance(sequence, list):
            quantiles = self.quantiles(target_data)
            subgroups = self.subgroup_values(sequence, quantiles)
        else:
            subgroups = 0 * ones

        data = pd.DataFrame({
            self.target: sequence,
            self.feature: estimation,
            PLOTTER.SUBGROUP: subgroups,
            PLOTTER.F_BASE_NAME: feature_data * ones})
        return data
    
    def _get_lower_estimation_(
        self, f_base: float | int, estim_upp: Series) -> Series:
        """Get the lower estimation of the kernel density. For the KDE 
        it is just a straight line at the location of the feature base. 
        For Violine the upper kernel density estimate is mirrored at the
        feature base."""
        if self.__class__.__name__ == 'Violine':
            estim_low = 2*f_base - estim_upp
        else:
            estim_low = pd.Series(
                f_base * np.ones(len(estim_upp)),
                index=estim_upp.index)
        return estim_low
        
    def __call__(self, kw_line: Dict[str, Any] = {}, **kwds) -> None:
        """Perform the plotting operation.

        The estimated kernel density is plotted as a line. The curves 
        are additionally filled if the "fill" option was set to "true" 
        during initialization.

        Parameters
        ----------
        kw_line : Dict[str, Any], optional
            Additional keyword arguments for the axes `plot` method,
            by default {}.
        **kwds : Dict[str, Any], optional
            Additional keyword arguments for the axes `fill_between`
            method, by default {}.
        """
        _kw_line = self.kw_default | kw_line
        for f_base, group in self.source.groupby(PLOTTER.F_BASE_NAME, observed=True):
            estim_upp = group[self.feature]
            estim_low = self._get_lower_estimation_(f_base, estim_upp) # type: ignore
            sequence = group[self.target]
            
            for agreement in group[PLOTTER.SUBGROUP].unique():
                _kwds = self._kw_fill(self.color, agreement, group) | kwds
                if self.target_on_y:
                    if self.fill:
                        self.ax.fill_betweenx(
                            sequence, estim_low, estim_upp, **_kwds)
                    self.ax.plot(
                        estim_upp, sequence, estim_low, sequence, **_kw_line)
                else:
                    if self.fill:
                        self.ax.fill_between(
                            sequence, estim_low, estim_upp, **_kwds)
                    self.ax.plot(
                        sequence, estim_upp, sequence, estim_low,**_kw_line)


class GaussianKDEContour(Plotter):
    """Class for creating contour plotters. Use this class for bivariate
    plots.

    Performs a 2 dimensional kernel density estimation using Gaussian
    Kernels. The estimation is then shown as contour lines. The x- and
    y-axes remain as feature and target axes.

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
    fill : bool, optional
        Flag indicating whether to fill between the contour lines,
        by default True
    fade_outers: bool, optional
        Flag indicating whether the outer lines of the contour plot
        should be faded. This has no effect if fill is True,
        by default True.
    n_points : int, optional
        Number of points the estimate and the sequence should have. 
        Note that the calculated points are equal to the square of the 
        given number (because the contour is two-dimensional).
        by default KD_SEQUENCE_LEN (defined in constants.py)
    margin : float, optional
        Margin for the sequence as factor of data range, by default 0.2.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on 
        the y-axis. If False, all contour lines have the same color. 
        by default True
    color : str | None, optional
        Color to be used to draw the artists. If None, the first 
        color is taken from the color cycle, by default None.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import matplotlib.pyplot as plt
    from daspi import GaussianKDEContour, load_dataset

    fig, ax = plt.subplots()
    colors = ('#1f77b4', '#ff7f0e', '#2ca02c')
    df = load_dataset('iris')
    for color, (name, group) in zip(colors, df.groupby('species')):
        kde = GaussianKDEContour(
            source=group, target='length', feature='width', color=color,
            fill=False, fade_outers=False, margin=0.3, ax=ax)
        kde()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import daspi as dsp

    df = load_dataset('iris')
    chart = dsp.SingleChart(
            source=df,
            target='length',
            feature='width',
            hue='leaf',
        ).plot(
            dsp.GaussianKDEContour,
            fill=False,
            fade_outers=False,
            margin=0.3
        ).label() # neded to add legend
    ```
    """
    __slots__ = ('cmap', 'shape', 'fill', 'n_points')

    cmap : LinearSegmentedColormap
    """The colormap to be used for the contour plot."""
    shape: Tuple[int, int]
    """Shape used to reshape data before plotting the contours."""
    fill: bool
    """Flag indicating whether to fill between the contour lines."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            fill: bool = True,
            fade_outers: bool = True,
            n_points: int = DEFAULT.KD_SEQUENCE_LEN,
            margin: float = 0.2,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.shape = (n_points, n_points)
        self.fill = fill
        feature_seq, target_seq, estimation = estimate_kernel_density_2d(
            feature= source[feature], target=source[target], n_points=n_points,
            margin=margin)
        data = pd.DataFrame({
            feature: feature_seq.ravel(),
            target: target_seq.ravel(),
            PLOTTER.TRANSFORMED_FEATURE: estimation.ravel()})
        super().__init__(
            source=data,
            target=target,
            feature=feature,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis,
            **kwds)
        if self.fill:
            colors = [COLOR.TRANSPARENT, self.color]
        elif fade_outers:
            rgba = mcolors.to_rgba(self.color)
            colors = [(*rgba[:3], 0.0), self.color]
        else:
            colors = [self.color, self.color]
        self.cmap = LinearSegmentedColormap.from_list('', colors)

    @property
    def kw_default(self) -> Dict[str, Any]:
        """Return the default keyword arguments for the plot."""
        kwds = dict(cmap=self.cmap)
        return kwds

    def __call__(self, **kwds) -> None:
        """Perform the plotting operation."""
        X = self.x.to_numpy().reshape(self.shape)
        Y = self.y.to_numpy().reshape(self.shape)
        estimation = self.source[PLOTTER.TRANSFORMED_FEATURE]
        Z = estimation.to_numpy().reshape(self.shape)
        _kwds = self.kw_default | kwds
        if self.fill:
            self.ax.contourf(X, Y, Z, **_kwds)
        else:
            self.ax.contour(X, Y, Z, **_kwds)
    
    def label_feature_ticks(self) -> None:
        warnings.warn(
            'Calling this method for this plotter is pointless.',
            UserWarning)


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
    margin : float, optional
        Margin for the sequence as factor of data range (max - min ). 
        If margin is 0, The two ends of the estimated density curve then 
        show the minimum and maximum value. Default is 0.
    fill : bool, optional
        Flag whether to fill in the curves, by default True
    agreements : Tuple[float, ...] or Tuple[int, ...], optional
        Specifies the tolerated process variation for calculating 
        quantiles. These quantiles are used to represent the filled area 
        with different opacity, thus highlighting the quantiles.If you 
        want the filled area to be uniform without highlighting the 
        quantiles, provide an empty tuple. This argument is only taken 
        into account if fill is set to True. The agreements can be either 
        integers or floats, determining the process variation tolerance 
        in the following ways:
        - If integers, the quantiles are determined using the normal 
          distribution (agreement * σ), e.g., agreement = 6 covers 
          ~99.75% of the data.
        - If floats, values must be between 0 and 1, interpreted as 
          acceptable proportions for the quantiles, e.g., 0.9973 
          corresponds to ~6σ.
        - If empty tuple, the filled area is uniform without 
          highlighting the quantiles.
        
        Default is `DEFAULT.AGREEMENTS` = (2, 4, 6), corresponding to 
        (±1σ, ±2σ, ±3σ).
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import Violine

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    violine = Violine(
        source=df, target='y', feature='x', fill=True, margin=0.3, 
        agreements=(), ax=ax)
    violine()
    violine.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True, # neded to label the feature tick labels
        ).plot(
            dsp.Violine,
            fill=False,
            margin=0.3
        ).label() # neded to label the feature tick labels
    ```
    """

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            width: float = CATEGORY.FEATURE_SPACE,
            margin: float = 0,
            fill: bool = True,
            agreements: Tuple[float, ...] | Tuple[int, ...] = DEFAULT.AGREEMENTS,
            target_on_y: bool = True,
            color: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            height=width/2,
            ignore_feature=False,
            margin=margin,
            fill=fill,
            agreements=agreements,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis,
            **kwds)

    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(color=self.color, alpha=COLOR.FILL_ALPHA)
        return kwds


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
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the center points. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from daspi import Errorbar
    from collections import defaultdict

    fig, ax = plt.subplots()
    data = defaultdict(list)
    data['x'] = ['first', 'second', 'third']
    for loc in [3, 4, 2]:
        temp = np.random.normal(loc=loc, scale=1, size=10)
        x_bar = np.mean(temp)
        sem = np.std(temp, ddof=1) / np.sqrt(len(temp))
        data['x_bar'].append(x_bar)
        data['lower'].append(x_bar - sem)
        data['upper'].append(x_bar + sem)

    df = pd.DataFrame(data)
    errorbar = Errorbar(
        source=df, target='x_bar', feature='x', lower='lower', upper='upper',
        show_center=True, ax=ax)
    errorbar(kw_center=dict(color='black', s=30, marker='_')
    errorbar.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    from collections import defaultdict

    data = defaultdict(list)
    data['x'] = ['first', 'second', 'third']
    for loc in [3, 4, 2]:
        temp = np.random.normal(loc=loc, scale=1, size=10)
        x_bar = np.mean(temp)
        sem = np.std(temp, ddof=1) / np.sqrt(len(temp))
        data['x_bar'].append(x_bar)
        data['lower'].append(x_bar - sem)
        data['upper'].append(x_bar + sem)

    chart = dsp.SingleChart(
            source=df,
            target='x_bar',
            hue='x',
            dodge=True,
        ).plot(
            dsp.Errorbar,
            lower='lower',
            upper='upper',
            bars_same_color=True,
            kw_call={'kw_center': dict(color='black', s=30, marker='_')}
        ).label() # neded to add legend
    ```
    """
    __slots__ = ('lower', 'upper', 'show_center', 'bars_same_color')

    lower: str
    """Column name of the lower error values."""
    upper: str
    """Column name of the upper error values."""
    show_center: bool
    """Flag indicating whether to show the center points."""
    bars_same_color: bool
    """Flag indicating whether to use same color for error bars as 
    markers for center."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            lower: str,
            upper: str,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.lower = lower
        self.upper = upper
        self.show_center = show_center
        self.bars_same_color = bars_same_color

        super().__init__(
            source=source,
            target=target,
            feature=feature,
            skip_na=skip_na,
            target_on_y=target_on_y,
            color=color,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
        
        if self.lower not in self.source:
            _source = source.set_index(self.feature).copy()
            idx = list(self._original_f_values)
            self.source[self.lower] = list(_source.loc[idx, self.lower])
            self.source[self.upper] = list(_source.loc[idx, self.upper])

        self._marker = marker

    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        _color = dict(color=self.color) if self.bars_same_color else {}
        kwds = KW.ERROR_BAR | _color
        return kwds
    
    @property
    def marker(self) -> str:
        """Get the marker style for the center points if show_center is
        True, otherwise '' is returned (read-only)."""
        if self._marker is None:
            self._marker = DEFAULT.MARKER
        return self._marker if self.show_center else ''
        
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
        err = np.abs(np.array([
            self.source[self.target] - self.source[self.lower],
            self.source[self.upper] - self.source[self.target]]))
        return err
    
    def __call__(self, kw_center: dict = {}, **kwds) -> None:
        """Perform the plotting operation.

        Parameters
        ----------
        kw_center : dict, optional
            Additional keyword arguments for the axes `scatter` method,
            by default {}.
        **kwds :
            Additional keyword arguments for the axes `errorbar` method.
        """
        if self.show_center:
            kw_center = dict(color=self.color, marker=self.marker) | kw_center
            self.ax.scatter(self.x, self.y, **kw_center)
        _kwds = self.kw_default | kwds
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
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

        by default False
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the center points. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import StandardErrorMean

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    sem = StandardErrorMean(
        source=df, target='y', feature='x',
        show_center=True, bars_same_color=True, ax=ax)
    sem(kw_center=dict(s=30, marker='_'))
    sem.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True, # neded to label the feature tick labels
        ).plot(
            dsp.StandardErrorMean,
            show_center=True,
            bars_same_color=True,
            kw_call=dict(kw_center=dict(s=30, marker='_'))
        ).label() # neded to label the feature tick labels
    ```
    """
    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        super().__init__(
            source=source,
            target=target,
            lower=PLOTTER.LOWER,
            upper=PLOTTER.UPPER,
            feature=feature,
            show_center=show_center,
            bars_same_color=bars_same_color,
            skip_na=skip_na, 
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)

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
    strategy : {'eval', 'fit', 'norm', 'data'}, optional
        Which strategy should be used to determine the control 
        limits (process spread):
        - `eval`: The strategy is determined according to the given 
          evaluate function. If none is given, the internal `evaluate`
          method is used.
        - `fit`: First, the distribution that best represents the 
          process data is searched for and then the agreed process 
          spread is calculated
        - `norm`: it is assumed that the data is subject to normal 
          distribution. The variation tolerance is then calculated as 
          agreement * standard deviation
        - `data`: The quantiles for the process variation tolerance 
          are read directly from the data.
        
        Default is 'norm'.
    agreement : int or float, optional
        Specify the tolerated process variation for which the 
        control limits are to be calculated. 
        - If int, the spread is determined using the normal 
          distribution agreement*sigma, 
          e.g. agreement = 6 -> 6*sigma ~ covers 99.75 % of the data. 
          The upper and lower permissible quantiles are then 
          calculated from this.
        - If float, the value must be between 0 and 1.This value is
          then interpreted as the acceptable proportion for the 
          spread, e.g. 0.9973 (which corresponds to ~ 6 sigma)
        
        Default is 6 because SixSigma ;-)
    possible_dists : tuple of strings or rv_continous, optional
        Distributions to which the data may be subject. Only 
        continuous distributions of scipy.stats are allowed,
        by default `DIST.COMMON`
    show_center : bool, optional
        Flag indicating whether to show the center points (see `kind` 
        option). Default is True.
    kind : Literal['mean', 'median'], optional
        The type of center to plot ('mean' or 'median'),
        by default 'mean'.
    bars_same_color : bool, optional
        Flag indicating whether to use same color for error bars as 
        markers for center. If False, the error bars are black,
        by default False
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the center points. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import SpreadWidth, Beeswarm

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    swarm = Beeswarm(source=df, target='y', feature='x')
    swarm(color=(0.3, )*4)
    spread = SpreadWidth(
        source=df, target='y', feature='x', strategy='data', agreement=1.0, 
        kind='median', show_center=True, bars_same_color=True,
        ax=ax)
    spread(kw_center=dict(s=30, marker='_'))
    spread.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True, # neded to label the feature tick labels
        ).plot(
            dsp.SpreadWidth,
            strategy='data',
            agreement=1.0,
            show_center=True,
            kind='median',
            bars_same_color=True,
            kw_call=dict(kw_center=dict(s=30, marker='_'))
        ).plot(
            dsp.Beeswarm,
            color=(0.3, ) * 4
        ).label() # neded to label the feature tick labels
    ```

    Notes
    -----
    Under the hood, the class `daspi.statistics.estimation.Estimator` is 
    used. The error bar then corresponds to the control limits `lcl` and 
    ucl` calculated with it.

    If you want to display the minimum and maximum values (the range), 
    set agreement to `1.0` (important: it must be a float) or to 
    `float('inf')` and strategy to 'data'. This way, the control limits 
    correspond to the minimum and maximum of the data.
    """
    __slots__ = (
        'strategy', 'agreement', 'possible_dists', '_kind', 'estimation')

    strategy: Literal['eval', 'fit', 'norm', 'data']
    """Strategy for estimating the spread width."""
    agreement: float | int
    """Agreement value for the spread width estimation."""
    possible_dists: Tuple[str | rv_continuous, ...]
    """Tuple of possible distributions for the spread width
    estimation."""
    _kind: Literal['mean', 'median']
    """The type of center to plot ('mean' or'median')."""
    estimation: Estimator
    """Estimator instance used for spread width and center estimation."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
            agreement: float | int = 6,
            possible_dists: Tuple[str | rv_continuous, ...] = DIST.COMMON,
            show_center: bool = True,
            kind: Literal['mean', 'median'] = 'mean',
            bars_same_color: bool = False,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self._kind = 'mean'
        self.kind = kind
        self.strategy = strategy
        self.agreement = agreement
        self.possible_dists = possible_dists
        
        super().__init__(
            source=source,
            target=target,
            lower=PLOTTER.LOWER,
            upper=PLOTTER.UPPER,
            feature=feature,
            show_center=show_center,
            bars_same_color=bars_same_color,
            skip_na=skip_na,
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)

    @property
    def marker(self) -> str:
        """Get the marker style for the center points if `show_center`
        is True, otherwise '' is returned. By default the marker is '_' 
        if `target_on_y` is True, '|' otherwise (read-only)."""
        if self._marker is None:
            self._marker = '_' if self.target_on_y else '|'
        return self._marker if self.show_center else ''
    
    @property
    def kind(self)-> Literal['mean', 'median']:
        """Get and set the type of location ('mean' or 'median') to 
        plot.
        
        Raises
        ------
        AssertionError
            If neither 'mean' or 'median' is given when setting `kind`.
        """
        return self._kind
    @kind.setter
    def kind(self, kind: Literal['mean', 'median']) -> None:
        assert kind in ('mean', 'median')
        self._kind = kind

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
        self.estimation = Estimator(
            samples=target_data, strategy=self.strategy, agreement=self.agreement,
            possible_dists=self.possible_dists)
        data = pd.DataFrame({
            self.target: [getattr(self.estimation, self.kind)],
            self.feature: [feature_data],
            self.lower: [self.estimation.lcl],
            self.upper: [self.estimation.ucl]})
        return data
    
    def __call__(self, kw_center: dict = {}, **kwds) -> None:
        """Perform the plotting operation.

        Parameters
        ----------
        kw_center : dict, optional
            Additional keyword arguments for the axes `scatter` method,
            by default {}.
        **kwds :
            Additional keyword arguments for the axes `errorbar` method.
        """
        kw_center = dict(marker=self.marker) | kw_center
        return super().__call__(kw_center, **kwds)


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
    n_groups : int, optional
        Number of groups (variable combinations) for the Bonferroni 
        adjustment. A good way to do this is to pass 
        `df.groupby(list_of_variates).ngroups`, where `list_of_variates` 
        is a list containing all the categorical columns in the source 
        that will be used for the chart to split the data into groups 
        (hue, categorical features, etc.). Specify 1 to not do a 
        Bonferroni adjustment. Default is 1
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
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    confidence_level : float, optional
        Confidence level for the confidence intervals,
        by default 0.95.
    ci_func : Callable, optional
        Function for calculating the confidence intervals. The following
        two arguments are passed to the function: The sample data and 
        the confidence level. The returned values must be three floats
        in order: center value, lower confidence limit and upper 
        confidence limit.
        Default is `daspi.statistics.conficence.mean_ci`.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the center points. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import ConfidenceInterval, variance_ci

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    ci = ConfidenceInterval(
        source=df, target='y', feature='x', show_center=True, ci_func=variance_ci,
        n_groups=df.x.nunique(), confidence_level=0.95, bars_same_color=True,
        ax=ax)
    ci(kw_center=dict(s=30, marker='_'))
    ci.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=1, scale=3, size=50))
            + list(np.random.normal(loc=1, scale=4, size=50))
            + list(np.random.normal(loc=1, scale=2, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True, # neded to label the feature tick labels
        ).plot(
            dsp.ConfidenceInterval,
            show_center=True,
            ci_func=dsp.variance_ci,
            n_groups=df.x.nunique(),
            confidence_level=0.95,
            bars_same_color=True,
            kw_call=dict(kw_center=dict(s=30, marker='_'))
        ).label() # neded to label the feature tick labels
    ```
    """
    __slots__ = ('confidence_level', 'ci_func', 'n_groups')

    confidence_level: float
    """Confidence level for the confidence intervals."""
    ci_func: Callable
    """Provided function for calculating the confidence intervals."""
    n_groups: int
    """Number of unique feature values."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            n_groups: int = 1,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            ci_func: Callable = mean_ci,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        assert n_groups >= 1 and isinstance(n_groups, int), (
            f'The given n_groups must be an integer and >= 1 got {n_groups}')
        self.confidence_level = confidence_level
        self.ci_func = ci_func
        self.n_groups = n_groups
        
        super().__init__(
            source=source,
            target=target,
            lower=PLOTTER.LOWER,
            upper=PLOTTER.UPPER,
            feature=feature,
            show_center=show_center,
            bars_same_color=bars_same_color,
            skip_na=skip_na, 
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
    
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
    n_groups : int, optional
        Number of groups (variable combinations) for the Bonferroni 
        adjustment. A good way to do this is to pass 
        `df.groupby(list_of_variates).ngroups`, where `list_of_variates` 
        is a list containing all the categorical columns in the source 
        that will be used for the chart to split the data into groups 
        (hue, categorical features, etc.). Specify 1 to not do a 
        Bonferroni adjustment. Default is 1
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
    marker : str | None, optional
        The marker style for the center points. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import MeanTest

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    test = MeanTest(
        source=df, target='y', feature='x', show_center=True,
        n_groups=df.x.nunique(), confidence_level=0.95, bars_same_color=True,
        ax=ax)
    test(kw_center=dict(s=30, marker='_'))
    test.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=3, scale=1, size=50))
            + list(np.random.normal(loc=4, scale=1, size=50))
            + list(np.random.normal(loc=2, scale=1, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True, # neded to label the feature tick labels
        ).plot(
            dsp.MeanTest,
            show_center=True,
            n_groups=df.x.nunique(),
            confidence_level=0.95,
            bars_same_color=True,
            kw_call=dict(kw_center=dict(s=30, marker='_'))
        ).label() # neded to label the feature tick labels
    ```
    """
    def __init__(
            self,
            source: DataFrame,
            target: str,
            n_groups: int = 1,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        super().__init__(
            source=source,
            target=target,
            n_groups=n_groups,
            feature=feature,
            show_center=show_center,
            bars_same_color=bars_same_color,
            target_on_y=target_on_y,
            confidence_level=confidence_level,
            ci_func=mean_ci,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)


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
    n_groups : int, optional
        Number of groups (variable combinations) for the Bonferroni 
        adjustment. A good way to do this is to pass 
        `df.groupby(list_of_variates).ngroups`, where `list_of_variates` 
        is a list containing all the categorical columns in the source 
        that will be used for the chart to split the data into groups 
        (hue, categorical features, etc.). Specify 1 to not do a 
        Bonferroni adjustment. Default is 1
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
    marker : str | None, optional
        The marker style for the center points. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import VariationTest

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=1, scale=3, size=50))
            + list(np.random.normal(loc=1, scale=4, size=50))
            + list(np.random.normal(loc=1, scale=2, size=50)))))
    test = VariationTest(
        source=df, target='y', feature='x', show_center=True,
        n_groups=df.x.nunique(), confidence_level=0.95, bars_same_color=True,
        ax=ax)
    test(kw_center=dict(s=30, marker='_'))
    test.label_feature_ticks()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 50 + ['second'] * 50 + ['third'] * 50,
        y = (
            list(np.random.normal(loc=1, scale=3, size=50))
            + list(np.random.normal(loc=1, scale=4, size=50))
            + list(np.random.normal(loc=1, scale=2, size=50)))))
    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True, # neded to label the feature tick labels
        ).plot(
            dsp.VariationTest,
            show_center=True,
            n_groups=df.x.nunique(),
            confidence_level=0.95,
            bars_same_color=True,
            kw_call=dict(kw_center=dict(s=30, marker='_'))
        ).label() # neded to label the feature tick labels
    ```
    """
    def __init__(
            self,
            source: DataFrame,
            target: str,
            n_groups: int = 1,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            kind: Literal['stdev', 'variance'] = 'stdev',
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        ci_func = stdev_ci if kind == 'stdev' else variance_ci
        super().__init__(
            source=source,
            target=target,
            n_groups=n_groups,
            feature=feature,
            show_center=show_center,
            bars_same_color=bars_same_color,
            target_on_y=target_on_y,
            confidence_level=confidence_level,
            ci_func=ci_func,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)


class ProportionTest(ConfidenceInterval):
    """Class for creating plotters with error bars representing 
    confidence intervals for proportion (events/observation).

    This class is specifically designed for testing the statistical
    significance of proportions. It uses confidence intervals
    to visually represent the uncertainty in the variation estimates and
    allows for a quick assessment of whether the intervals overlap or
    not.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name to use for the target variable. If falsy, the name
        will be formed from the specified `events` and `observations` 
        with a "/" character in between.
    n_groups : int, optional
        Number of groups (variable combinations) for the Bonferroni 
        adjustment. A good way to do this is to pass 
        `df.groupby(list_of_variates).ngroups`, where `list_of_variates` 
        is a list containing all the categorical columns in the source 
        that will be used for the chart to split the data into groups 
        (hue, categorical features, etc.). Specify 1 to not do a 
        Bonferroni adjustment. Default is 1
    events : str
        Column name containing the values of counted events for each
        feature.
    observations : str
        Column name containing the values of counted observations for
        each feature.
    feature : str, optional
        Column name of the feature variable for the plot,
        by default ''.
    method : Literal['sum', 'mean', 'median'], optional
        A pandas Series method to use for aggregating target values 
        within each feature level. This method is only required if there 
        is more than one value for number of observations and number of 
        events for each factor level. Otherwise it is ignored, 
        by default 'sum'.
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
    marker : str | None, optional
        The marker style for the center points. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import ProportionTest, ProcessEstimator, SpecLimits, Bar

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 100 + ['second'] * 100 + ['third'] * 100,
        y = (list(np.random.normal(loc=3, scale=1, size=100))
            + list(np.random.normal(loc=4, scale=1, size=100))
            + list(np.random.normal(loc=2, scale=1, size=100)))))
    spec_limits = SpecLimits(upper=4.3)

    # Create data that records how many are out of specification
    data = pd.DataFrame()
    for name, group in df.groupby('x'):
        y = ProcessEstimator(group['y'], spec_limits=spec_limits)
        temp = pd.DataFrame(dict(
            x = [name],
            proportion = y.n_nok / y.n_samples,
            events = y.n_nok,
            observations = y.n_samples))
        data = pd.concat([data, temp], ignore_index=True, axis=0)
    
    # Now plot it in combination with a bar chart
    bar = Bar(source=data, target='proportion', feature='x', ax=ax)
    bar()
    test = ProportionTest(
        source=data, target='proportion', feature='x', events='events', 
        observations='observations', show_center=False, n_groups=data.x.nunique(), 
        confidence_level=0.95, bars_same_color=True,)
    test()
    test.label_feature_ticks()
    test.target_as_percentage()
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 100 + ['second'] * 100 + ['third'] * 100,
        y = (list(np.random.normal(loc=3, scale=1, size=100))
            + list(np.random.normal(loc=4, scale=1, size=100))
            + list(np.random.normal(loc=2, scale=1, size=100)))))
    spec_limits = dsp.SpecLimits(upper=4.3)

    # Create data that records how many are out of specification
    data = pd.DataFrame()
    for name, group in df.groupby('x'):
        y = ProcessEstimator(group['y'], spec_limits=spec_limits)
        temp = pd.DataFrame(dict(
            x = [name],
            proportion = y.n_nok / y.n_samples,
            events = y.n_nok,
            observations = y.n_samples))
        data = pd.concat([data, temp], ignore_index=True, axis=0)

    # Now plot it in combination with a bar chart
    chart = dsp.SingleChart(
            source=data,
            target='proportion',
            feature='x',
            categorical_feature=True, # neded to label the feature tick labels
        ).plot(
            dsp.ProportionTest,
            events='events', 
            observations='observations',
            show_center=False,
            n_groups=data.x.nunique(), 
            confidence_level=0.95,
            bars_same_color=True,
        ).plot(
            dsp.Bar,
        ).label() # neded to label the feature tick labels
    ```
    
    Notes
    -----
    This class is a bit of a hack, as it creates its own target variable 
    using the ratio of events / observations. This allows it to 
    visualize proportions directly, even if a target column is not 
    explicitly provided.

    A recommended and robust approach is to precompute the proportion 
    yourself and pass it as the target variable. This gives you full 
    control over how the proportion is calculated and ensures 
    compatibility with other plotters or axes.

    See the Examples section for a demonstration of this approach, where 
    the proportion is computed manually and passed to both the Bar and 
    ProportionTest plotters. This method is especially useful when 
    combining multiple visualizations or when working with 
    pre-aggregated data.
    """

    __slots__ = ('method')

    method: Literal['sum', 'mean', 'median']
    """The provided Pandas Series method for aggregating events and
    observations (if there are multiple) per feature level."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            events: str,
            observations: str,
            n_groups: int = 1,
            feature: str = '',
            method: Literal['sum', 'mean', 'median'] = 'sum',
            show_center: bool = True,
            bars_same_color: bool = False,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        target = target if target else f'{events}/{observations}'
        data = source[[c for c in (events, observations, feature) if c]].copy()
        data[target] = list(zip(data[events], data[observations]))
        self.method = method
        if not feature:
            feature = PLOTTER.TRANSFORMED_FEATURE
            data[feature] = DEFAULT.FEATURE_BASE
        super().__init__(
            source=data,
            target=target,
            n_groups=n_groups,
            feature=feature,
            show_center=show_center,
            bars_same_color=bars_same_color,
            target_on_y=target_on_y,
            confidence_level=confidence_level,
            ci_func=proportion_ci,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
    
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
        events, observations = tuple(zip(*target_data))
        if len(target_data) > 1:
            events = getattr(pd.Series(events), self.method)()
            observations = getattr(pd.Series(observations), self.method)()
        else:
            events = events[0]
            observations = observations[0]
        center, lower, upper = self.ci_func(
            events, observations, self.confidence_level, self.n_groups)
        data = pd.DataFrame({
            self.target: [center],
            self.feature: [feature_data],
            self.lower: lower,
            self.upper: upper})
        return data


class CapabilityConfidenceInterval(ConfidenceInterval):
    """Class for creating plotters with error bars as confidence 
    interval for the process capability values Cp or Cpk.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        Column name of the target variable for the plot.
    spec_limits : SpecLimits
        Specification limits for the target variable. This can be
        created using the `SpecLimits` class.
    kind : {'cp', 'cpk'}, optional
        The capability index to be calculated. Cp can be used to compare 
        the process variability to a specification width, while Cpk 
        also considers the process mean. The Cp can only be calculated 
        if both specification limits are given.
    n_groups : int, optional
        Number of groups (variable combinations) for the Bonferroni 
        adjustment. A good way to do this is to pass 
        `df.groupby(list_of_variates).ngroups`, where `list_of_variates` 
        is a list containing all the categorical columns in the source 
        that will be used for the chart to split the data into groups 
        (hue, categorical features, etc.). Specify 1 to not do a 
        Bonferroni adjustment. Default is 1
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
    skip_na : Literal['none', 'all', 'any'], optional
        Flag indicating whether to skip missing values in the feature 
        grouped data, by default None
        - None, no missing values are skipped
        - all', grouped data is skipped if all values are missing
        - any', grouped data is skipped if any value is missing

    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on
        the y-axis, by default True.
    confidence_level : float, optional
        Confidence level for the confidence intervals,
        by default 0.95.
    color : str | None, optional
        Color to be used to draw the artists. If None, the first
        color is taken from the color cycle, by default None.
    marker : str | None, optional
        The marker style for the center points. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : matplotlib.axes.Axes | None, optional
        The axes object for the plot. If None, the current axes is 
        fetched using `plt.gca()`. If no axes are available, a new one 
        is created. Defaults to None.
    visible_spines : Literal['target', 'feature', 'none'] | None, optional
        Specifies which spines are visible, the others are hidden.
        If 'none', no spines are visible. If None, the spines are drawn
        according to the stylesheet. Defaults to None.
    hide_axis : Literal['target', 'feature', 'both'] | None, optional
        Specifies which axes should be hidden. If None, both axes 
        are displayed. Defaults to None.
    kw_estim : Dict[str, Any]
        Additional keyword arguments that are passed to the 
        `ProcessEstimator` class. Possible keword arguments are:
        - error_values: Tuple[float, ...] = (),
        - strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
        - agreement: float | int = 6,
        - possible_dists: Tuple[str | rv_continuous, ...] = DIST.COMMON
    
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).
    
    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import CapabilityConfidenceInterval, SpecLimits

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(
        x = ['first'] * 100 + ['second'] * 100 + ['third'] * 100,
        y = (list(np.random.normal(loc=3, scale=1, size=100))
            + list(np.random.normal(loc=4, scale=1, size=100))
            + list(np.random.normal(loc=2, scale=1, size=100)))))

    test = CapabilityConfidenceInterval(
        source=df, target='y', feature='x', spec_limits=SpecLimits(upper=4.3), 
        kind='cpk', show_center=True, n_groups=df.x.nunique(), 
        confidence_level=0.95, bars_same_color=True, ax=ax)
    test(kw_center=dict(s=30, marker='_'))
    test.label_feature_ticks()

    #If you are interested in the calculated values, you can get them like this:
    print(test.source)
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(
        x = ['first'] * 100 + ['second'] * 100 + ['third'] * 100,
        y = (list(np.random.normal(loc=3, scale=1, size=100))
            + list(np.random.normal(loc=4, scale=1, size=100))
            + list(np.random.normal(loc=2, scale=1, size=100)))))

    chart = dsp.SingleChart(
            source=df,
            target='y',
            feature='x',
            categorical_feature=True, # neded to label the feature tick labels
        ).plot(
            dsp.CapabilityConfidenceInterval,
            kind='cpk',
            spec_limits=dsp.SpecLimits(upper=4.3),
            show_center=True, 
            confidence_level=0.95,
            n_groups=df.x.nunique(),
            bars_same_color=True,
        ).label() # neded to label the feature tick labels

    #If you are interested in the calculated values, you can get them like this:
    df_cpk = chart.plots[0].source.copy()
    df_cpk.index = chart.dodging.pos_to_ticklabels(df_cpk['x'])
    print(df_cpk)
    ```
    """

    __slots__ = (
        'spec_limits', 'kind', 'show_feature_axis', 'processes', 'kw_estim')

    spec_limits: SpecLimits
    """Spec limits used for calculating the capability values."""
    kind: Literal['cp', 'cpk']
    """whether to calculate the confidence interval for Cp or Cpk 
    ('cp' or 'cpk')."""
    processes: List[ProcessEstimator]
    """ProcessEstimator classes used to calculate the cp and cpk values."""
    kw_estim: Dict[str, Any]
    """Additional keyword arguments that are passed to the 
    `ProcessEstimator` classes."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            spec_limits: SpecLimits,
            kind: Literal['cp', 'cpk'],
            n_groups: int = 1,
            feature: str = '',
            show_center: bool = True,
            bars_same_color: bool = False,
            skip_na: Literal['all', 'any'] | None = None,
            target_on_y: bool = True,
            confidence_level: float = 0.95,
            show_feature_axis: bool | None = None,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            kw_estim: Dict[str, Any] = {},
            **kwds) -> None:

        self.processes = []
        self.spec_limits = spec_limits
        self.kind = kind
        if show_feature_axis is None:
            show_feature_axis = bool(feature)
        self.show_feature_axis = show_feature_axis
        self.kw_estim = kw_estim
        
        super().__init__(
            source=source,
            target=target,
            n_groups=n_groups,
            feature=feature,
            show_center=show_center,
            bars_same_color=bars_same_color,
            skip_na=skip_na,
            target_on_y=target_on_y,
            confidence_level=confidence_level,
            ci_func=estimate_capability_confidence,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
    
    def hide_feature_axis(self) -> None:
        """Hide the density axis (spine, ticks and labels)."""
        axis = 'xaxis' if self.target_on_y else 'yaxis'
        spine = 'bottom' if self.target_on_y else 'left'
        getattr(self.ax, axis).set_visible(False)
        self.ax.spines[spine].set_visible(False)
    
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
        process = ProcessEstimator(
            samples=target_data,
            spec_limits=self.spec_limits,
            **self.kw_estim)
        center, lower, upper = self.ci_func(
            process=process,
            kind=self.kind,
            level=self.confidence_level,
            n_groups=self.n_groups)
        data = pd.DataFrame({
            self.target: [center],
            self.feature: [feature_data],
            self.lower: [lower],
            self.upper: [upper]})
        self.processes.append(process)
        return data 
    
    def __call__(self, kw_center: dict = {}, **kwds) -> None:
        """Perform the plotting operation.

        Parameters
        ----------
        kw_center : dict, optional
            Additional keyword arguments for the axes `scatter` method,
            by default {}.
        **kwds :
            Additional keyword arguments for the axes `errorbar` method.
        """
        super().__call__(kw_center, **kwds)
        if not self.show_feature_axis:
            self.hide_feature_axis()


class HideSubplot(Plotter):
    """Class for hiding all visual components of the x- and y-axis.
    
    Parameters
    ----------
    ax : Axes | None, optional
        The axes object (subplot) in the figure to hide.
    **kwds:
        Additional keyword arguments that have no effect and are
        only used to catch further arguments that have no use here
        (occurs when this class is used within chart objects).

    """

    def __init__(
            self,
            ax: Axes,
            **kwds) -> None:
        assert isinstance(ax, Axes)
        self.ax = ax
    
    @property
    def kw_default(self) -> Dict[str, Any]:
        return {}

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Hide all visual components of the x- and y-axis."""
        self.ax.set_axis_off()


class SkipSubplot(Plotter):
    """Class for skip plotting at current axes in a JointChart.
    """

    def __init__(self, *args, **kwds) -> None:
        """Initialize the class and store nothing."""
        pass

    @property
    def kw_default(self) -> Dict[str, Any]:
        """Get empty dict (read-only)."""
        return {}

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Do Nothing"""
        pass


class Stripe(ABC):
    """Abstract base class for drawing a stripe (line or area) on 
    Matplotlib Axes.

    A stripe is a visual element used to highlight a specific region or 
    value on a plot, such as a threshold line or a band of interest. 
    This class provides a flexible interface for defining the appearance 
    and behavior of such elements.

    Parameters
    ----------
    label : str
        The label of the stripe as it appears in the legend.
    position : float
        The central position of the stripe on the x- or y-axis.
    width : float
        The width of the stripe.
    orientation : {'horizontal', 'vertical'}, optional
        The orientation of the stripe. Defaults to 'horizontal'.
    color : str, optional
        The color of the stripe, specified as a named color or hex code.
        Defaults to `COLOR.STRIPE`.
    alpha : float or None, optional
        The transparency level of the stripe, between 0 (fully 
        transparent) and 1 (fully opaque). Defaults to None.
    lower_limit : float, optional
        The lower bound of the stripe relative to the plotting area, 
        expressed as a proportion between 0 and 1. Defaults to 0.0.
    upper_limit : float, optional
        The upper bound of the stripe relative to the plotting area, 
        expressed as a proportion between 0 and 1. Defaults to 1.0.
    zorder : float, optional
        The drawing order of the stripe relative to other plot elements.
        Higher values are drawn on top. Defaults to 0.7.
    show_position : bool, optional
        Whether to include the position value in the legend label.
        Defaults to False.
    decimals : int or None, optional
        Number of decimal places to use when formatting the position 
        value in the label. If None, the number is determined 
        automatically based on the magnitude of the position. 
        Defaults to None.

    Notes
    -----
    This is an abstract base class. Subclasses must implement the 
    `__call__` method to draw the stripe on a given Axes, and the 
    `handle` property to return the legend handle.

    The `label` property automatically wraps the label in dollar signs 
    (`$`) so that it is interpreted as a LaTeX-style math expression in 
    the legend. If `show_position` is True, the position value is 
    appended to the label.
    """

    __slots__ = (
        '_label', 'position', 'width', 'orientation', 'color', 'alpha',
        'lower_limit', 'upper_limit', 'zorder', 'show_position', '_decimals',
        '_axes')
    
    _label: str
    """The label of the stripe as it appears in the legend."""
    position: float
    """The position of the stripe on the x- or y-axis."""
    width: float
    """The width of the stripe."""
    orientation: Literal['horizontal', 'vertical']
    """The orientation of the stripe."""
    color: str
    """The color of the stripe as string or hex value."""
    alpha: float | None
    """Value used for blending the color."""
    lower_limit: float
    """The lower limit (start) of the stripe relative to the plotting
    area. Should be between 0 and 1."""
    upper_limit: float
    """The upper limit (end) of the stripe relative to the plotting
    area. Should be between 0 and 1."""
    linestyle: LineStyle
    """The linestyle of the stripe."""
    zorder: float
    """The zorder of the stripe."""
    show_position: bool
    """Whether position value of the stripe should be displayed the 
    label."""
    _decimals: int
    """The number of decimals for the position value showed in the 
    label. Only has an effect if show_position is True"""
    _axes: List[Axes]
    """The axes objects the stripe is plotted on. This attribute is 
    necessary so that each strip can only be drawn once on each axis."""

    def __init__(
            self,
            label: str,
            position: float,
            width: float,
            orientation: Literal['horizontal', 'vertical'] = 'horizontal',
            color: str = COLOR.STRIPE,
            alpha: float | None = None,
            lower_limit: float = 0.0,
            upper_limit: float = 1.0,
            zorder: float = 0.7,
            show_position: bool = False,
            decimals: int | None = None
            ) -> None:
        self.show_position = show_position
        self.position = position
        self.decimals= decimals
        self.label = label
        self.orientation = orientation
        self.width = width
        self.color = color
        self.alpha = alpha
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.zorder = zorder
        self._axes = []
    
    @property
    def decimals(self) -> int:
        """Get number of decimals used to format position value in
        label. If None is given for setting decimals, it is determined 
        based on the size of the position value see `determine_decimals`
        method."""
        return self._decimals
    @decimals.setter
    def decimals(self, decimals: int | None) -> None:
        if decimals is None:
            decimals = self.determine_decimals(self.position)
        self._decimals = decimals
    
    @staticmethod
    def determine_decimals(value: int | float) -> int:
        """Determine the number of decimal places to format values for 
        e.g. legend labels. The number of decimal places depends on the
        size of the provided value."""
        if value <= 0.5:
            return 4
        elif value <= 5:
            return 3
        elif value <= 50:
            return 2
        elif value <= 5000:
            return 1 
        else:
            return 0
    
    @property
    def label(self) -> str:
        """Get the label of the stripe. The label is always returned
        with a leading and trailing $ sign so that it is interpreted as
        a mathematical expression in the legend."""
        label = self._label
        if self.show_position:
            label = f'{label}={self.position:.{self.decimals}f}'
        return f'${label}$'
    @label.setter
    def label(self, label: str) -> None:
        self._label = label.strip('$')

    @property
    def identity(self) -> str:
        """Get the identity of the strip. Needed so that it only appears
        once in the legend. The identity is composed of the label and
        the color (read-only)."""
        return f'{self.label}_{self.color}'

    @property
    @abstractmethod
    def handle(self) -> Patch | Line2D:
        """Get the handle of the stripe used for legend."""
        raise NotImplementedError
    
    @abstractmethod
    def __call__(self, ax: Axes) -> Any:
        """Draw the stripe on the given Axes object."""
        raise NotImplementedError


class StripeLine(Stripe):
    """
    Concrete implementation of `Stripe` for drawing straight lines on Matplotlib Axes.

    This class is used to draw horizontal or vertical lines that highlight specific 
    positions on a plot, such as thresholds, limits, or reference values. It supports 
    customization of line style, color, transparency, and legend labeling.

    Parameters
    ----------
    label : str
        The label of the stripe as it appears in the legend.
    position : float
        The position of the stripe on the x- or y-axis.
    width : float, optional
        The line width of the stripe. Can be overridden by `lw` or `linewidth` in `**kwds`.
        Defaults to `LINE.WIDTH`.
    orientation : {'horizontal', 'vertical'}, optional
        The orientation of the stripe. Defaults to 'horizontal'.
    color : str, optional
        The color of the stripe, specified as a named color or hex code.
        Can be overridden by `c` in `**kwds`. Defaults to `COLOR.STRIPE`.
    alpha : float or None, optional
        Transparency level of the stripe, between 0 (transparent) and 1 (opaque).
        Defaults to None.
    lower_limit : float, optional
        The lower bound of the stripe relative to the plotting area (0 to 1).
        Defaults to 0.0.
    upper_limit : float, optional
        The upper bound of the stripe relative to the plotting area (0 to 1).
        Defaults to 1.0.
    zorder : float, optional
        Drawing order of the stripe. Higher values are drawn above lower ones.
        Defaults to 0.7.
    show_position : bool, optional
        Whether to include the position value in the legend label. Defaults to False.
    decimals : int or None, optional
        Number of decimal places to use when formatting the position in the label.
        If None, the number is determined automatically based on the value.
    linestyle : LineStyle, optional
        The line style of the stripe (e.g., solid, dashed). Can be overridden by `ls` in `**kwds`.
        Defaults to `LINE.DASHED`.
    **kwds :
        Additional keyword arguments for fine-tuning appearance. Priority order:
        - Line width: `lw`, `linewidth`, then `width`
        - Color: `c`, then `color`
        - Style: `ls`, then `linestyle`

    Examples
    --------
    Apply to an existing Axes object:

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from daspi import GaussianKDE, StripeLine, Estimator

    fig, ax = plt.subplots()
    df = pd.DataFrame(dict(x = np.random.weibull(a=1.5, size=1000)))
    kde = GaussianKDE(source=df, target='x', target_on_y=False)
    kde()
    x = Estimator(df.x)
    mean = StripeLine(
        label=r'\bar x = 'f'{x.mean:.2f}',
        position=x.mean,
        color="#145a5aa0",
        upper_limit=0.9,
        orientation='vertical')
    mean(ax=ax)
    median = StripeLine(
        label=r'\tilde x = 'f'{x.median:.2f}',
        position=x.median,
        color="#5a47149f",
        upper_limit=0.9,
        orientation='vertical')
    median(ax=ax)
    ax.set(ylim=(0, None))
    ax.legend([mean.handle, median.handle], [mean.label, median.label])
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(x = np.random.weibull(a=1.5, size=1000)))
    x = Estimator(df.x)
    mean = StripeLine(
        label=r'\bar x = 'f'{x.mean:.2f}',
        position=x.mean,
        color="#145a5aa0",
        upper_limit=0.9,
        orientation='vertical')
    median = StripeLine(
        label=r'\tilde x = 'f'{x.median:.2f}',
        position=x.median,
        color="#5a47149f",
        upper_limit=0.9,
        orientation='vertical')
    chart = dsp.SingleChart(
            source=df,
            target='x',
            target_on_y=False
        ).plot(dsp.GaussianKDE,
        ).stripes([mean, median]
        ).label() # neded to add the legend
    ```

    The last example only serves to demonstrate how custom lines can be 
    added. The median and mean are already predefined and can be easily 
    plotted by setting the appropriate flag. With the following example, 
    we get a similar result:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(x = np.random.weibull(a=1.5, size=1000)))
    chart = dsp.SingleChart(
            source=df,
            target='x',
            target_on_y=False
        ).plot(dsp.GaussianKDE,
        ).stripes(
            mean=True,
            median=True,
        ).label() # neded to add the legend
    ```

    Notes
    -----
    This class is part of the DaSPi visualization toolkit and is 
    designed to integrate seamlessly with other chart components. It 
    ensures that each stripe is only drawn once per Axes instance.
    """

    __slots__ = ('linestyle')

    linestyle: LineStyle
    """The linestyle of the stripe."""

    def __init__(
            self,
            label: str,
            position: float,
            width: float = LINE.WIDTH,
            orientation: Literal['horizontal', 'vertical'] = 'horizontal',
            color: str = COLOR.STRIPE,
            alpha: float | None = None,
            lower_limit: float = 0.0,
            upper_limit: float = 1.0,
            zorder: float = 0.7,
            show_position: bool = False,
            decimals: int | None = None,
            linestyle: LineStyle = LINE.DASHED,
            **kwds
            ) -> None:
        
        super().__init__(
            label=label,
            orientation=orientation,
            position=position,
            width=kwds.get('lw', kwds.get('linewidth', width)),
            color=kwds.get('c', color),
            alpha=alpha,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            zorder=zorder,
            show_position=show_position,
            decimals=decimals)
        self.linestyle = kwds.get('ls', linestyle)
    
    @property
    def handle(self) -> Line2D:
        """Get the handle of the stripe used for legend."""
        handle = Line2D(
            xdata=[],
            ydata=[],
            linewidth=self.width,
            linestyle=self.linestyle,
            color=self.color,
            markersize=0,
            alpha=self.alpha)
        return handle

    def __call__(self, ax: Axes) -> Any:
        """Draw the stripe on the given Axes object."""
        if ax in self._axes:
            return
        args = (
            self.position,
            self.lower_limit,
            self.upper_limit)
        kwds = dict(
            color=self.color,
            alpha=self.alpha,
            linewidth=self.width,
            linestyle=self.linestyle,
            zorder=self.zorder
            )
        if self.orientation == 'horizontal':
            ax.axhline(*args, **kwds)
        else:
            ax.axvline(*args, **kwds)
        self._axes.append(ax)


class StripeSpan(Stripe):
    """
    Concrete implementation of `Stripe` for drawing wide spans (bands) 
    on Matplotlib Axes.

    This class is used to highlight a continuous region between two 
    positions on a plot, such as confidence intervals, tolerance bands, 
    or shaded areas of interest. It supports both direct specification 
    of the lower and upper bounds, or a central position with width.

    Parameters
    ----------
    label : str
        The label of the stripe as it appears in the legend.
    lower_position : float or None, optional
        The lower bound of the stripe on the x- or y-axis. Must be 
        provided if `position` and `width` are not given.
        Defaults to None.
    upper_position : float or None, optional
        The upper bound of the stripe on the x- or y-axis. Must be 
        provided if `position` and `width` are not given.
        Defaults to None.
    position : float or None, optional
        The central position of the stripe. Must be provided if 
        `lower_position` and `upper_position` are not given.
        Defaults to None.
    width : float or None, optional
        The width of the stripe. Must be provided if `lower_position` 
        and `upper_position` are not given. Defaults to None.
    orientation : {'horizontal', 'vertical'}, optional
        The orientation of the stripe. Defaults to 'horizontal'.
    color : str, optional
        The fill color of the stripe. Can be a named color or hex code.
        Defaults to `COLOR.STRIPE`.
    alpha : float or None, optional
        Transparency level of the stripe, between 0 (transparent) and 1 
        (opaque). Defaults to `COLOR.CI_ALPHA`.
    lower_limit : float, optional
        The lower limit of the stripe relative to the plotting area 
        (0 to 1). Defaults to 0.0.
    upper_limit : float, optional
        The upper limit of the stripe relative to the plotting area 
        (0 to 1). Defaults to 1.0.
    zorder : float, optional
        Drawing order of the stripe. Higher values are drawn above lower 
        ones. Defaults to 0.7.
    border_linewidth : float, optional
        Width of the border line. Can also be set via `lw` or 
        `linewidth` in `**kwds`. Defaults to 0.
    **kwds :
        Additional keyword arguments for appearance customization. 
        Priority order:
        - Line width: `lw`, `linewidth`, then `border_linewidth`
        - Color: `c`, then `color`
        - Style: `ls`, then `linestyle`

    Examples
    --------
    Apply to an existing Axes object:

    ```python
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
    ```

    Apply using the plot method of a DaSPi Chart object:

    ```python
    import numpy as np
    import daspi as dsp
    import pandas as pd

    df = pd.DataFrame(dict(x = np.random.weibull(a=1.5, size=100)))
    spec_limits=dsp.SpecLimits(lower=0, upper=2.5)
    ok_area = StripeSpan(
        label=r'OK',
        lower_position=spec_limits.lower,
        upper_position=spec_limits.upper,
        color=COLOR.GOOD,
        orientation='horizontal')
    chart = dsp.SingleChart(
            source=df,
            target='x',
        ).plot(
            dsp.Line,
            kw_call={'marker': 'o'}
        ).stripes(
            [ok_area]
        ).label() # neded to add the legend
    ```

    Notes
    -----
    You must specify either:
    - `lower_position` and `upper_position`, or
    - `position` and `width`

    but not both. The class will compute the missing values accordingly.
    """

    __slots__ = ('lower_position', 'upper_position', 'border_linewidth')
    
    lower_position: float
    """Target position of the lower border of the stripe."""
    upper_position: float
    """Target position of the upper border of the stripe."""
    border_linewidth: float
    """Width of the border line. Could also be set during initialisation
    via `lw` or `linewidth`."""

    def __init__(
            self,
            label: str,
            lower_position: float | None = None,
            upper_position: float | None = None,
            position: float | None = None,
            width: float | None = None,
            orientation: Literal['horizontal', 'vertical'] = 'horizontal',
            color: str = COLOR.STRIPE,
            alpha: float = COLOR.CI_ALPHA,
            lower_limit: float = 0.0,
            upper_limit: float = 1.0,
            zorder: float = 0.7,
            border_linewidth: float = 0,
            **kwds) -> None:

        position, width = self._position_values_(
            lower_position, upper_position, position, width)
        
        super().__init__(
            label=label,
            orientation=orientation,
            position=position,
            width=width,
            color=color,
            alpha=alpha,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            zorder=zorder,
            show_position=False)
        self.border_linewidth = kwds.get(
            'lw', kwds.get('linewidth', border_linewidth))
    
    def _position_values_(
            self,
            lower: float | None,
            upper: float | None,
            pos: float | None, 
            width: float | None
            ) -> Tuple[float, float]:
        """Check if either the edges values or the position and width is
        given but not both. Then set the edges attributes and return
        position and width."""
        _msg = (
            'Either position and width or lower_position and upper_position '
            'must be given!')
        if lower is None and upper is None:
            assert pos is not None and width is not None, _msg
            lower = pos - abs(width/2)
            upper = pos + abs(width/2)
        else:
            assert pos is None and width is None, _msg
            assert lower is not None and upper is not None, _msg
            pos = (lower + upper) / 2
            width = abs(upper - lower)
        
        self.lower_position = lower
        self.upper_position = upper
        return pos, width

    @property
    def handle(self) -> Patch:
        """Get the handle of the stripe used for legend."""
        handle = Patch(color=self.color, alpha=self.alpha)
        return handle

    def __call__(self, ax: Axes) -> None:
        """Draw the stripe on the given Axes object."""
        if ax in self._axes:
            return
        
        args = (
            self.lower_position,
            self.upper_position,
            self.lower_limit,
            self.upper_limit)
        kwds = dict(
            color=self.color,
            alpha=self.alpha,
            linewidth=self.border_linewidth,
            zorder=self.zorder
            )
        if self.orientation == 'horizontal':
            ax.axhspan(*args, **kwds)
        else:
            ax.axvspan(*args, **kwds)
        self._axes.append(ax)


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
        Column name of the target variable.
    feature : str
        Column name indicating which is the first (reference 
        measurement) and which is the second measurement (the 
        measurement to be compared).
    identity : str
        Column name containing identities of each sample, must occur 
        once for each measurement.
    reverse : bool, optional
        Flag indicating if the order of the measurements should be 
        reversed, by default False
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
    marker : str | None, optional
        The marker style for the scatter plot. Available markers see:
        https://matplotlib.org/stable/api/markers_api.html, 
        by default None
    ax : Axes | None, optional
        The axes object for the plot. If None, an attempt is made to get
        the current one using `plt.gca`. If none is available, one is 
        created. The same applies to the Figure object. Defaults to 
        None.
    **kwds:
        Those arguments have no effect. Only serves to catch further
        arguments that have no use here (occurs when this class is 
        used within chart objects).
    
    Examples
    --------

    Apply using the plot method of a DaSPi Chart object:

    ```python
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
    ```

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
    __slots__ = ('identity', 'confidence', 'estimation', 'stripes',
        'lines_same_color')

    identity: str
    """Column name containing identities of each sample.""" 
    confidence: float
    """Confidence level of the confidence interval for mean and
    agreements."""
    estimation: Estimator
    """Estimator instance to estimate the mean and limits of agreement."""
    stripes: Dict[str, Stripe]
    """Dictionary of Stripe objects used for drawing lines and their
    confidence intervals."""
    lines_same_color: bool
    """Whether to use same color for lines and their confidence 
    intervals as for the points."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str,
            identity: str,
            reverse: bool = False,
            agreement: float = 3.92,
            confidence: float = 0.95,
            feature_axis: Literal['mean', 'data'] = 'mean',
            lines_same_color: bool = False,
            target_on_y: bool = True,
            color: str | None = None,
            marker: str | None = None,
            ax: Axes | None = None,
            visible_spines: Literal['target', 'feature', 'none'] | None = None,
            hide_axis: Literal['target', 'feature', 'both'] | None = None,
            **kwds) -> None:
        self.identity = identity
        
        first, second = sorted(list(source[feature].unique()), reverse=reverse)
        target1 = f'{target}-{first}'
        target2 = f'{target}-{second}'
        _target = f'{target2} - {target1}'
        df = pd.DataFrame()
        for name in (first, second):
            data = (source[source[feature] == name]
                [[self.identity, target]]
                .set_index(self.identity)
                .rename(columns={target: f'{target}-{name}'})
                .copy())
            assert len(data) == data.index.nunique(), (
                f'Duplicated measurements for {name}')
            df = pd.concat([df, data], axis=1)
        df = df.dropna(how='any', axis=0)
        df[_target] = df[target2] - df[target1]
        
        if feature_axis == 'mean':
            _feature = feature_axis
            df[_feature] = df[[target1, target2]].mean(axis=1)
        else:
            _feature = target1
        super().__init__(
            source=df,
            target=_target,
            feature=_feature,
            target_on_y=target_on_y,
            color=color,
            marker=marker,
            ax=ax,
            visible_spines=visible_spines,
            hide_axis=hide_axis)
        self.stripes = {}
        self.lines_same_color = lines_same_color
        self.confidence = confidence
        self.estimation = Estimator(
            samples=df[_target], strategy='norm', agreement=agreement)

    @property
    def kw_default(self) -> Dict[str, Any]:
        """Default keyword arguments for plotting (read-only)"""
        kwds = dict(color=self.color, marker=self.marker)
        return kwds
    
    def __call__(self, **kwds) -> None:
        """Perform the Bland-Altman plot operation.

        Parameters
        ----------
        **kwds : 
            Additional keyword arguments to be passed to the Axes
            `scatter` method.
        """
        _kwds = self.kw_default | kwds
        self.ax.scatter(self.x, self.y, **_kwds)

        orientation = 'horizontal' if self.target_on_y else 'vertical'
        kw_stripe: Dict[str, Any] = dict(
            orientation=orientation,
            show_position=True)
        if self.lines_same_color:
            kw_stripe['color'] = self.color
        span_label = f'{100*self.confidence:.0f} \\%-{STR["ci"]}'
        mean_ci = self.estimation.mean_ci(self.confidence)
        stdev_ci = self.estimation.stdev_ci(self.confidence)
        stdev_width = stdev_ci[1] - stdev_ci[0]
        stripes = (
            StripeLine(
                label=r'\bar x',
                position=self.estimation.mean,
                **(KW.MEAN_LINE | kw_stripe)),
            StripeSpan(
                label=span_label,
                lower_position=mean_ci[0],
                upper_position=mean_ci[1],
                **(KW.STRIPES_CONFIDENCE | kw_stripe)),
            StripeLine(
                label=STR['lcl'],
                position=self.estimation.lcl,
                **(KW.CONTROL_LINE | kw_stripe)),
            StripeSpan(
                label=span_label,
                position=self.estimation.lcl,
                width=stdev_width,
                **(KW.STRIPES_CONFIDENCE | kw_stripe)),
            StripeLine(
                label=STR['lcl'],
                position=self.estimation.ucl,
                **(KW.CONTROL_LINE | kw_stripe)),
            StripeSpan(
                label=span_label,
                position=self.estimation.ucl,
                width=stdev_width,
                **(KW.STRIPES_CONFIDENCE | kw_stripe)))
        for stripe in stripes:
            stripe(self.ax)
            self.stripes[stripe.identity] = stripe


__all__ = [
    'Plotter',
    'Scatter',
    'Line',
    'Stem',
    'LinearRegressionLine',
    'LoessLine',
    'Probability',
    'ParallelCoordinate',
    'TransformPlotter',
    'CenterLocation',
    'Bar',
    'Pareto',
    'Jitter',
    'Beeswarm',
    'QuantileBoxes',
    'GaussianKDE',
    'GaussianKDEContour',
    'Violine',
    'Errorbar',
    'StandardErrorMean',
    'SpreadWidth',
    'ConfidenceInterval',
    'MeanTest',
    'VariationTest',
    'ProportionTest',
    'CapabilityConfidenceInterval',
    'HideSubplot',
    'SkipSubplot',
    'Stripe',
    'StripeLine',
    'StripeSpan',
    'BlandAltman',
    ]
