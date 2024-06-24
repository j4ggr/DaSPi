"""
Module for creating various types of chart visualizations using 
Matplotlib and Pandas.

This module provides classes and utility functions to facilitate the 
creation of different types of charts and visualizations. It includes 
support for single-variable charts, joint charts combining multiple 
variables, and charts with multiple variables simultaneously.

All plotter objects from the plotter module can be used and combined 
with the chart classes from this module to create a preferred plot that
is optimal for the current analysis. These Chart objects are also used
to combine the LabelFacets, AxesFacets, and StripesFacets facet objects
to produce consistent charts for very different plots.

## Classes

- *Chart:* Abstract base class for creating chart visualizations.
- *SingleChart:* Represents a basic chart containing one Axes for 
  visualization with customizable features.
- *JointChart:* Represents a joint chart visualization combining
  multiple individual Axes.
- *MultipleVariateChart:* Represents a chart visualization handling 
  multiple variables simultaneously.

## Functionality

- Customization of chart attributes including target, feature, hue, 
  shape, size, etc.
- Layout setup for charts, including grid arrangements for joint charts.
- Adding stripes to highlight data patterns and labeling axes 
  appropriately.
- Saving charts to files and programmatically closing charts.

## Other Details

- Dependencies: NumPy, Matplotlib, Pandas.
- Typing annotations are extensively used for type hinting and 
documentation.
- Emphasizes modularity and extensibility, allowing users to create and 
customize a wide range of chart visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import Type
from typing import Self
from typing import Dict
from typing import List
from typing import Tuple
from typing import Literal
from typing import Generator
from pathlib import Path
from numpy.typing import NDArray
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from .classify import Dodger
from .classify import HueLabel
from .classify import SizeLabel
from .classify import ShapeLabel
from .facets import AxesFacets
from .facets import LabelFacets
from .facets import StripesFacets
from .plotter import Plotter
from matplotlib.axes import Axes
from matplotlib.axis import XAxis
from matplotlib.axis import YAxis
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from ..strings import STR
from .._typing import SpecLimit
from .._typing import SpecLimits
from .._typing import LegendHandlesLabels

from ..constants import KW
from ..constants import COLOR
from ..constants import PLOTTER


class Chart(ABC):
    """
    Abstract base class for creating chart visualizations.

    Parameters
    ----------
    source : pandas DataFrame
        A pandas DataFrame containing the data in long-format.
    target : str or Tuple[str]
        Column name for the target variable to be visualized.
    feature : str or Tuple[str]
        Column name for the feature variable to be visualized.
    target_on_y : bool
        If True, the target variable is plotted on the y-axis.
    axes_facets : AxesFacets
        An instance containing the subplots' Axes and their arrangement.
    **kwds
        Additional key word arguments to instantiate the `AxesFacets`
        object. Only taken into account if `axes_facets` is not
        provided.
    """
    __slots__ = (
        'source', 'target', 'feature', 'target_on_y', 'axes_facets',
        'label_facets', 'stripes_facets', 'nrows', 'ncols', '_data', '_xlabel',
        '_ylabel', '_plots')
    
    source: DataFrame
    """Pandas DataFrame containing the source data in long-format."""
    target: str
    """Column name for the target variable to be visualized."""
    feature: str
    """Column name for the feature variable to be visualized."""
    target_on_y: bool
    """Flag indicating whether the target variable is plotted on the 
    y-axis."""
    stripes_facets: StripesFacets
    """StripesFacets instance for creating location and spread width
    lines, specification limits and/or confidence interval areas as
    stripes on each Axes."""
    axes_facets: AxesFacets
    """AcesFacets instance for creating a grid of subplots with
    customizable sharing and sizing options."""
    label_facets: LabelFacets
    """LabelFacets instance for adding labels and titles to facets of a
    figure."""
    nrows: int
    """Number of rows of subplots in the grid."""
    ncols: int
    """Number of columns of subplots in the grid."""
    _data: DataFrame
    """Current source data subset used for current Axes."""
    _plots: List[Plotter]
    """All plotter objects used in `plot` method."""

    def __init__(
            self, source: DataFrame, target: str, feature: str = '',
            target_on_y: bool = True, axes_facets: AxesFacets | None = None, 
            **kwds) -> None:
        self.source = source
        self.target = target
        self.feature = feature
        self.nrows = kwds.pop('nrows', 1)
        self.ncols = kwds.pop('ncols', 1)
        if axes_facets is None:
            self.axes_facets = AxesFacets(self.nrows, self.ncols, **kwds)
        else:
            self.axes_facets = axes_facets
        self.target_on_y = target_on_y
        for ax in self.axes.flat:
            getattr(ax, f'set_{"x" if self.target_on_y else "y"}margin')(0)
        self._plots = []

    @property
    def figure(self) -> Figure:
        """Get the top level container for all the plot elements
        (read-only)."""
        return self.axes_facets.figure
    
    @property
    def axes(self) -> NDArray:
        """Get the created axes"""
        return self.axes_facets.axes
    
    @property
    def n_axes(self) -> int:
        """Get amount of axes"""
        return self.ncols * self.nrows
    
    @property
    def plots(self) -> List[Plotter]:
        """Get plotter objects used in `plot` method"""
        return self._plots
    
    @abstractmethod
    def _axis_label_(
            self, label: Any, is_target: bool) -> str | Tuple[str, ...]:
        """Helper method to get the axis label based on the provided 
        label and is_target flag.
        """
        
    @abstractmethod
    def axis_labels(
            self, feature_label: Any, target_label: Any
            ) -> Tuple[str | Tuple[str, ...], str | Tuple[str, ...]]:
        """Get the x and y axis labels based on the provided 
        `feature_label` and `target_label`.
        """
        
    @abstractmethod
    def _data_genenator_(self) -> Generator[DataFrame, Self, None]:
        """Implement the data generator and add the currently yielded 
        data to self._data so that it can be used internally.
        
        Returns
        -------
        Generator[DataFrame, Self, None]
            A generator object yielding DataFrames as subsets of the 
            source data, used as plotting data for each Axes.
        """
    
    def __iter__(self) -> Generator[DataFrame, Self, None]:
        """Iterate over the Chart object.

        Returns
        -------
        Generator[DataFrame, Self, None]
            A generator object yielding DataFrames as subsets of the 
            source data, used as plotting data for each Axes.
        """
        return self._data_genenator_()
        
    def __next__(self) -> DataFrame:
        """Get the next subset of the source data.

        Returns
        -------
        DataFrame
            The next subset of the source data.
        """
        return next(self)
    
    @abstractmethod
    def plot(
            self, plotter: Type[Plotter], kw_call: Dict[str, Any] = {}, **kwds
            ) -> Self:
        """Plot the chart using the specified plotter.

        Parameters
        ----------
        plotter : Type[Plotter]
            The plotter object.
        kw_call : Dict[str, Any]
            Additional keyword arguments for the plotter call method.
        **kwds:
            Additional keyword arguments for the plotter object.

        Returns
        -------
        Self
            The updated Chart object.
        """

    @abstractmethod
    def stripes(self, **kwds) -> Self:
        """Add stripes to the chart.

        Parameters
        ----------
        kwds : 
            Additional keyword arguments.

        Returns
        -------
        Self
            The updated Chart object.
        """

    @abstractmethod
    def label(
        self, fig_title: str = '', sub_title: str = '',
        feature_label: bool | str = '', target_label: bool | str = '',
        info: bool | str = False) -> Self:
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

        Returns
        -------
        Self
            The instance with updated labels and titles.

        Notes
        -----
        This method allows customization of chart labels and titles to
        enhance readability and provide context for the visualized data.
        """
    
    def save(self, file_name: str | Path, **kwds) -> Self:
        kw = KW.SAVE_CHART | kwds
        self.figure.savefig(file_name, **kw)
        return self

    def close(self) -> Self:
        """"Close figure"""
        plt.close(self.figure)
        return self


class SingleChart(Chart):
    """Represents a basic chart visualization with customizable
    features.

    This class provides a foundation for creating customizable chart
    visualizations. Customize the appearance and behavior of the chart
    using various parameters and keyword arguments.

    Parameters
    ----------
    source : pandas DataFrame
        Pandas long format DataFrame containing the data source for the
        plot.
    target : str
        The target variable (column) to visualize.
    feature : str, optional
        The feature variable (column) to use for the visualization,
        by default ''
    hue : str, optional
        The hue variable (column) for color differentiation,
        by default ''
    dodge : bool, optional
        Flag indicating whether to move overlapping categorical features
        with different colors along the axis so that they appear
        separately. Should only be set to True if given feature is
        categorical, by default False
    shape : str, optional
        The categorical shape variable (column) for marker
        differentiation, by default ''.
    size : str, optional
        The numeric variable (column) from which the marker size is
        derived. The range (minimum to maximum) of the variable is
        mapped to the marker size, by default ''
    categorical_feature : bool
        Flag indicating whether the features are categorical. If True,
        the feature values are transferred to the feature axis and the
        major grid is removed. However, a minor grid is created so that
        the categories are visually better separated. This attribute is
        set to True if `dodge` is set to True, by default False.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on the
        y-axis, by default True
    **kwds
        Additional key word arguments to instantiate the `AxesFacets`
        object.
    """

    __slots__ = (
        'hue', 'shape', 'size', 'sizing', 'shaping', 'hueing', 'dodging',
        'categorical_feature', '_variate_names', '_current_variate',
        '_last_variate', )
    
    hue: str
    """The hue variable (column) for color differentiation."""
    shape: str
    """The categorical shape variable (column) for marker differentiation."""
    size: str
    """The numeric variable (column) from which the marker size is derived."""
    sizing: SizeLabel
    """A label handler for marker sizes based on the given size
    variable (column)."""
    shaping: ShapeLabel
    """A label handler for marker shapes based on the given shape
    variable (column)."""
    hueing: HueLabel
    """A label handler for color differentiation based on the given hue
    variable column."""
    dodging: Dodger
    """A handler for dodging categorical features along the axis."""
    categorical_feature: bool
    """Flag indicating whether the features are categorical."""
    _current_variate: dict
    """Dictionary to store current variate information."""
    _last_variate: dict
    """Dictionary to store last variate information."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            hue: str = '',
            dodge: bool = False,
            shape: str = '',
            size: str = '', 
            categorical_feature: bool = False,
            target_on_y: bool = True,
            **kwds) -> None:
        self.categorical_feature = categorical_feature or dodge
        if feature == '' and dodge:
            feature = PLOTTER.FEATURE
            source[feature] = ''
        self.hue = hue
        self.shape = shape
        self.size = size
        super().__init__(
            source=source, target=target, feature=feature,
            target_on_y=target_on_y, **kwds)
        self.hueing = HueLabel(self.get_categorical_labels(self.hue))
        feature_tick_labels = ()
        if self.categorical_feature:
            assert self.feature in source, (
                'categorical_feature is True, but features is not present')
            feature_tick_labels = self.get_categorical_labels(self.feature)
        dodge_categories = tuple(self.hueing.labels) if dodge else ()
        self.dodging = Dodger(dodge_categories, feature_tick_labels)
        self.shaping = ShapeLabel(self.get_categorical_labels(self.shape))
        if self.size:
            self.sizing = SizeLabel(
                self.source[self.size].min(), self.source[self.size].max())
        self._variate_names = (self.hue, self.shape)
        self._current_variate = {}
        self._last_variate = {}
        self._reset_variate_()
    
    @property
    def ax(self) -> Axes:
        """Get the axes instance (read-only)."""
        if self.axes_facets.ax is None:
            raise AttributeError(
                'Chart has no Axes.')
        return self.axes_facets.ax
    
    @property
    def variate_names(self) -> List[str]:
        """Get names of all set variates (read-only)."""
        return [v for v in self._variate_names if v]
    
    @property
    def color(self) -> str:
        """Get color for current variate (read-only)."""
        return self.hueing[self._current_variate.get(self.hue, None)]
    
    @property
    def marker(self) -> str:
        """Get marker for current variate (read-only)"""
        return self.shaping[self._current_variate.get(self.shape, None)]

    @property
    def sizes(self) -> NDArray | None:
        """Get sizes for current variate, is set in grouped data 
        generator (read-only)."""
        if self.size not in self._data:
            return None
        return self.sizing(self._data[self.size])
    
    @property
    def legend_data(self) -> Dict[str, LegendHandlesLabels]:
        """Get dictionary of handles and labels (read-only).
        - keys: titles as str
        - values: handles and labels as tuple of tuples"""
        handle_label = {}
        if self.hue:
            handle_label[self.hue] = self.hueing.handles_labels()
        if self.shape:
            handle_label[self.shape] = self.shaping.handles_labels()
        if self.size:
            handle_label[self.size] = self.sizing.handles_labels()
        if hasattr(self, 'stripes_facets'):
            handle_label[STR['stripes']] = self.stripes_facets.handles_labels()
        return handle_label
    
    def _axis_label_(
            self, label: str | bool | None, is_target: bool) -> str:
        """Helper method to get the axis label based on the provided 
        label and is_target flag.

        Parameters
        ----------
        label: Any
            The label to use for the feature or target axis.
        is_target: bool
            Flag indicating whether the label is for the target variable.

        Returns
        -------
        str
            The axis label.
        """
        match label:
            case None | False: 
                return ''
            case True: 
                return self.target if is_target else self.feature
            case _:
                return str(label)
    
    def axis_labels(
            self, feature_label: bool | str | None, 
            target_label: bool | str | None
            ) -> Tuple[str, str]:
        """Get the x and y axis labels based on the provided 
        `feature_label` and `target_label`.
            - If a string is passed, it will be taken.
            - If True, labels of given feature or target name are used.
            - If False or None, empty string is used.

        Parameters
        ----------
        feature_label: bool | str | None
            Flag or label for the feature variable.
        target_label: bool | str | None
            Flag or label for the target variable.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the x-axis label (xlabel) and y-axis 
            label (ylabel).
        """
        xlabel = self._axis_label_(feature_label, is_target=False)
        ylabel = self._axis_label_(target_label, is_target=True)
        if not self.target_on_y:
            xlabel, ylabel = ylabel, xlabel
        return xlabel, ylabel
    
    def get_categorical_labels(self, colname: str) -> Tuple[Any, ...]:
        """Get sorted unique elements of given column name if in source.

        Parameters
        ----------
        colname : str
            The name of the column in the source DataFrame.

        Returns:
        --------
        Tuple:
            Sorted unique elements of the given column name.
        """
        if not colname:
            return ()
        return tuple(sorted(np.unique(self.source[colname])))
    
    def _reset_variate_(self) -> None:
        """Set values to None for current and last variate."""
        self._current_variate = {k: None for k in self.variate_names}
        self._last_variate = {k: None for k in self.variate_names}

    def update_variate(self, combination: Any) -> None:
        """Update current variate by given combination coming from 
        pandas DataFrame groupby function.

        Parameters
        ----------
        combination : Any
            The combination of variables coming from the DataFrame 
            groupby function.
        """
        if not isinstance(combination, tuple): 
            combination = (combination, )
        self._last_variate = deepcopy(self._current_variate)
        for key, name in zip(self.variate_names, combination):
            self._current_variate[key] = name
    
    def dodge(self) -> None:
        """Converts the feature data to tick positions, taking dodging 
        into account."""
        if not self.dodging:
            return
        hue_variate = self._current_variate.get(self.hue, None)
        self._data[self.feature] = self.dodging(
            self._data[self.feature], hue_variate)
        
    def _categorical_feature_grid_(self) -> None:
        """Hide major grid and set one minor grid for feature axis."""
        xy = 'x' if self.target_on_y else 'y'
        axis: XAxis | YAxis = getattr(self.axes_facets.ax, f'{xy}axis')
        axis.set_minor_locator(AutoMinorLocator(2))
        axis.grid(True, which='minor')
        axis.grid(False, which='major')
    
    def _categorical_feature_ticks_(self) -> None:
        """Set one major tick for each category and label it.
        
        Raises
        ------
        AttributeError :
            If axes_facets has no axes"""
        xy = 'x' if self.target_on_y else 'y'
        settings = {
            f'{xy}ticks': self.dodging.ticks,
            f'{xy}ticklabels': self.dodging.tick_lables,
            f'{xy}lim': self.dodging.lim}
        self.ax.set(**settings)
        self.ax.tick_params(which='minor', color=COLOR.TRANSPARENT)
        
    def _categorical_feature_axis_(self) -> None:
        """Set one major tick for each category and label it. Hide 
        major grid and set one minor grid for feature axis."""
        self._categorical_feature_grid_()
        self._categorical_feature_ticks_()

    def _data_genenator_(self) -> Generator[DataFrame, Self, None]:
        """Generate grouped data if `variate_names` are set, otherwise 
        yield the entire source DataFrame.

        This method serves as a generator function that yields grouped 
        data based on the `variate_names` attribute if it is set. 
        If no `variate_names` are specified, it yields the entire source 
        DataFrame.

        Yields:
        -------
        self._data : DataFrame
            Containing the grouped data or the entire source DataFrame.
        """
        if self.variate_names:
            for combination, data in self.source.groupby(self.variate_names):
                self._data = data
                self.update_variate(combination)
                self.dodge()
                yield self._data
        else:
            self._data = self.source
            yield self._data
        self._reset_variate_()

    def plot(
            self, plotter: Type[Plotter], kw_call: Dict[str, Any] = {}, **kwds
            ) -> Self:
        """Plot the chart.

        Parameters
        ----------
        plotter : Type[Plotter]
            The plotter object.
        kw_call : Dict[str, Any]
            Additional keyword arguments for the plotter call method.
        **kwds:
            Additional keyword arguments for the plotter object.

        Returns:
        --------
        Self:
            The SingleChart instance.
        """
        self.target_on_y = kwds.pop('target_on_y', self.target_on_y)
        _marker = kwds.pop('marker', None)
        for data in self:
            marker = _marker if _marker is not None else self.marker
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_on_y=self.target_on_y, color=self.color, 
                ax=self.axes_facets.ax, marker=marker, size=self.sizes,
                width=self.dodging.width,
                categorical_feature=self.categorical_feature, **kwds)
            plot(**kw_call)
            self._plots.append(plot)
        return self
    
    def stripes(
            self, mean: bool = False, median: bool = False,
            control_limits: bool = False, 
            spec_limits: Tuple[SpecLimit, SpecLimit] = (None, None), 
            confidence: float | None = None, **kwds) -> Self:
        """Plot location and spread width lines, specification limits 
        and/or confidence interval areas as stripes on each Axes. The
        location and spread (and their confidence bands) represent the 
        data per axes.

        Parameters
        ----------
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
            If provided, specifies the specification limits. 
            The tuple must contain two values for the lower and upper 
            limits. If a limit is not given, use None, by default ().
        confidence : float, optional
            The confidence level between 0 and 1, by default None.
        **kwds:
            Additional keyword arguments for configuring StripesFacets.

        Returns
        -------
        SingleChart:
            The instance of the SingleChart with the specified stripes 
            plotted on the axes.

        Notes
        -----
        This method plots stripes on the chart axes to represent 
        statistical measures such as mean, median, control limits, and 
        specification limits. The method provides options to customize 
        the appearance and behavior of the stripes using various 
        parameters and keyword arguments.
        """
        target = kwds.pop('target', self.source[self.target]) # TODO: consider target of bar and pareto
        single_axes = len(self.axes_facets) == 1
        self.stripes_facets = StripesFacets(
            target=target, single_axes=single_axes, mean=mean, median=median,
            control_limits=control_limits, spec_limits=spec_limits,
            confidence=confidence, **kwds)
        self.stripes_facets.draw(ax=self.ax, target_on_y=self.target_on_y)
        return self

    def label(
        self, fig_title: str = '', sub_title: str = '',
        feature_label: bool | str = '', target_label: bool | str = '',
        info: bool | str = False) -> Self:
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

        Returns
        -------
        SingleChart
            The instance of the SingleChart with updated labels and 
            titles.

        Notes
        -----
        This method allows customization of chart labels and titles to
        enhance readability and provide context for the visualized data.
        """
        if self.categorical_feature:
            self._categorical_feature_axis_()
        xlabel, ylabel = self.axis_labels(feature_label, target_label)

        self.label_facets = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=xlabel, ylabel=ylabel,
            info=info, legend_data=self.legend_data)
        self.label_facets.draw()

        return self


class JointChart(Chart):
    """
    Represents a joint chart visualization combining multiple 
    SingleCharts.

    Inherits from Chart.

    Parameters
    ----------
    source : DataFrame
        The source data.
    target : str or Tuple[str]
        The target variable(s).
    feature : str or Tuple[str]
        The feature variable(s).
    nrows : int
        Number of rows in the subplot grid.
    ncols : int
        Number of columns in the subplot grid.
    hue : str or Tuple[str], optional
        The hue variable(s), by default ''.
    shape : str or Tuple[str], optional
        The shape variable(s), by default ''.
    size : str or Tuple[str], optional
        The size variable(s), by default ''.
    dodge : bool or Tuple[bool], optional
        Flag indicating whether dodging is enabled, by default False.
    categorical_feature : bool or Tuple[str], optional
        Flag indicating whether feature is categorical, by default
        False.
    target_on_y : bool or List[bool], optional
        Flag indicating whether target is on y-axis, by default True.
    sharex : bool | Literal['none', 'all', 'row', 'col'], optional
        Flag indicating whether x-axis should be shared among subplots,
        by default False.
    sharey : bool | Literal['none', 'all', 'row', 'col'], optional
        Flag indicating whether y-axis should be shared among subplots,
        by default False.
    width_ratios : List[float], optional
        The width ratios for the subplot grid, by default None.
    height_ratios : List[float], optional
        The height ratios for the subplot grid, by default None.
    stretch_figsize : bool, optional
        Flag indicating whether figure size should be stretched to fill
        the grid, by default True.
    **kwds : dict
        Additional keyword arguments for Chart initialization.
    """
    __slots__ = (
        'charts', '_last_chart', '_chart_iterator', 'targets', 'features', 
        'hues', 'shapes', 'sizes', 'dodges', 'categorical_feature',
        'target_on_ys')

    charts: List[SingleChart]
    """List of SingleChart instances created for each Axis throughout
    the chart."""
    _last_chart: SingleChart | None
    """Last SingleChart instance worked on."""
    _chart_iterator: Generator[SingleChart, Self, None]
    """Iterator over SingleChart instances."""
    targets: Tuple[str, ...]
    """Column names for the target variable to be visualized for each
    axes."""
    features: Tuple[str, ...]
    """Column names for the feature variable to be visualized for each
    axes."""
    hues: Tuple[str, ...]
    """The hue variable (column) for color differentiation for each
    axes."""
    shapes: Tuple[str, ...]
    """The shape variables (column) for marker differentiation for each
    axes."""
    sizes: Tuple[str, ...]
    """The size variable (column) for marker size differentiation for
    each axes."""
    dodges: Tuple[bool, ...]
    """Flag indicating whether dodging is enabled for each axes."""
    categorical_feature: bool | Tuple[bool, ...]
    """Flags indicating if feature is categorical for each axes."""
    target_on_ys: Tuple[bool, ...]
    """Flags indicating whether target is on y-axis for each axes."""

    def __init__(
            self,
            source: DataFrame,
            target: str | Tuple[str, ...],
            feature: str | Tuple[str, ...],
            nrows: int,
            ncols: int,
            hue: str | Tuple[str, ...] = '',
            shape: str | Tuple[str, ...] = '',
            size: str | Tuple[str, ...] = '',
            dodge: bool | Tuple[bool, ...] = False,
            categorical_feature: bool | Tuple[bool, ...] = False,
            target_on_y: bool | Tuple[bool, ...] = True,
            sharex: bool | Literal['none', 'all', 'row', 'col'] = 'none', 
            sharey: bool | Literal['none', 'all', 'row', 'col'] = 'none', 
            width_ratios: List[float] | None = None,
            height_ratios: List[float] | None = None,
            stretch_figsize: bool = True,
            **kwds) -> None:

        self.charts = []
        self._last_chart = None
        self.ncols = ncols
        self.nrows = nrows

        super().__init__(
            source=source, target='', feature='', 
            sharex=sharex, sharey=sharey, width_ratios=width_ratios,
            height_ratios=height_ratios, stretch_figsize=stretch_figsize,
            nrows=nrows, ncols=ncols, **kwds)
        self.targets = self.ensure_tuple(target)
        self.features = self.ensure_tuple(feature)
        self.hues = self.ensure_tuple(hue)
        self.shapes = self.ensure_tuple(shape)
        self.sizes = self.ensure_tuple(size)
        self.dodges = self.ensure_tuple(dodge)
        self.categorical_feature = self.ensure_tuple(categorical_feature)
        self.target_on_ys = self.ensure_tuple(target_on_y)
        for i in [self.axes_facets.index for _ in self.axes_facets]:
            self.charts.append(SingleChart(
                source=self.source, target=self.targets[i],
                feature=self.features[i], hue=self.hues[i],
                dodge=self.dodges[i], shape=self.shapes[i], size=self.sizes[i],
                categorical_feature=self.categorical_feature[i],
                target_on_y=self.target_on_ys[i], axes_facets=self.axes_facets))

    @property
    def same_target_on_y(self) -> bool:
        """True if all target_on_y have the same boolean value."""
        reference = self.charts[0].target_on_y
        return all(chart.target_on_y == reference for chart in self.charts)
    
    @property
    def legend_data(self) -> Dict[str, LegendHandlesLabels]:
        """Get dictionary of handles and labels (read-only).
            - keys: titles as str
            - values: handles and labels as tuple of tuples"""
        legend_data = {}
        for chart in self.charts:
            legend_data = legend_data | chart.legend_data
        return legend_data
    
    def _single_label_allowed_(self, is_target: bool) -> bool:
        """Determines whether a single label is allowed for the 
        specified axis.
        
        This method checks whether a single axis label is allowed for 
        either target or feature dimensions based on certain conditions.
        The `same_target_on_y` attribute and the number of unique labels
        are considered.
        
        Parameters
        ----------
        is_target : bool
            If True, checks for target labels; otherwise, checks for 
            feature labels.

        Returns
        -------
        bool
            True if a single label is allowed, False otherwise.
        """
        allowed = all([
            self.same_target_on_y,
            len(set(self.targets if is_target else self.features)) == 1])
        return allowed 
    
    def _next_chart_(self) -> SingleChart:
        """Get next SingleChart instance to work on (read-only)."""
        if (self._last_chart in (None, self.charts[-1]) 
            or not hasattr(self, '_chart_iterator')):
            self._chart_iterator = self.itercharts()
        self._last_chart = next(self._chart_iterator)
        return self._last_chart
    
    def itercharts(self) -> Generator[SingleChart, Self, None]:
        """Iter over charts simultaneosly iters over axes of 
        `axes_facets`. That ensures that the current Axes to which the 
        current chart belongs is set."""
        for _, chart in zip(self.axes_facets, self.charts):
            yield chart
    
    def ensure_tuple(
            self, attribute: str | bool | Tuple | List) -> Tuple:
        """Ensures that the specified attribute is a tuple with the same
        length as the axes. If only one value is specified, it will be
        copied accordingly."""
        if isinstance(attribute, (str, bool)):
            _attribute = tuple(attribute for _ in range(self.n_axes))
        elif isinstance(attribute, tuple):
            _attribute = attribute
        elif isinstance(attribute, list):
            _attribute = tuple(attribute)
        else:
            raise ValueError(f'Not supported type {type(attribute)}')

        assert len(_attribute) == self.n_axes, (
            f'{attribute} does not have enough values, needed {self.n_axes}')

        return _attribute
    
    def _data_genenator_(self) -> Generator[DataFrame, Self, None]:
        return super()._data_genenator_()
    
    def _axis_label_(
            self, label: str | bool | None | Tuple[str | bool | None],
            is_target: bool) -> str | Tuple[str, ...]:
        """Helper method to get the axis label based on the provided 
        label and is_target flag.

        Parameters
        ----------
        label: str | bool | None | Tuple[str | bool | None]
            The label to use for the feature or target axis.
        is_target: bool
            Flag indicating whether the label is for the target variable.

        Returns
        -------
        str | Tuple[str, ...]
            The axis label as a string for a global axis label or a
            tuple containing an individual label for each axis.
        """
        if label in (False, None, ''):
            return tuple('' for _ in self.charts)
        elif isinstance(label, (tuple, list, set)):
            label_chart = zip(label, self.charts)
            return tuple(c._axis_label_(s, is_target) for s, c in label_chart)
        elif label is True:
            _label = self.targets[0] if is_target else self.features[0]
        else:
            _label = str(label)

        _kind = "target" if is_target else "feature"
        assert self._single_label_allowed_(is_target), (
            f'Single label not allowed for the {_kind} axis. '
            f'Ensure all subplots have the same orientation and {_kind}')
        return _label

    def _swap_labels_(
            self, xlabel: str | Tuple[str, ...], ylabel: str | Tuple[str, ...]
            ) -> Tuple[str | Tuple[str, ...], str | Tuple[str, ...]]:
        """Swaps axis labels based on certain conditions.

        If one of the label is a string, the method swaps the `xlabel` 
        and `ylabel` if `target_on_y` attribute is not True for the 
        first chart. If both labels are tuples, it processes each chart 
        in the list and swaps labels accordingly.

        Parameters
        ----------
        xlabel : str | Tuple[str, ...]
            The x-axis label(s) coming from `_axis_label_` method.
        ylabel : str | Tuple[str, ...]
            The y-axis label(s) coming from `_axis_label_` method.

        Returns
        -------
        Tuple[str | Tuple[str, ...], str | Tuple[str, ...]]
            The swapped x-axis label(s) and y-axis label(s).
        """
        if isinstance(xlabel, str) or isinstance(ylabel, str):
            if not self.target_on_ys[0]:
                xlabel, ylabel = ylabel, xlabel
        else:
            xy = []
            for x, y, chart in zip(xlabel, ylabel, self.charts):
                xy.append((x, y) if chart.target_on_y  else (y, x))
            xlabel, ylabel = tuple(zip(*xy))
        return xlabel, ylabel
    
    def axis_labels(
            self, feature_label: str | bool | None | Tuple[str | bool | None], 
            target_label: str | bool | None | Tuple[str | bool | None]
            ) -> Tuple[str | Tuple[str, ...], str | Tuple[str, ...]]:
        """Get the x and y axis labels based on the provided 
        `feature_label` and `target_label`."""
        xlabel = self._axis_label_(feature_label, False)
        ylabel = self._axis_label_(target_label, True)
        xlabel, ylabel = self._swap_labels_(xlabel, ylabel)
        return xlabel, ylabel

    def plot(
            self, plotter: Type[Plotter], kw_call: Dict[str, Any] = {}, **kwds
            ) -> Self:
        """Plot the data using the specified plotter.
        
        Parameters
        ----------
        plotter : Type[Plotter]
            The plotter object.
        kw_call : Dict[str, Any]
            Additional keyword arguments for the plotter call method.
        **kwds:
            Additional keyword arguments for the plotter object.

        Returns
        -------
        JointChart
            The updated JointChart instance after plotting.
        
        Notes
        -----
        Call the plot method for each individual Axes in the flattened 
        grid. You can specify the desired plotter for each chart 
        individually. The order of plotting corresponds to the flattened 
        arrangement of the subplots. Feel free to use this method to 
        create customized visualizations for each subplot.
        """
        chart = self._next_chart_()
        chart.plot(plotter, kw_call, **kwds)
        return self

    def stripes(
            self, mean: bool = False, median: bool = False,
            control_limits: bool = False, 
            spec_limits: Tuple[float | None, float | None] = (None, None), 
            confidence: float | None = None, **kwds) -> Self:
        """Plot location and spread width lines, specification limits 
        and/or confidence interval areas as stripes on each Axes. The
        location and spread (and their confidence bands) represent the 
        data per axes.

        Parameters
        ----------
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
            If provided, specifies the specification limits. 
            The tuple must contain two values for the lower and upper 
            limits. If a limit is not given, use None, by default ().
        confidence : float, optional
            The confidence level between 0 and 1, by default None.
        **kwds:
            Additional keyword arguments for configuring StripesFacets.

        Returns
        -------
        JointChart:
            The instance of the JointChart with the specified stripes 
            plotted on the axes.

        Notes
        -----
        The given arguments are applied to all axes!
        If stripes should only be drawn on selected axes, select the 
        desired subchart via `charts` attributes. Then use its `stripes`
        method.

        This method plots stripes on the chart axes to represent 
        statistical measures such as mean, median, control limits, and 
        specification limits. The method provides options to customize 
        the appearance and behavior of the stripes using various 
        parameters and keyword arguments.
        """
        for chart in self.itercharts():
            chart.stripes(
                mean=mean, median=median, control_limits=control_limits,
                spec_limits=spec_limits, confidence=confidence, **kwds)
        return self
    
    def label(
            self, fig_title: str = '', sub_title: str = '',
            feature_label: str | bool | Tuple = '', 
            target_label: str | bool | Tuple = '', 
            info: bool | str = False, axes_titles: Tuple[str, ...] = (),
            row_title: str = '', col_title: str = '') -> Self:
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
        row_title : str, optional
            The title of the rows, by default ''.
        col_title : str, optional
            The title of the columns, by default ''.

        Returns
        -------
        JointChart
            The instance of the JointChart with updated labels and 
            titles.

        Notes
        -----
        This method allows customization of chart labels and titles to
        enhance readability and provide context for the visualized data.
        """
        for chart in self.itercharts():
            if not chart.categorical_feature:
                continue
            chart._categorical_feature_axis_()
        xlabel, ylabel = self.axis_labels(feature_label, target_label)

        self.label_facets = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=xlabel, ylabel=ylabel,
            info=info, row_title=row_title, col_title=col_title,
            axes_titles=axes_titles, legend_data=self.legend_data)
        self.label_facets.draw()
        return self


class MultipleVariateChart(SingleChart):
    """Represents a chart visualization that handles multiple variables
    simultaneously.

    This class extends the functionality of SingleChart to create 
    visualizations for multiple variables, allowing for comparisons and
    insights across different dimensions.

    Parameters
    ----------
    source : DataFrame
        The source data for the chart.
    target : str
        The target variable (dependent variable).
    feature : str, optional
        The feature variable (independent variable), by default ''.
    hue : str, optional
        The hue variable (color grouping), by default ''.
    shape : str, optional
        The shape variable (marker grouping), by default ''.
    size : str, optional
        The size variable (marker size grouping), by default ''.
    col : str, optional
        The column variable for facetting, by default ''.
    row : str, optional
        The row variable for facetting, by default ''.
    dodge : bool, optional
        Whether to dodge categorical variables, by default False.
    stretch_figsize : bool, optional
        Whether to stretch the figure size, by default True.
    categorical_feature : bool, optional
        Whether the feature variable is categorical. If `dodge` is True,
        this will be automatically set to True, by default False.

    Examples
    --------
    >>> chart = MultipleVariateChart(source=data, target='sales', col='region', hue='product')
    >>> chart.plot(MyPlotter, param1=42, param2='abc')
    """

    __slots__ = ('col', 'row', 'row_labels', 'col_labels')
    col: str
    """The column variable for facetting (if applicable)."""
    row: str
    """The row variable for facetting (if applicable)."""
    row_labels: tuple
    """Labels corresponding to categorical values in the row variable."""
    col_labels: tuple
    """Labels corresponding to categorical values in the column 
    variable."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            hue: str = '',
            shape: str = '',
            size: str = '',
            col: str = '',
            row: str = '',
            dodge: bool = False,
            stretch_figsize: bool = True,
            categorical_feature: bool = False,
            ) -> None:
        self.target_on_y = True
        self.source = source
        self.col = col
        self.row = row
        self.row_labels = self.get_categorical_labels(self.row)
        self.col_labels = self.get_categorical_labels(self.col)
        nrows = max([1, len(self.row_labels)])
        ncols = max([1, len(self.col_labels)])
        super().__init__(
            source=self.source, target=target, feature=feature, hue=hue, 
            shape=shape, size=size, dodge=dodge, sharex=True, sharey=True,
            nrows=nrows, ncols=ncols, stretch_figsize=stretch_figsize,
            categorical_feature=categorical_feature)
        self._variate_names = (self.row, self.col, self.hue, self.shape)
        self._reset_variate_()
    
    @property
    def row_or_col_changed(self) -> bool:
        """Check whether the current variate belongs to a new row 
        or column relative to the last variate"""
        for key, old in self._last_variate.items():
            new = self._current_variate[key]
            if old != new and key in (self.row, self.col):
                return True
        return False
    
    def specification_limits_iterator(
            self, spec_limits: SpecLimits | Tuple[SpecLimits, ...]
            ) -> Generator[SpecLimits, Any, None]:
        """Generates specification limits based on the provided input.

        Parameters
        ----------
        spec_limits : SpecLimits | Tuple[SpecLimits, ...]
            The specification limits to generate from. If a single limit
            pair is provided, it will be used for all axes. If a tuple 
            of values is provided, each value corresponds to an axes.

        Yields
        ------
        SpecLimits
            The generated spec limits.

        Examples
        --------
        >>> generator = spec_limits_gen((1.0, 2.0))
        >>> next(generator)
        (1.0, 2.0)
        >>> next(generator)
        (1.0, 2.0)
        """
        if isinstance(spec_limits[0], tuple):
            _spec_limits = spec_limits
        else:
            _spec_limits = tuple([spec_limits]*self.n_axes)
        for axes_limits  in _spec_limits:
            yield axes_limits # type: ignore
    
    def _categorical_feature_axis_(self) -> None:
        """Set one major tick for each category and label it. Hide 
        major grid and set one minor grid for feature axis."""
        for ax in self.axes_facets:
            super()._categorical_feature_axis_()
    
    def _axes_data_(self) -> Generator[Series, Self, None]:
        """Generate all target data of each axes in one Series there are
        multiple axes, otherwise yield the entire target column. This
        function ensures also the current axes of `axes_facets`.

        This method serves as a generator function that yields grouped 
        data based on the `row` and `col` attribute if they are set. 
        If no `row` and no `col` are specified, it yields the entire
        target column.

        Yields:
        -------
        axes_data : Series
            Containing all target data for each axes.
        """
        names = [c for c in (self.row, self.col) if c]
        grouper = self.source.groupby(names) if names else [('', self.source)]
        for _, (_, data) in zip(self.axes_facets, grouper):
            axes_data = data[self.target]
            yield axes_data

    def plot(
            self, plotter: Type[Plotter], kw_call: Dict[str, Any] = {}, **kwds
            ) -> Self:
        """Plot the chart using the specified plotter.

        This method generates a subset of the source data specific to 
        each axes and then uses this data for the specified plotter.

        Parameters
        ----------
        plotter : Type[Plotter]
            The type of plotter to use.
        kw_call : Dict[str, Any]
            Additional keyword arguments for the plotter call method.
        **kwds : Any
            Additional keyword arguments to pass to the plotter.

        Returns
        -------
        Self
            The updated MultipleVariateChart object."""
        ax = None
        _ax = iter(self.axes_facets)
        for data in self:
            if self.row_or_col_changed or ax is None:
                ax = next(_ax)
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_on_y=self.target_on_y, color=self.color, ax=ax, 
                marker=self.marker, size=self.sizes,
                width=self.dodging.width, **kwds)
            plot(**kw_call)
            self._plots.append(plot)
        return self
    
    def stripes(
            self, mean: bool = False, median: bool = False,
            control_limits: bool = False,
            spec_limits: SpecLimits | Tuple[SpecLimits, ...] = (None, None),
            confidence: float | None = None, **kwds) -> Self:
        """Plot location and spread width lines, specification limits, 
        and/or confidence interval areas as stripes on each Axes. The
        location and spread (and their confidence bands) represent the
        data per axes.

        Parameters
        ----------
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
        SingleChart
            The instance of the SingleChart with the specified stripes
            plotted on the axes.

        Notes
        -----
        This method plots stripes on the chart axes to represent
        statistical measures such as mean, median, control limits, and
        specification limits. The method provides options to customize
        the appearance and behavior of the stripes using various
        parameters and keyword arguments.
        """
        _spec_limits = self.specification_limits_iterator(spec_limits)
        for axes_data, axes_limits in zip(self._axes_data_(), _spec_limits):
            super().stripes(
                target=axes_data, mean=mean, median=median, 
                control_limits=control_limits, spec_limits=axes_limits,
                confidence=confidence, **kwds)
        return self

    def label(
            self, fig_title: str = '', sub_title: str = '',
            feature_label: str | bool | None = '',
            target_label: str | bool | None = '', info: bool | str = False,
            row_title: str = '', col_title: str = '') -> Self:
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
        row_title : str, optional
            The title for the row facet (if applicable), by default ''.
        col_title : str, optional
            The title for the column facet (if applicable),
            by default ''.

        Returns
        -------
        MultiVariateChart
            The instance of the MultiVariateChart with updated labels
            and titles.

        Notes
        -----
        This method allows customization of chart labels and titles to
        enhance readability and provide context for the visualized data.
        """
        if self.categorical_feature:
            self._categorical_feature_axis_()
        if self.row and not row_title:
            row_title = self.row
        if self.col and not col_title:
            col_title = self.col
        xlabel, ylabel = self.axis_labels(feature_label, target_label)

        self.label_facets = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=xlabel, ylabel=ylabel,
            rows=self.row_labels, cols=self.col_labels,
            info=info, row_title=row_title, col_title=col_title,
            legend_data=self.legend_data)
        self.label_facets.draw()
        return self

__all__ = [
    'SingleChart',
    'JointChart',
    'MultipleVariateChart'
]