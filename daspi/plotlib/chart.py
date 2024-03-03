"""
Module for creating various types of chart visualizations using 
Matplotlib and Pandas.

This module provides classes and utility functions to facilitate the 
creation of different types of charts and visualizations. It includes 
support for single-variable charts, joint charts combining multiple 
variables, and charts with multiple variables simultaneously.

Classes:
- _Chart: Abstract base class for creating chart visualizations.
- SimpleChart: Represents a basic chart visualization with customizable 
features.
- JointChart: Represents a joint chart visualization combining multiple 
SimpleCharts.
- MultipleVariateChart: Represents a chart visualization handling 
multiple variables simultaneously.

Functionality:
- Customization of chart attributes including target, feature, hue, 
shape, size, etc.
- Layout setup for charts, including grid arrangements for joint charts.
- Adding stripes to highlight data patterns and labeling axes 
appropriately.
- Saving charts to files and programmatically closing charts.

Other Details:
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
from typing import Self
from typing import Dict
from typing import List
from typing import Tuple
from typing import Generator
from pathlib import Path
from numpy.typing import NDArray
from pandas.core.frame import DataFrame

from .utils import Dodger
from .utils import HueLabel
from .utils import SizeLabel
from .utils import ShapeLabel
from .facets import AxesFacets
from .facets import LabelFacets
from .facets import StripesFacets
from .plotter import _Plotter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from .._strings import STR
from .._constants import KW
from .._constants import COLOR


class _Chart(ABC):   
    """
    Abstract base class for creating chart visualizations.

    Attributes
    ----------
    source : pandas DataFrame
        A pandas DataFrame containing the data in long-format.
    target : str or Tuple[str]
        The target variable (column) to visualize.
    feature : str or Tuple[str]
        The feature variable (column) to use for the visualization.
    target_on_y : bool
        If True, the target variable is plotted on the y-axis.
    axes_facets : AxesFacets
        An instance containing the subplots' Axes and their arrangement.
    label_facets : LabelFacets, optional
        An optional instance for configuring and arranging figure and 
        subplot labels.
    stripes_facets : StripesFacets, optional
        An optional instance for adding stripes to the plot.
    nrows : int
        Number of rows in the subplot grid.
    ncols : int
        Number of columns in the subplot grid.
    figure : Figure (read-only)
        The top-level container for all plot elements.
    axes : NDArray (2D) (read-only)
        The axes of the subplot grid.
    n_axes : int (read-only)
        The total number of axes.
    xlabel, ylabel : str (read-only)
        Get the label for the x or y axis (set with the `set_axis_label` 
        method).
    plots : list of _Plotter (read-only)
        Plotter objects used in the `plot` method.
    """

    __slots__ = (
        'source', 'target', 'feature', 'target_on_y', 'axes_facets',
        'label_facets', 'stripes_facets', 'nrows', 'ncols', '_data', '_xlabel',
        '_ylabel', '_plots')
    source: DataFrame
    target: str
    feature: str
    target_on_y: bool
    stripes_facets: StripesFacets | None
    axes_facets: AxesFacets
    label_facets: LabelFacets | None
    nrows: int
    ncols: int
    _data: DataFrame
    _xlabel: str
    _ylabel: str
    _plots: List[_Plotter]

    def __init__(
            self, source: DataFrame, target: str | Tuple[str], 
            feature: str | Tuple[str]= '', target_on_y: bool = True, 
            axes_facets: AxesFacets | None = None, **kwds) -> None:
        self.source = source
        self.target = target
        self.feature = feature
        self.nrows = kwds.pop('nrows', 1)
        self.ncols = kwds.pop('ncols', 1)
        if axes_facets is None:
            self.axes_facets = AxesFacets(self.nrows, self.ncols, **kwds)
        else:
            self.axes_facets = axes_facets
        self.stripes_facets = None
        self.label_facets = None
        self.target_on_y = target_on_y
        for ax in self.axes.flat:
            getattr(ax, f'set_{"x" if self.target_on_y else "y"}margin')(0)
        self._data: DataFrame | None = None
        self._xlabel = ''
        self._ylabel = ''
        self._plots = []

    @property
    def figure(self) -> Figure:
        """Get the top level container for all the plot elements"""
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
    def xlabel(self) -> str:
        """Get label for x axis, set with `set_axis_label` method"""
        return self._xlabel
    
    @property
    def ylabel(self) -> str:
        """Get label for y axis, set with `set_axis_label` method"""
        return self._ylabel
    
    @property
    def plots(self) -> List[_Plotter]:
        """Get plotter objects used in `plot` method"""
        return self._plots
    
    def set_axis_label(
            self, label: Any, is_target: bool) -> str:
        """Set axis label according to given kind of label, taking into
        account the `target_on_y` attribute.
        
        Parameters
        ----------
        label : Any
            If a string is passed, it will be taken. If True, labels of 
            given feature or target name are used taking into account 
            the target_on_y attribute. If False or None, empty string is 
            used, by default ''
        is_target : Bool
            Set True if the label is for the target axis
        """

        match label:
            case None | False: 
                _label = ''
            case True: 
                _label = self.target if is_target else self.feature
            case _:
                _label = str(label)
        
        if is_target == self.target_on_y:
            self._ylabel = _label
        else:
            self._xlabel = _label
        
    @abstractmethod
    def _data_genenator_(self) -> Generator[Tuple, Self, None]:
        """Implement the data generator and add the currently yielded 
        data to self._data so that it can be used internally."""
    
    def __iter__(self) -> Generator[Tuple, Self, None]:
        return self._data_genenator_()
        
    def __next__(self) -> Axes:
        return next(self)
    
    @abstractmethod
    def plot(self, plotter: _Plotter): ...

    @abstractmethod
    def stripes(self, **stripes): ...

    @abstractmethod
    def label(self, **labels): ...
    
    def save(self, file_name: str | Path, **kwds) -> Self:
        kw = KW.SAVE_CHART | kwds
        self.figure.savefig(file_name, **kw)
        return self

    def close(self) -> Self:
        """"Close figure"""
        plt.close(self.figure)
        return self


class SimpleChart(_Chart):
    """
    Represents a basic chart visualization with customizable features.

    Inherits from _Chart.

    Attributes
    ----------
    hue : str
        The hue variable (column) for color differentiation.
    shape : str
        The shape variable (column) for marker differentiation.
    size : str
        The size variable (column) for marker size differentiation.
    marking : ShapeLabel
        Instance for configuring shape labels.
    sizing : SizeLabel
        Instance for configuring size labels.
    categorical_features : bool
        Flag indicating if the features are categorical.
    coloring : HueLabel
        Instance for configuring hue labels.
    dodging : Dodger
        Instance for configuring dodging attributes.
    variate_names : List[str] (read-only)
        Get names of all set variates
    color : str (read-only)
        Get color for current variate
    marker : str (read-only)
        Get marker for current variate
    sizes : NDArray (1D) or None (read-only)
        Get sizes for current variate, is set in grouped 
        `_data_generator_`
    legend_handles_labels : Dict[str, Tuple[tuple]] (read-only)
        Get dictionary of handles and labels
        - keys: titles as str
        - values: handles and labels as tuple of tuples
    """
    __slots__ = (
        'hue', 'shape', 'size', 'marking', 'sizing', '_sizes', 
        'categorical_features', 'coloring', 'dodging', '_variate_names',
        '_current_variate', '_last_variate', )
    hue: str
    shape: str
    size: str
    marking: ShapeLabel
    sizing: SizeLabel
    _sizes: NDArray
    categorical_features: bool
    coloring: HueLabel
    dodging: Dodger
    _current_variate: dict
    _last_variate: dict

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            hue: str = '',
            dodge: bool = False,
            shape: str = '',
            size: str = '', 
            categorical_features: bool = False,
            **kwds) -> None:
        self.categorical_features = categorical_features or dodge
        self.hue = hue
        self.shape = shape
        self.size = size
        super().__init__(source=source, target=target, feature=feature, **kwds)
        self.coloring = HueLabel(self.get_categorical_labels(self.hue))
        feature_tick_labels = ()
        if self.categorical_features:
            assert feature in source, (
                'categorical_features is True, but features is not present')
            feature_tick_labels = self.get_categorical_labels(feature)
        dodge_categories = self.coloring.labels if dodge else ()
        self.dodging = Dodger(dodge_categories, feature_tick_labels)
        self.marking = ShapeLabel(self.get_categorical_labels(self.shape))
        if self.size:
            self.sizing = SizeLabel(
                self.source[self.size].min(), self.source[self.size].max())
        else:
            self.sizing = None
        self._sizes: NDArray | None = None
        self._variate_names = (self.hue, self.shape)
        self._current_variate = {}
        self._last_variate = {}
        self._reset_variate_()
    
    @property
    def variate_names(self) -> List[str]:
        """Get names of all set variates"""
        return [v for v in self._variate_names if v]
    
    @property
    def color(self) -> str:
        """Get color for current variate"""
        hue_variate = self._current_variate.get(self.hue, None)
        return self.coloring[hue_variate]
    
    @property
    def marker(self) -> str:
        """Get marker for current variate"""
        marker_variate = self._current_variate.get(self.shape, None)
        return self.marking[marker_variate]

    @property
    def sizes(self) -> NDArray | None:
        """Get sizes for current variate, is set in grouped data 
        generator."""
        if not self.size: return None
        return self.sizing(self._data[self.size])
    
    @property
    def legend_handles_labels(self) -> Dict[str, Tuple[tuple]]:
        """Get dictionary of handles and labels
        - keys: titles as str
        - values: handles and labels as tuple of tuples"""
        handlers = (self.coloring, self.marking, self.sizing)
        titles = (self.hue, self.shape, self.size)
        if self.stripes_facets is not None:
            handlers = handlers + (self.stripes_facets,)
            titles = titles + (STR['stripes'], )
        return {t: h.handles_labels() for t, h in zip(titles, handlers) if t}
    
    def get_categorical_labels(self, colname: str) -> Tuple:
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
        if not colname: return ()
        return tuple(sorted(np.unique(self.source[colname])))
    
    def _reset_variate_(self):
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
        if not isinstance(combination, tuple): combination = (combination, )
        self._last_variate = deepcopy(self._current_variate)
        for key, name in zip(self.variate_names, combination):
            self._current_variate[key] = name
    
    def dodge(self) -> None:
        """Converts the feature data to tick positions, taking dodging 
        into account."""
        if not self.dodging: return
        hue_variate = self._current_variate.get(self.hue, None)
        self._data[self.feature] = self.dodging(
            self._data[self.feature], hue_variate)
        
    def _categorical_feature_grid_(self):
        """Hide major grid and set one minor grid for feature axis."""
        xy = 'x' if self.target_on_y else 'y'
        axis = getattr(self.axes_facets.ax, f'{xy}axis')
        axis.set_minor_locator(AutoMinorLocator(2))
        axis.grid(True, which='minor')
        axis.grid(False, which='major')
    
    def _categorical_feature_ticks_(self):
        """Set one major tick for each category and label it."""
        xy = 'x' if self.target_on_y else 'y'
        _ticks = self.dodging.ticks
        settings = {
            f'{xy}ticks': self.dodging.ticks,
            f'{xy}ticklabels': self.dodging.tick_lables,
            f'{xy}lim': (np.min(_ticks) - 0.5, np.max(_ticks) + 0.5)}
        self.axes_facets.ax.set(**settings)
        self.axes_facets.ax.tick_params(which='minor', color=COLOR.TRANSPARENT)
        
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
            self, plotter: _Plotter, **kwds) -> Self:
        """Plot the chart.

        Parameters
        ----------
        plotter : _Plotter
            The plotter object.
        **kwds:
            Additional keyword arguments for the plotter object.

        Returns:
        --------
        Self:
            The SimpleChart instance.
        """
        self.target_on_y = kwds.pop('target_on_y', self.target_on_y)
        for data in self:
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_on_y=self.target_on_y, color=self.color, 
                ax=self.axes_facets.ax, marker=self.marker, size=self.sizes,
                width=self.dodging.width, **kwds)
            plot()
            self._plots.append(plot)
        return self
    
    def stripes(
            self, mean: bool = False, median: bool = False,
            control_limits: bool = False, spec_limits: Tuple[float] = (), 
            confidence: float | None = None, **kwds) -> Self:
        """Plot stripes on the chart axes.

        Parameters
        ----------
        mean : bool, optional
            Whether to plot the mean value of the plotted data on the axes, 
            by default False.
        median : bool, optional
            Whether to plot the median value of the plotted data on the axes, 
            by default False.
        control_limits : bool, optional
            Whether to plot control limits representing the process spread,
            by default False.
        spec_limits : Tuple[float], optional
            If provided, specifies the specification limits. 
            The tuple must contain two values for the lower and upper limits.
            If a limit is not given, use None, by default ().
        confidence : float, optional
            The confidence level between 0 and 1, by default None.
        **kwds:
            Additional keyword arguments for configuring StripesFacets.

        Returns
        -------
        SimpleChart:
            The instance of the SimpleChart with the specified stripes plotted 
            on the axes.

        Notes
        -----
        This method plots stripes on the chart axes to represent statistical 
        measures such as mean, median, control limits, and specification limits. 
        The method provides options to customize the appearance and behavior 
        of the stripes using various parameters and keyword arguments.
        """
        self.stripes_facets = StripesFacets(
            target=self.source[self.target], mean=mean, median=median,
            control_limits=control_limits, spec_limits=spec_limits,
            confidence=confidence, **kwds)
        self.stripes_facets.draw(
            ax=self.axes_facets.ax, target_on_y=self.target_on_y)
        return self

    def label(
        self, fig_title: str = '', sub_title: str = '',
        feature_label: bool | str = '', target_label: bool | str = '',
        info: bool | str = False) -> Self:
        """Add labels to the chart.

        Parameters
        ----------
        fig_title : str, optional
            The figure title, by default ''.
        sub_title : str, optional
            The subtitle, by default ''.
        feature_label : bool | str, optional
            The feature label, by default ''.
        target_label : bool | str, optional
            The target label, by default ''.
        info : bool | str, optional
            Additional information label, by default False.

        Returns:
        --------
        Self:
            The SimpleChart instance.
        """
        if self.categorical_features:
            self._categorical_feature_axis_()
        self.set_axis_label(feature_label, is_target=False)
        self.set_axis_label(target_label, is_target=True)

        self.label_facets = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=self._xlabel, ylabel=self._ylabel,
            info=info, legends=self.legend_handles_labels)
        self.label_facets.draw()

        return self

class JointChart(_Chart):
    """
    Represents a joint chart visualization combining multiple SimpleCharts.

    Inherits from _Chart.

    Attributes
    ----------
    charts : List of SimpleChart
        List of SimpleChart instances to be combined.
    hue : str or Tuple[str]
        The hue variable (column) for color differentiation.
    shape : str or Tuple[str]
        The shape variable (column) for marker differentiation.
    size : str or Tuple[str]
        The size variable (column) for marker size differentiation.
    dodge : bool or Tuple[bool]
        Flag indicating if dodging is enabled.
    """
    __slots__ = ('charts', '_target', '_feature')
    charts: List[SimpleChart]
    hue: str | Tuple[str]
    shape: str | Tuple[str]
    size: str | Tuple[str]
    dodge: bool | Tuple[bool]

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str | Tuple[str],
            nrows: int,
            ncols: int,
            hue: str | Tuple[str] = '',
            shape: str | Tuple[str] = '',
            size: str | Tuple[str] = '',
            dodge: bool | Tuple[bool] = False,
            categorical_features: bool | Tuple[str] = False,
            target_on_y: bool | List[bool] = True,
            sharex: bool | str = False,
            sharey: bool | str = False,
            width_ratios: List[float] | None = None,
            height_ratios: List[float] | None = None,
            stretch_figsize: bool = True,
            **kwds) -> None:

        self.charts = []
        self.ncols = ncols
        self.nrows = nrows
        target = self.ensure_tuple(target)
        feature = self.ensure_tuple(feature)
        super().__init__(
            source=source, target=target, feature=feature, 
            sharex=sharex, sharey=sharey, width_ratios=width_ratios,
            height_ratios=height_ratios, stretch_figsize=stretch_figsize,
            nrows=nrows, ncols=ncols, **kwds)
        
        charts_data = dict(
            target = self._target,
            feature = self._feature,
            hue = self.ensure_tuple(hue),
            shape = self.ensure_tuple(shape),
            size = self.ensure_tuple(size),
            dodge = self.ensure_tuple(dodge),
            categorical_features = self.ensure_tuple(categorical_features),
            target_on_y = self.ensure_tuple(target_on_y))
        _kwds = dict(
            source=self.source, axes_facets=self.axes_facets)
        for values in zip(*charts_data.values()):
            kwds = _kwds | dict(zip(charts_data.keys(), values))
            self.charts.append(SimpleChart(**kwds))
    
    @property
    def _idx(self) -> int:
        """Get the index of current ax in flatten axes"""
        for idx, ax in enumerate(self.axes.flat):
            if self.axes_facets.ax == ax:
                return idx
        return 0
    
    @property
    def feature(self) -> str:
        return self._feature[self._idx]
    @feature.setter
    def feature(self, feature: str | Tuple[str]) -> None:
        self._feature: Tuple[str] = self.ensure_tuple(feature)
    
    @property
    def target(self) -> str:
        return self._target[self._idx]
    @target.setter
    def target(self, target: str | Tuple[str]) -> None:
        self._target: Tuple[str] = self.ensure_tuple(target)
    
    @property
    def target_on_y(self) -> List[bool]:
        """Get target_on_y of each sub chart as list

        Set target_on_y for all sub charts. if only one value is given, 
        it is adopted for all sub charts. If list or tuple, the values 
        are assigned to the subcharts in order"""
        return [c.target_on_y for c in self.charts]
    @target_on_y.setter
    def target_on_y(self, target_on_y: bool | List[bool] | Tuple[bool]) -> None:
        for toy, chart in zip(self.ensure_tuple(target_on_y), self.charts):
            chart.target_on_y = toy

    @property
    def same_target_on_y(self) -> bool:
        """True if all target_on_y have the same boolean value."""
        return all(toy == self.target_on_y[0] for toy in self.target_on_y)
    
    @property
    def legend_handles_labels(self) -> Dict:
        legend_hl = {}
        for chart in self.charts:
            legend_hl = legend_hl | chart.legend_handles_labels
        return legend_hl
    
    @property
    def xlabel(self) -> str | Tuple[str]:
        if not self._xlabel:
            self._xlabel = tuple((c.xlabel for c in self.charts))
        return self._xlabel
    
    @property
    def ylabel(self) -> str | Tuple[str]:
        if not self._ylabel: 
            self._ylabel = tuple((c.ylabel for c in self.charts))
        return self._ylabel
    
    def itercharts(self):
        """Iter over charts simultaneosly iters over axes of 
        `axes_facets`. That ensures that the current Axes to which the 
        current chart belongs is set."""
        for _, chart in zip(self.axes_facets, self.charts):
            yield chart
    
    def ensure_tuple(self, attribute: Any) -> Tuple:
        """Ensures that the specified attribute is a tuple with the same
        length as the axes. If only one value is specified, it will be
        copied accordingly."""
        if isinstance(attribute, tuple):
            new_attribute = attribute
        elif isinstance(attribute, list):
            new_attribute = tuple(attribute)
        else:
            new_attribute = tuple(attribute for _ in range(self.n_axes))
        assert len(new_attribute) == self.n_axes, (
            f'{attribute} does not have enough values, needed {self.n_axes}')
        return new_attribute

    def set_axis_label(
            self, label: Any, is_target: bool) -> None:
        if label and isinstance(label, str):
            assert self.same_target_on_y, (
                'For a single label, all axes must have the same orientation')
            if is_target == all(self.target_on_y):
                self._ylabel = label
            else:
                self._xlabel = label
        else:
            for _label, chart in zip(self.ensure_tuple(label), self.charts):
                chart.set_axis_label(_label, is_target=is_target)
    
    def _data_genenator_(self) -> Generator[Tuple, Self, None]:
        return super()._data_genenator_()
        
    def plot(
            self, plotters_kwds: List[Tuple[_Plotter | None, Dict]],
            hide_none: bool = True) -> Self:
        _axs = iter(self.axes_facets)
        for chart, (plotter, kwds) in zip(self.itercharts(), plotters_kwds):
            ax = next(_axs)
            if plotter is None:
                if hide_none: ax.set_axis_off()
                continue
            chart.plot(plotter, **kwds)
            self._plots.extend(chart.plots)
        return self
    
    def stripes(
            self, mean: bool = False, median: bool = False,
            control_limits: bool = False, spec_limits: Tuple[float] = (), 
            confidence: float | None = None, **kwds) -> Self:...
    
    def label(
            self, fig_title: str = '', sub_title: str = '',
            feature_label: str | bool | Tuple = '', 
            target_label: str | bool | Tuple = '', 
            row_title: str = '', col_title: str = '', info: bool | str = False
            ) -> Self:
        for chart in self.itercharts():
            if not chart.categorical_features: continue
            chart._categorical_feature_axis_()
        self._xlabel = ''
        self._ylabel = ''
        self.set_axis_label(feature_label, is_target=False)
        self.set_axis_label(target_label, is_target=True)

        self.label_facets = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=self.xlabel, ylabel=self.ylabel,
            info=info, row_title=row_title, col_title=col_title,
            legends=self.legend_handles_labels)
        self.label_facets.draw()
        return self


class MultipleVariateChart(SimpleChart):
    """
    Represents a chart visualization handling multiple variables simultaneously.

    Inherits from SimpleChart.

    Attributes
    ----------
    col : str
        The column used for column-wise differentiation.
    row : str
        The column used for row-wise differentiation.
    row_labels : tuple
        Tuple of sorted unique elements of the row column.
    col_labels : tuple
        Tuple of sorted unique elements of the column column.
    """

    __slots__ = ('col', 'row', 'row_labels', 'col_labels')
    col: str
    row: str
    row_labels: tuple
    col_labels: tuple

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
            categorical_features: bool = False,
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
            categorical_features=categorical_features)
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

    def plot(self, plotter: _Plotter, **kwds) -> Self:
        ax = None
        _ax = iter(self.axes_facets)
        for data in self:
            if self.row_or_col_changed or ax is None: ax = next(_ax)
            plot = plotter(
                source=data, target=self.target, feature=self.feature,
                target_on_y=self.target_on_y, color=self.color, ax=ax, 
                marker=self.marker, size=self.sizes,
                width=self.dodging.width, **kwds)
            plot()
            self._plots.append(plot)
        return self
    
    def stripes(
            self, mean: bool = False, median: bool = False,
            control_limits: bool = False, spec_limits: Tuple[float] = (), 
            confidence: float | None = None, **kwds) -> Self:
        super().stripes(
            mean=mean, median=median, control_limits=control_limits,
            spec_limits=spec_limits, confidence=confidence, **kwds)
                
    def label(
            self, feature_label: str, target_label: str,
            fig_title: str = '', sub_title: str = '', 
            row_title: str | None = None, col_title: str | None = None,
            info: bool | str = False) -> Self:
        if self.categorical_features:
            self._categorical_feature_axis_()
        self.set_axis_label(feature_label, is_target=False)
        self.set_axis_label(target_label, is_target=True)
        if self.row and row_title is None:
            row_title = self.row
        if self.col and col_title is None:
            col_title = self.col

        self.label_facets = LabelFacets(
            figure=self.figure, axes=self.axes, fig_title=fig_title,
            sub_title=sub_title, xlabel=self.xlabel, ylabel=self.ylabel,
            rows=self.row_labels, cols=self.col_labels,
            info=info, row_title=row_title, col_title=col_title,
            legends=self.legend_handles_labels)
        self.label_facets.draw()
        return self

__all__ = [
    SimpleChart.__name__,
    JointChart.__name__,
    MultipleVariateChart.__name__
]