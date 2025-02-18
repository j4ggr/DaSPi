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
- *MultivariateChart:* Represents a chart visualization handling 
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
import warnings
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
from typing import TypeVar
from typing import Literal
from typing import Callable
from typing import Sequence
from typing import Generator
from pathlib import Path
from functools import wraps
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.axis import XAxis
from matplotlib.axis import YAxis
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from .appearance import style
from .appearance import transpose_xy_axes_params

from .classify import Dodger
from .classify import HueLabel
from .classify import SizeLabel
from .classify import ShapeLabel

from .facets import AxesFacets
from .facets import LabelFacets
from .facets import StripesFacets

from .plotter import Stripe
from .plotter import Plotter
from .plotter import HideSubplot
from .plotter import SkipSubplot

from ..strings import STR

from .._typing import SpecLimit
from .._typing import SpecLimits
from .._typing import MosaicLayout
from .._typing import ShareAxisProperty
from .._typing import LegendHandlesLabels

from ..constants import KW
from ..constants import COLOR
from ..constants import PLOTTER
from ..constants import CATEGORY


T = TypeVar('T')

def check_label_order(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator that checks if a method is called after label().
    
    This decorator ensures that plot() and stripes() methods cannot be 
    called after the label() method has been called, enforcing the 
    correct order of method calls in Chart classes.

    Parameters
    ----------
    method : Callable[..., T]
        The method to be decorated.

    Returns
    -------
    Callable[..., T]
        The wrapped method that includes the order check.

    Raises
    ------
    ValueError
        If the decorated method is called after label() method.
    """
    @wraps(method)
    def wrapper(self: Chart, *args: Any, **kwargs: Any) -> T:
        assert not hasattr(self, 'label_facets'), (
            f'Cannot call {method.__name__}() after label(). '
            'The label() method must be called after stripes() and plot().')
        return method(self, *args, **kwargs)
    return wrapper


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
    target_on_y : bool, optional
        If True, the target variable is plotted on the y-axis.
    colors: Tuple[str, ...], optional
        Tuple of unique colors used for hue categories as hex or str,
        by default `CATEGORY.PALETTE`.
    markers : Tuple[str, ...], optional
        Tuple of markers used for shape marker categories as strings,
        by default `CATEGORY.MARKERS`.
    n_size_bins : int, optional
        Number of bins for the size range, by default 
        `CATEGORY.N_SIZE_BINS`.
    axes : AxesFacets, optional
        An instance containing the subplots' Axes and their arrangement.
        All further keyword arguments from here on will only be 
        considered if this one is not provided, by default None
    nrows : int, optional
        Number of rows of subplots in the grid, by default 1.
    ncols : int, optional
        Number of columns of subplots in the grid, by default 1.
    sharex : bool or {'none', 'all', 'row', 'col'}, optional
        Controls sharing of properties along the x-axis,
        by default 'none'.
    sharey : bool or {'none', 'all', 'row', 'col'}, optional
        Controls sharing of properties along the y-axis,
        by default 'none'.
    width_ratios : array-like of length ncols, optional
        Relative widths of the columns, by default None.
    height_ratios : array-like of length nrows, optional
        Relative heights of the rows, by default None.
    stretch_figsize : bool, optional
        If True, stretch the figure height and width based on the number 
        of rows and columns, by default False
    **kwds
        Additional key word arguments to instantiate the `AxesFacets`
        object.
    """
    __slots__ = (
        'source', 'target', 'feature', 'target_on_y', 'axes', '_ax',
        'label_facets', 'stripes_facets', '_data', '_plots', '_colors', 
        '_markers', '_n_size_bins', '_kw_where', '_follow_palette_order')
    
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
    axes: AxesFacets
    """AcesFacets instance for creating a grid of subplots with
    customizable sharing and sizing options."""
    label_facets: LabelFacets
    """LabelFacets instance for adding labels and titles to facets of a
    figure."""
    _data: DataFrame
    """Current source data subset used for current Axes."""
    _colors: Tuple[str, ...]
    """Tuple of colors used for hue categories as hex or str."""
    _markers: Tuple[str, ...]
    """Tuple of markers used for shape marker categories as strings."""
    _n_size_bins: int
    """Number of bins for the size range."""
    _plots: List[Plotter]
    """All plotter objects used in `plot` method."""
    _kw_where: Dict[str, Any]
    """Key word arguments to filter data in the plot method. This 
    argument is passed in the `kw_where` argument of the `plot` method.
    It must then be applied to `source` within `variate_data`
    method."""
    _ax: Axes | None
    """Axes object to which this chart belongs. This attribute is None 
    for parent charts such as JointChart and MultivariateChart. For 
    SingleChart this cannot be None."""

    def __init__(
            self,
            source: DataFrame,
            target: str,
            feature: str = '',
            *,
            target_on_y: bool = True,
            colors: Tuple[str, ...] | None = None,
            markers: Tuple[str, ...] | None = None,
            n_size_bins: int = CATEGORY.N_SIZE_BINS,
            axes: AxesFacets | None = None,
            nrows: int | None = None,
            ncols: int | None = None,
            mosaic: MosaicLayout| None = None,
            sharex: ShareAxisProperty = 'none', 
            sharey: ShareAxisProperty = 'none', 
            width_ratios: Sequence[float] | None = None,
            height_ratios: Sequence[float] | None = None, 
            stretch_figsize: bool = False,
            **kwds,
            ) -> None:
        self.source = source.copy()
        self.target = target
        self.feature = feature
        self._colors = colors if colors is not None else CATEGORY.PALETTE
        self._markers = markers if markers is not None else CATEGORY.MARKERS
        self._n_size_bins = n_size_bins
        self._plots = []
        self._kw_where = {}

        if axes is None:
            self.axes = AxesFacets(
                nrows=nrows,
                ncols=ncols,
                mosaic=mosaic,
                sharex=sharex,
                sharey=sharey,
                width_ratios=width_ratios,
                height_ratios=height_ratios,
                stretch_figsize=stretch_figsize,
                **kwds,)
        else:
            self.axes = axes
        self.target_on_y = target_on_y
        self._ax = self.axes.ax

    @property
    def figure(self) -> Figure:
        """Get the top level container for all the plot elements
        (read-only)."""
        return self.axes.figure
    
    @property
    def n_axes(self) -> int:
        """Get amount of axes"""
        return len(self.axes)
    
    @property
    def plots(self) -> List[Plotter]:
        """Get plotter objects used in `plot` method"""
        ignore_types = (HideSubplot, SkipSubplot)
        return [p for p in self._plots if not isinstance(p, ignore_types)]
    
    @abstractmethod
    def _axis_label_(
            self, label: Any, is_target: bool) -> str | Tuple[str, ...]:
        """Helper method to get the axis label based on the provided 
        label and is_target flag."""
        
    @abstractmethod
    def axis_labels(
            self, feature_label: Any, target_label: Any
            ) -> Tuple[str | Tuple[str, ...], str | Tuple[str, ...]]:
        """Get the x and y axis labels based on the provided 
        `feature_label` and `target_label`."""
        
    @abstractmethod
    def variate_data(
            self,
            skip_variate: List[str] = []
            ) -> Generator[DataFrame, Self, None]:
        """Implement the data generator and add the currently yielded 
        data to self._data so that it can be used internally. Also 
        consider the `_kw_where` attribute to filter the data here for 
        the plots.

        Parameters
        ----------
        skip_variate : List[str], optional
            A list of variate names to skip during the grouping. If 
            provided, these variates will not be included in the 
            groupby operation. Default is [].
        
        Returns
        -------
        Generator[DataFrame, Self, None]
            A generator object yielding DataFrames as subsets of the 
            source data, used as plotting data for each Axes.
        """
        raise NotImplementedError(
            'Generating data for each variate not implemented.')
    
    def _check_method_order(self, method_name: str) -> None:
        """Check if methods are called in the correct order.
        
        Parameters
        ----------
        method_name : str
            Name of the method being called.
            
        Raises
        ------
        ValueError
            If plot or stripes methods are called after label method.
        """
        if hasattr(self, 'label_facets') and method_name in ('plot', 'stripes'):
            raise ValueError(
                f'Cannot call {method_name}() after label(). '
                'The label() method must be called last.')
    
    @abstractmethod
    def plot(
            self,
            plotter: Type[Plotter],
            *,
            kw_call: Dict[str, Any] = {},
            **kwds
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
    def stripes(
            self,
            stripes: List[Stripe] = [],
            **kwds) -> Self:
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
            self,
            *,
            fig_title: str = '',
            sub_title: str = '',
            feature_label: bool | str = '',
            target_label: bool | str = '',
            info: bool | str = False
            ) -> Self:
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
    
    def specification_limits_iterator(
            self,
            spec_limits: SpecLimits | Tuple[SpecLimits, ...],
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
        assert len(_spec_limits) == self.n_axes, (
            f'spec_limits length {len(_spec_limits)} does not match '
            f'the number of axes {self.n_axes}')
        for axes_limits  in _spec_limits:
            yield axes_limits # type: ignore


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
    colors: Tuple[str, ...], optional
        Tuple of unique colors used for hue categories as hex or str,
        by default `CATEGORY.PALETTE`.
    markers : Tuple[str, ...], optional
        Tuple of markers used for shape marker categories as strings,
        by default `CATEGORY.MARKERS`.
    n_size_bins : int, optional
        Number of bins for the size range, by default 
        `CATEGORY.N_SIZE_BINS`.
    axes : AxesFacets, optional
        An instance containing the subplots' Axes and their arrangement.
        All further keyword arguments from here on will only be 
        considered if this one is not provided, by default None
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
            *,
            hue: str = '',
            dodge: bool = False,
            shape: str = '',
            size: str = '', 
            categorical_feature: bool = False,
            target_on_y: bool = True,
            colors: Tuple[str, ...] | None = None,
            markers: Tuple[str, ...] | None = None,
            n_size_bins: int = CATEGORY.N_SIZE_BINS,
            axes: AxesFacets | None = None,
            **kwds
            ) -> None:
        self.categorical_feature = categorical_feature or dodge
        if feature == '' and self.categorical_feature:
            feature = PLOTTER.FEATURE
        self.hue = hue
        self.shape = shape
        self.size = size
        super().__init__(
            source=source,
            target=target,
            feature=feature,
            target_on_y=target_on_y,
            colors=colors,
            markers=markers,
            n_size_bins=n_size_bins,
            axes = axes,
            **kwds)
        if self.categorical_feature:
            if self.feature not in self.source:
                self.source[self.feature] = ''
            self.source[self.feature].astype('category', copy=False)
        self.hueing = HueLabel(
            labels=self.unique_labels(self.hue),
            colors=self._colors,
            follow_order=style.follow_palette_order)
        dodge_labels = ()
        dodge_categories = ()
        if self.categorical_feature:
            dodge_labels = self.unique_labels(self.feature)
        if dodge:
            dodge_categories = tuple(self.hueing.labels)
        self.dodging = Dodger(dodge_categories, dodge_labels)
        self.shaping = ShapeLabel(
            self.unique_labels(self.shape), self._markers)
        if self.size:
            self.sizing = SizeLabel(
                self.source[self.size].min(),
                self.source[self.size].max(),
                self._n_size_bins)
        self._variate_names = (self.hue, self.shape)
        self._current_variate = {}
        self._last_variate = {}
        self._reset_variate_()
        self._transpose_xy_axes_params_()
    
    @property
    def ax(self) -> Axes:
        """Gets the current Axes instance that is currently being worked 
        on. This property raises an AttributeError if the current axis 
        is not set. To access the current axis without errors but with 
        the possibility of returning None, use the `ax` property of the 
        `AxesFacets` instance (e.g. `chart.axes.ax`) (read-only)."""
        if self._ax is None:
            raise AttributeError(
                'The current Axes instance is not set. Iterate over the '
                'AxesFacets instance and the sub charts simultaneously')
        return self._ax
    
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
    
    def _transpose_xy_axes_params_(self) -> None:
        """if target_on_y is false, all X- or Y-axis related rcParams 
        are swapped in pairs. If the plot is transposed, the set 
        parameters should also be swapped"""
        if self.target_on_y:
            return
        ticks = not bool(self.axes.sharex or self.axes.sharey)
        transpose_xy_axes_params(self.ax, ticks=ticks)
    
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
                for plotter in self.plots:
                    if type(plotter).__name__ == 'BlandAltman':
                        return plotter.target if is_target else plotter.feature
                return self.target if is_target else self. feature
            case _:
                return str(label)
    
    def axis_labels(
            self,
            feature_label: bool | str | None, 
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
    
    def unique_labels(self, colname: str) -> Tuple[Any, ...]:
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
        if not self.categorical_feature:
            return
        
        hue_variate = self._current_variate.get(self.hue, None)
        positions = self._data.pop(self.feature)
        self._data[self.feature] = self.dodging(positions, hue_variate)
        
    def _categorical_feature_grid_(self, ax: Axes) -> None:
        """Hide the major grid and set the subgrid between each category 
        for the feature axis. This feature is skipped if the main grid is not enabled.."""
        xy = 'x' if self.target_on_y else 'y'
        axis: XAxis | YAxis = getattr(ax, f'{xy}axis')
        if not axis.get_tick_params(which='major')['gridOn']:
            return
        axis.set_minor_locator(AutoMinorLocator(2))
        axis.grid(True, which='minor')
        axis.grid(False, which='major')
    
    def _categorical_feature_ticks_(self, ax: Axes) -> None:
        """Set one major tick for each category and label it.
        
        Raises
        ------
        AttributeError :
            If axes has no axes"""
        xy = 'x' if self.target_on_y else 'y'
        settings = {
            f'{xy}ticks': self.dodging.ticks,
            f'{xy}ticklabels': self.dodging.tick_lables,
            f'{xy}lim': self.dodging.lim}
        ax.set(**settings)
        ax.tick_params(which='minor', color=COLOR.TRANSPARENT)
        
    def _categorical_feature_axis_(self) -> None:
        """Set one major tick for each category and label it. Hide 
        major grid and set one minor grid for feature axis."""
        self._categorical_feature_grid_(self.ax)
        self._categorical_feature_ticks_(self.ax)

    def variate_data(
            self,
            skip_variate: List[str] = []
            ) -> Generator[DataFrame, Self, None]:
        """Generate grouped data if `variate_names` are set, otherwise 
        yield the entire source DataFrame.

        This method serves as a generator function that yields grouped 
        data based on the `variate_names` attribute if it is set. 
        If no `variate_names` are specified, it yields the entire source 
        DataFrame.

        Parameters
        ----------
        skip_variate : List[str], optional
            A list of variate names to skip during the grouping. If 
            provided, these variates will not be included in the groupby 
            operation. Default is []

        Yields:
        -------
        self._data : DataFrame
            Containing the grouped data or the entire source DataFrame.
        """
        source = self.source
        if self._kw_where:
            source = source.where(**self._kw_where)

        variate_names = [v for v in self.variate_names if v not in skip_variate]
        if variate_names:
            for combination, data in source.groupby(variate_names):
                self._data = data
                self.update_variate(combination)
                self.dodge()
                yield self._data
        else:
            self._data = source
            self.dodge()
            yield self._data
        self._reset_variate_()
        self._kw_where = {}

    @check_label_order
    def plot(
            self,
            plotter: Type[Plotter],
            *,
            skip_variate: List[str] = [],
            kw_call: Dict[str, Any] = {},
            kw_where: Dict[str, Any] = {},
            **kwds
            ) -> Self:
        """Apply a plotter with the specified data on the axis.

        Parameters
        ----------
        plotter : Type[Plotter]
            The plotter object.
        skip_variate : List[str], optional
            A list of variate names to skip during the grouping. If 
            provided, these variates will not be included in the groupby 
            operation. Default is []
        kw_call : Dict[str, Any]
            Additional keyword arguments for the plotter call method.
        kw_where : Dict[str, Any]
            Additional keyword arguments for the where method used to
            filter the data.
        **kwds:
            Additional keyword arguments for the plotter object. Here 
            you can set the plotter specific initialization parameters. 
            You can also override the standard parameters color, marker, 
            target_on_y, size (marker size) and width 
            (for categorical feature plots). These parameters are 
            handled automatically by the class. Only change them if you 
            know what you are doing.

        Returns:
        --------
        Self:
            The SingleChart instance.
        """
        self.target_on_y = kwds.pop('target_on_y', self.target_on_y)
        _color = kwds.pop('color', None)
        _marker = kwds.pop('marker', None)
        _size = kwds.pop('size', None)
        _width = kwds.pop('width', None)
        self._kw_where = kw_where
        for data in self.variate_data(skip_variate):
            plot = plotter(
                source=data,
                target=self.target,
                feature=self.feature,
                target_on_y=self.target_on_y,
                color=_color or self.color,
                marker=_marker or self.marker,
                size=_size or self.sizes,
                width=_width or self.dodging.width,
                ax=self.axes.ax,
                categorical_feature=self.categorical_feature,
                **kwds)
            plot(**kw_call)
            self._plots.append(plot)
        return self
    
    @check_label_order
    def stripes(
            self,
            stripes: List[Stripe] = [],
            *,
            mean: bool = False,
            median: bool = False,
            control_limits: bool = False, 
            spec_limits: Tuple[SpecLimit, SpecLimit] = (None, None), 
            confidence: float | None = None,
            strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
            agreement: float | int = 6,
            **kwds,
            ) -> Self:
        """Plot location and spread width lines, specification limits 
        and/or confidence interval areas as stripes on each Axes. The
        location and spread (and their confidence bands) represent the 
        data per axes.

        Parameters
        ----------
        stripes : List[Stripe], optional
            Additional non-predefined stripes to be added to the chart,
            by default [].
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
        target = kwds.pop('target', self.source[self.target])
        single_axes = kwds.pop('single_axes', len(self.axes) == 1)
        self.stripes_facets = StripesFacets(
            target=target,
            target_on_y=self.target_on_y,
            single_axes=single_axes,
            stripes=stripes,
            mean=mean,
            median=median,
            control_limits=control_limits,
            spec_limits=spec_limits,
            confidence=confidence,
            strategy=strategy,
            agreement=agreement,
            **kwds)
        self.stripes_facets.draw(ax=self.ax)
        return self

    def label(
            self,
            *,
            fig_title: str = '',
            sub_title: str = '',
            feature_label: bool | str = '',
            target_label: bool | str = '',
            info: bool | str = False
            ) -> Self:
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
            figure=self.figure,
            axes=self.axes,
            fig_title=fig_title,
            sub_title=sub_title,
            xlabel=xlabel,
            ylabel=ylabel,
            info=info,
            legend_data=self.legend_data)
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
    sharex : ShareAxisProperty, optional
        Flag indicating whether x-axis should be shared among subplots,
        by default False.
    sharey : ShareAxisProperty, optional
        Flag indicating whether y-axis should be shared among subplots,
        by default False.
    width_ratios : List[float], optional
        The width ratios for the subplot grid, by default None.
    height_ratios : List[float], optional
        The height ratios for the subplot grid, by default None.
    stretch_figsize : bool, optional
        Flag indicating whether figure size should be stretched to fill
        the grid, by default False.
    colors: Tuple[str, ...], optional
        Tuple of unique colors used for hue categories as hex or str,
        by default `CATEGORY.PALETTE`.
    markers : Tuple[str, ...], optional
        Tuple of markers used for shape marker categories as strings,
        by default `CATEGORY.MARKERS`.
    n_size_bins : int, optional
        Number of bins for the size range, by default 
        `CATEGORY.N_SIZE_BINS`.
    **kwds : dict
        Additional keyword arguments for Chart initialization.
    """
    __slots__ = (
        'charts', '_last_chart', '_chart_iterator', 'targets', 'features', 
        'hues', 'shapes', 'sizes', 'dodges', 'categorical_features',
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
    categorical_features: bool | Tuple[bool, ...]
    """Flags indicating if feature is categorical for each axes."""
    target_on_ys: Tuple[bool, ...]
    """Flags indicating whether target is on y-axis for each axes."""
    target_on_y: bool
    """Flag indicating whether target is on y-axis for current chart. 
    This flag is set during iterating over charts."""

    def __init__(
            self,
            source: DataFrame,
            target: str | Tuple[str, ...],
            feature: str | Tuple[str, ...],
            *,
            nrows: int | None = None,
            ncols: int | None = None,
            mosaic: MosaicLayout | None = None,
            hue: str | Tuple[str, ...] = '',
            shape: str | Tuple[str, ...] = '',
            size: str | Tuple[str, ...] = '',
            dodge: bool | Tuple[bool, ...] = False,
            categorical_feature: bool | Tuple[bool, ...] = False,
            target_on_y: bool | Tuple[bool, ...] = True,
            sharex: ShareAxisProperty = 'none', 
            sharey: ShareAxisProperty = 'none', 
            width_ratios: List[float] | None = None,
            height_ratios: List[float] | None = None,
            stretch_figsize: bool = False,
            colors: Tuple[str, ...] | None = None,
            markers: Tuple[str, ...] | None = None,
            n_size_bins: int = CATEGORY.N_SIZE_BINS,
            **kwds) -> None:

        self.charts = []
        self._last_chart = None

        super().__init__(
            source=source,
            target='',
            feature='', 
            sharex=sharex,
            sharey=sharey,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            stretch_figsize=stretch_figsize,
            nrows=nrows,
            ncols=ncols,
            colors=colors,
            markers=markers,
            n_size_bins=n_size_bins,
            mosaic=mosaic,
            **kwds)
        self.targets = self.normalize_to_tuple(target)
        self.features = self.normalize_to_tuple(feature)
        self.hues = self.normalize_to_tuple(hue)
        self.shapes = self.normalize_to_tuple(shape)
        self.sizes = self.normalize_to_tuple(size)
        self.dodges = self.normalize_to_tuple(dodge)
        self.categorical_features = tuple(
            dodge or categorical for dodge, categorical in 
            zip(self.dodges, self.normalize_to_tuple(categorical_feature)))
        self.target_on_ys = self.normalize_to_tuple(target_on_y)
        self.target_on_y = self.target_on_ys[0]

        if (not all(t == self.target_on_ys[0] for t in self.target_on_ys)
            and (sharex in (True, 'all') or sharey in (True, 'all'))):
            warnings.warn(
                'Shares axes along chart with mixed target_on_y!', UserWarning)

        for i, _ in enumerate(self.axes):
            self.charts.append(
                SingleChart(
                    source=self.source,
                    target=self.targets[i],
                    feature=self.features[i],
                    hue=self.hues[i],
                    dodge=self.dodges[i],
                    shape=self.shapes[i],
                    size=self.sizes[i],
                    categorical_feature=self.categorical_features[i],
                    target_on_y=self.target_on_ys[i],
                    colors=self._colors,
                    markers=self._markers,
                    n_size_bins = self._n_size_bins,
                    axes=self.axes))

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
            for key, handles_labels in chart.legend_data.items():
                if key not in legend_data:
                    legend_data[key] = handles_labels
                else:
                    for handle, label in zip(*handles_labels):
                        if label in legend_data[key][1]:
                            continue
                        legend_data[key] = (
                            legend_data[key][0] + (handle,),
                            legend_data[key][1] + (label,))
        return legend_data
    
    @property
    def plots(self) -> List[Plotter]:
        """Get plotter objects used in `plot` method"""
        return [p for c in self.charts for p in c.plots]
    
    @property
    def axes_share_feature(self) -> ShareAxisProperty:
        """Get the sharing of properties along the feature-axis
        (read-only)."""
        if self.target_on_y:
            share = self.axes.sharex
        else:
            share = self.axes.sharey
        return share

    @property
    def axes_share_target(self) -> ShareAxisProperty:
        """Get the sharing of properties along the target-axis
        (read-only)."""
        if self.target_on_y:
            share = self.axes.sharey
        else:
            share = self.axes.sharex
        return share
    
    def single_label_allowed(self, is_target: bool) -> bool:
        """Determines whether a single label is allowed for the 
        specified axis.
        
        This method checks whether a single axis label is allowed for 
        either target or feature dimensions based on certain conditions.
        The `same_target_on_y` attribute is allways considered. It is 
        also checked whether the number of unique values in the 
        respective dimension (features or targets) is 1 or whether the 
        sharing of the axis is set to True or 'all'.
        
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
        if is_target:
            share_axis = self.axes_share_target
            n_unique = len(set(self.targets))
        else:
            share_axis = self.axes_share_feature
            n_unique = len(set(self.features))
        allowed = all([
            self.same_target_on_y,
            (n_unique == 1 or share_axis in (True, 'all'))])
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
        `axes`. That ensures that the current Axes to which the 
        current chart belongs is set. The"""
        for ax, chart in zip(self.axes, self.charts):
            self._ax = ax
            self.target_on_y = chart.target_on_y
            yield chart
        self._ax = self.axes.ax
    
    def normalize_to_tuple(
            self,
            attribute: str | float | int | bool | None | List[Any] | Tuple[Any, ...]
            ) -> Tuple:
        """Normalize the input attribute to ensure it is a tuple with a length
        equal to the number of subplots (n_axes). If a single value is provided, 
        it is replicated to create a tuple with the same length as n_axes.

        Parameters
        ----------
        attribute : str, float, int, bool, list, tuple or None
            A single value or a list/tuple of values.

        Returns
        -------
        tuple
            A tuple containing the normalized values.

        Raises
        ------
        ValueError
            If the input type is not supported
        AssertionError
            If the length of the attribute does not match n_axes after 
            normalization.

        Examples
        --------
        Suppose n_axes is 3:
        - If attribute is 5, it will return (5, 5, 5).
        - If attribute is [], it will return ([], [], []).
        - If attribute is (1, 2, 3), it will return (1, 2, 3).
        - If attribute is (1, 2), it will raise a ValueError.
        """
        if isinstance(attribute, (str, float, int, bool, list, type(None))):
            normalized = (attribute,) * self.n_axes
        elif isinstance(attribute, tuple):
            normalized = attribute
        else:
            raise ValueError(f'Not supported type {type(attribute)}')

        assert len(normalized) == self.n_axes, (
            f'{attribute} does not have enough values, needed {self.n_axes}')

        return normalized
    
    def variate_data(
            self,
            skip_variate: List[str] = []
            ) -> Generator[DataFrame, Self, None]:
        raise NotImplementedError(
            'Generating data for each variate not implemented.')

    def _axis_label_(
            self,
            label: str | bool | None | Tuple[str | bool | None],
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

        _kind = 'target' if is_target else 'feature'
        assert self.single_label_allowed(is_target), (
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
            self,
            feature_label: str | bool | None | Tuple[str | bool | None], 
            target_label: str | bool | None | Tuple[str | bool | None]
            ) -> Tuple[str | Tuple[str, ...], str | Tuple[str, ...]]:
        """Get the x and y axis labels based on the provided 
        `feature_label` and `target_label`."""
        xlabel = self._axis_label_(feature_label, False)
        ylabel = self._axis_label_(target_label, True)
        xlabel, ylabel = self._swap_labels_(xlabel, ylabel)
        return xlabel, ylabel
    
    @check_label_order
    def plot(
            self,
            plotter: Type[Plotter],
            *,
            kw_call: Dict[str, Any] = {},
            kw_where: Dict[str, Any] = {},
            on_last_axes: bool = False,
            **kwds
            ) -> Self:
        """Plot the data using the specified plotter.
        
        Parameters
        ----------
        plotter : Type[Plotter]
            The plotter object.
        kw_call : Dict[str, Any]
            Additional keyword arguments for the plotter call method.
        kw_where : Dict[str, Any]
            Additional keyword arguments for the where method used to
            filter the data.
        on_last_axes : bool, optional
            If True, plot on the last axes in the grid. If False, plot 
            on the next axes in the grid, by default False.
        **kwds:
            Additional keyword arguments for the plotter object. Here 
            you can set the plotter specific initialization parameters. 
            You can also override the standard parameters color, marker, 
            target_on_y, size (marker size) and width 
            (for categorical feature plots).

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
        if on_last_axes and self._last_chart is not None:
            chart = self._last_chart
        else:
            chart = self._next_chart_()
        chart.plot(plotter, kw_call=kw_call, kw_where=kw_where, **kwds)
        return self

    @check_label_order
    def stripes(
            self,
            stripes: List[Stripe] | Tuple[List[Stripe], ...] = [],
            *,
            mean: bool | Tuple[bool, ...] = False,
            median: bool | Tuple[bool, ...] = False,
            control_limits: bool | Tuple[bool, ...] = False, 
            spec_limits: SpecLimits | Tuple[SpecLimits, ...] = (None, None),
            confidence: float | None | Tuple[float | None, ...] = None,
            strategy: Literal['eval', 'fit', 'norm', 'data'] | Tuple = 'norm',
            agreement: float | int | Tuple[int | float, ...] = 6,
            **kwds) -> Self:
        """Plot location and spread width lines, specification limits 
        and/or confidence interval areas as stripes on each Axes. The
        location and spread (and their confidence bands) represent the 
        data per axes.

        Parameters
        ----------
        stripes : List[Stripe] | Tuple[List[Stripe]], optional
            Additional non-predefined stripes to be added to the chart.
            Default is [].
        mean : bool or Tuple[bool, ...], optional
            Whether to plot the mean value of the plotted data on the 
            axes, by default False.
        median : bool or Tuple[bool, ...], optional
            Whether to plot the median value of the plotted data on the 
            axes, by default False.
        control_limits : bool or Tuple[bool, ...], optional
            Whether to plot control limits representing the process 
            spread, by default False.
        spec_limits : SpecLimits | Tuple[SpecLimits, ...], optional
            If provided, specifies the specification limits. 
            The tuple must contain two values for the lower and upper 
            limits. If a limit is not given, use None, by default ().
        confidence : float | None | Tuple[float | None, ...], optional
            The confidence level between 0 and 1, by default None.
        strategy : {'eval', 'fit', 'norm', 'data'} | Tuple, optional
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
        agreement : int | float | Tuple[int | float, ...], optional
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
        stripes = self.normalize_to_tuple(stripes)
        mean = self.normalize_to_tuple(mean)
        median = self.normalize_to_tuple(median)
        control_limits = self.normalize_to_tuple(control_limits)
        spec_limits = tuple(self.specification_limits_iterator(spec_limits))
        confidence = self.normalize_to_tuple(confidence)
        strategy = self.normalize_to_tuple(strategy)
        agreement = self.normalize_to_tuple(agreement)
        for i, chart in enumerate(self.itercharts()):
            chart.stripes(
                stripes=stripes[i],
                mean=mean[i],
                median=median[i],
                control_limits=control_limits[i],
                spec_limits=spec_limits[i],
                confidence=confidence[i],
                strategy=strategy[i],
                agreement=agreement[i],
                **kwds)
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
            figure=self.figure,
            axes=self.axes,
            fig_title=fig_title,
            sub_title=sub_title,
            xlabel=xlabel,
            ylabel=ylabel,
            info=info,
            rows=rows,
            cols=cols,
            row_title=row_title,
            col_title=col_title,
            axes_titles=axes_titles,
            legend_data=self.legend_data)
        self.label_facets.draw()
        return self


class MultivariateChart(SingleChart):
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
        Whether to stretch the figure size, by default False.
    categorical_feature : bool, optional
        Whether the feature variable is categorical. If `dodge` is True,
        this will be automatically set to True, by default False.
    target_on_y : bool, optional
        Flag indicating whether the target variable is plotted on the
        y-axis, by default True
    colors: Tuple[str, ...], optional
        Tuple of unique colors used for hue categories as hex or str,
        by default `CATEGORY.PALETTE`.
    markers : Tuple[str, ...], optional
        Tuple of markers used for shape marker categories as strings,
        by default `CATEGORY.MARKERS`.
    n_size_bins : int, optional
        Number of bins for the size range, by default 
        `CATEGORY.N_SIZE_BINS`.

    Examples
    --------
    
    ``` python
    import daspi as dsp
    df = dsp.load_dataset('iris')

    chart = dsp.MultivariateChart(
            source=df,
            target='length',
            feature='width',
            hue='species',
            col='leaf',
            markers=('x',)
        ).plot(
            dsp.GaussianKDEContour
        ).plot(
            dsp.Scatter
        ).label(
            feature_label='leaf width (cm)',
            target_label='leaf length (cm)',
        )
    ```
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
            stretch_figsize: bool = False,
            categorical_feature: bool = False,
            target_on_y: bool = True,
            colors: Tuple[str, ...] | None = None,
            markers: Tuple[str, ...] | None = None,
            n_size_bins: int = CATEGORY.N_SIZE_BINS,
            ) -> None:
        self.source = source
        self.col = col
        self.row = row
        self.row_labels = self.unique_labels(self.row)
        self.col_labels = self.unique_labels(self.col)
        nrows = max([1, len(self.row_labels)])
        ncols = max([1, len(self.col_labels)])
        super().__init__(
            source=self.source,
            target=target,
            feature=feature,
            hue=hue, 
            shape=shape,
            size=size,
            dodge=dodge,
            sharex=True,
            sharey=True,
            nrows=nrows,
            ncols=ncols,
            stretch_figsize=stretch_figsize,
            categorical_feature=categorical_feature,
            target_on_y=target_on_y,
            colors=colors,
            markers=markers,
            n_size_bins=n_size_bins)
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
    
    def _transpose_xy_axes_params_(self) -> None:
        """if target_on_y is false, all X- or Y-axis related rcParams 
        are swapped in pairs. If the plot is transposed, the set 
        parameters should also be swapped"""
        if self.target_on_y:
            return
        ticks = not bool(self.axes.sharex or self.axes.sharey)
        for ax in self.axes:
            transpose_xy_axes_params(ax, ticks=ticks)
    
    def _categorical_feature_axis_(self) -> None:
        """Set one major tick for each category and label it. Hide 
        major grid and set one minor grid for feature axis."""
        for ax in self.axes:
            self._ax = ax
            super()._categorical_feature_axis_()
    
    def _axes_data(self) -> Generator[Series, Self, None]:
        """Generate all target data of each axes in one Series there are
        multiple axes, otherwise yield the entire target column. This
        function ensures also the current axes of `axes`.

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
        for ax, (_, data) in zip(self.axes, grouper):
            self._ax = ax
            axes_data = data[self.target]
            yield axes_data
        self._ax = self.axes._default

    @check_label_order
    def plot(
            self,
            plotter: Type[Plotter],
            *,
            skip_variate: List[str] = [],
            kw_call: Dict[str, Any] = {},
            kw_where: dict = {},
            **kwds
            ) -> Self:
        """Plot the chart using the specified plotter.

        This method generates a subset of the source data specific to 
        each axes and then uses this data for the specified plotter.

        Parameters
        ----------
        plotter : Type[Plotter]
            The type of plotter to use.
        skip_variate : List[str], optional
            A list of variate names to skip during the grouping. If 
            provided, these variates will not be included in the groupby 
            operation. Default is None
        kw_call : Dict[str, Any]
            Additional keyword arguments for the plotter call method.
        kw_where : dict
            Do not use this argument in this instance.
        **kwds : Any
            Additional keyword arguments for the plotter object. Here 
            you can set the plotter specific initialization parameters. 
            Do not override the standard parameters color, marker, 
            target_on_y, size (marker size) and width 
            (for categorical feature plots). These parameters are 
            handled automatically by the class. For more flexibility, 
            use the JointChart class.

        Returns
        -------
        Self
            The updated MultivariateChart object."""
        if kw_where:
            raise ValueError(
                'Keyword argument "kw_where" is not allowed in this instance.')
        ax = None
        _ax = iter(self.axes)
        for data in self.variate_data(skip_variate):
            if self.row_or_col_changed or ax is None:
                ax = next(_ax)
            plot = plotter(
                source=data,
                target=self.target,
                feature=self.feature,
                target_on_y=self.target_on_y,
                color=self.color,
                ax=ax, 
                marker=self.marker,
                size=self.sizes,
                width=self.dodging.width,
                **kwds)
            plot(**kw_call)
            self._plots.append(plot)
        return self
    
    @check_label_order
    def stripes(
            self,
            stripes: List[Stripe] = [],
            *,
            mean: bool = False,
            median: bool = False,
            control_limits: bool = False, 
            spec_limits: Tuple[SpecLimit, SpecLimit] = (None, None), 
            confidence: float | None = None,
            strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
            agreement: float | int = 6,
            **kwds,
            ) -> Self:
        """Plot location and spread width lines, specification limits 
        and/or confidence interval areas as stripes on each Axes. The
        location and spread (and their confidence bands) represent the 
        data per axes.

        Parameters
        ----------
        stripes : List[Stripe], optional
            Additional non-predefined stripes to be added to the chart,
            by default [].
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
        for axes_data, axes_limits in zip(self._axes_data(), _spec_limits):
            super().stripes(
                stripes=stripes,
                target=axes_data,
                mean=mean,
                median=median, 
                control_limits=control_limits,
                spec_limits=axes_limits,
                confidence=confidence,
                strategy=strategy,
                agreement=agreement,
                **kwds)
        return self

    def label(
            self,
            *,
            fig_title: str = '',
            sub_title: str = '',
            feature_label: str | bool | None = '',
            target_label: str | bool | None = '',
            info: bool | str = False,
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
            figure=self.figure,
            axes=self.axes,
            fig_title=fig_title,
            sub_title=sub_title,
            xlabel=xlabel,
            ylabel=ylabel,
            rows=self.row_labels,
            cols=self.col_labels,
            info=info,
            row_title=row_title,
            col_title=col_title,
            legend_data=self.legend_data)
        self.label_facets.draw()
        return self

__all__ = [
    'SingleChart',
    'JointChart',
    'MultivariateChart'
]