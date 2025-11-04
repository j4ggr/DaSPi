import re
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any
from typing import Self
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from typing import overload
from typing import Sequence
from typing import Generator
from typing import Callable
from numpy.typing import NDArray
from matplotlib.text import Text
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.artist import Artist
from matplotlib.ticker import Formatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import StrMethodFormatter
from matplotlib.typing import HashableList
from matplotlib.patches import Patch

from .plotter import Stripe
from .plotter import StripeLine
from .plotter import StripeSpan

from .._typing import MosaicLayout
from .._typing import NumericSample1D
from .._typing import ShareAxisProperty
from .._typing import LegendHandlesLabels

from ..strings import STR

from ..constants import KW
from ..constants import LABEL
from ..constants import DEFAULT

from ..statistics import SpecLimits
from ..statistics import ProcessEstimator


__all__ = [
    "flat_unique",
    "LabelFacets",
    "AxesFacets",
    "StripesFacets",]


def flat_unique(nested: NDArray | List[List]) -> List:
    """Flatten the given array and return unique elements while
    preserving the order."""
    if isinstance(nested, list):
        nested = np.array(nested)
    return list(pd.Series(nested.flatten()).unique())


class AxesFacets:
    """A class for creating a grid of subplots with customizable sharing
    and sizing options.    
    
    The class provides flexible handling the created Axes instances 
    through its flat property and iteration capabilities:

    - **Flat property:** 
      Accesses Axes in a flattened view from left-to-right, 
      top-to-bottom
    - **Axes property:** 
      Maintains the 2D structure for grid-based operations
    - **Iteration:** 
      Yields each axis instance coming from the Flat property 
      sequentially to simplify plotting
    The class supports two different ways to create subplot layouts:

    Parameters
    ----------
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
    stretch_figsize : bool | float | Tuple[float, float], optional
        If True, the height and width of the figure are stretched based 
        on the number rows and columns in the axes grid. If a float is 
        provided, the figure size is stretched by the given factor. If a 
        tuple of two floats is provided, the figure size is stretched by 
        the given factors for the x and y axis, respectively.
        by default False.
    **kwds : dict, optional
        Additional keyword arguments to pass to the function 
        `plt.subplots` or to the function `plt.subplot_mosaic` if a 
        mosaic was specified.

    Examples
    --------
    1. Using nrows and ncols:
    Creates a regular grid of subplots with specified dimensions
    ```python
    # Creates a 2x3 grid of regular subplots
    facets = AxesFacets(nrows=2, ncols=3)
    ```

    2. Using mosaic:
    Creates a layout with custom subplot arrangements and spanning
    ```python
    import daspi as dsp
    # Creates a layout with spanning cells
    layout = (
        'AAB',
        'CDB')
    facets = dsp.AxesFacets(mosaic=layout)
    ```
    
    3. Create a seaborn-style jointplot layout with custom size ratios:
    ```python
    import daspi as dsp
    # Creates a layout with main scatter plot and marginal distributions
    layout = [
        ['hist_x', '.'],      # '.' creates an empty/blank Axes
        ['scatter', 'hist_y']]
    facets = AxesFacets(
        mosaic=layout,
        width_ratios=[4, 1],   # Make the marginal y-hist narrower
        height_ratios=[1, 4]   # Make the marginal x-hist shorter
        )
    ```

    This creates a figure with:
    - A main scatter plot in the bottom-left
    - A marginal histogram on top for x-distribution
    - A marginal histogram on right for y-distribution
    - Top-right cell is automatically empty using the '.' notation
    - Proportional spacing using width and height ratios

    The '.' character is a special notation in matplotlib's mosaic layout 
    that automatically creates an empty/invisible Axes, which is more 
    efficient than creating a visible Axes and then hiding it.
    """

    figsize: Tuple[float, float]
    """The figsize passed when creating the subplots."""
    figure: Figure
    """The figure instance to label."""
    axes: NDArray
    """A 2D array containing the Axes instances of the figure."""
    _nrows: int
    """Number of Axes rows in the grid"""
    _ncols: int
    """Number of Axes columns in the grid"""
    _sharex: ShareAxisProperty
    """Controls sharing of properties along the x-axis."""
    _sharey: ShareAxisProperty
    """Controls sharing of properties along the y-axis."""
    _ax: Axes | None
    """The current axes being worked on"""
    _default: Axes | None
    """The default Axes instance after iterating over this class. It is 
    the first Axes object in the grid if there is only one, otherwise 
    None."""
    mosaic: HashableList | None
    """A visual layout of how the Axes are arranged labeled as strings."""
    mosaic_pattern = re.compile(r'[\s+]?(\S+)[\s+]?')

    def __init__(
            self,
            *,
            nrows: int | None = None,
            ncols: int | None = None,
            mosaic: MosaicLayout | None = None,
            sharex: ShareAxisProperty = 'none', 
            sharey: ShareAxisProperty = 'none', 
            width_ratios: Sequence[float] | None = None,
            height_ratios: Sequence[float] | None = None, 
            stretch_figsize: bool | float | Tuple[float, float] = False,
            **kwds
            ) -> None:
        assert not all(arg is not None for arg in (nrows, ncols, mosaic)), (
            'Either nrows and ncols or mosaic must be provided, but not both.')
        
        if isinstance(mosaic, str):
            mosaic = self.mosaic_pattern.findall(mosaic)
        if isinstance(mosaic, (list, tuple)):
            self.mosaic = [[c for c in r] for r in mosaic]
            self._nrows = len(self.mosaic)
            self._ncols = len(self.mosaic[0])
            assert all(self._ncols == len(row) for row in self.mosaic), (
                f'Received a non-rectangular grid for the {mosaic=}.')
            self._sharex = True if sharex in (True, 'all') else False
            self._sharey = True if sharey in (True, 'all') else False
        else:
            self.mosaic = None
            self._sharex = sharex
            self._sharey = sharey
            self._nrows = nrows if isinstance(nrows, int) else 1
            self._ncols = ncols if isinstance(ncols, int) else 1

        figsize = kwds.pop('figsize', plt.rcParams['figure.figsize'])
        
        if stretch_figsize is False:
            stretch_x = stretch_y = 1
        elif stretch_figsize is True:
            stretch_x = (1 + math.log(self._ncols, math.e))
            stretch_y = (1 + math.log(self._nrows, math.e))
        elif isinstance(stretch_figsize, (tuple, list)):
            stretch_x, stretch_y = stretch_figsize
        elif isinstance(stretch_figsize, (int, float)):
            stretch_x = stretch_y = stretch_figsize
        else:
            raise TypeError(f'{stretch_figsize=} is not supported.')
        self.figsize = (
            (stretch_x * figsize[0], stretch_y * figsize[1]))

        _kwds: Dict[str, Any] = dict(
            sharex=self._sharex,
            sharey=self._sharey, 
            figsize=self.figsize,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            layout='tight',
            ) | kwds
        if self.mosaic:
            self.figure, axes = plt.subplot_mosaic(mosaic=self.mosaic, **_kwds)
            self.axes = np.array(
                [[axes.get(key, None) for key in row] for row in self.mosaic])
        else:
            self.figure, self.axes = plt.subplots(
                nrows=self._nrows, ncols=self._ncols, squeeze=False, **_kwds)

        self._default = self.axes[0, 0] if self.shape == (1, 1) else None
        self._ax = self._default

    @property
    def ax(self) -> Axes | None:
        """Get the axes that is currently being worked on. This property
        is automatically kept current when iterating through this 
        class (read-only)."""
        return self._ax
    
    @property
    def nrows(self) -> int:
        """Get the number of Axes rows in the grid (read-only)."""
        return self._nrows
    
    @property
    def ncols(self) -> int:
        """Get the number of Axes columns in the grid (read-only)."""
        return self._ncols
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the axes array (read-only)."""
        return self.axes.shape

    @property
    def sharex(self) -> ShareAxisProperty:
        """Get the sharing of properties along the x-axis (read-only)."""
        return self._sharex

    @property
    def sharey(self) -> ShareAxisProperty:
        """Get the sharing of properties along the y-axis (read-only)."""
        return self._sharey
    
    @property
    def flat(self) -> List[Axes]:
        """Get a list of all the Axes in the grid.

        Returns
        -------
        List[Axes]
            A list of all the Axes in the grid.
        """
        return [ax for ax in flat_unique(self.axes) if isinstance(ax, Axes)]
    
    def __iter__(self) -> Generator[Axes, Self, None]:
        """Iterate over the axes in the grid.

        Returns
        -------
        Generator[Axes, Self, None]
            The generator object that yields the axes in the grid.
        """
        def ax_gen(self) -> Generator[Axes, Self, None]:
            for ax in self.flat:
                self._ax = ax
                yield ax
            self._ax = self._default
        return ax_gen(self)
    
    def __next__(self) -> Axes:
        """Get the next axes in the grid.

        Returns
        -------
        Axes
            The next axes in the grid."""
        return next(self)
    
    @overload
    def __getitem__(self, index: Tuple[slice, slice]) -> NDArray:...

    @overload
    def __getitem__(self, index: Tuple[int, slice]) -> NDArray:...

    @overload
    def __getitem__(self, index: Tuple[slice, int]) -> NDArray:...
    
    @overload
    def __getitem__(self, index: int | Tuple[int, int]) -> Axes:...
    
    def __getitem__(self, index: int | Tuple[int | slice, int | slice]) -> Any:
        """Get the axes at the specified index in the grid. If the index
        is a number, the axes are fetched from left to right and from 
        top to bottom. If two numbers are specified, the axis is fetched 
        directly in the grid and it is then also possible to fetch an 
        axis across multiple indices if the axis is spanned across 
        multiple columns or rows.

        Parameters
        ----------
        index : int | Tuple[int, int]
            The index of the axes in the grid.

        Returns
        -------
        Axes
            The axes at the given index in the grid.
        """
        if isinstance(index, tuple):
            return self.axes[index]
        return self.flat[index]

    def __len__(self) -> int:
        """Get the total number of axes in the grid.

        Returns
        -------
        int
            The total number of axes in the grid.
        """
        return len(self.flat)


class StripesFacets:
    """A class for creating location and spread width lines, 
    specification limits and/or confidence interval areas as stripes on 
    each Axes. The location and spread (and their confidence bands) 
    represent the data per axes.

    Parameters
    ----------
    target : ArrayLike
        The target data for the estimation.
    target_on_y : bool
        Whether the target data is on the y-axis or not.
    single_axes : bool
        Whether a single axis is used for the chart or whether the 
        stripes are used on multiple axes. This affects how the 
        predefined legend labels are handled. If True, the corresponding
        position values are added to the label.
    stripes : List[Type[Stripe]], optional
        Additional non-predefined stripes to be added to the chart, 
        by default [].
    mean : bool, optional
        Whether to include a mean line on the chart, by default False.
    median : bool, optional
        Whether to include a median line on the chart, by default False.
    control_limits : bool, optional
        Whether to include control limits on the chart, by default False.
    spec_limits : SpecLimits, optional
        The specification limits for the chart, by default SpecLimits().
    confidence : float, optional
        The confidence level for the confidence intervals,
        by default None.
    strategy : {'eval', 'fit', 'norm', 'data'}, optional
        Which strategy should be used to determine the control limits
        (process spread):
        - eval: The strategy is determined according to the given 
          evaluate function. If none is given, the internal evaluate
          method is used.
        - fit: First, the distribution that best represents the process
          data is searched for and then the agreed process spread is 
          calculated
        - norm: it is assumed that the data is subject to normal
          distribution. The variation tolerance is then calculated as 
          agreement * standard deviation
        - data: The quantiles for the process variation tolerance are
          read directly from the data. by default 'norm'
    
    agreement : float or int, optional
        Specify the tolerated process variation for which the control 
        limits are to be calculated.
        If int, the spread is determined using the normal distribution 
        agreementsigma, e.g. agreement = 6 -> 6sigma ~ covers 99.75 % of 
        the data. The upper and lower permissible quantiles are then 
        calculated from this.
        If float, the value must be between 0 and 1.This value is then 
        interpreted as the acceptable proportion for the spread, e.g. 
        0.9973 (which corresponds to ~ 6 sigma) by default 6
    **kwds : dict, optional
        Additional keyword arguments to pass to the ProcessEstimator.
    """

    __slots__ = (
        'estimation', '_confidence', 'single_axes', 'stripes', 'target_on_y')
    
    estimation: ProcessEstimator
    """The process estimator object."""
    _confidence: float | None
    """The confidence level for the confidence intervals."""
    single_axes: bool
    """Whether a single axis is used for the chart or whether the 
    stripes are used on multiple axes. This affects how the predefined
    legend labels are handled. If True, the corresponding position
    values are added to the label."""
    stripes: Dict[str, Stripe]
    """The stripes to plot on the chart."""
    target_on_y: bool
    """Whether the target is on the y-axis or the x-axis."""
    
    def __init__(
        self,
        target: NumericSample1D,
        *,
        target_on_y: bool,
        single_axes: bool,
        stripes: List[Stripe] = [], 
        mean: bool = False,
        median: bool = False,
        control_limits: bool = False,
        spec_limits: SpecLimits = SpecLimits(),
        confidence: float | None = None,
        strategy: Literal['eval', 'fit', 'norm', 'data'] = 'norm',
        agreement: float | int = 6,
        **kwds) -> None:
        self.single_axes = single_axes
        self._confidence = confidence
        self.estimation = ProcessEstimator(
            samples=target,
            spec_limits=spec_limits,
            strategy=strategy,
            agreement=agreement,
            **kwds)
        self.target_on_y = target_on_y
        self.stripes = {s.identity: s for s in stripes}
        add_confidence_span = confidence is not None
        
        self.add_specification_limit_stripes()
        if mean:
            self.add_mean_stripes(add_confidence_span)
        if median:
            self.add_median_stripes(add_confidence_span)
        if control_limits:
            self.add_control_limit_stripes(add_confidence_span)
        
    @property
    def confidence(self) -> float:
        """Get the confidence level for the confidence intervals
        (read-only)."""
        if self._confidence is None:
            return DEFAULT.CONFIDENCE_LEVEL
        return self._confidence
    
    @property
    def orientation(self) -> Literal['horizontal', 'vertical']:
        """The orientation of the chart."""
        return 'horizontal' if self.target_on_y else 'vertical'
    
    @property
    def _kwds(self) -> Dict[str, Any]:
        kwds = dict(
            orientation=self.orientation,
            show_position=self.single_axes)
        return kwds

    @property
    def _confidence_label(self) -> str:
        """The lable used for confidence stripes."""
        if self._confidence is not None:
            label = f'{100*self._confidence:.0f} \\%-{STR["ci"]}'
        else:
            label = ''
        return label
    
    def add_mean_stripes(self, add_confidence_span: bool) -> None:
        """Add mean stripes to the plot.

        This method adds a stripe representing the mean of the target 
        data, using the `StripeLine` class. If `add_confidence_span` is 
        `True`, additional stripes representing the confidence interval 
        for the mean will also be added using the `StripeSpan` class.

        Parameters
        ----------
        add_confidence_span : bool
            Whether to add confidence stripes for the mean.
        """
        line = StripeLine(
            label=r'\bar x',
            position=self.estimation.mean,
            **(self._kwds | KW.MEAN_LINE))
        self.stripes[line.identity] = line

        if add_confidence_span:
            low, upp = self.estimation.mean_ci(self.confidence)
            span = StripeSpan(
                label=self._confidence_label,
                lower_position=low,
                upper_position=upp,
                color=line.color,
                **(self._kwds | KW.STRIPES_CONFIDENCE))
            self.stripes[f'{span.identity}_mean'] = span

    def add_median_stripes(self, add_confidence_span: bool) -> None:
        """Add median stripes to the plot.

        This method adds a stripe representing the median of the target 
        data, using the `StripeLine` class. If `add_confidence_span` is 
        `True`, additional stripes representing the confidence interval 
        for the median will also be added using the `StripeSpan` class.

        Parameters
        ----------
        add_confidence_span : bool
            Whether to add confidence stripes for the median.
        """
        line = StripeLine(
            label=r'\tilde x',
            position=self.estimation.median,
            **(self._kwds | KW.MEDIAN_LINE))
        self.stripes[line.identity] = line

        if add_confidence_span:
            low, upp = self.estimation.median_ci(self.confidence)
            span = StripeSpan(
                label=self._confidence_label,
                lower_position=low,
                upper_position=upp,
                color=line.color,
                **(self._kwds | KW.STRIPES_CONFIDENCE))
            self.stripes[f'{span.identity}_median'] = span
    
    def add_control_limit_stripes(self, add_confidence_span: bool) -> None:
        """Add control limit stripes to the plot.
        
        This method adds stripes representing the lower and upper 
        control limits for the target data.
        
        The stripes are added using the `StripeLine` class, with the 
        label and position set based on the control limits from the 
        `estimation` attribute. If `add_confidence_span` is `True`,
        additional stripes representing the confidence interval for the
        control limits will also be added using the `StripeSpan` class.
        
        Parameters
        ----------
        add_confidence_span : bool
            Whether to add confidence stripes for the control limits.
        """
                
        kw_line = self._kwds | KW.CONTROL_LINE
        line_low = StripeLine(
            label=r'x_{' + f'{self.estimation.q_low:.4f}' + '}',
            position=self.estimation.lcl,
            **kw_line)
        line_upp = StripeLine(
            label=r'x_{' + f'{self.estimation.q_upp:.4f}' + '}',
            position=self.estimation.ucl,
            **kw_line)
        self.stripes[line_low.identity] = line_low
        self.stripes[line_upp.identity] = line_upp
        
        if add_confidence_span:
            kw_span = self._kwds | KW.STRIPES_CONFIDENCE
            low, upp = self.estimation.stdev_ci(self.confidence)
            span_low = StripeSpan(
                label=self._confidence_label,
                position=self.estimation.lcl,
                width=upp - low,
                color=line_low.color,
                **kw_span)
            span_upp = StripeSpan(
                label=self._confidence_label,
                position=self.estimation.ucl,
                width=upp - low,
                color=line_upp.color,
                **kw_span)
            self.stripes[f'{span_low.identity}_low'] = span_low
            self.stripes[f'{span_upp.identity}_upp'] = span_upp
    
    def add_specification_limit_stripes(self) -> None:
        """Add specification limit stripes to the plot.
        
        This method adds stripes representing the lower and upper 
        specification limits to the plot. If a specification limit is
        `None`, it will be skipped.
        
        The stripes are added using the `StripeLine` class, with the 
        label and position set based on the specification limits from
        the `estimation` attribute.
        """
        kwds = self._kwds | KW.SPECIFICATION_LINE
        _labels = (STR['lsl'], STR['usl'])
        _limits = self.estimation.spec_limits.to_tuple()
        for label, limit in zip(_labels, _limits):
            if float('-inf') < limit < float('inf'):
                line = StripeLine(label=str(label), position=limit, **kwds)
                self.stripes[line.identity] = line
    
    def handles_labels(self) -> LegendHandlesLabels:
        """Get the legend handles and labels for the plot.
        
        This method processes the `self.stripes` dictionary to extract
        the legend handles and labels for the plot. It separates the 
        handles and labels for line stripes and span stripes, and 
        returns them as a tuple of handles and a tuple of labels.
        
        Returns
        -------
        LegendHandlesLabels:
            A tuple containing the legend handles and labels.
        """
        handles_labels_lines = {}
        handles_labels_spans = {}
        processed = set()
        for stripe in self.stripes.values():
            if stripe.identity in processed:
                continue
            if isinstance(stripe, StripeLine):
                handles_labels_lines[stripe.label] = stripe.handle
            else:
                handles_labels_spans[stripe.label] = stripe.handle
            processed.add(stripe.identity)
        handles_labels = handles_labels_lines | handles_labels_spans
        return tuple(handles_labels.values()), tuple(handles_labels.keys())

    def draw(self, ax: Axes) -> None:
        """Draw the stripes on the specified Axes.

        Parameters
        ----------
        ax : Axes
            The Axes on which to draw the stripes.
        """
        for stripe in self.stripes.values():
            stripe(ax)


class LabelFacets:
    """
    A class for adding labels and titles to facets of a figure with 
    advanced formatting.

    This class provides comprehensive labeling and formatting
    capabilities for matplotlib figures, including support for custom
    formatters, label  rotation, alignment control, and automatic margin
    adjustment for optimal layout.

    Parameters
    ----------
    axes : AxesFacets
        A AxesFacets instance containing the subplots' Axes and their 
        arrangement.
    fig_title : str, optional
        Main title that should be displayed at the top of the chart,
        by default ''.
    sub_title : str, optional
        Subtitle, which should appear directly below the main title and
        slightly smaller than it, by default ''.
    xlabel, ylabel: str or Tuple[str, ...], optional
        The axis label(s) of the figure. To label multiple axes with 
        different names, provide a tuple; otherwise, provide a string,
        by default ''.
    xlabel_formatter, ylabel_formatter : Formatter | Callable | str | None, optional
        Advanced formatters for axis tick labels with multiple input 
        types:
        
        - **String format templates**: Simple format strings using 
          Python's string formatting syntax (e.g., '{:.2f}', '{:.1e}',
          '${:.0f}', '{:.1%}') for quick and intuitive formatting
        - **Callable functions**: Custom functions that take one
          argument (value) or two arguments (value, position) and return 
          formatted strings for complete control
        - **Matplotlib Formatters**: Any matplotlib.ticker.Formatter
          instance for advanced formatting scenarios
        - **None**: Use matplotlib's default formatting
        
        The formatter automatically handles matplotlib compatibility by
        wrapping simple callables in FuncFormatter and converting string
        templates to appropriate formatter types. By default, None.
        
    xlabel_angle, ylabel_angle : float, optional
        Rotation angle for x and y axis tick labels in degrees. Positive
        values rotate counter-clockwise, negative values rotate
        clockwise. The chart automatically adjusts margins to
        accommodate rotated labels and prevent clipping. Common values: 
        0 (horizontal), 45 (diagonal), 90 (vertical). By default, 0.
        
    xlabel_align, ylabel_align : str, optional
        Alignment for x and y axis tick labels relative to their tick 
        marks:
        
        - **For x-axis**: 'left', 'center', 'right' control horizontal
          alignment of the label text relative to the tick position
        - **For y-axis**: 'bottom', 'center', 'top' control vertical
          alignment, where the string values map to vertical alignment
          for better API consistency
          
        Alignment is particularly useful with rotated labels to achieve
        optimal positioning and readability. By default, 'center'.
    
    info : bool or str, optional
        Indicates whether to include an info text at the lower left 
        corner in the figure. If True, the date and user are
        automatically  added. If a string is provided, it's appended to
        the automatic date/user information, separated by a comma.
        By default, False.
    rows: Tuple[str, ...], optional
        The row labels of the figure for faceted plots, by default ().
    cols: Tuple[str, ...], optional
        The column labels of the figure for faceted plots,
        by default ().
    row_title : str, optional
        The title of the rows for faceted plots, by default ''.
    col_title : str, optional
        The title of the columns for faceted plots, by default ''.
    axes_titles : Tuple[str, ...]
        Title for each Axes, useful for JointCharts with multiple
        subplots, by default ().
    legend_data : Dict[str, LegendHandlesLabels], optional
        The legends to be added to the figure. The key is used as the 
        legend title, and the values must be a tuple of tuples, where
        the inner tuple contains a handle as a Patch or Line2D artist
        and a label as a string, by default {}.

    Examples
    --------
    Basic usage with string formatters:
    
    ```python
    axes = AxesFacets(nrows=1, ncols=1)
    label_facets = LabelFacets(
        axes=axes,
        fig_title='Temperature Analysis',
        xlabel='Time (hours)',
        ylabel='Temperature',
        ylabel_formatter='{:.1f}°C',  # String template
        xlabel_formatter='{:.0f}h'    # String template
    )
    label_facets.draw()
    ```
    
    Custom callable formatters:
    
    ```python
    def scientific_formatter(value):
        if abs(value) >= 1000:
            return f'{value:.1e}'
        return f'{value:.2f}'
    
    label_facets = LabelFacets(
        axes=axes,
        ylabel_formatter=scientific_formatter,
        xlabel_formatter=lambda x: f'${x:,.0f}'  # Lambda formatter
    )
    ```
    
    Rotation and alignment for readability:
    
    ```python
    label_facets = LabelFacets(
        axes=axes,
        xlabel='Product Categories',
        xlabel_angle=45,      # Diagonal rotation
        xlabel_align='right', # Right-align for better rotation appearance
        ylabel_formatter='{:.1%}'  # Percentage format
    )
    ```
    
    Advanced faceted plot labeling:
    
    ```python
    label_facets = LabelFacets(
        axes=multi_axes,
        fig_title='Sales Analysis by Region',
        sub_title='Q3 2024 Performance',
        row_title='Geographic Region',
        col_title='Product Category',
        rows=('North', 'South', 'East', 'West'),
        cols=('Electronics', 'Clothing', 'Books'),
        xlabel_formatter='${:,.0f}',
        ylabel_formatter='{:.1f}%',
        info='Source: Sales database'
    )
    ```

    Notes
    -----
    - String formatters are automatically converted to 
      matplotlib-compatible formatters using intelligent type detection.
    - Margins are automatically adjusted when using rotated labels to
      prevent clipping and ensure optimal layout.
    - The class handles both single axes and complex multi-axes layouts.
    - All formatting options work consistently across different
      matplotlib backends and figure configurations.
    - Legend positioning is automatically calculated to avoid
      overlapping with other chart elements.
    """

    axes: AxesFacets
    """A AxesFacets instance containing the subplots' Axes and their 
    arrangement."""
    figure: Figure
    """The figure instance to label."""
    fig_title: str
    """The title to display at the top of the chart."""
    sub_title: str
    """The subtitle to display directly below the title of the chart."""
    xlabel: str | Tuple[str, ...]
    """The x-axis label(s) of the figure."""
    ylabel: str | Tuple[str, ...]
    """The y-axis label(s) of the figure."""
    xlabel_formatter: Callable | Formatter | None
    """Function to format the x-axis tick labels."""
    ylabel_formatter: Callable | Formatter | None
    """Function to format the y-axis tick labels."""
    xlabel_angle: float
    """Rotation angle for x-axis tick labels in degrees."""
    ylabel_angle: float
    """Rotation angle for y-axis tick labels in degrees."""
    xlabel_align: Literal['left', 'center', 'right']
    """Alignment for x-axis tick labels."""
    ylabel_align: Literal['bottom', 'center', 'top']
    """Alignment for y-axis tick labels."""
    info: bool | str
    """Indicates whether to include an info text in the figure."""
    rows: Tuple[str, ...]
    """The row labels of the figure."""
    cols: Tuple[str, ...]
    """The column labels of the figure."""
    row_title: str
    """The title of the rows."""
    col_title: str
    """The title of the columns."""
    axes_titles: Tuple[str, ...]
    """The titles of each axes."""
    legend_data: Dict[str, LegendHandlesLabels]
    """The legend_data to be added to the figure."""
    _legend: Legend | None
    """Figure legend if one is added"""
    _size: Tuple[int, int]
    """The size of the figure in pixels."""
    _margin: Dict[str, int]
    """The required margin for the labels of the image in pixels."""
    labels: Dict[str, Text]
    """The labels that were added to the image as text objects."""

    def __init__(
            self,
            axes: AxesFacets,
            *,
            fig_title: str = '', 
            sub_title: str = '',
            xlabel: str | Tuple[str, ...] = '',
            ylabel: str | Tuple[str, ...] = '',
            xlabel_formatter: Callable | None = None,
            ylabel_formatter: Callable | None = None,
            xlabel_angle: float = 0,
            ylabel_angle: float = 0,
            xlabel_align: Literal['left', 'center', 'right'] = 'center',
            ylabel_align: Literal['bottom', 'center', 'top'] = 'center',
            info: bool | str = False,
            rows: Tuple[str, ...] = (),
            cols: Tuple[str, ...] = (),
            row_title: str = '',
            col_title: str = '',
            axes_titles: Tuple[str, ...] = (),
            legend_data: Dict[str, LegendHandlesLabels] = {}
            ) -> None:
        self.axes = axes
        self.figure = self.axes.figure
        self.fig_title = fig_title
        self.sub_title = sub_title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlabel_formatter = self._prepare_formatter(xlabel_formatter)
        self.ylabel_formatter = self._prepare_formatter(ylabel_formatter)
        self.xlabel_angle = xlabel_angle
        self.ylabel_angle = ylabel_angle
        self.xlabel_align = xlabel_align
        self.ylabel_align = ylabel_align
        self.rows = rows
        self.cols = cols
        self.row_title = row_title
        self.col_title = col_title
        self.axes_titles = axes_titles
        self.info = info
        self.legend_data = legend_data
        self._legend = None
        self._size = (
            int(self.figure.get_figwidth() * self.figure.dpi),
            int(self.figure.get_figheight() * self.figure.dpi))
        self._margin = dict(left=0, bottom=0, right=0, top=0)
        self.labels = {}
    
    def _prepare_formatter(
            self,
            formatter: Formatter | Callable | str | None
            ) -> Formatter | None:
        """Prepare a formatter for use with matplotlib.
        
        If the formatter is a callable that takes only one argument,
        wrap it in a FuncFormatter. If it's already a Formatter or
        takes two arguments, use it as-is. If it's a string, convert it
        to a FuncFormatter for robust formatting.
        
        Parameters
        ----------
        formatter : Formatter | Callable | str | None
            The formatter to prepare. String formatters can use either
            positional format (e.g., '{:.2e}') or keyword format 
            (e.g., '{x:.2e}').
            
        Returns
        -------
        Formatter | None
            A matplotlib-compatible formatter or None
        """
        if formatter is None:
            return None
        
        if isinstance(formatter, Formatter):
            return formatter
        
        if isinstance(formatter, str):
            # Convert string formatters to FuncFormatter for robust handling
            # This handles both positional '{:.2e}' and keyword '{x:.2e}' formats
            if '{x' in formatter.lower() or '{pos' in formatter.lower():
                # Keyword format - use StrMethodFormatter
                return StrMethodFormatter(formatter)
            else:
                # Positional format - use FuncFormatter with string formatting
                return FuncFormatter(lambda x, pos, fmt=formatter: fmt.format(x))

        if callable(formatter):
            # Check if the callable accepts only one argument
            import inspect
            sig = inspect.signature(formatter)
            if len(sig.parameters) == 1:
                # Wrap single-argument callable in FuncFormatter
                return FuncFormatter(lambda x, pos, f=formatter: f(x))
            else:
                # Assume it's already a two-argument callable
                return FuncFormatter(formatter)
        
        return None
    
    @property
    def x_aligned(self) -> float:
        """Get aligned x position as fraction of figure width. This is
        used for title, subtitle and info (read-only)."""
        return KW._margin + LABEL.X_ALIGNEMENT / self._size[0]

    @property
    def margin_rectangle(self) -> Tuple[float, float, float, float]:
        """Get rectangle of margins around the subplots used for the
        additional labels as fraction of figure size (read-only)."""
        margins = (
            self._margin['left'] / self._size[0],
            self._margin['bottom'] / self._size[1],
            max(1 - (self._margin['right'] / self._size[0]), 0.5),
            max(1 - (self._margin['top'] / self._size[1]), 0.5))
        return margins
    
    @property
    def legend_fraction(self) -> float:
        """Get fraction of figure size for legend (read-only)."""
        return self.legend_width / self.figure.get_figwidth()

    @staticmethod
    def estimate_height(text: Text) -> int:
        """Get the estimated size of the text in the figure in pixels."""
        dpi = text.figure.dpi if text.figure else plt.rcParams['figure.dpi']
        return int(int(text.get_fontsize()) * LABEL.PPI * dpi) + LABEL.PADDING
    
    @staticmethod
    def estimate_rotation_margin(angle_degrees: float, base_margin: int = 25) -> int:
        """Estimate additional margin needed for rotated tick labels.
        
        Parameters
        ----------
        angle_degrees : float
            Rotation angle in degrees
        base_margin : int, optional
            Base margin for calculating rotated label space, by default 25
            
        Returns
        -------
        int
            Additional margin in pixels needed for the rotation
        """
        if abs(angle_degrees) < 5:  # No significant rotation
            return 0
        
        angle_rad = abs(angle_degrees) * np.pi / 180
        # Use sine to estimate the additional perpendicular space needed
        return int(base_margin * np.sin(angle_rad))
    
    @staticmethod
    def get_legend_artists(legend: Legend) -> List[Artist]:
        """Get the inner children of a legend.

        Parameters
        ----------
        legend : Legend
            The legend object.

        Returns
        -------
        List[Artist]:
            The artists representing the inner children of the legend.
        """
        return legend.get_children()[0].get_children()

    @property
    def legend(self) -> Legend | None:
        """Get legend added to figure (read-only)."""
        return self._legend
    
    @property
    def legend_artists(self) -> List[Artist]:
        """Get legend artists (read-only)."""
        if self.legend is None:
            return []
        return self.get_legend_artists(self.legend)
    
    @property
    def legend_width(self) -> int:
        """Get width of legend in pixels (read-only)."""
        if self.legend is None:
            return 0
        return int(self.legend.get_window_extent().width)

    def _add_legend(
            self, handles: Tuple[Patch |Line2D, ...], labels: Tuple[str, ...],
            title: str) -> None:
        """Adds a legend at the right side of the figure. If there is 
        already one, the existing one is extended with the new one
        
        Parameters
        ----------
        handles: Tuple[Patch | Line2D, ...]
            A list of Artists (lines, patches) to be added to the
            legend.
        labels : Tuple[str, ...]
            The labels must be in the same order as the corresponding 
            plots were drawn. If no labels are given, the handles and 
            labels of the first axes are used.
        title : str
            Title for the given handles and labels. 
        """
        legend = Legend(
            self.figure, handles, labels, title=title, **KW.LEGEND)
        
        if not self.legend:
            self.figure.legends.append(legend) # type: ignore
            self._legend = legend
        else:
            new_artists = self.get_legend_artists(legend)
            self.legend_artists.extend(new_artists)
    
    @staticmethod
    def not_shared(axis) -> bool:
        """Check if an axis is not shared."""
        return len(axis._get_shared_axis()) == 1
    
    def _add_axes_titles(self) -> None:
        """Add the axes titles to the figure. If cols are set, the
        axes titles are not drawn."""
        if not self.axes_titles:
            return
        
        if self.cols:
            warnings.warn('Drawing axes_titles ignored, because cols are set.')
            return

        for ax, title in zip(self.axes.flat, self.axes_titles):
            ax.set_title(title)

    def _apply_axis_formatting(self) -> None:
        """Apply formatters and rotation to all axes before calculating margins.
        This ensures that margin calculations account for the space needed by 
        rotated or formatted labels."""
        for ax in self.axes.flat:
            # Apply formatters
            if self.xlabel_formatter is not None:
                ax.xaxis.set_major_formatter(self.xlabel_formatter)
            if self.ylabel_formatter is not None:
                ax.yaxis.set_major_formatter(self.ylabel_formatter)
            
            # Apply rotation and alignment
            if self.xlabel_angle != 0 or self.xlabel_align != 'center':
                ax.tick_params(axis='x', rotation=self.xlabel_angle)
                # Apply horizontal alignment for x-axis labels
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment(self.xlabel_align)
                    
            if self.ylabel_angle != 0 or self.ylabel_align != 'center':
                ax.tick_params(axis='y', rotation=self.ylabel_angle)
                # Apply vertical alignment for y-axis labels
                for label in ax.get_yticklabels():
                    label.set_verticalalignment(self.ylabel_align)

    def _add_left_labels(self) -> None:
        """Add the ylabel or ylabs. If only one is present, a label will
        be added centrally for all axes. Otherwise, a label will be added
        for all axes that do not have a shared axes."""
        self._margin['left'] = 0

        if self.ylabel:
            if isinstance(self.ylabel, str):
                _kwds = KW.YLABEL
                _kwds['x'] += self._margin['left'] / self._size[0]
                _text = self.figure.text(s=self.ylabel, **_kwds)
                self._margin['left'] += self.estimate_height(_text)
                self.labels['ylabel'] = _text
            else:
                for ax, ylabel in zip(self.axes.flat, self.ylabel):
                    if self.not_shared(ax.yaxis) or ax in self.axes[:, 0]:
                        ax.set(ylabel=ylabel)
        
        # Add extra margin for rotated y-axis labels
        if self.ylabel_angle != 0:
            extra_margin = self.estimate_rotation_margin(self.ylabel_angle, base_margin=20)
            self._margin['left'] += extra_margin

    def _add_bottom_labels(self) -> None:
        """Insert the xlabel or xlabs. If only one is present, a label 
        will be added centrally for all axes. Otherwise, a label will be 
        added for all axes that do not have a shared axes.
        Also insert an info text in the bottom left-hand corner of the 
        figure. By default, the info text contains today's date and the 
        user name. If attribute `info` is a string, it is added to the 
        info text separated by a comma."""
        self._margin['bottom'] = 0

        if self.info:
            _kwds = KW.INFO | dict(x=self.x_aligned)
            info_text = f'{STR.TODAY} {STR.USERNAME}'
            if isinstance(self.info, str):
                info_text = f'{info_text}, {self.info}'
            _text = self.figure.text(s=info_text, **_kwds)
            self._margin['bottom'] += self.estimate_height(_text)
            self.labels['info'] = _text
        
        if self.xlabel:
            if isinstance(self.xlabel, str):
                _kwds = KW.XLABEL
                _kwds['y'] += self._margin['bottom'] / self._size[1]
                _text = self.figure.text(s=self.xlabel, **_kwds)
                self._margin['bottom'] += self.estimate_height(_text)
                self.labels['xlabel'] = _text
            else:
                for ax, xlabel in zip(self.axes.flat, self.xlabel):
                    if self.not_shared(ax.xaxis) or ax in self.axes[-1, :]: 
                        ax.set(xlabel=xlabel)
        
        # Add extra margin for rotated x-axis labels
        if self.xlabel_angle != 0:
            extra_margin = self.estimate_rotation_margin(self.xlabel_angle, base_margin=30)
            self._margin['bottom'] += extra_margin

    def _add_right_labels(self) -> None:
        """Add row labels and row title to the figure. The title is 
        added to figure object, the rows to the axes objects."""
        self._margin['right'] = self.legend_width
        if self.legend_width > 0:
            self._margin['right'] += 2 * LABEL.PADDING

        if self.row_title:
            _kwds = KW.ROW_TITLE
            _kwds['x'] -= self._margin['right'] / self._size[0]
            _text = self.figure.text(s=self.row_title, **_kwds)
            self._margin['right'] += self.estimate_height(_text)
            self.labels['row_title'] = _text

        if self.rows:
            for ax, label in zip(self.axes[:, -1], self.rows):
                _kwds = KW.ROW_LABEL | {'transform': ax.transAxes}
                _text = ax.text(s=label, **_kwds)
                self.labels[f'row_{label}'] = _text

    def _add_top_labels(self) -> None:
        """Add all provided labels at the top of the charts. These 
        labels include the figure title, sub title, column title and 
        column labels."""
        self._margin['top'] = 0

        if self.fig_title:
            _kwds = KW.FIG_TITLE | dict(x=self.x_aligned)
            _text = self.figure.text(s=self.fig_title, **_kwds)
            self._margin['top'] += self.estimate_height(_text) + LABEL.PADDING
            self.labels['fig_title'] = _text

        if self.sub_title:
            _kwds = KW.SUB_TITLE | dict(x=self.x_aligned)
            _kwds['y'] -= self._margin['top'] / self._size[1]
            _text = self.figure.text(s=self.sub_title, **_kwds)
            self._margin['top'] += self.estimate_height(_text) + LABEL.PADDING
            self.labels['sub_title'] = _text
        
        if self.col_title:
            _kwds = KW.COL_TITLE
            _kwds['y'] -= self._margin['top'] / self._size[1]
            _text = self.figure.text(s=self.col_title, **_kwds)
            self._margin['top'] += self.estimate_height(_text)
            self.labels['col_title'] = _text

        if self.cols:
            for ax, label in zip(self.axes[0, :], self.cols):
                _kwds = KW.COL_LABEL | {'transform': ax.transAxes}
                _text = ax.text(s=label, **_kwds)
                self.labels[f'col_{label}'] = _text
    
    def _remove_last_labelpad(self) -> None:
        """Remove the padding added for last added labels."""
        for pos, margin in self._margin.items():
            if margin > LABEL.PADDING:
                self._margin[pos] = margin - LABEL.PADDING
    
    def _adjust_centered_labels(self) -> None:
        """Adjust the position of all centered labels according to the 
        current margin settings."""
        lr_adjustment = (
            (self._margin['left'] - self._margin['right']) / self._size[0] / 2)
        bt_adjustment = (
            (self._margin['bottom'] - self._margin['top']) / self._size[1] / 2)
        if 'xlabel' in self.labels:
            self.labels['xlabel'].set_x(KW.XLABEL['x'] + lr_adjustment)
        if 'ylabel' in self.labels:
            self.labels['ylabel'].set_y(KW.YLABEL['y'] + bt_adjustment)
        if 'col_title' in self.labels:
            self.labels['col_title'].set_x(KW.COL_TITLE['x'] + lr_adjustment)
        if 'row_title' in self.labels:
            self.labels['row_title'].set_y(KW.ROW_TITLE['y'] + bt_adjustment)
        if self.legend:
            bbox = self.legend.get_bbox_to_anchor()
            self.legend.set_bbox_to_anchor((
                bbox.x1 / self._size[0],
                (bbox.y1 - self._margin['top']) / self._size[1]))
    
    # TODO: this does not work as expected
    def clear(self) -> None:
        """Remove all the label facets."""
        for label in self.labels.values():
            label.remove()
            if label in self.figure.texts:
                self.figure.texts.remove(label)
            del(label)
        self.labels = {}

        if self.legend:
            self.legend.remove()
            if self.legend in self.figure.legends:
                self.figure.legends.remove(self.legend)
            self._legend = None
        
        self._margin = dict(left=0, bottom=0, right=0, top=0)
    
    def draw(self) -> None:
        """Draw all the label facets to the figure."""
        self.clear()

        # Apply axis formatting first, before calculating margins
        self._apply_axis_formatting()
        
        self._add_axes_titles()
        for title, (handles, labels) in self.legend_data.items():
            self._add_legend(handles, labels, title)
        self._add_left_labels()
        self._add_bottom_labels()
        self._add_right_labels()
        self._add_top_labels()

        self._remove_last_labelpad()
        self._adjust_centered_labels()
            
        self.figure.tight_layout(rect=self.margin_rectangle)
