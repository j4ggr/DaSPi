import math
import pandas as pd
import matplotlib.pyplot as plt

from typing import Self
from typing import List
from typing import Dict
from typing import Tuple
from typing import Callable
from typing import Sequence
from typing import Generator
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.artist import Artist
from matplotlib.patches import Patch

from .._typing import SpecLimit
from .._typing import ShareAxisProperty
from .._typing import NumericSample1D
from .._typing import LegendHandlesLabels

from ..strings import STR
from ..constants import KW
from ..constants import LABEL
from ..constants import DEFAULT
from ..statistics.estimation import ProcessEstimator


class LabelFacets:
    """
    A class for adding labels and titles to facets of a figure.

    Parameters
    ----------
    figure : Figure
        The figure to label.
    axes : 2DArray
        A 2D array of Axes instances. Setting squeeze=False when using 
        the `plt.subplots` method ensures that it is always a 2D array.
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
    info : bool or str, optional
        Indicates whether to include an info text at the lower left 
        corner in the figure. The date and user are automatically added,
        by default False.
    rows: Tuple[str, ...], optional
        The row labels of the figure, by default ().
    cols: Tuple[str, ...], optional
        The column labels of the figure, by default ().
    row_title : str, optional
        The title of the rows, by default ''.
    col_title : str, optional
        The title of the columns, by default ''.
    axes_titles : Tuple[str, ...]
        Title for each Axes, usefull for JointCharts, by default ()
    legend_data : Dict[str, LegendHandlesLabels], optional
        The legends to be added to the figure. The key is used as the 
        legend title, and the values must be a tuple of tuples, where
        the inner tuple contains a handle as a Patch or Line2D artist
        and a label as a string, by default {}.
    """

    figure: Figure
    """The figure instance to label."""
    axes: NDArray
    """A 2D array containing the Axes instances of the figure."""
    fig_title: str
    """The title to display at the top of the chart."""
    sub_title: str
    """The subtitle to display directly below the title of the chart."""
    xlabel: str | Tuple[str, ...]
    """The x-axis label(s) of the figure."""
    ylabel: str | Tuple[str, ...]
    """The y-axis label(s) of the figure."""
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

    def __init__(
            self,
            figure: Figure,
            axes: NDArray,
            fig_title: str = '', 
            sub_title: str = '',
            xlabel: str | Tuple[str, ...] = '',
            ylabel: str | Tuple[str, ...] = '',
            info: bool | str = False,
            rows: Tuple[str, ...] = (),
            cols: Tuple[str, ...] = (),
            row_title: str = '',
            col_title: str = '',
            axes_titles: Tuple[str, ...] = (),
            legend_data: Dict[str, LegendHandlesLabels] = {}
            ) -> None:
        self.figure = figure
        self.axes = axes
        self.fig_title = fig_title
        self.sub_title = sub_title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.rows = rows
        self.cols = cols
        self.row_title = row_title
        self.col_title = col_title
        self.axes_titles = axes_titles
        self.info = info
        self.legend_data = legend_data
        self._legend = None
    
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
    def shift_text_y(self) -> float:
        """Get offset to move text based on the fig height (read-only)."""
        return LABEL.SHIFT_BASE / self.figure.get_figheight()

    @property
    def shift_text_x(self) -> float:
        """Get offset to move text based on the fig width (read-only)."""
        return LABEL.SHIFT_BASE / self.figure.get_figwidth()

    @property
    def shift_fig_title(self) -> float:
        """Get offset in y direction for fig title (read-only)."""
        labels = (self.axes_titles, self.col_title, self.sub_title)
        n = (LABEL.AXES_PADDING
             + sum(map(bool, labels)))
        return n * self.shift_text_y

    @property
    def shift_sub_title(self) -> float:
        """Get offset in y direction for sub title (read-only)."""
        labels = (self.col_title, self.axes_titles)
        n = (LABEL.AXES_PADDING
             + LABEL.LABEL_PADDING * int(any(map(bool, labels)))
             + sum(map(bool, labels)))
        return n * self.shift_text_y

    @property
    def shift_legend(self) -> float:
        """Get offset in x direction for legend (read-only)."""
        labels = (self.row_title, self.rows)
        n = (LABEL.AXES_PADDING 
             + LABEL.LABEL_PADDING * sum(map(bool, labels))
             + sum(map(bool, labels)))
        return n * self.shift_text_x

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

    def add_legend(
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
        kw = KW.LEGEND
        bbox = kw['bbox_to_anchor']
        kw['bbox_to_anchor'] = (bbox[0] + self.shift_legend, bbox[1])
        legend = Legend(
            self.axes[0, -1], handles, labels, title=title, **kw)
        if not self.legend:
            self.figure.legends.append(legend) # type: ignore
            self._legend = legend
        else:
            new_artists = self.get_legend_artists(legend)
            self.legend_artists.extend(new_artists)

    def add_xlabel(self) -> None:
        """Add x-axis label(s) to the figure."""
        if not self.xlabel:
            return
        if isinstance(self.xlabel, str):
            kw = KW.XLABEL
            kw['y'] = kw['y'] - LABEL.AXES_PADDING * self.shift_text_y
            self.figure.text(s=self.xlabel, **kw)
        else:
            for ax, xlabel in zip(self.axes.flat, self.xlabel):
                if (len(ax.xaxis._get_shared_axis()) == 1
                    or ax in self.axes[-1]): 
                    ax.set(xlabel=xlabel)

    def add_ylabel(self) -> None:
        """Add y-axis label(s) to the figure."""
        if not self.ylabel:
            return
        if isinstance(self.ylabel, str):
            kw = KW.YLABEL
            kw['x'] = kw['x'] - LABEL.AXES_PADDING * self.shift_text_x
            self.figure.text(s=self.ylabel, **kw)
        else:
            for ax, ylabel in zip(self.axes.flat, self.ylabel):
                if (len(ax.yaxis._get_shared_axis()) == 1
                    or ax in self.axes.T[0]): 
                    ax.set(ylabel=ylabel)

    def add_row_labels(self) -> None:
        """Add row labels and row title to the figure."""
        if not self.rows:
            return
        for axs, label in zip(self.axes, self.rows):
            ax = axs[-1]
            kwds = KW.ROW_LABEL | {'transform': ax.transAxes}
            ax.text(s=label, **kwds)
        self.figure.text(s=self.row_title, **KW.ROW_TITLE)
    
    def add_col_labels(self) -> None:
        """Add column labels and column title to the figure."""
        if not self.cols:
            return
        for ax, label in zip(self.axes[0], self.cols):
            kwds = KW.COL_LABEL | {'transform': ax.transAxes}
            ax.text(s=label, **kwds)
        self.figure.text(s=self.col_title, **KW.COL_TITLE)

    def add_titles(self) -> None:
        """Add the figure and sub-title at the top of the chart."""
        if not self.fig_title and not self.sub_title:
            return
        kw_fig = KW.FIG_TITLE
        kw_sub = KW.SUB_TITLE
        kw_sub['y'] = kw_sub['y'] + self.shift_sub_title
        kw_fig['y'] = kw_fig['y'] + self.shift_fig_title
        if self.sub_title:
            self.figure.text(s=self.sub_title, **kw_sub)
        if self.fig_title:
            self.figure.text(s=self.fig_title, **kw_fig)
    
    def add_axes_titles(self) -> None:
        """Add the provided axes titles."""
        if not self.axes_titles:
            return
        for ax, title in zip(self.axes.flat, self.axes_titles):
            ax.set_title(title)
    
    def add_info(self) -> None:
        """Insert an info text in the bottom left-hand corner of the 
        figure. By default, the info text contains today's date and the 
        user name. If attribute `info` is a string, it is added to the 
        info text separated by a comma."""
        if not self.info:
            return
        info_text = f'{STR.TODAY} {STR.USERNAME}'
        if isinstance(self.info, str):
            info_text = f'{info_text}, {self.info}'
        kwds = KW.INFO
        if self.xlabel:
            kwds['y'] = kwds['y'] - self.shift_text_y
        self.figure.text(s=info_text, **kwds)
    
    def draw(self) -> None:
        """Draw all the label facets to the figure."""
        self.add_xlabel()
        self.add_ylabel()
        self.add_axes_titles()
        self.add_row_labels()
        self.add_col_labels()
        for title, (handles, labels) in self.legend_data.items():
            self.add_legend(handles, labels, title)
        self.add_info()
        self.add_titles()


class AxesFacets:
    """A class for creating a grid of subplots with customizable sharing
    and sizing options.

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
    stretch_figsize : bool, optional
        If True, stretch the figure height and width based on the number of
        rows and columns, by default True.
    **kwds : dict, optional
        Additional keyword arguments to pass to the `plt.subplots` function.
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
    index: int
    """The axes flatten array index of current axes being worked on. """

    def __init__(
            self, nrows: int = 1, ncols: int = 1, 
            sharex: ShareAxisProperty = 'none', 
            sharey: ShareAxisProperty = 'none', 
            width_ratios: Sequence[float] | None = None,
            height_ratios: Sequence[float] | None = None, 
            stretch_figsize: bool = True, **kwds
            ) -> None:

        figsize = kwds.pop('figsize', plt.rcParams['figure.figsize'])
        if stretch_figsize:
            figsize = ((1 + math.log(ncols, math.e)) * figsize[0],
                       (1 + math.log(nrows, math.e)) * figsize[1])
        self.figsize = figsize
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, 
            squeeze=False, figsize=self.figsize, width_ratios=width_ratios,
            height_ratios=height_ratios, **kwds,)
        self.figure: Figure = fig
        self.axes: NDArray = axes
        self._nrows = nrows
        self._ncols = ncols
        self._sharex = sharex
        self._sharey = sharey
        self._ax = self.axes[0, 0] if self.nrows == self.ncols == 1 else None
        self.index = 0

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
    def sharex(self) -> ShareAxisProperty:
        """Get the sharing of properties along the x-axis (read-only)."""
        return self._sharex

    @property
    def sharey(self) -> ShareAxisProperty:
        """Get the sharing of properties along the y-axis (read-only)."""
        return self._sharey
    
    def __iter__(self) -> Generator[Axes, Self, None]:
        """Iterate over the axes in the grid.

        Returns
        -------
        Generator[Axes, Self, None]
            The generator object that yields the axes in the grid.
        """
        def ax_gen(self) -> Generator[Axes, Self, None]:
            for index, ax in enumerate(self.axes.flat):
                self._ax = ax
                self.index = index
                yield ax
            self._ax = None
            self.index = 0
        return ax_gen(self)
    
    def __next__(self) -> Axes:
        """Get the next axes in the grid.

        Returns
        -------
        Axes
            The next axes in the grid."""
        return next(self)
    
    def __getitem__(self, index: int) -> Axes:
        """Get the axes at the given index in the grid.

        Parameters
        ----------
        index : int
            The index of the axes in the grid.

        Returns
        -------
        Axes
            The axes at the given index in the grid.
        """
        return self.axes.flat[index]

    def __len__(self) -> int:
        """Get the total number of axes in the grid.

        Returns
        -------
        int
            The total number of axes in the grid.
        """
        return len(self.axes.flat)


class StripesFacets:
    """A class for creating location and spread width lines, 
    specification limits and/or confidence interval areas as stripes on 
    each Axes. The location and spread (and their confidence bands) 
    represent the data per axes.

    Parameters
    ----------
    target : ArrayLike
        The target data for the estimation.
    single_axes : bool
        Whether to use a single axes for the chart or multiple axes for
        each facet.
    mean : bool, optional
        Whether to include a mean line on the chart, by default False.
    median : bool, optional
        Whether to include a median line on the chart, by default False.
    control_limits : bool, optional
        Whether to include control limits on the chart, by default False.
    spec_limits : Tuple[float | None, float | None], optional
        The specification limits for the chart, by default (None, None).
    confidence : float, optional
        The confidence level for the confidence intervals,
        by default None.
    **kwds : dict, optional
        Additional keyword arguments to pass to the ProcessEstimator.
    """

    __slots__ = (
        'estimation', 'mask', '_confidence', 'spec_limits', 'single_axes')
    
    estimation: ProcessEstimator
    """The process estimator object."""
    mask: Tuple[bool, ...]
    """The mask indicating which lines to include on the chart."""
    _confidence: float | None
    """The confidence level for the confidence intervals."""
    spec_limits: Tuple[SpecLimit, SpecLimit]
    """The specification limits for the chart."""
    single_axes: bool
    """Whether to use a single axes for the chart or multiple axes for each facet."""
    
    def __init__(
        self,
        target: NumericSample1D,
        single_axes: bool, 
        mean: bool = False,
        median: bool = False,
        control_limits: bool = False,
        spec_limits: Tuple[SpecLimit, SpecLimit] = (None, None),
        confidence: float | None = None,
        **kwds) -> None:
        assert len(spec_limits) == 2, (
            f'Specification limits must contain 2 values, got {spec_limits}. '
            'Set None for limits that do not exist')
        self.single_axes = single_axes
        self.spec_limits = tuple(l if pd.notna(l) else None for l in spec_limits) # type: ignore
        self._confidence = confidence
        self.mask = (
            mean, median, control_limits, control_limits,
            self.spec_limits[0] is not None, self.spec_limits[1] is not None)
        self.estimation = ProcessEstimator(
            samples=target, lsl=spec_limits[0], usl=spec_limits[1], **kwds)
    
    @property
    def _d(self) -> int:
        """Get the number of decimal places to format values for legend
        labels. The number of decimal places depends on the size of the
        estimated median value (read-only)."""
        median = self.estimation.median
        if median <= 0.5:
            return 4
        elif median <= 5:
            return 3
        elif median <= 50:
            return 2
        else:
            return 1 
    
    @property
    def kwds(self) -> Tuple[dict, ...]:
        """Get keyword arguments for all lines that are plotted"""
        kwds = self._filter((
            KW.MEAN_LINE, KW.MEDIAN_LINE, KW.CONTROL_LINE, KW.CONTROL_LINE,
            KW.SPECIFICATION_LINE, KW.SPECIFICATION_LINE))
        return kwds
    
    @property
    def ci_functions(self) -> Tuple[Callable, ...]:
        """Get functions to calculate the confidence interval(s) 
        (read-only)."""
        ci = self._filter((
            self.estimation.mean_ci, self.estimation.median_ci, 
            self.estimation.stdev_ci, self.estimation.stdev_ci, None, None))
        return ci
    
    @property
    def values(self) -> Tuple[float | int, ...]:
        """Get the values for all lines that are plotted (read-only)."""
        attrs = ('mean', 'median', 'lcl', 'ucl')
        values = self._filter(
            [getattr(self.estimation, a) for a in attrs] 
            + list(self.spec_limits))
        return values
    
    @property
    def confidence(self) -> float:
        """Get the confidence level for the confidence intervals
        (read-only)."""
        if self._confidence is None:
            return DEFAULT.CONFIDENCE_LEVEL
        return self._confidence
    
    @property
    def labels(self) -> Tuple[str, ...]:
        """Get legend labels for added lines and spans (read-only)."""
        if self.estimation.strategy == 'norm':
            lcl = r'\bar x-' + f'{self.estimation._k}' + r'\sigma'
            ucl = r'\bar x+' + f'{self.estimation._k}' + r'\sigma'
        else:
            lcl = r'x_{' + f'{self.estimation._q_low:.4f}' + '}'
            ucl = r'x_{' + f'{self.estimation._q_upp:.4f}' + '}'
        labels = self._filter(
            (r'\bar x', r'x_{0.5}', lcl, ucl, STR['lsl'], STR['usl']))
        if self.single_axes:
            labels = tuple(
                f'${L}={v:.{self._d}f}$' for L, v in zip(labels, self.values))
        else:
            labels = tuple(f'${label}$' for label in labels)
        if self._confidence is not None:
            labels = labels + (f'{100*self.confidence:.0f} %-{STR["ci"]}',)
        return labels
    
    @property
    def handles(self) -> Tuple[Patch |Line2D, ...]:
        """Get the legend handles for added lines and spans (read-only)."""
        handles = tuple(
            Line2D([], [], markersize=0, **kwds) for kwds in self.kwds)
        if self._confidence is not None:
            handles = handles + (Patch(**KW.CI_HANDLE), )
        return handles
    
    def handles_labels(self) -> LegendHandlesLabels:
        """Get the legend handles and labels for added lines and spans."""
        return self.handles, self.labels
    
    def _filter(self, values: tuple | list) -> tuple:
        """Filter the given values according to the given boolean 
        attributes.

        Parameters
        ----------
        values : tuple | list
            The values to be filtered.

        Returns
        -------
        tuple
            The filtered values."""
        return tuple(v for v, m in zip(values, self.mask) if m)

    def draw(self, ax: Axes, target_on_y: bool) -> None:
        """Draw the stripes on the specified Axes.

        Parameters
        ----------
        ax : Axes
            The Axes on which to draw the stripes.
        target_on_y : bool
            Whether the target data is plotted on the y-axis.
        """
        for kwds, value, ci in zip(self.kwds, self.values, self.ci_functions):
            if target_on_y:
                ax.axhline(value, **kwds)
            else:
                ax.axvline(value, **kwds)
            if ci is not None and self._confidence is not None:
                low, upp = ci(self.confidence)
                if target_on_y:
                    ax.axhspan(low, upp, **KW.STRIPES_CONFIDENCE)
                else:
                    ax.axvspan(low, upp, **KW.STRIPES_CONFIDENCE)


__all__ = [
    "LabelFacets",
    "AxesFacets",
    "StripesFacets",
]