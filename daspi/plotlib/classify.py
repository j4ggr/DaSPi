"""The provided Python code defines several classes and a list of
strings, all related to handling categorical features in data
visualization. Here's a brief description of each component:

1. _CategoryLabel: An abstract base class representing a category label 
handler for plotted categorical values. It has properties for available 
categories, default category, labels, and number of used categories. It
also includes methods for getting category items, retrieving legend
handles and labels, and checking if the object is empty.
2. HueLabel: A class representing category labels for hue values in
plots. It inherits from _CategoryLabel and adds a property for available
hue colors. It also overrides the handles_labels method to provide
specific legend handles and labels for hue categories.
3. ShapeLabel: A class representing category labels for shape markers in
plots. It inherits from _CategoryLabel and overrides the handles_labels
method to provide specific legend handles and labels for shape markers.
4. SizeLabel: A class representing category labels for marker sizes in
plots. It inherits from _CategoryLabel and adds properties for the
offset and factor for value-to-size transformation. It also overrides
the handles_labels method to provide specific legend handles and labels
for marker sizes. Additionally, it includes methods for getting size
values and converting values into size values for markers.
5. Dodger: A class for handling dodging of categorical features in
plots. It has properties for categories, ticks, tick labels, width of
each category bar, number of categories, dodge values, and default dodge
value. It also includes methods for getting dodge values, replacing
source values with dodged ticks, and checking if the object has more
than one category.
"""

import numpy as np
import pandas as pd

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Sequence
from numpy.typing import NDArray
from numpy.typing import ArrayLike
from pandas._typing import Scalar
from pandas.core.series import Series
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .._typing import LegendHandlesLabels
from ..constants import KW
from ..constants import DEFAULT
from ..constants import CATEGORY


__all__ = [
    'HueLabel',
    'ShapeLabel',
    'SizeLabel',
    'Dodger',
    ]


class _CategoryLabel(ABC):
    """Abstract base class representing a handler for category labels 
    in plotted categorical values.

    This abstract base class is intended to be subclassed for managing 
    and manipulating category labels in various types of visualizations. 
    It provides a common interface for handling labels associated with 
    categorical data.

    Parameters
    ----------
    labels : Sequence[Scalar]
        A sequence of labels corresponding to the categories. These 
        labels will be used to identify and differentiate the various
        categories in the plot. The type of labels can vary, including 
        strings, integers, or other scalar types.
    categories : Sequence[Scalar]
        A sequence of categories that the labels correspond to. This 
        parameter provides context for the labels, indicating which 
        categorical data they represent.
    """

    __slots__ = ('_default', '_categories', '_labels', '_n')

    _default: Any
    """Default category item."""
    _categories: Tuple[Any, ...]
    """Available categories."""
    _labels: Tuple[str, ...]
    """Labels corresponding to the categories."""
    _n: int
    """Number of used categories."""
    
    def __init__(
            self,
            labels: Sequence[Scalar],
            categories: Tuple[Any, ...]) -> None:
        self._n = len(labels)
        self._default = None
        self._categories = categories
        self.labels = labels

    @property
    def categories(self) -> Tuple[Any, ...]:
        """Get the available categories (read-only)."""
        return self._categories[:self.n_used]

    @property
    def default(self) -> Any:
        """Get default category item (read-only)."""
        if self._default is None:
            self._default = self._categories[0]
        return self._default

    @property
    def labels(self) -> Tuple[str, ...]:
        """Get and set the labels corresponding to the categories."""
        return self._labels
    @labels.setter
    def labels(self, labels: Sequence[Scalar]) -> None:
        assert self.n_used <= self.n_allowed, (
            f'{self} can handle {self.n_allowed} categories, got {len(labels)}')
        assert self.n_used == len(set(labels)), (
            'Labels occur more than once, only unique labels are allowed')
        self._labels = tuple(map(str, labels))

    @property
    def n_used(self) -> int:
        """Get the number of used categories (read-only)."""
        return self._n

    @property
    def n_allowed(self) -> int:
        """Get the allowed amount of categories (read-only)."""
        return len(self._categories)
    
    def __getitem__(self, label: Any) -> Any:
        """Get the category item corresponding to the given label.

        Parameters
        ----------
        label : Any
            The label for which to retrieve the category item.

        Returns
        -------
        Any
            The category item corresponding to the label.
        """
        if label is None:
            return self.default
        
        _label = str(label)
        assert _label in self.labels, (
            f"Can't get category for label {_label}, got {self.labels}")
        item = self.categories[self.labels.index(_label)]
        return item

    def __str__(self) -> str:
        """Get a string representation of the _CategoryLabel object."""
        return self.__class__.__name__
    
    def __bool__(self) -> bool:
        """Check if the _CategoryLabel object is non-empty."""
        return bool(self._n)
    
    @abstractmethod
    def handles_labels(self) -> LegendHandlesLabels:
        """Abstract method to retrieve legend handles and labels.

        Returns
        -------
        LegendHandlesLabels
            A tuple containing legend handles and labels.
        """
        pass


class HueLabel(_CategoryLabel):
    """A class representing category labels for hue values in plots.

    This class is used to manage and display hue labels in 
    visualizations, allowing for customization of color representation 
    based on categorical data.

    Parameters
    ----------
    labels : Sequence[Scalar]
        Labels corresponding to the hue categories. These labels will be 
        used to identify different categories in the plot.
    colors : Tuple[str, ...], optional
        A tuple of available hue categories represented as hex codes or
        named colors. By default, this is set to `CATEGORY.PALETTE`, 
        which provides a predefined set of colors for the categories.
    follow_order : bool, optional
        Determines how colors are assigned to hue categories. If set to 
        `True`, the colors will follow the order specified in the
        `colors` parameter. If set to `False`, the colors will be 
        distributed evenly across the entire palette, inspired by 
        ggplot's approach. The default value is `False`, which promotes 
        an even distribution of colors across categories for better 
        visual clarity.

    Examples
    --------
    ```python
    hue_labels = HueLabel(
        labels=['A', 'B', 'C'],
        colors=('red', 'green', 'blue'),
        follow_order=True)
    ```
    """
    __slots__ = ('follow_order')

    follow_order: bool
    """Determines how colors are assigned to hue categories. If set to 
    `True`, the colors will follow the order specified in the `colors` 
    parameter. If set to `False`, the colors will be distributed evenly 
    across the entire palette"""

    def __init__(
            self,
            labels: Sequence[Scalar],
            colors: Tuple[str, ...],
            follow_order: bool = False
            ) -> None:
        self.follow_order = follow_order
        super().__init__(labels, colors)
        if not follow_order:
            self._default = DEFAULT.PLOTTING_COLOR
    
    @property
    def categories(self) -> Tuple[str, ...]:
        """Tuple containing the available hue categories as the current
        color palette as defined in constants `CATEGORY.PALETTE` or
        the given colors during initialization (read-only)."""
        if self.follow_order or self.n_used in (0, 1):
            return super().categories
        
        step = self.n_allowed // self.n_used
        return self._categories[::step][:self.n_used]

    @property
    def colors(self) -> Tuple[str, ...]:
        """Get the available hue colors (read-only)."""
        return self.categories

    def handles_labels(self) -> LegendHandlesLabels:
        """Retrieve legend handles and labels for hue categories.

        Returns
        -------
        LegendHandlesLabels
            A tuple containing legend handles and labels.
        """
        handles = tuple(Patch(color=c, **KW.HUE_HANDLES) for c in self.colors)
        return handles, self.labels


class ShapeLabel(_CategoryLabel):
    """A class representing category labels for shape markers in plots.

    This class is designed to manage and display shape markers 
    associated with different categories in visualizations. It allows 
    for the customization of markers to enhance the clarity and 
    effectiveness of data representation.

    Parameters
    ----------
    labels : Sequence[Scalar]
        A sequence of labels corresponding to the shape marker 
        categories. These labels will be used to identify and 
        differentiate the shapes in the plot.
    markers : Tuple[str, ...], optional
        A tuple of available shape marker categories represented as 
        strings. This defines the shapes that can be used for the 
        corresponding categories. By default, this is set to 
        `CATEGORY.MARKERS`, which provides a predefined set of markers 
        for various categories.

    Examples
    --------
    ```python
    shape_labels = ShapeLabel(
        labels=['A', 'B', 'C'], 
        markers=('o', 's', '^'))
    ```
    """

    def __init__(
            self,
            labels: Sequence[Scalar],
            markers: Tuple[str, ...]) -> None:
        super().__init__(tuple(map(str, labels)), markers)
    
    @property
    def categories(self) -> Tuple[str, ...]:
        """Tuple containing the available shape markers as defined
        in constants `CATEGORY.MARKERS` or the given markers during
        initialization (read-only)."""
        return super().categories
    
    @property
    def markers(self) -> Tuple[str, ...]:
        """Get the used shape markers (read-only)."""
        return self.categories

    def handles_labels(self) -> LegendHandlesLabels:
        """Retrieve legend handles and labels for shape markers.

        Returns
        -------
        LegendHandlesLabels
            A tuple containing legend handles and labels.
        """
        handles = tuple(
            Line2D(marker=m, **KW.SHAPE_HANDLES) for m in self.markers)
        return handles, self.labels


class SizeLabel(_CategoryLabel):
    """A class representing category labels for marker sizes in plots.

    This class is designed to manage and display marker sizes associated 
    with different categories in visualizations. It allows for the 
    customization of sizes to effectively represent data magnitudes.

    Parameters
    ----------
    min_value : int or float
        The minimum value for the size range. This defines the smallest 
        marker size that can be used in the plot.
    max_value : int or float
        The maximum value for the size range. This defines the largest 
        marker size that can be used in the plot.
    n_bins : int, optional
        The number of bins to divide the size range into. This 
        determines how many distinct size categories will be created for
        the markers. By default, this is set to `CATEGORY.N_SIZE_BINS`, 
        which provides a predefined number of bins for size 
        categorization.

    Examples
    --------
    ```python
    size_labels = SizeLabel(min_value=5, max_value=100, n_bins=5)
    ```
    """

    __slots__ = ('_min', '_max')

    _min: int | float
    """Minimum value for the size range."""
    _max: int | float
    """Maximum value for the size range."""

    def __init__(
            self, 
            min_value: int | float,
            max_value: int | float,
            n_bins: int) -> None:
        assert max_value > min_value
        self._min = min_value
        self._max = max_value
        use_integer = isinstance(self._min, int) and isinstance(self._max, int)
        labels = np.linspace(
                self._min, self._max, CATEGORY.N_SIZE_BINS, 
                dtype = int if use_integer else float)
        if use_integer:
            labels = tuple(map(str, labels))
        else:
            labels = tuple(map(lambda x: f'{x:.3f}', labels))
        handle_sizes = tuple(np.linspace(
            *CATEGORY.MARKERSIZE_LIMITS, n_bins, dtype=int))
        super().__init__(labels, handle_sizes)
    
    @property
    def categories(self) -> Tuple[int, ...]:
        """Tuple containing the available marker sizes (read-only)."""
        return super().categories
    
    @property
    def offset(self) -> int:
        """Get the offset for value-to-size transformation (read-only)."""
        return CATEGORY.MARKERSIZE_LIMITS[0]

    @property
    def factor(self) -> float:
        """Get the factor for value-to-size transformation (read-only)."""
        low, upp = CATEGORY.MARKERSIZE_LIMITS
        return (upp - low)/(self._max - self._min)
    
    def handles_labels(self) -> LegendHandlesLabels:
        """Retrieve legend handles and labels for marker sizes.

        Returns
        -------
        LegendHandlesLabels
            A tuple containing legend handles and labels.
        """
        handles = tuple(
            Line2D(markersize=s, **KW.SIZE_HANDLES) for s in self.categories)
        return handles, self.labels
    
    def __getitem__(self, item: int | float | None) -> float:
        """Get the size value corresponding to the given item.

        Parameters
        ----------
        item : int | float | None
            The item for which to retrieve the size value.

        Returns
        -------
        float
            The size value corresponding to the item."""
        if item is None:
            return self.default
        return self([item])[0]
    
    def __call__(self, values: ArrayLike) -> NDArray:
        """Convert values into size values for markers.

        Parameters
        ----------
        values : ArrayLike
            Values to be converted into marker sizes.

        Returns
        -------
        NDArray
            Size values for markers.
        """
        sizes = self.factor * (np.array(values) - self._min) + self.offset
        sizes = np.square(sizes)
        return sizes


class Dodger:
    """A class for handling the dodging of categorical features in plots.

    This class is designed to facilitate the visual separation of 
    overlapping categorical features in plots by adjusting their 
    positions. Dodging helps improve clarity and readability of 
    categorical data representations.

    Parameters
    ----------
    categories : Tuple[str, ...]
        A tuple of categories corresponding to the features being 
        plotted. These categories will be used to determine the 
        positions of the dodged elements in the visualization.
    tick_labels : Tuple[str, ...]
        A tuple of labels for the ticks on the axis. These labels will 
        correspond to the categories and will be displayed along the 
        axis to enhance the interpretability of the plot.

    Examples
    --------
    ```python
    dodger = Dodger(
        categories=('A', 'B', 'C'),
        tick_labels=('Category A', 'Category B', 'Category C'))
    ```
    """

    __slots__ = (
        'categories', 'ticks', 'tick_labels', 'amount', 'width', 'dodge',
        '_default', '_pos_to_label_map')
    
    categories: Tuple[str, ...]
    """Categories corresponding to the features."""
    ticks: NDArray[np.int_]
    """Numeric positions of the ticks."""
    tick_labels: Tuple[str, ...]
    """Labels for the ticks on the axis."""
    width: float
    """Width of each category bar."""
    amount: int
    """Number of categories."""
    dodge: Dict[str, float]
    """Dictionary mapping categories to dodge values."""
    _default: int
    """Default dodge value."""
    _pos_to_label_map: Dict[str, str]
    """Dictionary mapping tick positions to tick labels."""

    def __init__(
            self,
            categories: Tuple[str, ...],
            tick_labels: Tuple[Any, ...]) -> None:
        self.categories = categories
        self.tick_labels = tuple(map(str, tick_labels))
        self.ticks = np.arange(len(tick_labels)) + DEFAULT.FEATURE_BASE
        self.amount = max(len(self.categories), 1)
        space = CATEGORY.FEATURE_SPACE/self.amount
        self.width = space - CATEGORY.FEATURE_PAD
        offset = (space - CATEGORY.FEATURE_SPACE) / 2
        _dodge = tuple(i*space + offset for i in range(self.amount))
        self.dodge = {c: d for c, d in zip(self.categories, _dodge)}
        self._default = 0
        self._pos_to_label_map = {}
    
    @property
    def lim(self) -> Tuple[float, float]:
        """Get the required axis limits (read-only)."""
        return (min(self.ticks) - 0.5, max(self.ticks) + 0.5)
    
    def __getitem__(self, category: str | None) -> float | int:
        """Get the dodge value for a given category.

        Parameters
        ----------
        category : str | None
            The category for which to retrieve the dodge value.

        Returns
        -------
        float | int
            The dodge value corresponding to the category.
        """
        if category is None:
            return self._default
        return self.dodge.get(str(category), self._default)
    
    def __call__(self, values: Series, category: str | None) -> 'Series[float]':
        """Replace source values with dodged ticks using the given 
        category to get the right offset.

        First it ensurse that the tick values are strings like the once 
        in the `tick_labels` attribute. Then it replaces the values with
        the positions (basic ticks + doddging offset). Finally it 
        converts the values to floats.

        Parameters
        ----------
        values : Series
            Source values to be replaced.
        category : str | None
            The category for which to apply the dodge.

        Returns
        -------
        Series[float]
            Series with replaced values.
        """
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        offset = self[category]
        positions = (self.ticks + offset).astype(float).astype(str)
        mapper = dict(zip(self.tick_labels, positions))
        self._pos_to_label_map |= {v: k for k, v in mapper.items()}
        return values.astype(str).replace(mapper).astype(float)
    
    def __bool__(self) -> bool:
        """Check if the Dodger object has more than one category.

        Returns
        -------
        bool
            True if there are multiple categories, False otherwise.
        """
        return len(self.categories) > 1
    
    def pos_to_ticklabels(self, positions: Series) -> Series:
        """Convert numeric or encoded feature axis positions to their 
        corresponding tick labels.

        This method uses an internal mapping (`_pos_to_label_map`) to 
        translate position values (typically numeric or categorical 
        codes) into human-readable tick labels for display on the 
        feature axis.

        Parameters
        ----------
        positions : Series
            A pandas Series containing the axis positions to be 
            converted.

        Returns
        -------
        Series
            A Series of the same shape with positions replaced by their 
            corresponding labels.
        """
        return positions.astype(str).replace(self._pos_to_label_map)
