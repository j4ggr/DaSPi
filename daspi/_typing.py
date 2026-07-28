from collections.abc import Hashable, Sequence
from datetime import date, datetime
from typing import Literal, TypeVar

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import NDArray
from pandas.core.series import Series

__all__ = [
    'FloatOrArray',
    'LegendHandlesLabels',
    'LineStyle',
    'MosaicLayout',
    'NumericSample1D',
    'Sample1D',
    'ShareAxisProperty',
]


type Sample1D = (
    Sequence[int | float | str | datetime | date]
    | 'Series[int | float | str | datetime | date]'
    | NDArray[np.integer | np.floating | np.str_ | np.datetime64])

type NumericSample1D = (
    Sequence[int | float]
    | 'Series[int | float]'
    | NDArray[np.integer | np.floating])

type ShareAxisProperty = (
    bool | Literal['none', 'all', 'row', 'col'])
"""Type alias for matplotlib share axis property.
    - True: share all axes
    - False: do not share axes
    - 'none': do not share axes
    - 'all': share all axes
    - 'row': share axes within each row
    - 'col': share axes within each column
"""

type LegendHandlesLabels = (
    tuple[tuple[Patch |Line2D, ...], tuple[str, ...]])
"""Type alias as tuple of maptlotlib legend handles and labels."""

type LineStyle = (
    Literal[
        '-', 'solid', '--', 'dashed', '-.', 'dashdot', ':', 'dotted', 
        'none', 'None', ' ', '']
    | tuple[int, tuple[int, ...]])
"""Type alias for matplotlib line styles.
    - solid line: '-' or 'solid'
    - dashed line: '--' or 'dashed'
    - dash-dotted line: '-.' or 'dashdot'
    - dotted line: ':' or 'dotted'
    - draw nothing: 'none', 'None', ' ' or ''

Alternatively, a dash tuple of the following form can be provided: 
`(Offset, (On, Off, ...))`, where the on-off sequence can appear as 
empty tuple (the same as continuous), once (containing 2 numbers On and
Off), or multiple times with different On and Off numbers."""

type MosaicLayout = (
    list[Sequence[Hashable]]
    | tuple[Sequence[Hashable], ...]
    | list[list[str]]
    | str | None)
"""Type alias for mosaic layout. From the author's point of view, the 
best option is a tuple of strings. Dor example:
    ```python
    layout: MosaicLayout = (
        'AA.',
        '°°D',
        '°°D')
    ```
"""

FloatOrArray = TypeVar('FloatOrArray', float, NDArray, Series)
"""Type alias for float, numpy array, or pandas series."""

type LevelType = float | int | str
"""Type alias for factor level values in design of experiments.
Can be numeric (float, int) or categorical (str, int)."""
