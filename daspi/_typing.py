import numpy as np

from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Literal
from typing import Sequence
from typing import Hashable
from typing import TypeAlias
from datetime import date
from datetime import datetime
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas.core.series import Series

Sample1D: TypeAlias = (
    Sequence[int | float | str | datetime | date]
    | 'Series[int | float | str | datetime | date]'
    | NDArray[np.integer | np.floating | np.str_ | np.datetime64])

NumericSample1D: TypeAlias = (
    Sequence[int | float]
    | 'Series[int | float]'
    | NDArray[np.integer | np.floating])

ShareAxisProperty: TypeAlias = (
    bool | Literal['none', 'all', 'row', 'col'])
"""Type alias for matplotlib share axis property.
    - True: share all axes
    - False: do not share axes
    - 'none': do not share axes
    - 'all': share all axes
    - 'row': share axes within each row
    - 'col': share axes within each column
"""

LegendHandlesLabels: TypeAlias = (
    Tuple[Tuple[Patch |Line2D, ...], Tuple[str, ...]])
"""Type alias as tuple of maptlotlib legend handles and labels."""

LineStyle: TypeAlias = (
    Literal[
        '-', 'solid', '--', 'dashed', '-.', 'dashdot', ':', 'dotted', 
        'none', 'None', ' ', '']
    | Tuple[int, Tuple[int, ...]])
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

MosaicLayout: TypeAlias = (
    List[Sequence[Hashable]]
    | Tuple[Sequence[Hashable], ...]
    | List[List[str]]
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

__all__ = [
    'Sample1D',
    'NumericSample1D',
    'ShareAxisProperty',
    'LegendHandlesLabels',
    'LineStyle',
    'MosaicLayout',
    'FloatOrArray',
]
