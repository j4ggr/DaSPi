import numpy as np

from typing import Tuple
from typing import Literal
from typing import Sequence
from typing import TypeAlias
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas.core.series import Series

NumericSample1D: TypeAlias = (
    Sequence[int | float]
    | 'Series[int | float]'
    | NDArray[np.integer | np.floating])

SpecLimit: TypeAlias = float | int | None
SpecLimits: TypeAlias = Tuple[SpecLimit, SpecLimit]

ShareAxisProperty: TypeAlias = bool | Literal['none', 'all', 'row', 'col']
""""""

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

__all__ = [
    'NumericSample1D',
    'SpecLimit',
    'SpecLimits',
    'ShareAxisProperty',
    'LegendHandlesLabels',
    'LineStyle'
]
