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

LegendHandlesLabels: TypeAlias = Tuple[
    Tuple[Patch |Line2D, ...], Tuple[str, ...]]


__all__ = [
    'NumericSample1D',
    'SpecLimit',
    'SpecLimits',
    'ShareAxisProperty',
    'LegendHandlesLabels',
]
