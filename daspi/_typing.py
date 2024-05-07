from typing import List
from typing import Tuple
from typing import TypeAlias
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SpecLimit: TypeAlias = float | int | None
SpecLimits: TypeAlias = Tuple[SpecLimit, SpecLimit]
LegendHandlesLabels: TypeAlias = Tuple[Tuple[Patch |Line2D, ...], Tuple[str, ...]]

__all__ = [
    'SpecLimit',
    'SpecLimits',
    'LegendHandlesLabels',
]
