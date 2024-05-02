from typing import Tuple
from typing import TypeAlias
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SpecLimit: TypeAlias = float | int | None
LegendHandles: TypeAlias = Tuple[*Tuple[Line2D | Patch, ...]]
HandlesLabels: TypeAlias = Tuple[LegendHandles, Tuple[str, ...]]


__all__ = [
    'SpecLimit',
    'LegendHandles',
    'HandlesLabels',
]
