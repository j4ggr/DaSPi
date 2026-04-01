
"""Plotting library for data analysis, statistics, and process improvement.

This package provides a composable system for building publication-ready
charts from raw pandas DataFrames. Four layers of abstraction stack cleanly
on top of each other:

``appearance``
    Style management (``Style``), colormap registration, and axis
    utility helpers (``get_shared_axes``, ``transpose_xy_axes_params``).

``classify``
    Category-label helpers that map data labels to visual properties:
    ``HueLabel`` (colours), ``ShapeLabel`` (markers), ``SizeLabel``
    (marker sizes), and ``Dodger`` (categorical axis dodging).

``plotter``
    Low-level *Plotter* classes. Each class is responsible for a
    single mark type (e.g. ``Scatter``, ``Line``, ``GaussianKDE``,
    ``Probability``, ``ErrorBar``). Plotters are designed to be
    instantiated and then called on an ``Axes`` object. They can be
    freely combined inside a ``Chart``.

``facets``
    Layout and annotation helpers: ``AxesFacets`` (subplot grid +
    mosaic), ``LabelFacets`` (axis and legend labelling), and
    ``StripesFacets`` (reference lines and bands).

``chart``
    High-level *Chart* classes. ``SingleChart``, ``JointChart``, and
    ``MultivariateChart`` each accept a source DataFrame and expose a
    fluent ``plot()`` / ``label()`` / ``stripes()`` interface that
    wires together the lower layers automatically.

``precast``
    Ready-to-use composite charts built on top of the chart layer.
    Pass a ``LinearModel`` or a DataFrame and get a complete
    multi-panel figure in one call — examples:
    ``ResidualsCharts``, ``ParameterRelevanceCharts``,
    ``ProcessCapabilityAnalysisCharts``, ``GageRnRCharts``.

All public names from each submodule are re-exported at the package
level, so ``from daspi.plotlib import Scatter`` works without knowing
which submodule it lives in.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import List
from pathlib import Path

from .appearance import *

from .classify import *

from .plotter import *

from .facets import *

from .chart import *

from .precast import *

# TODO: some alpha fails for vspan when using daspi and ggplot2 styles
style.use('daspi')

