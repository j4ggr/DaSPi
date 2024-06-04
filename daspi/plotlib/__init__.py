"""This plotting library allows you to create uniform and attractive 
charts using a simple syntax.
Furthermore, the statistical estimators such as the spread, mean values,
kernel density estimation can be combined as desired with scatter plots
or error bar plots. These plots can be divided based on multiple
variables through axis position, colors, marker symbols, marker size, 
Axes rows and columns.
Threshold values such as the mean and standard deviation or 
specification limits can also be drawn for all values per axis.
Finally, image titles, subtitles and axis labels and even a small 
information text at the bottom with the date and author can also be 
added.

## Examples

### Chart with one Axes


### Chart with multiple Axes


### Joint Chart


## Notes

All plotters from the plotter module can also be applied to an Axes
object. The following are available:

    - Plotter
    - Scatter
    - Line
    - LinearRegression
    - Probability
    - BlandAltman
    - TransformPlotter
    - CenterLocation
    - Bar
    - Pareto
    - Jitter
    - GaussianKDE
    - Violine
    - Errorbar
    - StandardErrorMean
    - SpreadWidth
    - ConfidenceInterval
    - MeanTest
    - VariationTest

 You can also create your own plotter by inheriting from the `Plotter` 
 class. If the data needs to be transformed, like with KDE, you can also
 inherit from TransformPlotter. Then override the `transform` method.
 The `__call__` method must be overwritten in any case.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

from .classify import Dodger
from .classify import HueLabel
from .classify import SizeLabel
from .classify import ShapeLabel

from .plotter import Bar
from .plotter import Line
from .plotter import Pareto
from .plotter import Jitter
from .plotter import Plotter
from .plotter import Scatter
from .plotter import Violine
from .plotter import Errorbar
from .plotter import MeanTest
from .plotter import HideSubplot
from .plotter import SkipSubplot
from .plotter import SpreadWidth
from .plotter import Probability
from .plotter import BlandAltman
from .plotter import GaussianKDE
from .plotter import VariationTest
from .plotter import ProportionTest
from .plotter import CenterLocation
from .plotter import LinearRegression
from .plotter import TransformPlotter
from .plotter import StandardErrorMean
from .plotter import ConfidenceInterval

from .facets import AxesFacets
from .facets import LabelFacets
from .facets import StripesFacets

from .chart import Chart
from .chart import JointChart
from .chart import SingleChart
from .chart import MultipleVariateChart

from .templates import ResiduesCharts

plt.style.use(Path(__file__).parent/'daspi.mplstyle')
