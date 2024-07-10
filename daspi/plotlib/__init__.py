

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
from .plotter import ParallelCoordinate

from .facets import AxesFacets
from .facets import LabelFacets
from .facets import StripesFacets

from .chart import Chart
from .chart import JointChart
from .chart import SingleChart
from .chart import MultipleVariateChart

from .templates import ResiduesCharts
from .templates import ParameterRelevanceCharts

plt.style.use(Path(__file__).parent/'daspi.mplstyle')
