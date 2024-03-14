import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

plt.style.use(Path(__file__).parent/'daspi.mplstyle')

from .utils import Dodger
from .utils import HueLabel
from .utils import SizeLabel
from .utils import ShapeLabel
from .utils import shared_axes

from .plotter import Bar
from .plotter import Line
from .plotter import Pareto
from .plotter import Jitter
from .plotter import Plotter
from .plotter import Scatter
from .plotter import Violine
from .plotter import Errorbar
from .plotter import MeanTest
from .plotter import SpreadWidth
from .plotter import Probability
from .plotter import BlandAltman
from .plotter import GaussianKDE
from .plotter import VariationTest
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
from .chart import SimpleChart
from .chart import MultipleVariateChart
