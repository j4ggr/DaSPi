from .strings import STR


from .constants import KW
from .constants import RE
from .constants import DIST
from .constants import COLOR
from .constants import LABEL
from .constants import ANOVA
from .constants import PLOTTER
from .constants import DEFAULT
from .constants import CATEGORY


from .statistics.confidence import sem
from .statistics.confidence import fit_ci
from .statistics.confidence import mean_ci
from .statistics.confidence import stdev_ci
from .statistics.confidence import median_ci
from .statistics.confidence import variance_ci
from .statistics.confidence import prob_points
from .statistics.confidence import proportion_ci
from .statistics.confidence import bonferroni_ci
from .statistics.confidence import delta_mean_ci
from .statistics.confidence import prediction_ci
from .statistics.confidence import dist_prob_fit_ci
from .statistics.confidence import delta_variance_ci
from .statistics.confidence import confidence_to_alpha
from .statistics.confidence import delta_proportions_ci

from .statistics.hypothesis import f_test
from .statistics.hypothesis import skew_test
from .statistics.hypothesis import all_normal
from .statistics.hypothesis import levene_test
from .statistics.hypothesis import position_test
from .statistics.hypothesis import variance_test
from .statistics.hypothesis import kurtosis_test
from .statistics.hypothesis import proportions_test
from .statistics.hypothesis import mean_stability_test
from .statistics.hypothesis import anderson_darling_test
from .statistics.hypothesis import kolmogorov_smirnov_test
from .statistics.hypothesis import variance_stability_test

from .statistics.estimation import Estimator
from .statistics.estimation import ProcessEstimator
from .statistics.estimation import estimate_distribution
from .statistics.estimation import estimate_kernel_density


from .plotlib.classify import Dodger
from .plotlib.classify import HueLabel
from .plotlib.classify import SizeLabel
from .plotlib.classify import ShapeLabel

from .plotlib.plotter import Bar
from .plotlib.plotter import Line
from .plotlib.plotter import Pareto
from .plotlib.plotter import Jitter
from .plotlib.plotter import Plotter
from .plotlib.plotter import Scatter
from .plotlib.plotter import Violine
from .plotlib.plotter import Errorbar
from .plotlib.plotter import MeanTest
from .plotlib.plotter import HideSubplot
from .plotlib.plotter import SkipSubplot
from .plotlib.plotter import SpreadWidth
from .plotlib.plotter import Probability
from .plotlib.plotter import BlandAltman
from .plotlib.plotter import GaussianKDE
from .plotlib.plotter import VariationTest
from .plotlib.plotter import ProportionTest
from .plotlib.plotter import CenterLocation
from .plotlib.plotter import LinearRegression
from .plotlib.plotter import TransformPlotter
from .plotlib.plotter import StandardErrorMean
from .plotlib.plotter import ConfidenceInterval

from .plotlib.facets import AxesFacets
from .plotlib.facets import LabelFacets
from .plotlib.facets import StripesFacets

from .plotlib.chart import Chart
from .plotlib.chart import JointChart
from .plotlib.chart import SingleChart
from .plotlib.chart import MultipleVariateChart

from .plotlib.templates import ResiduesCharts


from .anova.utils import uniques
from .anova.utils import anova_table
from .anova.utils import hierarchical
from .anova.utils import get_term_name
from .anova.utils import is_main_feature
from .anova.utils import variance_inflation_factor

from .anova.model import LinearModel


from .datasets import load_dataset
from .datasets import list_dataset
