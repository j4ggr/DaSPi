"""
DaSPi — Process Analytics & Six Sigma in Python

DaSPi helps engineers analyze and improve processes using statistical workflows.

## Four Flagship Workflows

1. **Gage R&R Analysis** — Verify measurement system capability (MSA Type 1 & 2)
2. **Process Capability Analysis** — Evaluate if process meets specifications (Cp, Cpk)
3. **Root Cause Analysis** — Identify factors that drive variation (ANOVA, regression)
4. **Statistical Process Control** — Monitor process stability (SPC charts)

Each workflow produces visual output + interpretation in under 20 lines of code.

For detailed examples and documentation, visit: https://j4ggr.github.io/DaSPi/
"""

from ._version import __version__ as __version__

from .config import CONFIG as CONFIG
from .strings import STR as STR

from .constants import ANOVA as ANOVA
from .constants import CATEGORY as CATEGORY
from .constants import COLOR as COLOR
from .constants import DEFAULT as DEFAULT
from .constants import DIST as DIST
from .constants import KW as KW
from .constants import LABEL as LABEL
from .constants import LINE as LINE
from .constants import PLOTTER as PLOTTER
from .constants import RE as RE
from .constants import SIGMA_DIFFERENCE as SIGMA_DIFFERENCE

from .doe import Factor as Factor
from .doe import FullFactorialDesignBuilder as FullFactorialDesignBuilder
from .doe import FullFactorial2kDesignBuilder as FullFactorial2kDesignBuilder
from .doe import FractionalFactorialDesignBuilder as FractionalFactorialDesignBuilder
from .doe import get_default_generators as get_default_generators

from .statistics.montecarlo import Binning as Binning
from .statistics.montecarlo import SpecLimits as SpecLimits
from .statistics.montecarlo import Specification as Specification
from .statistics.montecarlo import UNBOUNDED as UNBOUNDED
from .statistics.montecarlo import round_to_nearest as round_to_nearest
from .statistics.montecarlo import RandomProcessValue as RandomProcessValue
from .statistics.montecarlo import inclination_displacement as inclination_displacement

from .statistics.confidence import sem as sem
from .statistics.confidence import cp_ci as cp_ci
from .statistics.confidence import cpk_ci as cpk_ci
from .statistics.confidence import fit_ci as fit_ci
from .statistics.confidence import mean_ci as mean_ci
from .statistics.confidence import stdev_ci as stdev_ci
from .statistics.confidence import median_ci as median_ci
from .statistics.confidence import variance_ci as variance_ci
from .statistics.confidence import proportion_ci as proportion_ci
from .statistics.confidence import bonferroni_ci as bonferroni_ci
from .statistics.confidence import delta_mean_ci as delta_mean_ci
from .statistics.confidence import prediction_ci as prediction_ci
from .statistics.confidence import delta_stdev_ci as delta_stdev_ci
from .statistics.confidence import delta_variance_ci as delta_variance_ci
from .statistics.confidence import confidence_to_alpha as confidence_to_alpha
from .statistics.confidence import delta_proportions_ci as delta_proportions_ci

from .statistics.hypothesis import f_test as f_test
from .statistics.hypothesis import t_test as t_test
from .statistics.hypothesis import skew_test as skew_test
from .statistics.hypothesis import all_normal as all_normal
from .statistics.hypothesis import dunn_test as dunn_test
from .statistics.hypothesis import levene_test as levene_test
from .statistics.hypothesis import position_test as position_test
from .statistics.hypothesis import variance_test as variance_test
from .statistics.hypothesis import kurtosis_test as kurtosis_test
from .statistics.hypothesis import pairwise_tests as pairwise_tests
from .statistics.hypothesis import proportions_test as proportions_test
from .statistics.hypothesis import mean_stability_test as mean_stability_test
from .statistics.hypothesis import anderson_darling_test as anderson_darling_test
from .statistics.hypothesis import kolmogorov_smirnov_test as kolmogorov_smirnov_test
from .statistics.hypothesis import variance_stability_test as variance_stability_test

from .statistics.estimation import Loess as Loess
from .statistics.estimation import Lowess as Lowess
from .statistics.estimation import GageEstimator as GageEstimator
from .statistics.estimation import ProcessEstimator as ProcessEstimator
from .statistics.estimation import root_sum_squares as root_sum_squares
from .statistics.estimation import estimate_resolution as estimate_resolution
from .statistics.estimation import DistributionEstimator as DistributionEstimator
from .statistics.estimation import estimate_distribution as estimate_distribution
from .statistics.estimation import MeasurementUncertainty as MeasurementUncertainty
from .statistics.estimation import estimate_kernel_density as estimate_kernel_density
from .statistics.estimation import estimate_kernel_density_2d as estimate_kernel_density_2d
from .statistics.estimation import LocationDispersionEstimator as LocationDispersionEstimator
from .statistics.estimation import estimate_capability_confidence as estimate_capability_confidence

from .plotlib import style as style

from .plotlib.classify import Dodger as Dodger
from .plotlib.classify import HueLabel as HueLabel
from .plotlib.classify import SizeLabel as SizeLabel
from .plotlib.classify import ShapeLabel as ShapeLabel

from .plotlib.plotter import Box as Box
from .plotlib.plotter import Bar as Bar
from .plotlib.plotter import Line as Line
from .plotlib.plotter import Stem as Stem
from .plotlib.plotter import Pareto as Pareto
from .plotlib.plotter import Jitter as Jitter
from .plotlib.plotter import Plotter as Plotter
from .plotlib.plotter import Scatter as Scatter
from .plotlib.plotter import Violin as Violin
from .plotlib.plotter import Beeswarm as Beeswarm
from .plotlib.plotter import ErrorBar as ErrorBar
from .plotlib.plotter import MeanTest as MeanTest
from .plotlib.plotter import LoessLine as LoessLine
from .plotlib.plotter import StripeLine as StripeLine
from .plotlib.plotter import StripeSpan as StripeSpan
from .plotlib.plotter import HideSubplot as HideSubplot
from .plotlib.plotter import SkipSubplot as SkipSubplot
from .plotlib.plotter import SpreadWidth as SpreadWidth
from .plotlib.plotter import Probability as Probability
from .plotlib.plotter import BlandAltman as BlandAltman
from .plotlib.plotter import GaussianKDE as GaussianKDE
from .plotlib.plotter import QuantileBoxes as QuantileBoxes
from .plotlib.plotter import VariationTest as VariationTest
from .plotlib.plotter import ProportionTest as ProportionTest
from .plotlib.plotter import CenterLocation as CenterLocation
from .plotlib.plotter import TransformPlotter as TransformPlotter
from .plotlib.plotter import StandardErrorMean as StandardErrorMean
from .plotlib.plotter import ConfidenceInterval as ConfidenceInterval
from .plotlib.plotter import ParallelCoordinate as ParallelCoordinate
from .plotlib.plotter import GaussianKDEContour as GaussianKDEContour
from .plotlib.plotter import LinearRegressionLine as LinearRegressionLine
from .plotlib.plotter import CategoricalObservation as CategoricalObservation
from .plotlib.plotter import CapabilityConfidenceInterval as CapabilityConfidenceInterval
from .plotlib.plotter import GaussianKDEContourUnivariate as GaussianKDEContourUnivariate

from .plotlib.facets import AxesFacets as AxesFacets
from .plotlib.facets import flat_unique as flat_unique
from .plotlib.facets import LabelFacets as LabelFacets
from .plotlib.facets import StripesFacets as StripesFacets

from .plotlib.chart import Chart as Chart
from .plotlib.chart import JointChart as JointChart
from .plotlib.chart import SingleChart as SingleChart
from .plotlib.chart import MultivariateChart as MultivariateChart

from .plotlib.precast import GageRnRCharts as GageRnRCharts
from .plotlib.precast import GageStudyCharts as GageStudyCharts
from .plotlib.precast import ResidualsCharts as ResidualsCharts
from .plotlib.precast import PairComparisonCharts as PairComparisonCharts
from .plotlib.precast import PairwiseMatrixCharts as PairwiseMatrixCharts
from .plotlib.precast import ParameterRelevanceCharts as ParameterRelevanceCharts
from .plotlib.precast import BivariateUnivariateCharts as BivariateUnivariateCharts
from .plotlib.precast import ProcessCapabilityAnalysisCharts as ProcessCapabilityAnalysisCharts

from .anova.convert import get_term_name as get_term_name
from .anova.convert import frames_to_html as frames_to_html

from .anova.tables import uniques as uniques
from .anova.tables import anova_table as anova_table
from .anova.tables import terms_effect as terms_effect
from .anova.tables import terms_probability as terms_probability
from .anova.tables import variance_inflation_factor as variance_inflation_factor

from .anova.model import GeneralizedLinearModel as GeneralizedLinearModel
from .anova.model import LinearModel as LinearModel
from .anova.model import GageRnRModel as GageRnRModel
from .anova.model import hierarchical as hierarchical
from .anova.model import GageStudyModel as GageStudyModel
from .anova.model import is_main_parameter as is_main_parameter

from .datasets import load_dataset as load_dataset
from .datasets import list_dataset as list_dataset

