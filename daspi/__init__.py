"""
## Usage

### Visualization

To use DaSPi, you can import the package and start exploring your data. Here is an example of how to use DaSPi to visualize a dataset:

```python
import daspi as dsp
df = dsp.load_dataset('iris')

chart = dsp.MultivariateChart(
        source=df,
        target='length',
        feature='width',
        hue='species',
        col='leaf',
        markers=('x',)
    ).plot(
        dsp.GaussianKDEContour
    ).plot(
        dsp.Scatter
    ).label(
        feature_label='leaf width (cm)',
        target_label='leaf length (cm)',
    )
```

![Iris sepal length species](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/iris_contour_size-leaf-species.png)

### ANOVA

Do some ANOVA and statistics on a dataset. Run the example below in a Jupyther Notebook to see the results.

```python
df = dsp.load_dataset('painkillers-dissolution')
model = dsp.LinearModel(
    source=df,
    target='dissolution',
    factors=['employee', 'stirrer', 'brand', 'catalyst', 'water'],
    covariates=['temperature', 'preparation'],
    order=2)
df_gof = pd.concat(model.recursive_elimination())

dsp.ResidualsCharts(model).plot().stripes().label(info=True)
dsp.ParameterRelevanceCharts(model).plot().label(info=True)
model
```

**Formula:**

dissolution ~ 16.0792 + 2.3750 employee[T.B] + 0.8375 employee[T.C] + 10.7500 brand[T.OuchAway] - 3.8000 water[T.tap] - 5.7167 brand[T.OuchAway]:water[T.tap]

**Model Summary**

| Hierarchical | Least Parameter |  P Least |        S |        AIC |       R² |   R² Adj |  R² Pred |
| -----------: | --------------: | -------: | -------: | ---------: | -------: | -------: | -------: |
|         True |        employee | 0.023298 | 2.374693 | 224.835935 | 0.857379 | 0.840400 | 0.813719 |

**Parameter Statistics**

|                               |      Coef |  Std Err |         T |        P |    CI Low |    CI Upp |
| ----------------------------: | --------: | -------: | --------: | -------: | --------: | --------: |
|                     Intercept | 16.079167 | 0.839581 | 19.151424 | 0.000000 | 14.384824 | 17.773509 |
|                 employee[T.B] |  2.375000 | 0.839581 |  2.828793 | 0.007133 |  0.680657 |  4.069343 |
|                 employee[T.C] |  0.837500 | 0.839581 |  0.997522 | 0.324224 | -0.856843 |  2.531843 |
|              brand[T.OuchAway] | 10.750000 | 0.969464 | 11.088598 | 0.000000 |  8.793542 | 12.706458 |
|                  water[T.tap] | -3.800000 | 0.969464 | -3.919690 | 0.000321 | -5.756458 | -1.843542 |
| brand[T.OuchAway]:water[T.tap] | -5.716667 | 1.371030 | -4.169616 | 0.000149 | -8.483516 | -2.949817 |

**Analysis of Variance**

|      Source |   DF |         SS |         MS |          F |        P |       n² |
| ----------: | ---: | ---------: | ---------: | ---------: | -------: | -------: |
|    employee |    2 |  46.431667 |  23.215833 |   4.116891 | 0.023298 | 0.027960 |
|       brand |    1 | 747.340833 | 747.340833 | 132.526821 | 0.000000 | 0.450027 |
|       water |    1 | 532.000833 | 532.000833 |  94.340328 | 0.000000 | 0.320355 |
| brand:water |    1 |  98.040833 |  98.040833 |  17.385695 | 0.000149 | 0.059037 |
|    Residual |   42 | 236.845000 |   5.639167 |        nan |      nan | 0.142621 |

**Variance Inflation Factor**

|             |   DF |      VIF |     GVIF | Threshold | Collinear |              Method |
| ----------: | ---: | -------: | -------: | --------: | --------: | ------------------: |
|   Intercept |    1 | 5.000000 | 2.236068 |  2.236068 |      True |           R_squared |
|    employee |    2 | 1.000000 | 1.000000 |  1.495349 |     False |         generalized |
|       brand |    1 | 1.000000 | 1.000000 |  2.236068 |     False |           R_squared |
|       water |    1 | 1.000000 | 1.000000 |  2.236068 |     False |           R_squared |
| brand:water |    1 | 1.000000 | 1.000000 |  2.236068 |     False | single_order-2_term |

![ANOVA dissolution residuals](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/anova_dissolution_residues.png)

![ANOVA dissolution parameters](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/anova_dissolution_params.png)

### Process capability

Analyze process variation and other key performance indicators for process capacity.

```python
df = dsp.load_dataset('drop_card')
spec_limits = dsp.SpecLimits(0, float(df.loc[0, 'usl']))
target = 'distance'

chart = dsp.ProcessCapabilityAnalysisCharts(
        source=df,
        target=target,
        spec_limits=spec_limits,
        hue='method'
    ).plot(
    ).stripes(
    ).label(
        fig_title='Process Capability Analysis',
        sub_title='Drop Card Experiment',
        target_label='Distance (cm)',
        info=True
    )

samples_parallel = df[df['method']=='parallel'][target]
samples_series = df[df['method']=='perpendicular'][target]
pd.concat([
    dsp.ProcessEstimator(samples_parallel, spec_limits).describe(),
    dsp.ProcessEstimator(samples_series, spec_limits).describe()],
    axis=1,
    ignore_index=True,
).rename(
    columns={0: 'parallel', 1: 'perpendicular'}
)
```

|           |   parallel | perpendicular |
| --------: | ---------: | ------------: |
| n_samples |         20 |            20 |
| n_missing |          0 |             0 |
|      n_ok |         18 |            20 |
|     n_nok |          2 |             0 |
|  n_errors |          0 |             0 |
|        ok |    90.00 % |      100.00 % |
|       nok |    10.00 % |        0.00 % |
|  nok_norm |     8.01 % |        3.73 % |
|   nok_fit |     7.24 % |        5.77 % |
|       min |        8.5 |          17.5 |
|       max |       83.0 |          73.0 |
|      mean |     42.935 |        48.485 |
|    median |      40.75 |          52.5 |
|       std |  22.666583 |     17.359489 |
|       sem |   5.068402 |        3.8817 |
|    excess |  -0.900801 |     -1.236078 |
|  p_excess |   0.288757 |      0.072573 |
|      skew |    0.19252 |     -0.377538 |
|    p_skew |   0.690373 |      0.438723 |
|      p_ad |   0.754044 |      0.098371 |
|      dist |    lognorm |      logistic |
|    p_ks |   0.964797 |      0.744326 |
|  strategy |       norm |          norm |
|       lcl | -25.064748 |     -3.593468 |
|       ucl | 110.934748 |    100.563468 |
|       lsl |          0 |             0 |
|       usl |       80.0 |          80.0 |
|        cp |   0.588237 |      0.768072 |
|       cpk |   0.545076 |      0.605145 |
|         Z |   1.635227 |      1.815434 |
|      Z_lt |   0.135227 |      0.315434 |

![Process Capability Analysis](https://raw.githubusercontent.com/j4ggr/DaSPi/main/docs/img/cpk-analysis_drop-card.png)

## About DaSPi

DaSPi was created and is actively maintained by **Reto Jäggli**, a Data Scientist at Festo Microtechnology AG. 
Much of the development happens during spare time, driven by a passion for making data analysis, statistics, and process improvement more accessible and integrated.

Contributions to DaSPi are very welcome!
If you find bugs or have ideas for improvements, please report them or submit pull requests on the [GitHub repository](https://github.com/j4ggr/DaSPi), where the full source code is also available for review.

**Important Notice:**  
DaSPi is still under heavy development and may contain hidden bugs. 
While every effort is made to ensure reliability, no warranty is provided. 
The results obtained using DaSPi should be double-checked with other trusted statistical software whenever possible. 
Where applicable, DaSPi acts as a convenient wrapper around well-established packages such as pandas, numpy, matplotlib, scipy, and statsmodels, leveraging their robustness and functionality.
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

