"""
## Data analysis, Statistics and Process improvements (DaSPi)

Visualize and analyze your data with DaSPi. This package is designed for users who want to find relevant influencing factors in processes and validate improvements.
This package offers many [Six Sigma](https://en.wikipedia.org/wiki/Six_Sigma) tools based on the following packages:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://scipy.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)

The goal of this package is to be easy to use and flexible so that it can be adapted to a wide array of data analysis tasks.

# Why DaSPi?

There are great packages for data analysis and visualization in Python, such as [Pandas](https://pandas.pydata.org/pandas-docs/stable), [Seaborn](https://seaborn.pydata.org/index.html), [Altair](https://altair-viz.github.io/), [Statsmodels](https://www.statsmodels.org/stable/), [Scipy](https://docs.scipy.org/doc/scipy/), [Pinguins](https://pingouin-stats.org/). But most of the time they work not directly with each other. Wouldn't it be great if you could use all of these packages together in one place? That's where DaSPi comes in. DaSPi is a Python package that provides a unified interface for data analysis, statistics and visualization. It allows you to use all of the great packages mentioned above together in one place, making it easier to explore and understand your data.

# Features

- **Ease of Use:** DaSPi is designed to be easy to use, even for beginners. It provides a simple and intuitive interface that makes it easy to get started with data analysis.
- **Visualization:** DaSPi provides a wide range of visualization options, including multivariate charts, joint charts, and useful templates. This makes it easy to explore and understand your data in a visual way.
- **Statistics:** DaSPi provides a wide range of statistical functions and tests, including hypothesis testing, confidence intervals, and regression analysis. This makes it easy to explore and understand your data in a statistical way.
- **Open Source:** DaSPi is open source, which means that it is free to use and modify. This makes it a great option for users who want to customize the package to their specific needs.

This Package contains following submodules:

- **plotlib:** Visualizations with Matplotlib, where the division by color, marker size or shape as well as rows and columns subplots are automated depending on the given categorical data. Any plots can also be combined, such as scatter with contour plot, violin with error bars or other creative combinations.
- **anova:** analysis of variance (ANOVA), which is used to compare the variance within and between of two or more groups, or the effects of different treatments on a response variable. It also includes a function for calculating the variance inflation factor (VIF) for linear regression models. The main class is LinearModel, which provides methods for fitting linear regression with interactions and automatically elimiinating insignificant variables.
- **statistics:** applied statistics, hypothesis test and confidence calculations. It also includes estimation for process capability and capability index.
- **datasets:** data for exersices. It includes different datasets that can be used for testing and experimentation.

# Usage

To use DaSPi, you can import the package and start exploring your data. Here is an example of how to use DaSPi to analyze a dataset:

``` py
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

![Iris sepal length species](img/iris_contour_size-leaf-species.png)
"""

from ._version import __version__

from .strings import STR


from .constants import KW
from .constants import RE
from .constants import DIST
from .constants import LINE
from .constants import COLOR
from .constants import LABEL
from .constants import ANOVA
from .constants import PLOTTER
from .constants import DEFAULT
from .constants import CATEGORY
from .constants import SIGMA_DIFFERENCE


from .statistics.confidence import sem
from .statistics.confidence import cp_ci
from .statistics.confidence import cpk_ci
from .statistics.confidence import fit_ci
from .statistics.confidence import mean_ci
from .statistics.confidence import stdev_ci
from .statistics.confidence import median_ci
from .statistics.confidence import variance_ci
from .statistics.confidence import proportion_ci
from .statistics.confidence import bonferroni_ci
from .statistics.confidence import delta_mean_ci
from .statistics.confidence import prediction_ci
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

from .statistics.estimation import Loess
from .statistics.estimation import Lowess
from .statistics.estimation import Estimator
from .statistics.estimation import ProcessEstimator
from .statistics.estimation import estimate_distribution
from .statistics.estimation import estimate_kernel_density
from .statistics.estimation import estimate_kernel_density_2d
from .statistics.estimation import estimate_capability_confidence


from .plotlib import style

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
from .plotlib.plotter import Beeswarm
from .plotlib.plotter import Errorbar
from .plotlib.plotter import MeanTest
from .plotlib.plotter import LoessLine
from .plotlib.plotter import QuantileBoxes
from .plotlib.plotter import StripeLine
from .plotlib.plotter import StripeSpan
from .plotlib.plotter import HideSubplot
from .plotlib.plotter import SkipSubplot
from .plotlib.plotter import SpreadWidth
from .plotlib.plotter import Probability
from .plotlib.plotter import BlandAltman
from .plotlib.plotter import GaussianKDE
from .plotlib.plotter import VariationTest
from .plotlib.plotter import ProportionTest
from .plotlib.plotter import CenterLocation
from .plotlib.plotter import TransformPlotter
from .plotlib.plotter import StandardErrorMean
from .plotlib.plotter import ConfidenceInterval
from .plotlib.plotter import ParallelCoordinate
from .plotlib.plotter import GaussianKDEContour
from .plotlib.plotter import LinearRegressionLine
from .plotlib.plotter import CapabilityConfidenceInterval

from .plotlib.facets import AxesFacets
from .plotlib.facets import flat_unique
from .plotlib.facets import LabelFacets
from .plotlib.facets import StripesFacets

from .plotlib.chart import Chart
from .plotlib.chart import JointChart
from .plotlib.chart import SingleChart
from .plotlib.chart import MultivariateChart

from .plotlib.templates import ResidualsCharts
from .plotlib.templates import PairComparisonCharts
from .plotlib.templates import ParameterRelevanceCharts
from .plotlib.templates import BivariateUnivariateCharts


from .anova.convert import get_term_name
from .anova.convert import frames_to_html

from .anova.tables import uniques
from .anova.tables import anova_table
from .anova.tables import terms_effect
from .anova.tables import terms_probability
from .anova.tables import variance_inflation_factor

from .anova.model import hierarchical
from .anova.model import LinearModel
from .anova.model import is_main_feature


from .datasets import load_dataset
from .datasets import list_dataset

