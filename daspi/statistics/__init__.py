"""Statistics package — confidence intervals, hypothesis tests,
estimators, and Monte Carlo simulation.

This package bundles four complementary submodules that cover the full
statistical workflow from raw samples to process-capability reporting:

`confidence`
    Two-sided confidence interval functions for mean, median, variance,
    standard deviation, proportions, Cp/Cpk, and regression predictions.
    All functions return ``(point_estimate, lower, upper)`` tuples.

`hypothesis`
    Hypothesis tests for normality (Anderson-Darling, KS), variance
    equality (F-test, Levene), location (t-test, Mann-Whitney U),
    proportions, and distribution shape (skewness, kurtosis).  Every
    function returns ``(p_value, statistic, ...)``.

`estimation`
    High-level estimator classes (`LocationDispersionEstimator`,
    `DistributionEstimator`, `ProcessEstimator`, `GageEstimator`) and
    standalone helpers for kernel density estimation, LOESS/LOWESS
    smoothing, and GUM measurement-uncertainty propagation.

`montecarlo`
    Data structures for encoding engineering specifications
    (`SpecLimits`, `Specification`) and classes for Monte Carlo process
    simulation (`RandomProcessValue`, `Binning`).

All public names from each submodule are re-exported at the package
level, so ``from daspi.statistics import mean_ci`` works without
knowing which submodule contains it.
"""
from .confidence import *

from .hypothesis import *

from .estimation import *

from .montecarlo import *
