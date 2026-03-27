"""ANOVA and measurement system analysis package.

This package provides tools for fitting, simplifying, and reporting
linear models in the context of Analysis of Variance (ANOVA), designed
experiments (DOE), and measurement system analysis (MSA).

`convert`
    Utility functions for reversing patsy categorical encoding and
    serialising DataFrames to HTML.

`tables`
    Stateless functions that compute ANOVA tables, effect sizes,
    variance inflation factors, and term p-values from statsmodels
    regression result objects.

`model`
    The three main model classes:

    - `LinearModel` — general OLS model with backward elimination,
      ANOVA reporting, and response optimisation.
    - `GageStudyModel` — MSA Type-1 study with full GUM uncertainty
      budget (CAL, RE, BI, LIN, EVR).
    - `GageRnRModel` — Gage R&R study decomposing measurement
      variation into repeatability (EV) and reproducibility (AV).

All public names from each submodule are re-exported at the package
level.
"""
from .convert import *

from .tables import *

from .model import *
