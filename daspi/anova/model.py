import patsy

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from typing import Self
from typing import List
from typing import Dict
from typing import Literal
from typing import DefaultDict
from collections import defaultdict
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.optimize._optimize import OptimizeResult
from statsmodels.regression.linear_model import RegressionResults


class LinearModel:
    """This class is used to create and simplify linear models so that 
    only significant features describe the model.
    
    Balanced models (DOEs or EVOPs) including covariates can be 
    analyzed. With this class you can create an encoded design matrix
    with all factor levels including their interactions. All
    non-significant factors can then be automatically eliminated.
    Furthermore, this class allows the main effects, the sum of squares
    (explained variation) and the Anova table to be examined in more 
    detail.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas DataFrame as tabular data in a long format used for the
        model.
    target : str
        Column name of endogene variable
    features : List[str]
        Column names of exogene variables
    covariates : List[str], optional
        Column names for covariates. Covariates are logged confounding
        variables (features that cannot be specifically adjusted). No
        interactions are taken into account in the model for these
        features, by default []
    alpha : float, optional
        Threshold as alpha risk. All features including covariate and 
        intercept that are smaller than alpha are removed during the
        automatic elimination of the factors, by default 0.05
    """
    __slots__ = (
        'source', '_df', 'target', 'features', 'covariates', 'model', '_alpha',
        'input_map', 'output_map', 'gof_metrics', '_df')
    source: DataFrame
    _df: DataFrame
    target: str
    features: List[str]
    covariates: List[str]
    _alpha: float
    model: RegressionResults
    input_map: Dict[str, str]
    output_map: Dict[str, str]
    gof_metrics: DefaultDict

    def __init__(
            self,
            source: DataFrame,
            target: str,
            features: List[str],
            covariates: List[str] = [],
            alpha: float = 0.05) -> None:
        self.source = source
        self.target = target
        self.features = features
        self.covariates = covariates
        self.output_map = {target: 'y'}
        self.input_map = (
            {f: f'x_{i}' for i, f in enumerate(features)} 
            | {c: f'e_{i}' for i, c in enumerate(covariates)})
        self.alpha = alpha
        self.gof_metrics = defaultdict(list)

    @property
    def alpha(self) -> float:
        """Alpha risk as significance threshold for p-value of exegenous
        factors."""
        return self._alpha
    @alpha.setter
    def alpha(self, alpha: float) -> None:
        assert 0 < alpha < 1, 'Alpha risk must be between 0 and 1'
        self._alpha = alpha
    
    def construct_design_matrix(self) -> None:
        data = (self.source.rename(
            columns = self.input_map | self.output_map
            ))