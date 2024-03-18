import patsy

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from typing import List
from typing import Dict
from typing import DefaultDict
from collections import defaultdict
from pandas.core.frame import DataFrame
from scipy.optimize._optimize import OptimizeResult
from statsmodels.regression.linear_model import RegressionResults

from .utils import encoded_dmatrices
from .utils import goodness_of_fit_metrics

from ..constants import ANOVA


class LinearModel:
    """
    
    Parameters
    ----------
    source : pandas DataFrame
        Tabular data for the model
    target : str
        Column name of endogene variable
    features : List[str]
        Column names of exogene variables
    covariate
    """

    __slots__ = (
        'source', '_target', '_features', '_covariates', 'formula',
        'design_matrix', 'level_map', 'results')
    source: DataFrame
    _target: str
    _features: List[str]
    _covariates: List[str]
    formula: str
    design_matrix: DataFrame
    level_map: dict
    results: RegressionResults | None
    metrics: DefaultDict

    def __init__(
            self,
            source: DataFrame,
            target: str,
            features: List[str],
            covariates: List[str] = []) -> None:
        self.source = source
        self._target = target
        self._features = features
        self._covariates = covariates
        self.formula = ''
        self.design_matrix = pd.DataFrame()
        self.level_map = {}
        self._results = None
        self.metrics = defaultdict(list)
    
    @property
    def target(self) -> str:
        """Get cleaned target name (read-only)."""
        return self.clean_column_name(self._target)
    
    @property
    def features(self) -> List[str]:
        """Get cleaned features (read-only)."""
        return list(map(self.clean_feature_name, self._features))
    
    @property
    def covariates(self) -> List[str]:
        """Get cleaned covariates (read-only)."""
        return list(map(self.clean_feature_name, self._covariates))

    @property
    def columns_map(self) -> Dict[str, str]:
        """Get the column mapping dict where the keys are the original
        names and the values are the cleaned names (read-only)."""
        originals = [self._target] + self._features + self._covariates
        cleaned = [self.target] + self.features + self.covariates
        return {o: c for o, c in zip(originals, cleaned)}
    
    @property
    def results(self) -> RegressionResults:
        """Get fitted regression results. Raises AssertionError if no
        model is fitted yet."""
        assert self._results is not None, (
            'Model not fitted yet, pls call `fit` method first.')
        return self._results
    
    @property
    def exogenous(self) -> List[str]:
        """Get column names of exogene values of constructed design 
        matrix (read-only)."""
        skip = (ANOVA.INTERCEPT, self.target)
        return [c for c in self.design_matrix.columns if c not in skip]
    
    @property
    def effects(self) -> pd.Series:
        """Calculates the impact of each factor on the target. The effects 
        are described as twice the parameter coefficients"""
        effects = 2 * self.results.params
        if ANOVA.INTERCEPT in effects.index:
            effects.loc[ANOVA.INTERCEPT] = np.nan
        return effects
    
    @staticmethod
    def clean_column_name(name: str) -> str:
        """Clean column name so it can be used in patsy formula"""
        name = (name
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('[', '')
            .replace(']', ''))
        return name
    
    def _least_param_by_effect_(self) -> str:
        """Get the parameter that has the smallest effect on the target 
        variable. If it has multiple with the smallest effect, the highest 
        interaction is returned"""
        effects = self.effects
        smallest = np.where(effects == effects.min())[0]
        return effects.index[smallest[-1]]

    def _least_param_by_pvalue_(model: RegressionResults) -> str:
        """Get the parameter with the least two-tailed p value for the t-stats."""
        return model.pvalues.index[model.pvalues.argmax()]

    def least_param(self) -> str:
        """Get the least significant parameter for the model"""
        if any(self.results.pvalues.isna()):
            return self._least_param_by_effect_()
        else:
            return self._least_param_by_pvalue_()
    
    def construct_design_matrix(
            self, encoded: bool = True, complete: bool = False) -> None:
        """Construct a design matrix by given target, features and 
        covariates. The created design matrix is then set to the 
        `design_matrix` attribute.

        Parameters
        ----------
        encode : bool, optional
            Flag whether to encode the exogenous such that the lowest
            level is -1 and the highest level is 1. The interactions are
            the product of the main factors. Categorical values are 
            one-hot encoded, by default True
        complete : bool, optional
            Flag whether to construct a predictor matrix that consists 
            a complete model, i.e. all combinations of interactions up 
            to the length of the features. No interactions are created
            for the covariates, by default False
        """
        self.formula = (
            f'{self.target} ~ '
            + '*'.join(self.features) if complete else '+'.join(self.features)
            + '+'.join(['', *self.covariates]) if self.covariates else '')
        if encoded:
            y, X, self.level_map = encoded_dmatrices(self.source, self.formula)
        else:
            y, X = patsy.dmatrices(
                self.formula, self.source, return_type='dataframe')
        self.design_matrix = pd.concat([X, y], axis=1)
    
    def fit(self, **kwds) -> None:
        self._results = smf.ols(self.formula, self.design_matrix, **kwds).fit()
        self.metrics = goodness_of_fit_metrics(
            self.results, metrics=self.metrics, formula=self.formula, 
            least=self.least_param)
