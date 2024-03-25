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
from scipy.optimize._optimize import OptimizeResult
from statsmodels.regression.linear_model import RegressionResults

from .utils import hierarchical
from .utils import is_main_feature
from .utils import encoded_dmatrices

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
        '_source', '_target', '_features', '_covariates', 'formula',
        'design_matrix', 'level_map', '_results', 'gof_metrics')
    _source: DataFrame
    _target: str
    _features: List[str]
    _covariates: List[str]
    formula: str
    design_matrix: DataFrame
    level_map: dict
    _results: RegressionResults | None
    gof_metrics: DefaultDict

    def __init__(
            self,
            source: DataFrame,
            target: str,
            features: List[str],
            covariates: List[str] = []) -> None:
        self._source = source
        self._target = target
        self._features = features
        self._covariates = covariates
        self.formula = ''
        self.design_matrix = pd.DataFrame()
        self.level_map = {}
        self._results = None
        self.gof_metrics = defaultdict(list)
    
    @property
    def source(self) -> DataFrame:
        """Get source data with cleaned column names"""
        return self._source.rename(columns=self.columns_map)
    
    @property
    def target(self) -> str:
        """Get cleaned target name (read-only)."""
        return self.clean_column_name(self._target)
    
    @property
    def features(self) -> List[str]:
        """Get cleaned features (read-only)."""
        return list(map(self.clean_column_name, self._features))
    
    @property
    def covariates(self) -> List[str]:
        """Get cleaned covariates (read-only)."""
        return list(map(self.clean_column_name, self._covariates))

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
        model is fitted yet (read-only)."""
        assert self._results is not None, (
            'Model not fitted yet, pls call `fit` method first.')
        return self._results

    @property
    def p_least(self) -> float:
        """Get highest p-value (read-only)."""
        return self.results.pvalues.max()

    @property
    def exogenous(self) -> List[str]:
        """Get column names of exogene values of fitted model
        (read-only)."""
        return self.exogene_names
    
    @property
    def main_features(self) -> List[str]:
        """Get all main parameters of current regression results
        excluding intercept (read-only)."""
        return [n for n in self.exogenous if is_main_feature(n)]
    
    @property
    def effects(self) -> pd.Series:
        """Calculates the impact of each factor on the target. The effects 
        are described as twice the parameter coefficients (read-only)."""
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
    
    def is_hierarchical(self) -> bool:
        """Check if current fitted model is hierarchical."""
        features = list(self.results.params.index)
        hierarchical_features = hierarchical(features)
        return all([f in features for f in hierarchical_features])
    
    def add_gof_metrics(self) -> None:
        """Add different goodness-of-fit metric to `gof_metrics`
        attribute.
        
        Keys:
        - 'formula' = the corresponding regression formula
        - 'least_feature' = the least significant feature
        - 'aic' = Akaike's information criteria
        - 'rsquared' = R-squared of the model
        - 'rsquared_adj' = adjusted R-squared
        - 'p_least' = The least two-tailed p value for the t-stats of the 
        params.
        - 'hierarchical' = boolean value if model is hierarchical
        """
        self.gof_metrics['formula'].append(self.formula)
        self.gof_metrics['least_feature'].append(self.least_feature())
        for metric in ['aic', 'rsquared', 'rsquared_adj']:
            self.gof_metrics[metric].append(getattr(self.results, metric))
        self.gof_metrics['p_least'].append(self.p_least)
        self.gof_metrics['hierarchical'].append(self.is_hierarchical())
    
    def _least_by_effect_(self) -> str:
        """Get the feature that has the smallest effect on the target
        variable. If it has multiple with the smallest effect, the
        highest interaction is returned."""
        effects = self.effects
        smallest = np.where(effects == effects.min())[0]
        return effects.index[smallest[-1]]

    def _least_by_pvalue_(self) -> str:
        """Get the feature with the least two-tailed p value for the t-stats."""
        index = self.results.pvalues.argmax()
        return self.results.pvalues.index[index]

    def least_feature(self) -> str:
        """Get the least significant feature (parameter) for the model"""
        no_p = any(self.results.pvalues.isna())
        return self._least_by_effect_() if no_p else self._least_by_pvalue_()
    
    def construct_design_matrix(
            self, encoded: bool = False, complete: bool = False) -> Self:
        """Construct a design matrix by given target, features and 
        covariates. The created design matrix is then set to the 
        `design_matrix` attribute.

        Parameters
        ----------
        encode : bool, optional
            Flag whether to encode the exogenous such that the lowest
            level is -1 and the highest level is 1. The interactions are
            the product of the main factors. Categorical values are 
            one-hot encoded, by default False
        complete : bool, optional
            Flag whether to construct a predictor matrix that consists 
            a complete model, i.e. all combinations of interactions up 
            to the length of the features. No interactions are created
            for the covariates, by default False
        """
        self.formula = f'{self.target}~'
        formula = (
            self.formula
            + ('*'.join(self.features) if complete else '+'.join(self.features))
            + ('+'.join(['', *self.covariates]) if self.covariates else ''))
        if encoded:
            y, X, self.level_map = encoded_dmatrices(self.source, formula)
        else:
            y, X = patsy.dmatrices(
                formula, self.source, return_type='dataframe')
        self.design_matrix = pd.concat([X, y], axis=1)
        self.formula += (
            f'{"+".join([c for c in X.columns if c != ANOVA.INTERCEPT])}')
        return self
    
    def fit(self, **kwds) -> Self:
        """First create design matrix if not allready done. Then create
        and fit a ordinary least squares model using current formula. 
        Finally calculate some goodness-of-fit metrics and store them
        in `metrics` attribute
        
        Parameters
        ----------
        **kwds
            Additional keyword arguments for `ols` function of 
            `statsmodels.formula.api`.
        """
        if self.design_matrix.empty:
            self.construct_design_matrix()
        self._results = smf.ols(self.formula, self.design_matrix, **kwds).fit()
        self.add_gof_metrics()
        return self
    
    def recursive_feature_elimination(
            self, alpha: float = 0.5, rsquared_max: float = 0.99,
            ensure_hierarchy: bool = True) -> Self:
        """Perform a linear regression starting with complete model.
        Then recursive features are eliminated according to the highest
        p-value (two-tailed p values for the t-stats of the params).
        Features are eliminated until only significant features
        (p-value < given threshold) remain in the model.

        Parameters
        ----------
        alpha : float in (0, 1), optional
            Significance threshold for p-value, by default 0.05
        rsquared_max : float in (0, 1), optional
            If given, the model must have a lower R^2 value than the given 
            threshold, by default 0.99
        ensure_hierarchy : bool, optional
            Adds features at the end to ensure model is hierarchical, 
            by default True
        """
        self.gof_metrics = defaultdict(list)
        _features = self.formula.split('~')[1].split('+')
        while len(_features) > 1:
            self.fit()
            if self.p_least <= alpha and self.results.rsquared <= rsquared_max:
                break
            if self.gof_metrics['least_feature'][-1] == ANOVA.INTERCEPT:
                _features.append('-1')
            else:
                _features.remove(self.gof_metrics['least_feature'][-1])
            self.formula = f'{self.target}~{"+".join(_features)}'

        if ensure_hierarchy and not self.is_hierarchical:
            self.formula = f'{self.target}~{"+".join(hierarchical(_features))}'
            self.fit()
        return self

    def highest_features(self) -> List[str]:
        """Get all main and interaction features that do not appear in a 
        higher interaction. Covariates are not taken into account here."""
        exclude = [[e] for e in [ANOVA.INTERCEPT] + self.covariates]
        features_splitted = sorted(
            list(map(lambda x: x.split(ANOVA.SEP), self.exogenous)), 
            key=len, reverse=True)
        
        features = []
        highest_level = len(features_splitted[0])
        for fs in features_splitted:
            level = len(fs)
            if level == highest_level:
                features.append(fs)
            else:
                check = []
                for f in features:
                    check.extend(set(fs) & set(f))
                if len(check) < level:
                    features.append(fs)
        return [ANOVA.SEP.join(f) for f in features if f not in exclude]
    
    def predict(
            self, xs: List[float], intercept: Literal[0, 1] = 1,
            negate: bool = False) -> float:
        """Predict y with given xs. Ensure that all non interactions are 
        given in xs
        
        Parameters
        ----------
        model : statsmodels RegressionResults
            fitted OLS model
        xs : array_like
            The values for which you want to predict. Make sure the 
            order matches the `main_features` property.
        negate : bool, optional
            If True, the predicted value is negated (used for 
            optimization), by default False
            
        Returns
        -------
        y : float
            Predicted value
        """
        assert len(xs) == len(self.main_features), (
            f'Please provide a value for each main feature')
        
        X = np.zeros(len(self.exogenous))
        for i, name in enumerate(self.exogenous):
            if ANOVA.SEP in name:
                continue
            X[i] = xs[i] if name != ANOVA.INTERCEPT else intercept
        y = self.results.predict(pd.DataFrame([X], columns=self.exogenous))
        return -y if negate else y
    #TODO optimizer

__all__ = [
    LinearModel.__name__
]
