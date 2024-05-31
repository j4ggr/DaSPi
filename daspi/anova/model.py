import patsy

import numpy as np
import pandas as pd
import patsy.highlevel
import statsmodels.api as sm
import statsmodels.formula.api as smf

from typing import Any
from typing import Set
from typing import Self
from typing import List
from typing import Dict
from typing import Literal
from typing import DefaultDict
from collections import defaultdict
from pandas.core.frame import DataFrame
from patsy.design_info import DesignInfo
from pandas.core.series import Series
from scipy.optimize._optimize import OptimizeResult
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .utils import hierarchical
from .utils import is_main_feature
from .utils import encoded_dmatrices
from .utils import clean_categorical_names

from ..constants import ANOVA


class LinearModel:
    """This class is used to create and simplify linear models so that 
    only significant features describe the model.
    
    Balanced models (DOEs or EVOPs) including covariates can be 
    analyzed. With this class, you can create an encoded design matrix
    with all factor levels, including their interactions. All
    non-significant factors can then be automatically eliminated.
    Furthermore, this class allows the examination of main effects, 
    the sum of squares (explained variation), and the Anova table in 
    more detail.
    
    Parameters
    ----------
    source : pandas DataFrame
        Pandas DataFrame as tabular data in a long format used for the
        model.
    target : str
        Column name of the endogenous variable.
    features : List[str]
        Column names of the exogenous variables.
    covariates : List[str], optional
        Column names for covariates. Covariates are logged confounding
        variables (features that cannot be specifically adjusted). No
        interactions are taken into account in the model for these
        features. Default is an empty list.
    alpha : float, optional
        Threshold as alpha risk. All features, including covariates and 
        intercept, that have a p-value smaller than alpha are removed 
        during the automatic elimination of the factors. Default is 0.05.
    """
    __slots__ = (
        'source', 'dmatrix', 'target', 'features', 'covariates', '_model', 
        '_alpha', 'input_map', 'output_map', 'gof_metrics', 'exclude',
        'gof_metrics', 'design_info', 'level_map')
    source: DataFrame
    dmatrix: DataFrame
    target: str
    features: List[str]
    covariates: List[str]
    _alpha: float
    _model: RegressionResultsWrapper | None
    input_map: Dict[str, str]
    output_map: Dict[str, str]
    gof_metrics: DefaultDict
    exclude: Set[str]
    gof_metrics: DefaultDict
    design_info: DesignInfo
    level_map: Dict[Any, Any]

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
            {f: f'x{i}' for i, f in enumerate(features)} 
            | {c: f'e{i}' for i, c in enumerate(covariates)})
        self.alpha = alpha
        self.gof_metrics = defaultdict(list)
        self.dmatrix = pd.DataFrame()
        self.exclude = set()
        self._model = None
        self.gof_metrics = defaultdict(list)
    
    @property
    def model(self) -> RegressionResultsWrapper:
        """Get regression results of fitted model. Raises AssertionError
        if no model is fitted yet (read-only)."""
        assert self._model is not None, (
            'Model not fitted yet, call `fit` method first.')
        return self._model
    
    @property
    def p_values(self) -> 'Series[float]':
        if not isinstance(self.model.pvalues, Series):
            return pd.Series(self.model.pvalues)
        return self.model.pvalues

    @property
    def p_least(self) -> float:
        """Get highest p-value (read-only)."""
        return self.p_values.max()
    
    @property
    def main_features(self) -> List[str]:
        """Get all main parameters of current model excluding intercept
        (read-only)."""
        return [n for n in self.exogenous if is_main_feature(n)]

    @property
    def alpha(self) -> float:
        """Alpha risk as significance threshold for p-value of exegenous
        factors."""
        return self._alpha
    @alpha.setter
    def alpha(self, alpha: float) -> None:
        assert 0 < alpha < 1, 'Alpha risk must be between 0 and 1'
        self._alpha = alpha
    
    @property
    def endogenous(self) -> str:
        """Get column name of endogenous values of design matrix
        (read-only)."""
        return self.output_map[self.target]
    
    @property
    def exogenous(self) -> List[str]:
        """Get column names of all exogene values of design matrix for
        current model (read-only)."""
        ignore = set(list(self.exclude) + [self.endogenous])
        return [c for c in self.dmatrix.columns if c not in ignore]
    
    def construct_design_matrix(
            self, encode: bool = False, complete: bool = False) -> Self:
        """Construct a design matrix by given target, features and 
        covariates. The created design matrix is then set to the 
        `dmatrix` attribute.

        Parameters
        ----------
        encode : bool, optional
            Flag indicating whether to encode the exogenous variables.
            If set to True, the categorical variables will be one-hot
            encoded, and the numerical variables will be scaled such
            that the lowest level is -1 and the highest level is 1.
            The interactions between the main factors will be
            represented as the product of the main factors.
            Covariates will not be encoded. Default is False.
        complete : bool, optional
            Flag whether to construct a predictor matrix that consists 
            a complete model, i.e. all combinations of interactions up 
            to the length of the features. No interactions are created
            for the covariates. Default is False.
        """
        y: DataFrame
        X: DataFrame
        data = self.source.rename(columns = self.input_map | self.output_map)
        features = [x for x in self.input_map.values() if x.startswith('x')]
        covariates = [e for e in self.input_map.values() if e.startswith('e')]
        formula = (
            f'{self.output_map[self.target]}~'
            + ('*'.join(features) if complete else '+'.join(features))
            + ('+'.join(['', *covariates]) if covariates else ''))
        if encode:
            y, X, self.level_map, self.design_info = encoded_dmatrices(
                data, formula, covariates)
        else:
            y, X = patsy.highlevel.dmatrices(
                formula, data, return_type='dataframe') # type: ignore
            self.design_info = X.design_info # type: ignore
            X = X.rename(
                columns={c: clean_categorical_names(c) for c in X.columns})
        self.dmatrix = pd.concat([y, X], axis=1)
        return self
    
    def fit(self, **kwds) -> Self:
        """First create design matrix if not allready done. Then create
        and fit a ordinary least squares model using current formula. 
        Finally calculate some goodness-of-fit metrics and store them
        in `metrics` attribute
        
        Parameters
        ----------
        **kwds
            Additional keyword arguments for `OLS` class of 
            `statsmodels.api`.
        """
        if self.dmatrix.empty:
            self.construct_design_matrix()

        self._model = sm.OLS(
            self.dmatrix[self.endogenous],
            self.dmatrix[self.exogenous], **kwds).fit()
        self.store_gof_metrics()
        return self
    
    def is_hierarchical(self) -> bool:
        """Check if current fitted model is hierarchical."""
        features = list(self.model.params.index)
        return all([hf in features for hf in hierarchical(features)])
    
    def effects(self) -> Series:
        """Calculates the impact of each factor on the target. The
        effects are described as twice the parameter coefficients and 
        occur as an absolute number."""
        params: Series = 2 * self.model.params
        if ANOVA.INTERCEPT in params.index:
            params.pop(ANOVA.INTERCEPT)
        effects = params.abs()
        effects.name = ANOVA.EFFECTS
        effects.index.name = ANOVA.FEATURES
        return effects
    
    def store_gof_metrics(self) -> None:
        """Add different goodness-of-fit metric to `gof_metrics`
        attribute.
        
        Keys:
        
        - 'least_feature' = the least significant feature
        - 'aic' = Akaike's information criteria
        - 'rsquared' = R-squared of the model
        - 'rsquared_adj' = adjusted R-squared
        - 'p_least' = The least two-tailed p value for the t-stats of the 
        params.
        - 'hierarchical' = boolean value if model is hierarchical
        """
        self.gof_metrics['exogenous'].append(self.exogenous)
        self.gof_metrics['least_feature'].append(self.least_feature())
        for metric in ['aic', 'rsquared', 'rsquared_adj']:
            self.gof_metrics[metric].append(getattr(self.model, metric))
        self.gof_metrics['p_least'].append(self.p_least)
        self.gof_metrics['hierarchical'].append(self.is_hierarchical())

    def _least_by_effect_(self) -> str:
        """Get the feature that has the smallest effect on the target
        variable. If it has multiple with the smallest effect, the
        highest interaction is returned."""
        effects = self.effects()
        smallest = np.where(effects == effects.min())[0]
        return str(effects.index[smallest[-1]])

    def _least_by_pvalue_(self) -> str:
        """Get the feature with the least two-tailed p value for the t-stats."""
        return str(self.p_values.index[self.p_values.argmax()])

    def least_feature(self) -> str:
        """Get the least significant feature (parameter) for the model"""
        no_p = any(self.p_values.isna())
        return self._least_by_effect_() if no_p else self._least_by_pvalue_()

    def recursive_feature_elimination(
            self, rsquared_max: float = 0.99, ensure_hierarchy: bool = True
            ) -> Self:
        """Perform a linear regression starting with complete model.
        Then recursive features are eliminated according to the highest
        p-value (two-tailed p values for the t-stats of the params).
        Features are eliminated until only significant features
        (p-value < given threshold) remain in the model.

        Parameters
        ----------
        rsquared_max : float in (0, 1), optional
            If given, the model must have a lower R^2 value than the given 
            threshold, by default 0.99
        ensure_hierarchy : bool, optional
            Adds features at the end to ensure model is hierarchical, 
            by default True

        Notes
        -----
        The attributes `gof_metrics` and `exclude` are reset (and thus 
        also the exogenous).
        """
        self.exclude = set()
        self.gof_metrics = defaultdict(list)
        max_steps = len(self.exogenous)
        for step in range(max_steps):
            self.fit()
            if (self.p_least <= self.alpha 
                and self.model.rsquared <= rsquared_max
                or len(self.exogenous) == 1):
                break
            self.exclude.add(self.gof_metrics['least_feature'][-1])

        if ensure_hierarchy and not self.is_hierarchical:
            h_features = hierarchical(self.exogenous)
            self.exclude = {e for e in self.exclude if e not in h_features}
            self.fit()
        return self
    
    def anova(self) -> DataFrame:
        """Calculate anova table for fitted model. If the fitted model
        has sigificant interactions anova typ II is performed, otherwise
        typ I (https://stats.stackexchange.com/a/93031)."""
        typ = 'II' if self.has_significant_interactions() else 'I'
        anova = (sm.stats
            .anova_lm(self.model, typ=typ)
            .reset_index(drop=False)
            .rename(columns={'index': ANOVA.FEATURES}))
        return anova
    
    def anova_typ1(self) -> DataFrame:
        anova = self.effects().to_frame()
        return anova

    def highest_features(self) -> List[str]:
        """Get all main and interaction features that do not appear in a 
        higher interaction. Covariates are not taken into account here."""
        _features = [f for f in self.exogenous if not f.startswith('e_')]
        features_splitted = sorted(
            [f.split(ANOVA.SEP) for f in _features], 
            key=len, reverse=True)
        
        features = []
        highest_level = len(features_splitted[0])
        for f_split in features_splitted:
            level = len(f_split)
            if level == highest_level:
                features.append(f_split)
            else:
                intersect = [i for f in features for i in set(f_split) & set(f)]
                if len(intersect) < level:
                    features.append(f_split)
        return [ANOVA.SEP.join(f) for f in features]

    def has_significant_interactions(self) -> bool:
        """True if fitted model has significant interactions."""
        for feature in self.highest_features():
            if ANOVA.SEP not in feature:
                continue
            if self.p_values[feature] < self.alpha:
                return True
        return False
    
    def predict(
            self, xs: List[float], intercept: Literal[0, 1] = 1,
            negate: bool = False) -> float:
        """Predict y with given xs. Ensure that all non interactions are 
        given in xs
        
        Parameters
        ----------
        xs : array_like
            The values for which you want to predict. Make sure the 
            order matches the `main_features` property.
        intercept : Literal[0, 1], optional
            Factor level for the intercept, either 0 or 1, by default 1
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
        
        features = self.exogenous + [ANOVA.INTERCEPT]
        X = np.zeros(len(features))
        for i, feature in enumerate(self.exogenous):
            if ANOVA.SEP not in feature:
                X[i] = xs[i]
        X[-1] = intercept
        y = float(self.model.predict(pd.DataFrame([X], columns=features))) # type: ignore
        return -y if negate else y
