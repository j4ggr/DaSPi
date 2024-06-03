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

from .utils import uniques
from .utils import hierarchical
from .utils import get_term_name
from .utils import is_main_feature

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
        'data', 'dmatrix', 'target', 'features', 'covariates', '_model', 
        '_alpha', 'input_map', 'input_rmap', 'output_map', 'gof_metrics',
        'exclude', 'level_map', 'initial_formula', '_anova')
    data: DataFrame
    target: str
    features: List[str]
    covariates: List[str]
    _alpha: float
    _model: RegressionResultsWrapper | None
    input_map: Dict[str, str]
    input_rmap: Dict[str, str]
    output_map: Dict[str, str]
    gof_metrics: DefaultDict
    exclude: Set[str]
    level_map: Dict[Any, Any]
    initial_formula: str
    _anova: DataFrame

    def __init__(
            self,
            source: DataFrame,
            target: str,
            features: List[str],
            covariates: List[str] = [],
            alpha: float = 0.05,
            complete: bool = False) -> None:
        self.target = target
        self.features = features
        self.covariates = covariates
        self.output_map = {target: 'y'}
        _features = tuple(f'x{i}' for i in range(len(features)))
        _covariates = tuple(f'e{i}' for i in range(len(covariates)))
        self.input_map = (
            {f: _f for f, _f in zip(features, _features)}
            | {c: _c for c, _c in zip(covariates, _covariates)})
        self.input_rmap = {v: k for k, v in self.input_map.items()}
        self.alpha = alpha
        self.gof_metrics = defaultdict(list)
        self.dmatrix = pd.DataFrame()
        self.exclude = set()
        self._model = None
        self.data = (source
            .copy()
            .rename(columns=self.input_map|self.output_map))
        self.initial_formula = (
            f'{self.output_map[self.target]}~'
            + ('*'.join(_features) if complete else '+'.join(_features))
            + ('+'.join(['', *_covariates]) if _covariates else ''))
    
    @property
    def model(self) -> RegressionResultsWrapper:
        """Get regression results of fitted model. Raises AssertionError
        if no model is fitted yet (read-only)."""
        assert self._model is not None, (
            'Model not fitted yet, call `fit` method first.')
        return self._model
    
    @property
    def p_values(self) -> 'Series[float]':
        """P-value for significance of adding model terms using anova
        typ III table for current model (read-only)."""
        if self._anova.empty:
            return pd.Series({t: np.nan for t in self.design_info.term_names})
        return self._anova['PR(>F)'].iloc[:-2]

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
    def design_info(self) -> DesignInfo:
        """Get the DesignInfo instance of current fitted model
        (read-only)."""
        return self.model.model.data.design_info
    
    @property
    def term_names(self) -> List[str]:
        """Get the names of all terms variables for the current fitted 
        model (read-only)."""
        return self.model.model.data.design_info.term_names
    
    @property
    def main_features(self) -> List[str]:
        """Get all main parameters of current model excluding intercept
        (read-only)."""
        return [n for n in self.term_names if is_main_feature(n)]
    
    @property
    def formula(self) -> str:
        """Get the formula used for the linear model, excluding any
        factors specified in the `exclude` attribute. The formula is
        constructed based on the excluded factors. If the intercept is
        excluded, the formula will include '-1' as the first term.
        Otherwise, the formula will include the original terms
        (excluding the excluded factors) separated by '+' (read-only)."""
        if self._model is None:
            return self.initial_formula
    
        ignore = list(self.exclude) + [ANOVA.INTERCEPT]
        terms = [t for t in self.term_names if t not in ignore]
        if ANOVA.INTERCEPT in self.exclude:
            terms = ['-1'] + terms
        return f'{self.output_map[self.target]}~{"+".join(terms)}'
    
    def _reverse_single_term_name_(self, term_name: str) -> str:
        """Reverse the single term name using the original names stored
        in `input_rmap`.

        Parameters
        ----------
        term_name : str
            The term name (no interaction).

        Returns
        -------
        str
            The reversed term name.
        """
        split = term_name.split('_')
        split[0] = self.input_rmap.get(split[0], split[0])
        return '_'.join(split)

    def _reverse_term_name_(self, term_name: str) -> str:
        """Reverse the term name using the feature or covariate names
        provided when initializing.

        Parameters
        ----------
        term_name : str
            The term name of the design info.

        Returns
        -------
        str
            The reversed term name.
        """
        if term_name == ANOVA.INTERCEPT:
            original_name = term_name
        else:
            original_name = ANOVA.SEP.join(map(
                self._reverse_single_term_name_,
                term_name.split(ANOVA.SEP)))
        return original_name
    
    def is_hierarchical(self) -> bool:
        """Check if current fitted model is hierarchical."""
        features = list(self.model.params.index)
        return all([hf in features for hf in hierarchical(features)])
    
    def effects(self) -> Series:
        """Calculates the impact of each term on the target. The
        effects are described as twice the parameter coefficients and 
        occur as an absolute number."""
        params: Series = 2 * self.model.params
        if ANOVA.INTERCEPT in params.index:
            params.pop(ANOVA.INTERCEPT)
        names_map = {n: get_term_name(n) for n in params.index}
        effects = params.abs()
        effects = (params
            .abs()
            .rename(index=names_map)
            .groupby(level=0, axis=0)
            .sum()
            [uniques(names_map.values())])
        effects.name = ANOVA.EFFECTS
        effects.index.name = ANOVA.FEATURES
        return effects
    
    def store_gof_metrics(self) -> None:
        """Add different goodness-of-fit metric to `gof_metrics`
        attribute.
        
        Keys:
        
        - 'least_term' = the least significant term
        - 'aic' = Akaike's information criteria
        - 'rsquared' = R-squared of the model
        - 'rsquared_adj' = adjusted R-squared
        - 'p_least' = The least two-tailed p value for the t-stats of the 
        params.
        - 'hierarchical' = boolean value if model is hierarchical
        """
        self.gof_metrics['formula'].append(self.formula)
        self.gof_metrics['least_term'].append(self.least_term())
        for metric in ['aic', 'rsquared', 'rsquared_adj']:
            self.gof_metrics[metric].append(getattr(self.model, metric))
        self.gof_metrics['p_least'].append(self.p_values.max())
        self.gof_metrics['hierarchical'].append(self.is_hierarchical())

    def least_term(self) -> str:
        """Get the term name with the least effect or the least p-value.

        Returns
        -------
        str
            The term name with the least effect or the least p-value.

        Notes
        -----
        This method checks if any p-values are missing (NaN). If there
        are missing p-values, it returns the term name that has the 
        smallest effect on the target variable. Otherwise, it returns
        the term name with the least p-value for the F-stats coming from
        current ANOVA table.
        """
        if any(self.p_values.isna()):
            effects = self.effects()
            idx_smallest = np.where(effects == effects.min())[0][-1]
            least = str(effects.index[idx_smallest])
        else:
            least = str(self.p_values.index[self.p_values.argmax()])
        return least
    
    def fit(self, **kwds) -> Self:
        """Create and fit a ordinary least squares model using current 
        formula. Then perform an analysis of variance (ANOVA typ III)
        on the fitted model and store it in `_anova`. Finally calculate 
        some goodness-of-fit metrics and store them in `metrics` 
        attribute.
        
        Parameters
        ----------
        **kwds
            Additional keyword arguments for `ols` function of 
            `statsmodels.formula.api`.
        """
        self._model = smf.ols(self.formula, self.data).fit()
        if all(self.model.pvalues.isna()):
            self._anova = pd.DataFrame()
        else:
            self._anova = self.anova()
        self.store_gof_metrics()
        return self

    def recursive_feature_elimination(
            self, rsquared_max: float = 0.99, ensure_hierarchy: bool = True,
            **kwds) -> Self:
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
        **kwds
            Additional keyword arguments for `ols` function of 
            `statsmodels.formula.api`.

        Notes
        -----
        The attributes `gof_metrics` and `exclude` are reset (and thus 
        also the exogenous).
        """
        self._model = None
        self.exclude = set()
        self.gof_metrics = defaultdict(list)
        self.fit(**kwds)
        max_steps = len(self.term_names)
        for _ in range(max_steps):
            if (self.p_values.max() <= self.alpha 
                and self.model.rsquared <= rsquared_max
                or len(self.term_names) == 1):
                break
            self.exclude.add(self.gof_metrics['least_term'][-1])
            self.fit(**kwds)

        if ensure_hierarchy and not self.is_hierarchical:
            h_features = hierarchical(self.term_names)
            self.exclude = {e for e in self.exclude if e not in h_features}
            self.fit(**kwds)
        return self
    
    def anova(self, typ: Literal['', 'I', 'II', 'III'] = 'III') -> DataFrame:
        """Perform an analysis of variance (ANOVA) on the fitted model.

        Parameters
        ----------
        typ : Literal['', 'I', 'II', 'III'], optional
            The type of ANOVA to perform. Default is 'III', see notes
            for more informations about the types.
            - '' : If no type is specified, 'II' is used if the model 
                has significant interactions, otherwise 'I' is used.
            - 'I' : Type I sum of squares ANOVA.
            - 'II' : Type II sum of squares ANOVA.
            - 'III' : Type III sum of squares ANOVA.

        Returns
        -------
        DataFrame
            The ANOVA table containing the sum of squares, degrees of 
            freedom, mean squares, F-statistic, and p-value.

        Notes
        -----
        The Minitab software uses Type III by default, so that is what 
        we will use here. A discussion on which one to use can be found 
        here:
        https://stats.stackexchange.com/a/93031
        
        The ANOVA table provides information about the significance of 
        each factor and interaction in the model. The type of ANOVA 
        determines how the sum of squares is partitioned among the 
        factors.

        If the model does not have a 'design_info' attribute, it is set 
        to the value of 'self.design_info' before performing the ANOVA.

        Examples
        --------
        >>> import daspi
        >>> df = daspi.load_dataset('anova3')
        >>> lm = LinearModel(df, 'Cholesterol', ['Sex', 'Risk', 'Drug'])
        >>> lm.fit()
        >>> anova_table = lm.anova(typ='III')
        >>> print(anova_table)
                  df     sum_sq    mean_sq          F    PR(>F)
        Sex        1   2.074568   2.074568   2.406818  0.126543
        Risk       1  11.332130  11.332130  13.147017  0.000631
        Drug       2   0.815743   0.407872   0.473194  0.625522
        Residual  55  47.407498   0.861955        NaN       NaN
        Total     59  61.629940   1.044575        NaN       NaN
        """
        total_columns = ['df', 'sum_sq']
        if not typ:
            typ = 'II' if self.has_significant_interactions() else 'I'
        if not hasattr(self.model.model.data, 'design_info'):
            self.model.model.data.design_info = self.design_info
        anova = sm.stats.anova_lm(self.model, typ=typ)
        anova.index = (anova.index
            .set_names(ANOVA.SOURCE)
            .map(self._reverse_term_name_))
        if ANOVA.INTERCEPT in anova.index:
            anova = anova.drop(ANOVA.INTERCEPT, axis=0)
        anova.loc[ANOVA.TOTAL, total_columns] = anova[total_columns].sum()
        anova['mean_sq'] = anova['sum_sq']/anova['df']
        anova['df'] = anova['df'].astype(int)
        return anova[ANOVA.COLNAMES]

    def highest_features(self) -> List[str]:
        """Get all main and interaction features that do not appear in a 
        higher interaction. Covariates are not taken into account here."""
        _features = [f for f in self.term_names if not f.startswith('e_')]
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
        
        X = np.zeros(len(self.term_names))
        for i, feature in enumerate(self.term_names):
            if ANOVA.SEP not in feature:
                X[i] = xs[i]
        X[-1] = intercept
        y = float(self.model.predict(pd.DataFrame([X], columns=features))) # type: ignore
        return -y if negate else y
