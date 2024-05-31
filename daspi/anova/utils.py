import re
import patsy
import itertools

import numpy as np
import pandas as pd 

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Callable
import patsy.highlevel
from scipy.optimize import Bounds
from scipy.optimize import minimize
from pandas.api.types import is_numeric_dtype
from pandas.core.frame import DataFrame
from patsy.design_info import DesignInfo
from pandas.core.series import Series
from scipy.optimize._optimize import OptimizeResult
from statsmodels.regression.linear_model import RegressionResults

from ..constants import RE
from ..constants import ANOVA


def is_encoded_categorical(feature_name: str) -> bool:
    """True if given feature is a patsy encoded categorical feature."""
    return bool(RE.ENCODED_VALUE.findall(feature_name))

def clean_categorical_names(feature_name: str) -> str:
    """Clean feature name of patsy encoded categorical features. Such 
    names are structured as follows '<term name>[T.<value>]'.
    Cleaned name looks like '<term name>_<value>'."""
    if ANOVA.SEP in feature_name:
        split = feature_name.split(ANOVA.SEP)
        return ANOVA.SEP.join([clean_categorical_names(f) for f in split])
    else:
        name = RE.ENCODED_NAME.findall(feature_name)
        value = RE.ENCODED_VALUE.findall(feature_name)
        return f'{name[0]}_{value[0]}' if name and value else feature_name

def get_term_name(feature_name: str) -> str:
    """Get original term name of a patsy encoded categorical feature.
    Such names are structured as follows '<term name>[T.<value>]'."""
    match = RE.ENCODED_NAME.findall(feature_name)
    return feature_name if not match else match[0]

def decode_cat_main(
        feature_name: str, code: int, di: DesignInfo) -> str:
    """Decode 0 or 1 value of encoded categorical main feature. 
    The design info coming from patsy is used to get original values."""
    matches = RE.ENCODED_VALUE.findall(feature_name)
    sep = ''
    values = []
    if code == 1:
        sep = ' & '
        values = matches
    else:
        term_name = get_term_name(feature_name)
        for factor, info in di.factor_infos.items(): # type: ignore
            if term_name == factor.code:
                sep = ' | '
                values = [c for c in info.categories if str(c) not in matches]
                break
    return sep.join(map(str, values))

def encoded_dmatrices(
        data: DataFrame, formula: str, covariates: List[str] = []
        ) -> Tuple[DataFrame, DataFrame, Dict[str, tuple], DesignInfo]:
    """Encode all levels of the main factors such that the lowest level 
    is -1 and the highest level is 1. The interactions between the main 
    factors are represented as the product of the main factors. 
    Categorical values are one-hot encoded. Covariates are not encoded 
    if they are continuous variables.
    
    Parameters
    ----------
    data : pandas DataFrame
        Data with the real factor levels.
    formula : str
        The formula specifying the model used by patsy.
    covariates : List[str], optional
        List of column names for covariates. Covariates are not encoded 
        if they are continuous variables. Default is an empty list.
    
    Returns
    -------
    y : pandas DataFrame
        Target values.
    X_code : pandas DataFrame
        Encoded feature values.
    mapper : dict
        Information about how the levels are encoded.
        - key: str = feature name of the main level.
        - value: dict = key: original level, value: encoded level.
    design_info : DesignInfo
        A DesignInfo object that holds metadata about a design matrix.
    """
    y: DataFrame
    X: DataFrame
    design_info: DesignInfo
    y, X = patsy.highlevel.dmatrices(formula, data, return_type='dataframe') # type: ignore
    design_info = X.design_info # type: ignore
    features = list(design_info.column_name_indexes.keys())
    X_code = pd.DataFrame(0, index=X.index, columns=features)
    mapper = {}
    columns = {}
    for f in features:
        columns[f] = clean_categorical_names(f)
        if f == ANOVA.INTERCEPT or any([(e in f) for e in covariates]):
            x_encoded = X[f]
        elif is_encoded_categorical(f) and ANOVA.SEP not in f:
            x_encoded = X[f]
            mapper[columns[f]] = {
                decode_cat_main(f, i, design_info): i for i in (0, 1)}
        elif ANOVA.SEP not in f:
            uniques = sorted(X[f].unique())
            codes = np.linspace(-1, 1, len(uniques))
            mapper[f] = {u: c for u, c in zip(uniques, codes)}
            x_encoded = X[f].replace(mapper[f])
        else:
            split = f.split(ANOVA.SEP)
            x_encoded = X_code[split[0]]
            for s in split[1:]:
                x_encoded = x_encoded*X_code[s]
        X_code[f] = x_encoded
    X_code = X_code.rename(columns=columns)
    X_code.design_info = design_info
    return y, X_code, mapper, design_info

def hierarchical(features: List[str]) -> List[str]:
    """Get all features such that all lower interactions and main 
    effects are present with the same features that appear in the higher 
    interactions

    Parameters
    ----------
    features : list of str
        Columns of exogene variables
    
    Returns
    -------
    h_features : list of str
        Sorted features for hierarchical model"""

    h_features = set(features)
    for feature in features:
        split = feature.split(ANOVA.SEP)
        n_splits = len(split)
        for s in split:
            h_features.add(s)
        if n_splits <= ANOVA.SMALLEST_INTERACTION:
            continue

        for i in range(ANOVA.SMALLEST_INTERACTION, n_splits):
            for combo in map(ANOVA.SEP.join, itertools.combinations(split, i)):
                h_features.add(combo)

    return sorted(sorted(list(h_features)), key=lambda x: x.count(ANOVA.SEP))

def is_main_feature(feature: str) -> bool:
    """Check if given feature is a main parameter (intercept is 
    excluded)."""
    return feature != ANOVA.INTERCEPT and ANOVA.SEP not in feature

def decode(value: str|float, mapper: dict, feature: str) -> Any:
    """Get original value of encoded value for given feature

    Parameters
    ----------
    value : str or float
        encoded value used as 2. level key in mapper
    mapper : dict
        information about how the levels are encoded
        - key: str = feature name of main level
        - value: dict = key: originals, value: codes
    feature : str
        feature name used as 1. level key in mapper
    
    Returns
    -------
    orig : string, float or None
        value used in original data bevore encoding
    """
    for orig, code in mapper[feature].items():
        if isinstance(value, str): code = str(code)
        if code == value: return orig
    return None

def optimize(
        fun: Callable, x0: List[float], negate: bool, columns: List[str], 
        mapper: dict, bounds: Bounds|None = None, **kwds
        ) -> Tuple[List[float], float, OptimizeResult]:
    """Base function for optimize output with scipy.optimize.minimize 
    function

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,), where n is 
        the number of independent
    negate : bool
        If the function was created for maximization, the prediction is 
        negated here again
    mapper : dict or None
        - key: str = feature name of main level
        - value: dict = key: originals, value: codes
    bounds : scipy optimizer Bounds
        Bounds on variables
    **kwds
        Additional keyword arguments for `scipy.optimize.minimize`
        function.
    
    Returns
    -------
    xs : ndarray
        Optimized values for independents
    y : float
        predicted output
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.
    """
    if not bounds:
        bounds = Bounds(-np.ones(len(x0)), np.ones(len(x0))) # type: ignore
    res: OptimizeResult = minimize(fun, x0, bounds=bounds, **kwds)
    xs = [decode(x, mapper, c) for x, c in zip(res.x, columns)]
    y = -res.fun if negate else res.fun
    return xs, y, res


__all__ = [
    'is_encoded_categorical',
    'clean_categorical_names',
    'get_term_name',
    'decode_cat_main',
    'encoded_dmatrices',
    'hierarchical',
    'is_main_feature',
    'decode',
    'optimize',
]
