import itertools

import numpy as np

from typing import Any
from typing import List
from typing import Tuple
from typing import Callable
from typing import Iterable
import patsy.highlevel
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

from ..constants import RE
from ..constants import ANOVA


def uniques(seq: Iterable) -> list[Any]:
    """Get a list of unique elements from a sequence while preserving 
    the original order.

    Parameters
    ----------
    seq : Iterable
        The input sequence.

    Returns
    -------
    List[Any]
        A list of unique elements from the input sequence, preserving 
        the original order.

    Notes
    -----
    This function is based on the 'uniqify' algorithm by Peter Bengtsson.
    Source: https://www.peterbe.com/plog/uniqifiers-benchmark

    Examples
    --------
    >>> sequence = [1, 2, 3, 2, 1, 4, 5, 4]
    >>> unique_elements = preserved_order_uniques(sequence)
    >>> print(unique_elements)
    [1, 2, 3, 4, 5]
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_term_name(feature_name: str) -> str:
    """Get the original term name of a patsy encoded categorical
    feature, including interactions.

    Parameters
    ----------
    feature_name : str
        The encoded feature name.

    Returns
    -------
    str
        The original term name of the categorical feature.

    Notes
    -----
    Patsy encodes categorical features by appending '[T.<value>]' to the
    original term name. Interactions between features are represented by
    separating the feature names with ':'. This function extracts the
    original term name from the encoded feature name, taking into
    account interactions.

    Examples
    --------
    >>> encoded_name = 'Category[T.Value]:OtherCategory[T.OtherValue]'
    >>> term_name = get_term_name(encoded_name)
    >>> print(term_name)
    'Category:OtherCategory'
    """
    names = feature_name.split(ANOVA.SEP)
    matches = list(map(RE.ENCODED_NAME.findall, names))
    return ANOVA.SEP.join([(m[0] if m else n) for m, n in zip(matches, names)])

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
    'uniques',
    'get_term_name',
    'hierarchical',
    'is_main_feature',
    'optimize',
]
