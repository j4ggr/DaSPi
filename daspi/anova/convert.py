from typing import List
from typing import Tuple
from pandas.core.frame import DataFrame

from ..constants import RE
from ..constants import ANOVA


def get_term_name(name: str) -> str:
    """Get the original term name of a patsy encoded categorical
    column name, including interactions.

    Parameters
    ----------
    name : str
        The encoded column name.

    Returns
    -------
    str
        The original term name of the categorical column name.

    Notes
    -----
    Patsy encodes categorical columns by appending '[T.<value>]' to the
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
    if not isinstance(name, str):
        return name
    
    names = name.split(ANOVA.SEP)
    matches = list(map(RE.ENCODED_NAME.findall, names))
    return ANOVA.SEP.join([(m[0] if m else n) for m, n in zip(matches, names)])

def frames_to_html(
        dfs: DataFrame | List[DataFrame] | Tuple[DataFrame, ...],
        captions: str | List[str] | Tuple[str, ...]) -> str:
    """Converts one or more DataFrames to HTML tables with captions.

    Parameters
    ----------
    dfs : DataFrame or list/tuple of DataFrames
        The DataFrame(s) to be converted to HTML.
    captions : str or list/tuple of str
        The captions to be used for the HTML tables. The number of 
        captions must match the number of DataFrames.

    Returns
    -------
    str
        The HTML representation of the DataFrames with captions.
    """
    if isinstance(captions, str):
        captions = (captions,)
    if isinstance(dfs, DataFrame):
        dfs = (dfs,)
    assert len(dfs) <= len(captions), (
        "There must be at most as many captions as DataFrames.")
    spacing = 2*'</br>'
    html = ''
    for (df, caption) in zip(dfs, captions):
        if html:
            html += spacing
        html += (df
            .style
            .set_table_attributes("style='display:inline'")
            .set_caption(caption)
            .to_html())
    return html


# TODO: implement optimizer in model
# def optimize(
#         fun: Callable, x0: List[float], negate: bool, columns: List[str], 
#         mapper: dict, bounds: Bounds|None = None, **kwds
#         ) -> Tuple[List[float], float, OptimizeResult]:
#     """Base function for optimize output with scipy.optimize.minimize 
#     function

#     Parameters
#     ----------
#     fun : callable
#         The objective function to be minimized.
#     x0 : ndarray, shape (n,)
#         Initial guess. Array of real elements of size (n,), where n is 
#         the number of independent
#     negate : bool
#         If the function was created for maximization, the prediction is 
#         negated here again
#     mapper : dict or None
#         - key: str = feature name of main level
#         - value: dict = key: originals, value: codes
#     bounds : scipy optimizer Bounds
#         Bounds on variables
#     **kwds
#         Additional keyword arguments for `scipy.optimize.minimize`
#         function.
    
#     Returns
#     -------
#     xs : ndarray
#         Optimized values for independents
#     y : float
#         predicted output
#     res : OptimizeResult
#         The optimization result represented as a ``OptimizeResult`` object.
#         Important attributes are: ``x`` the solution array, ``success`` a
#         Boolean flag indicating if the optimizer exited successfully and
#         ``message`` which describes the cause of the termination. See
#         `OptimizeResult` for a description of other attributes.
#     """
#     if not bounds:
#         bounds = Bounds(-np.ones(len(x0)), np.ones(len(x0))) # type: ignore
#     res: OptimizeResult = minimize(fun, x0, bounds=bounds, **kwds)
#     xs = [decode(x, mapper, c) for x, c in zip(res.x, columns)]
#     y = -res.fun if negate else res.fun
#     return xs, y, res


__all__ = [
    'get_term_name',
    'frames_to_html',
]
