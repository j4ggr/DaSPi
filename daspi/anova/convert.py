from typing import List
from typing import Tuple
from pandas.core.frame import DataFrame

from ..constants import RE
from ..constants import ANOVA

__all__ = [
    'get_term_name',
    'frames_to_html',
]

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
    
    ```python
    encoded_name = 'Category[T.Value]:OtherCategory[T.OtherValue]'
    term_name = get_term_name(encoded_name)
    print(term_name)
    ```
    
    ```console
    'Category:OtherCategory'
    ```
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
