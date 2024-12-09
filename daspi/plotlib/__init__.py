

import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

from .classify import *

from .plotter import *

from .facets import *

from .chart import *

from .templates import *


_styles_dir = Path(__file__).parent/'styles'
_daspi_styles = tuple(style.stem for style in _styles_dir.glob('*.mplstyle'))
_mpl_styles = tuple(plt.style.available)

STYLES = _daspi_styles + _mpl_styles

def use_style(name: str | Path) -> None:
    """Apply a matplotlib style for plotting.

    Parameters
    ----------
    name : str or Path
        Name of the style to use. Can be:
        - A built-in DaSPi style name ('daspi', 'daspi-dark' or 'ggplot2)
        - A built-in matplotlib style name
        - Path to a custom .mplstyle file

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the provided style name is not valid or file doesn't exist

    Examples
    --------
    Change plotting style:
    ``` python
    use_style('daspi')                      # Use default DaSPi style
    use_style('seaborn')                    # Use built-in matplotlib style
    use_style('path/to/custom.mplstyle')    # Use custom style file
    ```
    
    Show available styles:
    ``` python
    import daspi as dsp
    print(dsp.STYLES)
    ```
    """
    if name in _daspi_styles:
        plt.style.use(_styles_dir/f'{name}.mplstyle')
    elif name in _mpl_styles:
        plt.style.use(name)
    elif Path(name).is_file() and str(name).endswith('.mplstyle'):
        plt.style.use(name)
    else:
        raise ValueError(
            f'{name} is not a valid matplotlib style. Available styles are: '
            f'{STYLES}')


use_style('daspi')

