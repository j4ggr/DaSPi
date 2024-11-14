

import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

from .classify import *

from .plotter import *

from .facets import *

from .chart import *

from .templates import *

plt.style.use(Path(__file__).parent/'daspi.mplstyle')
