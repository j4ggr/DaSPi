import sys
import pytest

import numpy as np
import pandas as pd

from pytest import approx
from pathlib import Path
from pandas.core.series import Series
from pandas.core.frame import DataFrame

sys.path.append(str(Path(__file__).parent.resolve()))

from daspi.anova.model import *
from daspi.anova.utils import *

class TestLinearModel:

    def test_alpha(self) -> None:
        with pytest.raises(AssertionError, match=r'\d+ not in \(0, 1\)'):
            confidence_to_alpha(5)
