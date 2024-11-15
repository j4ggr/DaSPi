import sys
import pytest

import numpy as np
import pandas as pd

from typing import Any
from typing import Dict
from pytest import approx
from pathlib import Path
from pandas.core.frame import DataFrame

sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi.statistics.hypothesis import *

source = Path(__file__).parent/'data'
KW_READ: Dict[str, Any] = dict(sep=';', index_col=0)

df_dist10: DataFrame = pd.read_csv(
    source/f'dists_10-samples.csv', skiprows=1, nrows=10, **KW_READ)
df_valid10: DataFrame = pd.read_csv(
    source/f'dists_10-samples.csv', skiprows=14, **KW_READ)
df_dist25: DataFrame = pd.read_csv(
    source/f'dists_25-samples.csv', skiprows=1, nrows=25, **KW_READ)
df_valid25: DataFrame = pd.read_csv(
    source/f'dists_25-samples.csv', skiprows=29, **KW_READ)


class TestChunker:

    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_with_valid_input(self) -> None:
        sections = 3
        result = list(chunker(self.samples, sections))
        assert len(result) == sections
        assert np.array_equal(result[0], np.array([1, 2, 3, 4]))
        assert np.array_equal(result[1], np.array([5, 6, 7]))
        assert np.array_equal(result[2], np.array([8, 9, 10]))

    def test_with_single_section(self) -> None:
        sections = 1
        result = list(chunker(self.samples, sections))
        assert len(result) == sections
        assert np.array_equal(result[0], self.samples)

    def test_with_single_sample(self) -> None:
        sections = 2
        samples = [1]
        result = list(chunker(samples, sections))
        assert len(result) == sections
        assert np.array_equal(result[0], samples)
        assert result[1].size == 0

    def test_with_zero_sections(self) -> None:
        sections = 0
        with pytest.raises(AssertionError):
            list(chunker(self.samples, sections))

    def test_with_negative_sections(self) -> None:
        sections = -2
        with pytest.raises(AssertionError):
            list(chunker(self.samples, sections))

    def test_with_non_integer_sections(self) -> None:
        sections = 2.5
        with pytest.raises(AssertionError):
            list(chunker(self.samples, sections)) # type: ignore

