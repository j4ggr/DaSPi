import sys
import pytest

import numpy as np
import pandas as pd

from pytest import approx
from pathlib import Path
from numpy.typing import ArrayLike 

sys.path.append(Path(__file__).parent.resolve())

from daspi.plotlib.utils import *
from daspi._constants import CATEGORY


class TestCategoryLabelHandler:
    colors = HueLabelHandler(('alpha', 'beta', 'gamma'))
    markers = ShapeLabelHandler(('foo', 'bar', 'bazz'))
    sizes_s = SizeLabelHandler(1.5, 3.5, 'scatter')
    sizes_l = SizeLabelHandler(1.5, 3.5, 'line')

    def test_str(self):
        assert str(self.colors) == 'HueLabelHandler'
        assert str(self.markers) == 'ShapeLabelHandler'
        assert str(self.sizes_s) == 'SizeLabelHandler'

    def test_errrors(self):
        with pytest.raises(KeyError) as err:
            self.colors['psi']
        assert str(err.value) == f"\"Can't get category for label 'psi', got {self.colors.labels}\""
        
        with pytest.raises(AssertionError) as err:
            n = self.colors.n_allowed
            HueLabelHandler([i for i in range(n+1)])
        assert str(err.value) == f'HueLabelHandler can handle {n} categories, got {n+1}'

        with pytest.raises(AssertionError) as err:
            ShapeLabelHandler(('foo', 'foo', 'bar', 'bazz'))
        assert str(err.value) == f'One or more labels occur more than once, only unique labels are allowed'

    def test_handles_labels(self):
        handles, labels = self.colors.handles_labels()
        assert len(self.colors.colors) == self.colors.n_used
        assert len(handles) == len(labels)
        
        handles, labels = self.markers.handles_labels()
        assert len(self.markers.markers) == self.markers.n_used
        assert len(handles) == len(labels)
        
        handles, labels = self.sizes_s.handles_labels()
        assert len(self.sizes_s.categories) == self.sizes_s.n_used
        assert len(handles) == len(labels)
        assert len(handles) == CATEGORY.N_SIZE_BINS
    
    def test_sizes(self):
        s_min, s_max = CATEGORY.MARKERSIZE_LIMITS
        assert self.sizes_l[1.5] == s_min
        assert self.sizes_l[3.5] == s_max
        assert self.sizes_s[1.5] == s_min**2
        assert self.sizes_s[3.5] == s_max**2

        sizes = self.sizes_l.sizes([1.5, 3.5])
        assert np.array_equal(sizes, CATEGORY.MARKERSIZE_LIMITS)

        values = np.linspace(
            self.sizes_l._min, self.sizes_l._max, CATEGORY.N_SIZE_BINS)
        sizes = self.sizes_l.sizes(values)
        assert np.array_equal(sizes, self.sizes_l.categories)

        sizes = self.sizes_l.sizes([1.5, 3.5])
        assert np.array_equal(sizes, CATEGORY.MARKERSIZE_LIMITS)

        values = np.linspace(
            self.sizes_l._min, self.sizes_l._max, CATEGORY.N_SIZE_BINS)
        sizes = self.sizes_s.sizes(values)
        assert np.array_equal(sizes, np.square(self.sizes_s.categories))

