import sys
import pytest
import random
import numpy as np

from pathlib import Path
from pandas.core.frame import DataFrame

from daspi.statistics.montecarlo import Specification

sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi.statistics.montecarlo import *


class TestSpecLimits:
    def test_initialization_with_limits(self) -> None:
        limits = SpecLimits(lower=10, upper=20)
        assert limits.lower == 10
        assert limits.upper == 20

    def test_initialization_with_unbounded_limits(self) -> None:
        limits = SpecLimits()
        assert limits.lower == float('-inf')
        assert limits.upper == float('inf')

    def test_is_unbounded(self) -> None:
        limits = SpecLimits(lower=10, upper=float('inf'))
        assert limits.is_unbounded is True
        
        limits = SpecLimits(lower=float('-inf'), upper=20)
        assert limits.is_unbounded is True

        limits = SpecLimits(lower=10, upper=20)
        assert limits.is_unbounded is False

    def test_are_both_finite(self) -> None:
        limits = SpecLimits(lower=10, upper=20)
        assert limits.are_both_finite is True

        limits = SpecLimits(lower=float('-inf'), upper=20)
        assert limits.are_both_finite is False

        limits = SpecLimits(lower=10, upper=float('inf'))
        assert limits.are_both_finite is False

    def test_is_both_unbounded(self) -> None:
        limits = SpecLimits(lower=float('-inf'), upper=float('inf'))
        assert limits.both_unbounded is True

        limits = SpecLimits(lower=10, upper=20)
        assert limits.both_unbounded is False

    def test_range(self) -> None:
        limits = SpecLimits(lower=10, upper=20)
        assert limits.tolerance == 10

        limits = SpecLimits(lower=5, upper=5)
        assert limits.tolerance == 0

        limits = SpecLimits()
        assert limits.tolerance == float('inf')

    def test_contains(self) -> None:
        limits = SpecLimits(lower=10, upper=20)
        assert (15 in limits) is True
        assert (5 in limits) is False
        assert (25 in limits) is False

        limits_unbounded = SpecLimits()
        assert (0 in limits_unbounded) is True

    def test_to_tuple(self) -> None:
        limits = SpecLimits(lower=10, upper=20)
        assert limits.to_tuple() == (10, 20)

        limits_unbounded = SpecLimits()
        assert limits_unbounded.to_tuple() == (float('-inf'), float('inf'))


class TestSpecification:

    def test_specification_with_limits_only(self) -> None:
        spec = Specification(limits=(0, 1))
        assert spec.LIMITS == (0, 1)
        assert spec.TOLERANCE == 1.0
        assert spec.NOMINAL == 0.5
    
    def test_specification_with_tolerance_and_nominal(self) -> None:
        spec = Specification(tolerance=0.1, nominal=0.5)
        assert spec.LIMITS == (0.45, 0.55)
        assert spec.TOLERANCE == 0.1
        assert spec.NOMINAL == 0.5
    
    def test_specification_with_all_values(self) -> None:
        spec = Specification(limits=(0, 1), tolerance=1.0, nominal=0.5)
        assert spec.LIMITS == (0, 1)
        assert spec.TOLERANCE == 1.0
        assert spec.NOMINAL == 0.5
    
    def test_specification_with_negative_values(self) -> None:
        spec = Specification(limits=(-10, -5))
        assert spec.LIMITS == (-10, -5)
        assert spec.TOLERANCE == 5.0
        assert spec.NOMINAL == -7.5
    
    def test_specification_with_same_limits(self) -> None:
        spec = Specification(limits=(5, 5))
        assert spec.LIMITS == (5, 5)
        assert spec.TOLERANCE == 0.0
        assert spec.NOMINAL == 5.0
    
    def test_specification_with_decimal_values(self) -> None:
        spec = Specification(limits=(0.25, 0.75))
        assert spec.LIMITS == (0.25, 0.75)
        assert spec.TOLERANCE == 0.5
        assert spec.NOMINAL == 0.5
    
    def test_specification_repr_and_str(self) -> None:
        spec = Specification(limits=(0, 1))
        repr_str = repr(spec)
        str_str = str(spec)
        
        assert "Specification(limits=(0, 1), tolerance=1, nominal=0.5)" == repr_str
        assert "Specification(limits=(0, 1), tolerance=1, nominal=0.5)" == str_str
    
    def test_specification_invalid_limits(self) -> None:
        with pytest.raises(AssertionError):
            Specification(limits=(1, 0))  # Lower limit greater than upper limit
    
    def test_specification_missing_required_specs(self) -> None:
        with pytest.raises(AssertionError):
            Specification()  # No specifications provided
        
        with pytest.raises(AssertionError):
            Specification(tolerance=0.1)  # Missing nominal
        
        with pytest.raises(AssertionError):
            Specification(nominal=0.5)  # Missing tolerance
    
    def test_specification_properties_read_only(self) -> None:
        spec = Specification(limits=(0, 1))
        
        # Verify properties are read-only by attempting to set them
        with pytest.raises(AttributeError):
            spec.LIMITS = (1, 2) # type: ignore
        
        with pytest.raises(AttributeError):
            spec.TOLERANCE = 2.0 # type: ignore
        
        with pytest.raises(AttributeError):
            spec.NOMINAL = 1.5 # type: ignore


class TestRandomProcessValue:
    
    @pytest.fixture
    def SPEC(self) -> Specification:
        # Setup a specification with limits and tolerance
        return Specification(limits=(5, 10))
    
    def test_initialization_with_invalid_distribution(self, SPEC: Specification) -> None:
        with pytest.raises(AssertionError):
            RandomProcessValue(SPEC, 'invalid_distribution') # type: ignore

    def test_normal_distribution(self, SPEC: Specification) -> None:
        rpv = RandomProcessValue(SPEC, 'normal')
        value = rpv()
        assert rpv.lower <= value <= rpv.upper, "Value not within limits"

    def test_uniform_distribution(self, SPEC: Specification) -> None:
        rpv = RandomProcessValue(SPEC, 'uniform')
        value = rpv()
        assert rpv.lower <= value <= rpv.upper, "Value not within limits"

    def test_circular_distribution(self, SPEC: Specification) -> None:
        rpv = RandomProcessValue(SPEC, 'circular')
        value = rpv()
        assert -rpv.specification.TOLERANCE <= value <= rpv.specification.TOLERANCE, "Value not within circular tolerance"

    def test_clipping_function(self, SPEC: Specification) -> None:
        rpv = RandomProcessValue(SPEC, 'normal', clip=True)
        # Generate a value outside the limits
        random.seed(0)  # For reproducibility
        value = rpv.clip(15, rpv.lower, rpv.upper)
        assert value == rpv.upper, "Clipping failed for upper limit"
        
        value = rpv.clip(-5, rpv.lower, rpv.upper)
        assert value == rpv.lower, "Clipping failed for lower limit"

    def test_generate_method(self, SPEC: Specification) -> None:
        rpv = RandomProcessValue(SPEC, 'normal', clip=True)
        values = rpv.generate(1000)
        assert len(values) == 1000, "Generated array length mismatch"
        assert all(rpv.lower <= v <= rpv.upper for v in values), "Generated values out of bounds"


class TestBinning:
    
    def test_linear_binning(self) -> None:
        data = np.array([1.5, 2.3, 3.7, 4.1, 5.0])
        binning = Binning(data, num_bins=3, distance=1.0, kind='linear')
        expected_nominals = np.array([2.32, 3.32, 4.32])
        expected_indices = np.array([0, 0, 1, 2, 2])
        expected_binned_values = np.array([2.32, 2.32, 3.32, 4.32, 4.32])
        
        np.testing.assert_allclose(binning.nominals, expected_nominals)
        assert (binning.indices == expected_indices).all()
        np.testing.assert_allclose(binning.values(), expected_binned_values)

    def test_quantile_binning(self) -> None:
        data = np.array([1.5, 2.3, 3.7, 4.1, 5.0])
        binning = Binning(data, num_bins=3, kind='quantile')
        expected_nominals = np.array([2.3, 3.7, 4.1])
        expected_indices = np.array([0, 0, 1, 2, 2])
        expected_binned_values = np.array([2.3, 2.3, 3.7, 4.1, 4.1])
        
        np.testing.assert_allclose(binning.nominals, expected_nominals)
        assert (binning.indices == expected_indices).all()
        np.testing.assert_allclose(binning.values(), expected_binned_values)

    def test_round_to_nearest(self) -> None:
        data = np.array([1.5, 2.3, 3.7, 4.1, 5.0])
        binning = Binning(data, num_bins=3, distance=1.0, kind='linear')
        expected_nominals = np.array([2.32, 3.32, 4.32])
        np.testing.assert_allclose(binning.nominals, expected_nominals)
        binning.round_to_nearest(nearest=5, digit=2)
        expected_nominals = np.array([2.30, 3.30, 4.30])
        expected_indices = np.array([0, 0, 1, 2, 2])
        expected_binned_values = np.array([2.30, 2.30, 3.30, 4.30, 4.30])
        np.testing.assert_allclose(binning.values(), expected_binned_values)
        assert (binning.indices == expected_indices).all()
        np.testing.assert_allclose(binning.nominals, expected_nominals)
        
        binning = Binning(data, num_bins=3, distance=1.0, kind='quantile')
        expected_nominals = np.array([2.3, 3.7, 4.1])
        np.testing.assert_allclose(binning.nominals, expected_nominals)
        binning.round_to_nearest(nearest=5, digit=1)
        expected_nominals = np.array([2.5, 3.5, 4.0])
        expected_indices = np.array([0, 0, 1, 2, 2])
        expected_binned_values = np.array([2.5, 2.5, 3.5, 4.0, 4.0])
        np.testing.assert_allclose(binning.values(), expected_binned_values)
        assert (binning.indices == expected_indices).all()
        np.testing.assert_allclose(binning.nominals, expected_nominals)

    def test_assert_distance_linear(self) -> None:
        data = np.array([1.5, 2.3, 3.7, 4.1, 5.0])
        with pytest.raises(AssertionError, match='Specify a distance between the bins'):
            Binning(data, num_bins=3, kind='linear')
