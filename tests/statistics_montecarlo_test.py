import sys
import pytest

from pathlib import Path
from pandas.core.frame import DataFrame

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
        assert limits.range == 10

        limits = SpecLimits(lower=5, upper=5)
        assert limits.range == 0

        limits = SpecLimits()
        assert limits.range == float('inf')

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


class TestParameter:

    def test_parameter_with_limits_only(self) -> None:
        param = Parameter(limits=(0, 1))
        assert param.LIMITS == (0, 1)
        assert param.TOLERANCE == 1.0
        assert param.NOMINAL == 0.5
    
    def test_parameter_with_tolerance_and_nominal(self) -> None:
        param = Parameter(tolerance=0.1, nominal=0.5)
        assert param.LIMITS == (0.45, 0.55)
        assert param.TOLERANCE == 0.1
        assert param.NOMINAL == 0.5
    
    def test_parameter_with_all_values(self) -> None:
        param = Parameter(limits=(0, 1), tolerance=1.0, nominal=0.5)
        assert param.LIMITS == (0, 1)
        assert param.TOLERANCE == 1.0
        assert param.NOMINAL == 0.5
    
    def test_parameter_with_negative_values(self) -> None:
        param = Parameter(limits=(-10, -5))
        assert param.LIMITS == (-10, -5)
        assert param.TOLERANCE == 5.0
        assert param.NOMINAL == -7.5
    
    def test_parameter_with_same_limits(self) -> None:
        param = Parameter(limits=(5, 5))
        assert param.LIMITS == (5, 5)
        assert param.TOLERANCE == 0.0
        assert param.NOMINAL == 5.0
    
    def test_parameter_with_decimal_values(self) -> None:
        param = Parameter(limits=(0.25, 0.75))
        assert param.LIMITS == (0.25, 0.75)
        assert param.TOLERANCE == 0.5
        assert param.NOMINAL == 0.5
    
    def test_parameter_repr_and_str(self) -> None:
        param = Parameter(limits=(0, 1))
        repr_str = repr(param)
        str_str = str(param)
        
        assert "Parameter(limits=(0, 1), tolerance=1, nominal=0.5)" == repr_str
        assert "Parameter(limits=(0, 1), tolerance=1, nominal=0.5)" == str_str
    
    def test_parameter_invalid_limits(self) -> None:
        with pytest.raises(AssertionError):
            Parameter(limits=(1, 0))  # Lower limit greater than upper limit
    
    def test_parameter_missing_required_params(self) -> None:
        with pytest.raises(AssertionError):
            Parameter()  # No parameters provided
        
        with pytest.raises(AssertionError):
            Parameter(tolerance=0.1)  # Missing nominal
        
        with pytest.raises(AssertionError):
            Parameter(nominal=0.5)  # Missing tolerance
    
    def test_parameter_properties_read_only(self) -> None:
        param = Parameter(limits=(0, 1))
        
        # Verify properties are read-only by attempting to set them
        with pytest.raises(AttributeError):
            param.LIMITS = (1, 2) # type: ignore
        
        with pytest.raises(AttributeError):
            param.TOLERANCE = 2.0 # type: ignore
        
        with pytest.raises(AttributeError):
            param.NOMINAL = 1.5 # type: ignore
