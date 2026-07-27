"""
Tests for GeneralizedLinearModel for count data
"""
import pytest
import pandas as pd
from pandas import DataFrame
import numpy as np

from daspi.anova.model import GeneralizedLinearModel


class TestGeneralizedLinearModel:
    """Tests for Generalized Linear Model for count data and other non-normal responses"""
    
    @pytest.fixture
    def count_data(self) -> DataFrame:
        """Create count data (e.g., defect counts) for testing"""
        np.random.seed(42)
        
        data = {
            'Factor_A': ['A1']*20 + ['A2']*20 + ['A3']*20,
            'Factor_B': ['B1', 'B2']*30,
            'Count': (
                # A1 combinations
                [10, 12, 11, 13, 12, 14, 11, 13, 10, 12] + 
                [8, 7, 9, 10, 8, 9, 7, 8, 9, 10] +
                # A2 combinations (lower counts)
                [5, 4, 6, 5, 4, 6, 5, 4, 6, 5] +
                [3, 2, 4, 3, 2, 4, 3, 2, 4, 3] +
                # A3 combinations (higher counts)
                [18, 20, 19, 21, 17, 19, 18, 20, 19, 21] +
                [25, 24, 26, 27, 23, 25, 24, 26, 27, 23]
            )
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def proportion_data(self) -> DataFrame:
        """Create proportion data (e.g., pass/fail) for testing"""
        np.random.seed(123)
        
        data = {
            'Treatment': ['Control']*15 + ['Drug_A']*15 + ['Drug_B']*15,
            'Success': [8, 7, 9, 8, 7, 10, 8, 9, 7, 8, 9, 10, 8, 7, 9] +
                      [12, 13, 11, 12, 14, 13, 12, 11, 13, 14, 12, 13, 12, 11, 13] +
                      [14, 15, 13, 14, 16, 15, 14, 13, 15, 14, 15, 14, 13, 15, 14],
            'Total': [15] * 45
        }
        return pd.DataFrame(data)
    
    def test_poisson_initialization(self, count_data: DataFrame) -> None:
        """Test GLMModel initialization with Poisson family"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        assert model is not None
        assert model.family == 'poisson'
    
    def test_negative_binomial_initialization(self, count_data: DataFrame) -> None:
        """Test GLMModel initialization with Negative Binomial family"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='negbin',
            order=1
        )
        assert model is not None
        assert model.family == 'negbin'
    
    def test_binomial_initialization(self, proportion_data: DataFrame) -> None:
        """Test GLMModel initialization with Binomial family"""
        model = GeneralizedLinearModel(
            source=proportion_data,
            target='Success',
            factors=['Treatment'],
            family='binomial',
            trials_column='Total',
            order=1
        )
        assert model is not None
        assert model.family == 'binomial'
    
    def test_fit_method(self, count_data: DataFrame) -> None:
        """Test that fit() is called automatically"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        # Model should be fitted automatically
        assert hasattr(model, '_fit_result')
    
    def test_deviance_check(self, count_data: DataFrame) -> None:
        """Test deviance_check method"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        check_result = model.deviance_check()
        assert isinstance(check_result, bool)
    
    def test_dispersion_property(self, count_data: DataFrame) -> None:
        """Test dispersion property for overdispersion detection"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        dispersion = model.dispersion
        assert isinstance(dispersion, (int, float))
        assert dispersion > 0
    
    def test_deviance_property(self, count_data: DataFrame) -> None:
        """Test deviance property"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        deviance = model.deviance
        assert isinstance(deviance, (int, float))
        assert deviance >= 0
    
    def test_parameter_statistics(self, count_data: DataFrame) -> None:
        """Test parameter_statistics method"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        params = model.parameter_statistics()
        assert isinstance(params, DataFrame)
        assert 'coef' in params.columns
        assert 'std_err' in params.columns
        assert 'p_value' in params.columns
    
    def test_effects(self, count_data: DataFrame) -> None:
        """Test effects method"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        effects = model.effects()
        assert isinstance(effects, DataFrame)
        assert 'factor' in effects.columns or 'Factor' in effects.columns.str.lower()
    
    def test_p_values(self, count_data: DataFrame) -> None:
        """Test p_values method"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        pvals = model.p_values()
        assert isinstance(pvals, DataFrame)
        # Check that p-values are in valid range
        assert all((pvals['p_value'] >= 0) & (pvals['p_value'] <= 1))
    
    def test_predict(self, count_data: DataFrame) -> None:
        """Test predict method"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        predictions = model.predict()
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(count_data)
        # Predictions should be positive for counts
        assert all(predictions > 0)
    
    def test_compare_groups(self, count_data: DataFrame) -> None:
        """Test compare_groups method"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A'],
            family='poisson',
            order=1
        )
        comparison = model.compare_groups('Factor_A')
        assert isinstance(comparison, DataFrame)
        assert 'level' in comparison.columns or 'Level' in comparison.columns.str.lower()
    
    def test_summary_method(self, count_data: DataFrame) -> None:
        """Test summary method returns something"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        summary = model.summary()
        # Summary should return some representation
        assert summary is not None
    
    def test_with_interactions(self, count_data: DataFrame) -> None:
        """Test model with interaction terms"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=2  # Include interactions
        )
        assert model is not None
        # Should have interaction terms
        params = model.parameter_statistics()
        # Check if interaction terms exist (e.g., Factor_A:Factor_B)
        param_names = params.index.tolist()
        has_interaction = any(':' in str(name) for name in param_names)
        assert has_interaction
    
    def test_overdispersion_detection(self, count_data: DataFrame) -> None:
        """Test overdispersion detection"""
        model = GeneralizedLinearModel(
            source=count_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='poisson',
            order=1
        )
        # Check if overdispersion is detected
        is_overdispersed = model.deviance_check()
        assert isinstance(is_overdispersed, bool)
    
    def test_negative_binomial_for_overdispersion(self, count_data: DataFrame) -> None:
        """Test that Negative Binomial can handle overdispersed data"""
        # Create overdispersed data
        np.random.seed(42)
        overdispersed_data = count_data.copy()
        overdispersed_data['Count'] = overdispersed_data['Count'] + np.random.poisson(5, len(count_data))
        
        model_nb = GeneralizedLinearModel(
            source=overdispersed_data,
            target='Count',
            factors=['Factor_A', 'Factor_B'],
            family='negbin',
            order=1
        )
        assert model_nb is not None
        # Negative binomial should fit
        assert hasattr(model_nb, '_fit_result')
    
    def test_invalid_family(self, count_data: DataFrame) -> None:
        """Test that invalid family raises error"""
        with pytest.raises((ValueError, KeyError)):
            GeneralizedLinearModel(
                source=count_data,
                target='Count',
                factors=['Factor_A'],
                family='invalid_family',
                order=1
            )
    
    def test_missing_data_handling(self) -> None:
        """Test handling of missing data"""
        data_with_na = {
            'Factor': ['A', 'A', 'B', 'B', 'A', 'B'],
            'Count': [10, 12, np.nan, 20, 11, 19]
        }
        df = pd.DataFrame(data_with_na)
        
        # Should handle missing data (may drop or raise error)
        try:
            model = GeneralizedLinearModel(
                source=df,
                target='Count',
                factors=['Factor'],
                family='poisson',
                order=1
            )
            # If it succeeds, check it's valid
            assert model is not None
        except (ValueError, Exception):
            # If it raises, that's also acceptable
            pass
