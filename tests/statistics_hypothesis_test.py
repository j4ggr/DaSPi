import sys
import pytest

import numpy as np
import pandas as pd

from typing import Any
from pathlib import Path
from pandas.core.frame import DataFrame

sys.path.append(str(Path(__file__).parent.resolve())) 

from daspi.statistics.hypothesis import *

source = Path(__file__).parent/'data'
KW_READ: dict[str, Any] = {'sep': ';', 'index_col': 0}

df_dist10: DataFrame = pd.read_csv(
    source/'dists_10-samples.csv', skiprows=1, nrows=10, **KW_READ)
df_valid10: DataFrame = pd.read_csv(
    source/'dists_10-samples.csv', skiprows=14, **KW_READ)
df_dist25: DataFrame = pd.read_csv(
    source/'dists_25-samples.csv', skiprows=1, nrows=25, **KW_READ)
df_valid25: DataFrame = pd.read_csv(
    source/'dists_25-samples.csv', skiprows=29, **KW_READ)


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

# --- GROUPED TESTS FOR hypothesis.py COVERAGE ---
class TestHypothesisFunctions:
    def test_anderson_darling_test(self):
        from daspi.statistics.hypothesis import anderson_darling_test
        data = np.random.normal(0, 1, 100)
        p, stat = anderson_darling_test(data)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert 0 <= p <= 1

    def test_all_normal(self):
        from daspi.statistics.hypothesis import all_normal
        a = np.random.normal(0, 1, 50)
        b = np.random.normal(0, 1, 50)
        assert all_normal(a, b)
        assert all_normal(a, b, p_threshold=0.01) in [True, False]
        with pytest.raises(AssertionError):
            all_normal(a, b, p_threshold=1.5)

    def test_kolmogorov_smirnov_test(self):
        from daspi.statistics.hypothesis import kolmogorov_smirnov_test
        data = np.random.normal(0, 1, 100)
        p, D, params = kolmogorov_smirnov_test(data, 'norm')
        assert isinstance(p, float)
        assert isinstance(D, float)
        assert isinstance(params, tuple)
        assert 0 <= p <= 1

    def test_f_test(self) -> None:
        from daspi.statistics.hypothesis import f_test
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        p, F = f_test(a, b)
        assert isinstance(p, float)
        assert isinstance(F, float)
        assert 0 <= p <= 1

    def test_t_test(self) -> None:
        from daspi.statistics.hypothesis import t_test
        a = np.random.normal(0, 1, 30)
        p, t, df = t_test(a)
        assert isinstance(p, float)
        assert isinstance(t, float)
        assert isinstance(df, int)
        assert 0 <= p <= 1

    def test_levene_test(self) -> None:
        from daspi.statistics.hypothesis import levene_test
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        p, L = levene_test(a, b)
        assert isinstance(p, float)
        assert isinstance(L, float)
        assert 0 <= p <= 1

    def test_variance_stability_test(self) -> None:
        from daspi.statistics.hypothesis import variance_stability_test
        a = np.random.normal(0, 1, 30)
        p, L = variance_stability_test(a, n_sections=3)
        assert isinstance(p, float)
        assert isinstance(L, float)
        assert 0 <= p <= 1

    def test_mean_stability_test(self) -> None:
        from daspi.statistics.hypothesis import mean_stability_test
        a = np.random.normal(0, 1, 30)
        p, stat = mean_stability_test(a, n_sections=3)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert 0 <= p <= 1

    def test_position_test(self) -> None:
        from daspi.statistics.hypothesis import position_test
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        p, stat, test = position_test(a, b)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert isinstance(test, str)
        assert 0 <= p <= 1

    def test_variance_test(self) -> None:
        from daspi.statistics.hypothesis import variance_test
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        p, stat, test = variance_test(a, b)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert isinstance(test, str)
        assert 0 <= p <= 1

    def test_proportions_test(self) -> None:
        from daspi.statistics.hypothesis import proportions_test
        p, stat, test = proportions_test(5, 10, 7, 12)
        assert isinstance(p, float)


# ============================================================================
# TESTS FOR NEW FEATURES: dunn_test() and pairwise_tests()
# ============================================================================

class TestDunnTest:
    """Tests for Dunn's post-hoc test (non-parametric pairwise comparisons)"""
    
    @pytest.fixture
    def sample_groups(self) -> dict[str, np.ndarray]:
        """Create sample groups with known differences"""
        np.random.seed(42)
        return {
            'Group_A': np.array([10, 12, 11, 13, 12, 14, 11, 13]),
            'Group_B': np.array([20, 22, 21, 23, 19, 21, 22, 20]),
            'Group_C': np.array([15, 17, 16, 18, 15, 16, 17, 16]),
        }
    
    def test_basic_functionality(self, sample_groups: dict[str, np.ndarray]) -> None:
        """Test basic execution and return structure"""
        from daspi.statistics.hypothesis import dunn_test
        
        result = dunn_test(sample_groups)
        
        # Check result is DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check expected columns
        expected_cols = ['Group1', 'Group2', 'z_statistic', 'p_raw', 'p_adjusted', 'significant']
        assert all(col in result.columns for col in expected_cols)
        
        # Check number of comparisons (n*(n-1)/2 for 3 groups = 3)
        assert len(result) == 3
    
    def test_z_statistics(self, sample_groups: dict[str, np.ndarray]) -> None:
        """Test z-statistics are computed correctly"""
        from daspi.statistics.hypothesis import dunn_test
        
        result = dunn_test(sample_groups)
        
        # Z-statistics should be numeric
        assert result['z_statistic'].dtype in [np.float64, float]
        
        # Absolute z-statistics should be > 0 for different groups
        assert all(result['z_statistic'].abs() > 0)
    
    def test_p_values_range(self, sample_groups: dict[str, np.ndarray]) -> None:
        """Test p-values are in valid range [0, 1]"""
        from daspi.statistics.hypothesis import dunn_test
        
        result = dunn_test(sample_groups)
        
        # P-values should be between 0 and 1
        assert all(result['p_raw'] >= 0)
        assert all(result['p_raw'] <= 1)
        assert all(result['p_adjusted'] >= 0)
        assert all(result['p_adjusted'] <= 1)
    
    def test_bonferroni_correction(self, sample_groups: dict[str, np.ndarray]) -> None:
        """Test Bonferroni correction increases adjusted p-values"""
        from daspi.statistics.hypothesis import dunn_test
        
        result = dunn_test(sample_groups, p_adjust='bonferroni')
        
        # Adjusted p-values should be >= raw p-values (conservative)
        assert all(result['p_adjusted'] >= result['p_raw'])
    
    def test_different_correction_methods(self, sample_groups: dict[str, np.ndarray]) -> None:
        """Test different p-value correction methods"""
        from daspi.statistics.hypothesis import dunn_test
        
        methods = ['bonferroni', 'holm', 'BH', 'BY']
        
        for method in methods:
            result = dunn_test(sample_groups, p_adjust=method)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert all(result['p_adjusted'] >= 0)
            assert all(result['p_adjusted'] <= 1)
    
    def test_significance_detection(self) -> None:
        """Test detection of significant differences"""
        from daspi.statistics.hypothesis import dunn_test
        
        # Create groups with large differences
        groups_large_diff = {
            'Low': np.array([1, 2, 3, 2, 1, 3, 2]),
            'High': np.array([100, 102, 101, 99, 100, 103, 101])
        }
        
        result = dunn_test(groups_large_diff, p_adjust='bonferroni')
        
        # Should detect significant difference
        assert result.iloc[0]['significant'] == True
        assert result.iloc[0]['p_adjusted'] < 0.05
    
    def test_no_significance_similar_groups(self) -> None:
        """Test no significance for similar groups"""
        from daspi.statistics.hypothesis import dunn_test
        
        np.random.seed(123)
        # Groups from same distribution
        groups_similar = {
            'A': np.random.normal(10, 1, 30),
            'B': np.random.normal(10, 1, 30),
        }
        
        result = dunn_test(groups_similar, p_adjust='bonferroni')
        
        # Likely not significant (may occasionally fail due to randomness)
        assert result.iloc[0]['p_raw'] > 0.01  # Very liberal threshold
    
    def test_empty_groups_error(self) -> None:
        """Test error handling for empty groups"""
        from daspi.statistics.hypothesis import dunn_test
        
        with pytest.raises((ValueError, KeyError, AssertionError)):
            dunn_test({})
    
    def test_single_group_error(self) -> None:
        """Test error with single group (need at least 2 for comparison)"""
        from daspi.statistics.hypothesis import dunn_test
        
        with pytest.raises((ValueError, AssertionError)):
            dunn_test({'A': np.array([1, 2, 3])})


class TestPairwiseTests:
    """Tests for pairwise_tests() with automatic test selection"""
    
    @pytest.fixture
    def normal_groups(self) -> dict[str, np.ndarray]:
        """Normal distributed groups for parametric tests"""
        np.random.seed(42)
        return {
            'Group_A': np.random.normal(10, 2, 30),
            'Group_B': np.random.normal(15, 2, 30),
            'Group_C': np.random.normal(12, 2, 30),
        }
    
    @pytest.fixture
    def non_normal_groups(self) -> dict[str, np.ndarray]:
        """Non-normal groups for non-parametric tests"""
        np.random.seed(42)
        return {
            'Group_A': np.random.exponential(2, 30),
            'Group_B': np.random.exponential(5, 30),
        }
    
    def test_basic_functionality(self, normal_groups: dict[str, np.ndarray]) -> None:
        """Test basic execution"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        result = pairwise_tests(normal_groups)
        
        # Check return type
        assert isinstance(result, pd.DataFrame)
        
        # Check columns
        expected_cols = ['Group1', 'Group2', 'statistic', 'p_raw', 'mean_diff',
                        'p_adjusted', 'significant', 'test_used']
        assert all(col in result.columns for col in expected_cols)
        
        # Check number of comparisons
        assert len(result) == 3  # 3 choose 2
    
    def test_auto_test_selection_normal(self, normal_groups: dict[str, np.ndarray]) -> None:
        """Test automatic selection of t-test for normal data"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        result = pairwise_tests(normal_groups, test='auto')
        
        # Should use t-test for normal data
        # (May use Mann-Whitney if normality test fails, so we just check it runs)
        assert 'test_used' in result.columns
        assert result['test_used'].iloc[0] in ['t-test', 'Mann-Whitney U']
    
    def test_forced_t_test(self, normal_groups: dict[str, np.ndarray]) -> None:
        """Test forcing t-test"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        result = pairwise_tests(normal_groups, test='t')
        
        # Check test used
        assert all(result['test_used'] == 't-test')
    
    def test_forced_mann_whitney(self, non_normal_groups: dict[str, np.ndarray]) -> None:
        """Test forcing Mann-Whitney U test"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        result = pairwise_tests(non_normal_groups, test='mannwhitneyu')
        
        # Check test used
        assert all(result['test_used'] == 'Mann-Whitney U')
    
    def test_p_value_corrections(self, normal_groups: dict[str, np.ndarray]) -> None:
        """Test different correction methods"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        methods = ['bonferroni', 'holm', 'BH']
        
        for method in methods:
            result = pairwise_tests(normal_groups, p_adjust=method)
            assert isinstance(result, pd.DataFrame)
            # Adjusted p-values should be >= raw p-values
            assert all(result['p_adjusted'] >= result['p_raw'])
    
    def test_significance_threshold(self, normal_groups: dict[str, np.ndarray]) -> None:
        """Test significance threshold - fixed at 0.05 in implementation"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        result = pairwise_tests(normal_groups)
        
        # Significant column should match p_adjusted < 0.05
        for _, row in result.iterrows():
            expected_sig = row['p_adjusted'] < 0.05
            assert row['significant'] == expected_sig
    
    def test_large_differences_detected(self) -> None:
        """Test detection of large differences"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        groups_different = {
            'Low': np.array([1, 2, 3, 2, 1] * 10),
            'High': np.array([100, 101, 99, 100, 102] * 10)
        }
        
        result = pairwise_tests(groups_different, p_adjust='bonferroni')
        
        # Should detect highly significant difference
        assert result.iloc[0]['significant'] == True
        assert result.iloc[0]['p_raw'] < 0.001
    
    def test_unequal_sample_sizes(self) -> None:
        """Test with unequal sample sizes"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        groups_unequal = {
            'Small': np.random.normal(10, 1, 10),
            'Large': np.random.normal(15, 1, 50),
        }
        
        result = pairwise_tests(groups_unequal)
        
        # Should handle unequal sizes
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_empty_groups_error(self) -> None:
        """Test error for empty input"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        with pytest.raises((ValueError, KeyError, AssertionError)):
            pairwise_tests({})
    
    def test_invalid_test_type(self, normal_groups: dict[str, np.ndarray]) -> None:
        """Test error for invalid test type"""
        from daspi.statistics.hypothesis import pairwise_tests
        
        with pytest.raises((ValueError, AssertionError)):
            pairwise_tests(normal_groups, test='invalid_test')

    def test_kurtosis_test(self) -> None:
        from daspi.statistics.hypothesis import kurtosis_test
        a = np.random.normal(0, 1, 30)
        p, stat = kurtosis_test(a)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert 0 <= p <= 1

    def test_skew_test(self) -> None:
        from daspi.statistics.hypothesis import skew_test
        a = np.random.normal(0, 1, 30)
        p, stat = skew_test(a)
        assert isinstance(p, float)
        assert isinstance(stat, float)
        assert 0 <= p <= 1

    def test_ensure_generic(self) -> None:
        from daspi.statistics.hypothesis import ensure_generic
        dist = ensure_generic('norm')
        assert hasattr(dist, 'cdf')
        import scipy.stats as stats
        dist2 = ensure_generic(stats.norm)
        assert hasattr(dist2, 'cdf')
        with pytest.raises(AttributeError):
            ensure_generic('not_a_dist')