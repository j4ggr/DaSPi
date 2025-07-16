import sys
import pytest
import warnings

from pathlib import Path
from pandas.core.frame import DataFrame

sys.path.append(str(Path(__file__).parent.resolve()))

from daspi.doe.build import *
from daspi.constants import DOE


class TestFactor:

    def test_factor_init_basic(self) -> None:
        """Test basic factor initialization."""
        factor = Factor('temperature', (20, 30, 40))
        assert factor.name == 'temperature'
        assert factor.levels == (20, 30, 40)
        assert factor.central_point == 30  # middle level for 3-level factor
        assert not factor.is_categorical
        
    def test_factor_init_with_central_point(self) -> None:
        """Test factor initialization with explicit central point."""
        factor = Factor('pressure', (1, 5))
        assert factor.name == 'pressure'
        assert factor.levels == (1, 5)
        assert factor.central_point == 3
        assert not factor.is_categorical
        
    def test_factor_init_categorical(self) -> None:
        """Test categorical factor initialization."""
        factor = Factor('material', ('A', 'B', 'C'), is_categorical=True)
        assert factor.name == 'material'
        assert factor.levels == ('A', 'B', 'C')
        assert factor.central_point is None
        assert factor.is_categorical
        
    def test_factor_auto_detect_categorical(self) -> None:
        """Test automatic detection of categorical factors."""
        factor = Factor('type', ('low', 'medium', 'high'))
        assert factor.is_categorical
        assert factor.central_point is None
        
    def test_factor_two_level_central_point(self) -> None:
        """Test automatic central point calculation for 2-level numeric factor."""
        factor = Factor('speed', (10, 20))
        assert factor.central_point == 15  # (10 + 20) / 2
        assert not factor.is_categorical
        
    def test_factor_three_level_central_point(self) -> None:
        """Test automatic central point selection for 3-level numeric factor."""
        factor = Factor('voltage', (5, 10, 15))
        assert factor.central_point == 10  # middle level
        assert not factor.is_categorical
        
    def test_factor_n_levels_property(self) -> None:
        """Test n_levels property."""
        factor = Factor('temp', (20, 30, 40))
        assert factor.n_levels == 3
        
    def test_factor_mixed_types_warning(self) -> None:
        """Test warning when string levels are detected in numeric factor."""
        with pytest.warns(UserWarning, match='Found string level.*Assuming factor is categorical'):
            factor = Factor('mixed', (1, 'medium', 3))
            assert factor.is_categorical
            
    def test_factor_categorical_no_central_point(self) -> None:
        """Test that categorical factors don't get central points."""
        factor = Factor('category', ('X', 'Y', 'Z'))
        assert factor.central_point is None
        assert factor.is_categorical
    
    def test_corrected_central_points_numeric(self) -> None:
        """Test corrected central point for numeric factors."""
        factor = Factor('numeric', (10, 20, 30))
        assert factor.corrected_central_points == (DOE.CORRECTED_CENTRAL,)

    def test_corrected_central_points_categorical(self) -> None:
        factor = Factor('category', ('X', 'Y', 'Z'), is_categorical=True)
        assert factor.corrected_central_points != (DOE.CORRECTED_CENTRAL,)
        assert factor.central_point is None  # No central point for categorical
        assert factor.corrected_central_points == factor.corrected_levels


class TestFullFactorialDesignBuilder:

    def test_basic_full_factorial(self) -> None:
        factor_a = Factor('A', (1, 2))
        factor_b = Factor('B', (10, 20))
        builder = FullFactorialDesignBuilder(factor_a, factor_b)
        df = builder.build_design(corrected=False)
        # 2x2 = 4 runs
        assert df.shape[0] == 4
        assert set(df['A']) == {1, 2}
        assert set(df['B']) == {10, 20}
        combos = set(tuple(row) for row in df[['A', 'B']].values)
        assert combos == {(1, 10), (1, 20), (2, 10), (2, 20)}

    def test_full_factorial_with_replicates(self) -> None:
        factor_a = Factor('A', (1, 2))
        factor_b = Factor('B', (10, 20))
        builder = FullFactorialDesignBuilder(factor_a, factor_b, replicates=3)
        df = builder.build_design(corrected=False)
        # 2x2x3 = 12 runs
        assert df.shape[0] == 12
        assert set(df['A']) == {1, 2}
        assert set(df['B']) == {10, 20}
        # Each combination appears 3 times
        combos = df.groupby(['A', 'B']).size()
        assert all(combos == 3)

    def test_full_factorial_shuffle(self) -> None:
        fA = Factor('A', (1, 2))
        fB = Factor('B', (10, 20))
        fC = Factor('C', (100, 200))
        fD = Factor('D', (1000, 2000))
        builder = FullFactorialDesignBuilder(fA, fB, fC, fD, shuffle=True)
        df1 = builder.build_design(corrected=False)
        builder2 = FullFactorialDesignBuilder(fA, fB, fC, fD, shuffle=True)
        df2 = builder2.build_design(corrected=False)
        # Shuffling should result in different run orders (not always, but likely)
        assert not df1.equals(df2) or df1.shape == df2.shape
        assert df1[DOE.RUN_ORDER].equals(df2[DOE.RUN_ORDER])
        assert not df1[DOE.STD_ORDER].equals(df1[DOE.RUN_ORDER])
        assert not df2[DOE.STD_ORDER].equals(df2[DOE.RUN_ORDER])

    def test_full_factorial_column_names(self):
        factor_a = Factor('A', (1, 2))
        factor_b = Factor('B', (10, 20))
        builder = FullFactorialDesignBuilder(factor_a, factor_b)
        df = builder.build_design(corrected=False)
        # Check for required columns
        from daspi.constants import DOE
        for col in [DOE.STD_ORDER, DOE.RUN_ORDER, DOE.REPLICA, DOE.BLOCK, 'A', 'B']:
            assert col in df.columns

    def test_evenly_spaced_blocks(self) -> None:
        factor_a = Factor('A', (1, 2, 3, 4))
        factor_b = Factor('B', (10, 20))
        builder = FullFactorialDesignBuilder(factor_a, factor_b, blocks=4)
        df = builder.build_design(corrected=False)
        # 4x2 = 8 runs, 4 blocks, each block should have 2 runs
        assert df[DOE.BLOCK].nunique() == 4
        block_counts = df[DOE.BLOCK].value_counts().sort_index().tolist()
        assert block_counts == [2, 2, 2, 2]

    def test_blocks_by_highest_interaction(self) -> None:
        factor_a = Factor('A', (-1, 1))
        factor_b = Factor('B', (-1, 1))
        builder = FullFactorialDesignBuilder(factor_a, factor_b, blocks='highest')  # type: ignore
        df = builder.build_design(corrected=True)
        # 2x2 = 4 runs, blocks assigned by A*B interaction
        assert df[DOE.BLOCK].nunique() == 2
        # Check block assignment by product of A and B
        ab = df['A'] * df['B']
        block_map = dict(zip(ab.unique(), df[DOE.BLOCK].unique()))
        for _, row in df.iterrows():
            assert row[DOE.BLOCK] == block_map[row['A'] * row['B']]

    def test_blocks_by_user_interaction(self) -> None:
        factor_a = Factor('A', (-1, 1))
        factor_b = Factor('B', (-1, 1))
        factor_c = Factor('C', (-1, 1))
        builder = FullFactorialDesignBuilder(factor_a, factor_b, factor_c, blocks=['A', 'B'])  # type: ignore
        df = builder.build_design(corrected=True)
        # 2x2x2 = 8 runs, blocks assigned by A*B interaction
        assert df[DOE.BLOCK].nunique() == 2
        ab = df['A'] * df['B']
        block_map = dict(zip(ab.unique(), df[DOE.BLOCK].unique()))
        for _, row in df.iterrows():
            assert row[DOE.BLOCK] == block_map[row['A'] * row['B']]

    def test_blocks_by_replica(self) -> None:
        factor_a = Factor('A', (1, 2))
        factor_b = Factor('B', (10, 20))
        builder = FullFactorialDesignBuilder(factor_a, factor_b, replicates=3, blocks='replica')  # type: ignore
        df = builder.build_design(corrected=False)
        # Each block should correspond to a replica
        assert set(df[DOE.BLOCK].unique()) == set(df[DOE.REPLICA].unique())
        # Each block should have 4 runs (2x2)
        block_counts = df[DOE.BLOCK].value_counts().sort_index().tolist()
        assert block_counts == [4, 4, 4]


class TestFullFactorial2kDesignBuilder:

    def test_2k_basic(self) -> None:
        # 2 factors, 2 levels each
        factor_a = Factor('A', (0, 1))
        factor_b = Factor('B', (0, 1))
        builder = FullFactorial2kDesignBuilder(factor_a, factor_b)
        df = builder.build_design(corrected=False)
        assert df.shape[0] == 4
        combos = set(tuple(row) for row in df[['A', 'B']].values)
        assert combos == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_2k_with_replicates(self) -> None:
        factor_a = Factor('A', (0, 1))
        factor_b = Factor('B', (0, 1))
        builder = FullFactorial2kDesignBuilder(factor_a, factor_b, replicates=2)
        df = builder.build_design(corrected=False)
        assert df.shape[0] == 8
        combos = df.groupby(['A', 'B']).size()
        assert all(combos == 2)

    def test_2k_shuffle(self) -> None:
        factor_a = Factor('A', (0, 1))
        factor_b = Factor('B', (0, 1))
        replicates = 4
        builder = FullFactorial2kDesignBuilder(
            factor_a, factor_b, replicates=replicates, shuffle=True)
        df1 = builder.build_design(corrected=False)
        builder2 = FullFactorial2kDesignBuilder(
            factor_a, factor_b, replicates=replicates, shuffle=True)
        df2 = builder2.build_design(corrected=False)
        # Shuffling should result in different run orders (not always, but likely)
        assert not df1.equals(df2) or df1.shape == df2.shape
        assert df1[DOE.RUN_ORDER].equals(df2[DOE.RUN_ORDER])
        assert not df1[DOE.STD_ORDER].equals(df1[DOE.RUN_ORDER])
        assert not df2[DOE.STD_ORDER].equals(df2[DOE.RUN_ORDER])

    def test_2k_column_names(self) -> None:
        from daspi.constants import DOE
        factor_a = Factor('A', (0, 1))
        factor_b = Factor('B', (0, 1))
        builder = FullFactorial2kDesignBuilder(factor_a, factor_b)
        df = builder.build_design(corrected=False)
        for col in [DOE.STD_ORDER, DOE.RUN_ORDER, DOE.REPLICA, DOE.BLOCK, 'A', 'B']:
            assert col in df.columns

    def test_full_factorial_with_central_points(self) -> None:
        factor_a = Factor('A', (0, 10))
        factor_b = Factor('B', (100, 200))
        builder = FullFactorial2kDesignBuilder(factor_a, factor_b, central_points=2)
        df = builder.build_design(corrected=False)
        # There should be 4 factorial runs + 2 central points = 6 rows
        assert df.shape[0] == 6
        # Central points should have the central_point attribute value
        central_a = factor_a.central_point
        central_b = factor_b.central_point
        central_mask = (df['A'] == central_a) & (df['B'] == central_b)
        assert central_mask.sum() == 2
        assert all(df[central_mask][DOE.CENTRAL_POINT] == DOE.CORRECTED_CENTRAL)
        # The rest are factorial runs
        factorial_rows = df[~central_mask]
        assert all(factorial_rows[DOE.CENTRAL_POINT] != DOE.CORRECTED_CENTRAL)
        assert set(tuple(row) for row in factorial_rows[['A', 'B']].values) == {
            (0, 100), (0, 200), (10, 100), (10, 200)}

class TestFractionalFactorialDesignBuilder:

    def test_basic(self) -> None:
        fA = Factor('A', (-1, 1))
        fB = Factor('B', (-1, 1))
        fC = Factor('C', (-1, 1))
        builder = FractionalFactorialDesignBuilder(fA, fB, fC, generators=['C=AB'])
        df = builder.build_design(corrected=False)
        # Should be 4 runs, C = A*B
        assert df.shape[0] == 4
        for _, row in df.iterrows():
            assert row['C'] == row['A'] * row['B']

    def test_foldover_all(self) -> None:
        fA = Factor('A', (-1, 1))
        fB = Factor('B', (-1, 1))
        fC = Factor('C', (-1, 1))
        builder = FractionalFactorialDesignBuilder(
            fA, fB, fC, generators=['C=AB'], fold=True, shuffle=False)
        assert hasattr(builder, 'fold') and builder.fold == True
        # Foldover should double the runs
        df = builder.build_design(corrected=False)
        # Should be 8 runs, with foldover (all reversed)
        assert df.shape[0] == 8
        # Folded runs: for each original, there should be a reversed run
        original = df.iloc[:4]
        folded = df.iloc[4:]
        for col in ['A', 'B', 'C']:
            for i in range(4):
                assert folded.iloc[i][col] == -original.iloc[i][col]

    def test_foldover_by_factor(self) -> None:
        fA = Factor('A', (-1, 1))
        fB = Factor('B', (-1, 1))
        fC = Factor('C', (-1, 1))
        builder = FractionalFactorialDesignBuilder(
            fA, fB, fC, generators=['C=AB'], fold='A', shuffle=False)
        assert hasattr(builder, 'fold') and builder.fold == 'A'
        # Foldover should double the runs
        df = builder.build_design(corrected=False)
        # Should be 8 runs, with foldover (all reversed)
        assert df.shape[0] == 8
        # Folded runs: for each original, there should be a run with A reversed
        original = df.iloc[:4]
        folded = df.iloc[4:]
        for col in ['A', 'B', 'C']:
            for i in range(4):
                if col == 'A':
                    assert folded.iloc[i][col] == -original.iloc[i][col]
                else:
                    assert folded.iloc[i][col] == original.iloc[i][col]

    def test_full_fact_warning(self) -> None:
        # Foldover that just replicates the design should warn
        fA = Factor('A', (-1, 1))
        fB = Factor('B', (-1, 1))
        fC = Factor('C', (-1, 1))
        builder = FractionalFactorialDesignBuilder(
            fA, fB, fC, generators=['C=AB'], fold='A', shuffle=False)
        # If folding over A, but all combinations already present, should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            df = builder.build_design(corrected=False)
            assert len(w) > 0
            assert 'Foldover creates a full factorial design' in str(w[0].message)

    def test_duplicate_fold_warning(self) -> None:
        # Foldover on full factorial should warn and not duplicate
        fA = Factor('A', (-1, 1))
        fB = Factor('B', (-1, 1))
        builder = FullFactorial2kDesignBuilder(fA, fB, shuffle=False)
        df = builder.build_design(corrected=True)
        # Simulate foldover logic
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            builder.fold = True
            df_folded = builder.build_design(corrected=True)
            assert len(w) > 0
            assert 'Foldover does not add new runs' in str(w[-1].message)
            assert df_folded.equals(df)

    def test_invalid_generator(self) -> None:
        fA = Factor('A', (-1, 1))
        fB = Factor('B', (-1, 1))
        fC = Factor('C', (-1, 1))
        # Invalid generator string (no dependent factor)
        try:
            FractionalFactorialDesignBuilder(fA, fB, fC, generators=['CAB'])
            assert False, 'Should raise AssertionError for invalid generator format'
        except AssertionError:
            pass

    def test_decode_values(self) -> None:
        # Test decoding from corrected to original values
        fA = Factor('A', (10, 20))
        fB = Factor('B', (100, 200))
        builder = FullFactorialDesignBuilder(fA, fB)
        df = builder.build_design(corrected=True)
        df_decoded = builder._decode_values(df, builder.factors)
        assert set(df_decoded['A']) == {10, 20}
        assert set(df_decoded['B']) == {100, 200}