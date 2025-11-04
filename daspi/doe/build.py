import warnings

import numpy as np
import pandas as pd

from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal

from abc import ABC
from abc import abstractmethod
from itertools import product
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from .._typing import LevelType

from ..constants import DOE


__all__ = [
    'Factor',
    'BaseDesignBuilder',
    'FullFactorialDesignBuilder',
    'FullFactorial2kDesignBuilder',
    'FractionalFactorialDesignBuilder',
    'get_default_generators',]


class Factor:
    """Represents a factor in a design of experiments.

    Parameters
    ----------
    name : str
        Name of the factor.
    levels : Tuple[LevelType, ...]
        Levels of the factor (numeric or categorical).
    is_categorical : bool, optional
        Whether the factor is categorical. This is automatically 
        inferred from the levels if not specified. Categorical factors
        are assumed to be unordered and have no central point.
    """
    
    _name: str
    """Name of the factor."""
    _levels: Tuple[LevelType, ...]
    """Levels of the factor (numeric or categorical)."""
    _central_point: LevelType | None = None
    """Optional central point level."""
    _is_categorical: bool = False
    """Whether the factor is categorical."""
    _corrected_levels: Tuple[float | int, ...]
    """Corrected levels for float-coded designs."""
    _corrected_level_map: Dict[float | int, LevelType]
    """Mapping from float or int codes to their corresponding factor 
    levels. This is used for float-coded designs where levels are 
    represented by floats. Set this from outside the class, e.g. in a 
    design builder."""
    _n_levels: int
    
    def __init__(
            self,
            name: str,
            levels: Tuple[LevelType, ...],
            is_categorical: bool = False
            ) -> None:
        self._n_levels = len(levels)
        assert self.n_levels > 1, 'At least two levels are required.'

        self._name = name
        
        if any(isinstance(lvl, str) for lvl in levels) and not is_categorical:
            warnings.warn(
                f'Found string level(s) in factor {name}. Assuming factor is '
                'categorical. To avoid this warning, set is_categorical=True.')
            self._is_categorical = True
        else:
            self._is_categorical = is_categorical
        
        if self.is_categorical:
            self._levels = tuple(levels)
        else:
            self._levels = tuple(sorted(levels))

        if not self.is_categorical:
            self._central_point = (self._levels[0] + self._levels[-1]) / 2 # type: ignore
        else:
            self._central_point = None
        
        if self.is_categorical:
            self._corrected_levels = tuple(range(self.n_levels))
        else:
            lim = self.n_levels // 2
            self._corrected_levels = tuple(
                np.linspace(-lim, lim, self.n_levels, dtype=int))
        
        self._corrected_level_map = dict(
            zip(self._corrected_levels, self._levels))
        
        if self.central_point is not None:
            self._corrected_level_map |= {
                DOE.CORRECTED_CENTRAL: self.central_point}

    @property
    def n_levels(self) -> int:
        """Return number of levels (read-only)."""
        return self._n_levels

    @property
    def name(self) -> str:
        """Name of the factor (read-only)."""
        return self._name

    @property
    def is_categorical(self) -> bool:
        """Return whether the factor is categorical (read-only)."""
        return self._is_categorical
    
    @property
    def levels(self) -> Tuple[LevelType, ...]:
        """Levels of the factor (read-only)."""
        return self._levels

    @property
    def central_point(self) -> LevelType | None:
        """Central point of the factor (read-only)."""
        return self._central_point
    
    @property
    def is_centralized(self) -> bool:
        """Check if the corrected levels are centered around 0 
        (read-only)."""
        return sum(self.corrected_levels) == 0

    @property
    def corrected_central_points(self) -> Tuple[float | int, ...]:
        """Corrected central points as tuple, used for float-coded 
        designs (read-only).
        
        If the factor has a central point, it returns a tuple with the
        central coded value, otherwise it returns the corrected levels.

        Notes
        -----
        If this property is accessed, it will also update the 
        `corrected_level_map`.
        """
        if self.central_point is not None:
            self._corrected_level_map |= {
                DOE.CORRECTED_CENTRAL: self.central_point}
            return (DOE.CORRECTED_CENTRAL,)
        else:
            return self.corrected_levels

    @property
    def corrected_levels(self) -> Tuple[float | int, ...]:
        """Corrected levels for float-coded designs (read-only)."""
        return self._corrected_levels
    
    @property
    def corrected_level_map(self) -> Dict[float | int, LevelType]:
        """Mapping from float or int codes to their corresponding factor 
        levels (read-only)."""
        return self._corrected_level_map


class BaseDesignBuilder(ABC):
    """Abstract base class for DOE builders.
    
    Parameters
    ----------
    factors : Iterable[Factor]
        Factors defining the design space.
    replicates : int, optional
        Number of replicates. Must be positive, by default 1.
    blocks : int | str | List[str] | Literal['highest', 'replica'], optional
        Block assignment: integer for evenly spaced blocks, 
        str | List[str] for user-defined block generator, or Literal 
        'highest'/'replica'. Must be positive or 'highest'/'replica', by 
        default 1.
    central_points : int, optional
        Number of central points to be added to each block. These are
        used to test linear effects. Must be non-negative. If set to 0,
        no central points are added. by default 0.
    shuffle : bool, optional
        Whether to shuffle the design, by default True.
    
    Raises
    ------
    AssertionError
        If any of the parameters are invalid:
        - At least one factor is required.
        - Factor names must not conflict with standard columns in the 
          design matrix.
        - All factors must be instances of Factor class.
        - Number of replicates must be positive.
        - Number of central points must be non-negative.
        - At least one factor must provide a central point, or set 
          central_points to 0.
        - Number of blocks must be positive.
        - Factor names must be unique.
    """

    _factors : Tuple[Factor, ...]
    _factor_names: List[str]
    _replicates : int
    _central_points : int
    _blocks : int | str | List[str] | Literal['highest', 'replica']
    _level_counts : Tuple[int, ...]
    fold: bool | str
    """Whether to add a foldover to the design."""
    shuffle : bool
    """Whether to shuffle the design."""

    def __init__(
            self, 
            *factors: Factor,
            replicates: int = 1,
            blocks: int | str | List[str] | Literal['highest', 'replica'] = 1,
            central_points: int = 0,
            shuffle: bool = True,
            ) -> None:
        self.factors = factors
        self.replicates = replicates
        self.central_points = central_points
        self.blocks = blocks
        self.shuffle = shuffle

    @property
    def factors(self) -> Tuple[Factor, ...]:
        """Tuple of factors defining the design space."""
        return self._factors
    @factors.setter
    def factors(self, factors: Tuple[Factor, ...]) -> None:
        factor_names = [f.name for f in factors]
    
        assert factors, 'At least one factor is required.'
        assert all(isinstance(f, Factor) for f in factors), (
            'All factors must be instances of Factor class.')
        assert not any(n in self.standard_columns for n in factor_names), (
            'Factor names must not conflict with standard columns in the '
            f'design matrix. Standard columns are: {self.standard_columns}')
        assert len(set(factor_names)) == len(factor_names), (
            'Factor names must be unique.')
        
        self._factors = factors
        self._factor_names = factor_names
        self._level_counts = tuple(f.n_levels for f in self.factors)

    @property
    def factor_names(self) -> List[str]:
        """List of factor names (read-only)."""
        return self._factor_names

    @property
    def level_counts(self) -> Tuple[int, ...]:
        """Tuple of level counts for each factor (read-only)."""
        return self._level_counts

    @property
    def replicates(self) -> int:
        """Number of replicates (read-only)."""
        return self._replicates
    @replicates.setter
    def replicates(self, replicates: int) -> None:
        assert replicates > 0, 'Number of replicates must be positive.'
        self._replicates = replicates
    
    @property
    def central_points(self) -> int:
        """Number of central points (read-only)."""
        return self._central_points
    @central_points.setter
    def central_points(self, central_points: int) -> None:
        assert central_points >= 0, (
            'Number of central points must be non-negative.')
        if central_points > 0:
            assert any(f.central_point is not None for f in self.factors), (
                'At least one factor must provide a central point, '
                'or set central_points to 0.')
        self._central_points = central_points

    @property
    def blocks(self) -> int | str | List[str] | Literal['highest','replica']:
        """Block assignment: integer for evenly spaced blocks, 
        str | List[str] for user-defined block generator, or Literal 
        'highest'/'replica'."""
        return self._blocks
    @blocks.setter
    def blocks(
            self,
            blocks: int | str | List[str]  | Literal['highest', 'replica']
            ) -> None:
        if blocks in ('highest', 'replica'):
            self._blocks = blocks
        
        elif isinstance(blocks, int):
            assert blocks > 0, 'Number of blocks must be positive.'
            self._blocks = blocks
        
        elif isinstance(blocks, (str, list)):
            assert all(b in self.factor_names for b in blocks), (
                'All block generator elements must be str.')
            assert len(blocks) > 0, 'Block generator must not be empty.'
            self._blocks = blocks
        
        else:
            raise ValueError(
                'blocks must be int, Iterable[str], or '
                f"Literal['highest', 'replica'], got {blocks}.")

    @property
    def standard_columns(self) -> List[str]:
        """List of standard columns in the design matrix."""
        return [
            DOE.STD_ORDER,
            DOE.RUN_ORDER,
            DOE.CENTRAL_POINT,
            DOE.REPLICA,
            DOE.BLOCK]
    
    @property
    def columns(self) -> List[str]:
        """List of columns in the design matrix."""
        return self.standard_columns + self.factor_names
    
    @property
    def is_2k(self) -> bool:
        """Check if the design is a 2^k design."""
        return all(count == 2 for count in self.level_counts)

    @staticmethod
    def _decode_values(
            df_design: DataFrame,
            factors: Tuple[Factor, ...],
            ) -> DataFrame:
        """
        Map integer-coded design matrix to original factor values.

        Parameters
        ----------
        df_design : NDArray
            Matrix with integer indices for levels.
        factors : Tuple[Factor, ...]
            Factors for mapping indices to original values.

        Returns
        -------
        DataFrame
            DataFrame with original factor values.
        
        Raises
        ------
        AssertionError
            -If the design matrix columns do not match factor names.
        """
        assert all(f.name in df_design.columns for f in factors), (
            'Design matrix columns must match factor names.')

        df_design = df_design.copy()
        for factor in factors:
            df_design[factor.name] = df_design[factor.name].replace(
                factor.corrected_level_map)
        return df_design

    @abstractmethod
    def _build_corrected_base_design(self) -> DataFrame:
        """
        Abstract method to generate the corrected base design matrix.

        This method must be implemented by subclasses to generate the
        base design matrix with corrected values of factor levels.
        It should return a DataFrame where each column corresponds to a
        factor, and each row represents an experimental run.

        Returns
        -------
        DataFrame
            Base design matrix with corrected values of factor levels.
        """
        pass

    def _fold_design(self, df_design: DataFrame) -> DataFrame:
        """Fold the design matrix if fold is True.

        This method adds a foldover to the design matrix by reversing
        the levels of all factors. If fold is a string, it should be
        the name of the factor to fold over. If fold is False, no
        folding is applied.

        Parameters
        ----------
        df_design : DataFrame
            Design matrix to fold.

        Returns
        -------
        DataFrame
            Folded design matrix.

        Raises
        AssertionError
            If fold is not in factor names or is not a boolean.
        """
        fold = getattr(self, 'fold', False)
        assert fold in self.factor_names or isinstance(fold, bool), (
            f'Fold factor "{fold}" not found in factor names: '
            f'{self.factor_names}')
        if not fold:
            return df_design

        assert self.is_2k, (
            'Foldover is only applicable to 2^k designs. '
            'Use FullFactorial2kDesignBuilder for 2^k designs.')
        
        folded = df_design.copy()
        names = fold if fold in self.factor_names else self.factor_names
        folded.loc[:, names] = -folded.loc[:, names]
        combined = pd.concat([df_design, folded], ignore_index=True)
        if len(combined[self.factor_names].drop_duplicates()) == len(df_design):
            warnings.warn(
                'Foldover does not add new runs, the folded design is already '
                'present in the original design. Foldover will simply '
                'replicate the design, so it will not be applied.')
            return df_design
        
        elif len(combined) >= np.prod(self.level_counts):
            warnings.warn(
                'Foldover creates a full factorial design with all possible '
                'runs. This may not be intended. Consider using a different '
                'design builder.')

        df_design = combined
        return df_design

    def _replicate_design(self, df_design: DataFrame) -> DataFrame:
        """Replicate the design matrix according to the replicates 
        parameter.

        If replicates > 1, each row in the design matrix is
        repeated according to the number of replicates, and a new
        column is added to indicate the replicate number.
        
        Parameters
        ----------
        df_design : DataFrame
            Design matrix to replicate.

        Returns
        -------
        DataFrame
            Replicated design matrix.
        """
        if self.replicates > 1:
            df_design = pd.concat(
                [df_design] * self.replicates,
                ignore_index=True)
        return df_design
    
    def _count_replicates(self, df_design: DataFrame) -> Series:
        """Count the number of replicates for each factor combination.

        Parameters
        ----------
        df_design : DataFrame
            Design matrix to count replicates for.
        
        Returns
        -------
        Series
            Series with the count of replicates for each factor 
            combination.
        """
        return df_design.groupby(self.factor_names).cumcount() + 1
    
    def _add_block_column(self, df_design: DataFrame) -> DataFrame:
        """Add a block column to the design matrix based on the blocks 
        option.

        - If blocks is an int > 1: assign blocks evenly 
          (not statistically confounded).
        - If blocks is an Iterable[str]: assign blocks by confounding 
          with the specified interaction (statistically correct).
        - If blocks is 'highest': assign blocks by confounding with the 
          highest-order interaction (all centralized factors).
        - If blocks is 'replica': assign blocks based on the replicate 
          number.
        - If blocks == 1: assign all runs to block 1.

        Returns
        -------
        DataFrame
            Design matrix with an additional column for the block 
            assignments.
        """
        if self.blocks == 1:
            _blocks = [1 for _ in range(len(df_design))]

        elif isinstance(self.blocks, int):
            n = len(df_design)
            sizes = [
                n // self.blocks + (1 if x < n % self.blocks else 0)
                for x in range(self.blocks)]
            _blocks = [
                i for i, n in enumerate(sizes, start=1) for _ in range(n)]

        elif self.blocks == 'replica':
            _blocks = self._count_replicates(df_design).to_list()

        elif self.blocks == 'highest' or isinstance(self.blocks, (str, list)):
            if self.blocks == 'highest':
                factors = [
                    f.name for f in self.factors if f.is_centralized]
                assert factors, (
                    'No factors with centralized levels found for highest '
                    'order interaction. Use a different blocks option.')
                if len(factors) == 1:
                    warnings.warn(
                        f'Only one centralized factor found {factors[0]}. '
                        'Using it as the block generator. The block will be '
                        'confounded with this factor. Consider using a '
                        'different blocks option for more complex designs.')
            else:
                factors = list(self.blocks)
            product = df_design[factors].prod(axis=1)
            mapper = {v: i for i, v in enumerate(product.unique(), start=1)}
            _blocks = [mapper[v] for v in product]

        else:
            raise ValueError(f'Invalid blocks option: {self.blocks}')

        blocks = pd.Series(_blocks, dtype=int, index=df_design.index)
        df_design[DOE.BLOCK] = blocks
        if blocks.nunique() > 1 and not isinstance(self.blocks, int):
            df_design = df_design.sort_values(DOE.BLOCK, ignore_index=True)
        return df_design

    def _add_central_points(self, df_design: DataFrame) -> DataFrame:
        """Add central points to the design matrix if specified.

        If central points are specified, they are added to each block in
        the design matrix as additional rows with the central point 
        level for each factor.
        
        Parameters
        ----------
        df_design : DataFrame
            Design matrix to add central points to.
        
        Returns
        -------
        DataFrame
            Design matrix with central points added.
        """
        df_design[DOE.CENTRAL_POINT] = 1
        if not self.central_points > 0:
            return df_design

        df_central = pd.concat([
            pd.DataFrame(
                product(*[f.corrected_central_points for f in self.factors]),
                columns=self.factor_names)]
            * self.central_points, 
            ignore_index=True)
        df_central[DOE.CENTRAL_POINT] = 0
        
        df_design = (
            df_design.groupby(DOE.BLOCK, group_keys=False)
            .apply(lambda group: pd.concat([group, df_central]))
            .reset_index(drop=True))
        df_design[DOE.BLOCK] = df_design[DOE.BLOCK].ffill()
        return df_design

    def _shuffle_design(self, df_design: DataFrame) -> DataFrame:
        """
        Shuffle the design matrix if shuffle is True.

        This method adds a standard order column to the design matrix,
        and a run order column after shuffling.

        Parameters
        ----------
        df_design : DataFrame
            Design matrix to shuffle.

        Returns
        -------
        DataFrame
            Shuffled design matrix.
        """
        df_design[DOE.STD_ORDER] = np.arange(len(df_design))
        if self.shuffle:
            df_design = (
                df_design.groupby(DOE.BLOCK, group_keys=False)
                .apply(lambda group: group.sample(frac=1))
                .reset_index(drop=True))
        df_design[DOE.RUN_ORDER] = np.arange(len(df_design))
        return df_design
    
    def _clean_design(self, df_design: DataFrame) -> DataFrame:
        """Clean the design matrix by sorting columns and converting
        standard columns to integer type."""
        df_design = df_design[self.columns]
        df_design[self.standard_columns] = (
            df_design[self.standard_columns].astype(int))
        return df_design

    def build_design(
            self,
            corrected: bool = True
            ) -> DataFrame:
        """Generate the design matrix with original factor values.

        This method builds the design matrix by generating the
        corrected (integer-coded) design, replicating it according to
        the replicates parameter, shuffling it if specified, and 
        adding central points if specified. Block assignment is
        controlled by the `blocks` option at initialization:

        - If `blocks` is an int > 1: blocks are assigned evenly 
          (not statistically confounded).
        - If `blocks` is an str or List[str]: blocks are assigned by 
          confounding with the specified interaction 
          (statistically correct).
        - If `blocks` is 'highest': blocks are assigned by confounding
          with the highest-order interaction (all centralized factors).
        - If `blocks` is 'replica': blocks are assigned based on the
          replicate number.
        - If `blocks` == 1: all runs are assigned to block 1.

        Returns
        -------
        DataFrame
            Design matrix with original factor values if corrected is
            False, otherwise with integer codes representing factor
            levels.

        Notes
        -----
        Statistically correct block assignment is performed by 
        confounding the block effect with a specified interaction 
        (block generator). For each run, the value of the block 
        generator (the product of the coded levels of the specified 
        factors) is computed, and unique values are mapped to block 
        numbers. If `blocks` is 'highest', the highest-order interaction 
        (all factors) is used. This ensures that block effects are 
        orthogonal to main effects and lower-order interactions when 
        using a confounding generator, as recommended in DOE literature 
        (see Montgomery, 2017). If `blocks` is a str or List[str], the
        specified interaction is used. If `blocks` is 'replica', blocks
        are assigned by replicate. If `blocks` is an int, blocks are
        assigned evenly (not statistically confounded).
        """
        df_design = self._build_corrected_base_design()
        df_design = self._fold_design(df_design)
        df_design = self._replicate_design(df_design)
        df_design = self._add_block_column(df_design)
        df_design = self._add_central_points(df_design)
        df_design = self._shuffle_design(df_design)
        df_design[DOE.REPLICA] = self._count_replicates(df_design)
        df_design = self._clean_design(df_design)

        if not corrected:
            df_design = self._decode_values(df_design, self.factors)
        return df_design


class FullFactorialDesignBuilder(BaseDesignBuilder):
    """Builder for full factorial designs.
    
    Parameters
    ----------
    factors : Iterable[Factor]
        Factors defining the design space.
    replicates : int, optional
        Number of replicates. Must be positive, by default 1.
    blocks : int | str | List[str] | Literal['highest', 'replica'], optional
        Number of blocks or block assignment strategy. For more
        information, see the docstring of the `build_design` method.
        Defaults to 1.
    central_points : int, optional
        Number of central points to be added to each block. These are
        used to test linear effects. Must be non-negative. If set to 0,
        no central points are added. by default 0.
    shuffle : bool, optional
        Whether to shuffle the design, by default True.
    
    Examples
    --------

    Create a full factorial design with two factors, each with two 
    levels:

    ```python
    import daspi as dsp

    factor_a = dsp.Factor('A', (1, 2))
    factor_b = dsp.Factor('B', (10, 20))
    builder = dsp.FullFactorialDesignBuilder(factor_a, factor_b)
    df = builder.build_design(corrected=False)
    print(df)
    ```

    ```console
       std_order  run_order  central_point  replica  block  A   B
    0          0          0              1        1      1  1  10
    1          1          1              1        1      1  1  20
    2          2          2              1        1      1  2  10
    3          3          3              1        1      1  2  20
    ```

    Create a full factorial corrected design with two factors, each with 
    two levels, and 3 replicates and split into blocks using the highest 
    interaction:
    
    ```python
    import daspi as dsp
    import numpy as np

    np.random.seed(42)  # optional for reproducibility

    factor_a = dsp.Factor('A', (1, 2))
    factor_b = dsp.Factor('B', (10, 20))
    builder = dsp.FullFactorialDesignBuilder(
        factor_a, factor_b, replicates=3, blocks='highest')
    df = builder.build_design(corrected=True)
    print(df)
    ```

    ```console
        std_order  run_order  central_point  replica  block    A    B
    0           0          0              1        1      1 -1.0 -1.0
    1           1          1              1        1      1  1.0  1.0
    2           5          2              1        2      1  1.0  1.0
    3           2          3              1        2      1 -1.0 -1.0
    4           4          4              1        3      1 -1.0 -1.0
    5           3          5              1        3      1  1.0  1.0
    6           9          6              1        1      2  1.0 -1.0
    7           6          7              1        1      2 -1.0  1.0
    8           7          8              1        2      2  1.0 -1.0
    9           8          9              1        2      2 -1.0  1.0
    10         11         10              1        3      2  1.0 -1.0
    11         10         11              1        3      2 -1.0  1.0
    ```

    Raises
    ------
    AssertionError
        If any of the parameters are invalid:
        - At least one factor is required.
        - All factors must be instances of Factor class.
        - Number of replicates must be positive.
        - Number of blocks must be positive.
    """

    def __init__(
            self,
            *factors: Factor,
            replicates: int = 1,
            blocks: int | str | List[str] | Literal['highest', 'replica'] = 1,
            central_points: int = 0,
            shuffle: bool = True
            ) -> None:
        super().__init__(
            *factors,
            replicates=replicates,
            blocks=blocks,
            central_points=central_points,
            shuffle=shuffle)

    def _build_corrected_base_design(self) -> DataFrame:
        """Generate the integer-coded design matrix for full factorial 
        design.

        Full factorial design means all possible combinations of factor 
        levels. First generate all combinations using itertools.product, 
        then convert to numpy array for easier manipulation, and finally 
        convert to DataFrame with factor names as columns.
        
        Returns
        -------
        DataFrame
            Design matrix with integer indices representing factor 
            levels. Each column corresponds to a factor, each row to an 
            experimental run. Values are indices into the corresponding 
            factor's levels tuple.
        """
        df_design = pd.DataFrame(
            product(*[f.corrected_levels for f in self.factors]),
            columns=self.factor_names)
        return df_design


class FullFactorial2kDesignBuilder(FullFactorialDesignBuilder):
    """Builder for full factorial designs with 2-level factors.
    This class is a specialization of FullFactorialDesignBuilder for
    the case where all factors have exactly 2 levels.

    This design also supports replicates, central points, and blocks,
    and can be used to explore interactions between factors.

    Parameters
    ----------
    factors : Iterable[Factor]
        Factors defining the design space. All factors must have exactly
        2 levels.
    replicates : int, optional
        Number of replicates. Must be positive, by default 1.
    blocks : int | str | List[str] | Literal['highest', 'replica'], optional
        Block assignment: integer for evenly spaced blocks, 
        str | List[str] for user-defined block generator, or Literal 
        'highest'/'replica'. Must be positive or 'highest'/'replica', by 
        default 1.
    central_points : int, optional
        Number of central points to be added to each block. These are
        used to test linear effects. Must be non-negative. If set to 0,
        no central points are added. by default 0.
    shuffle : bool, optional
        Whether to shuffle the design, by default True.
    
    Examples
    --------
    Create a full factorial design with two factors, each with two 
    levels:
    ```python
    import daspi as dsp

    factor_a = dsp.Factor('A', (0, 1))
    factor_b = dsp.Factor('B', (0, 1))
    builder = dsp.FullFactorial2kDesignBuilder(
        factor_a, factor_b, shuffle=False)
    df = builder.build_design(corrected=False)
    print(df)
    ```

    ```console
       std_order  run_order  central_point  replica  block  A  B
    0          0          0              1        1      1  0  0
    1          1          1              1        1      1  0  1
    2          2          2              1        1      1  1  0
    3          3          3              1        1      1  1  1
    ``` 

    Create a randomized full factorial corrected design with two
    factors, each with two levels, and 3 replicates, blocks by the
    highest interaction and 2 central points per block to test 
    linear effects:

    ```python
    import daspi as dsp
    import numpy as np

    np.random.seed(42) # optional for reproducibility

    factor_a = dsp.Factor('A', (0, 1))
    factor_b = dsp.Factor('B', (0, 1))
    builder = dsp.FullFactorial2kDesignBuilder(
        factor_a, factor_b, replicates=3, central_points=2,
        blocks='highest', shuffle=True)
    df = builder.build_design(corrected=True)
    print(df)
    ```

    ```console
        std_order  run_order  central_point  replica  block  A  B
    0           1          0              1        1      1  1  1
    1           5          1              1        2      1  1  1
    2           0          2              1        1      1 -1 -1
    3           7          3              0        1      1  0  0
    4           2          4              1        2      1 -1 -1
    5           4          5              1        3      1 -1 -1
    6           3          6              1        3      1  1  1
    7           6          7              0        2      1  0  0
    8          11          8              1        1      2  1 -1
    9          15          9              0        3      2  0  0
    10          8         10              1        1      2 -1  1
    11         12         11              1        2      2 -1  1
    12         13         12              1        2      2  1 -1
    13         10         13              1        3      2 -1  1
    14          9         14              1        3      2  1 -1
    15         14         15              0        4      2  0  0
    ``` 

    Raises
    ------
    AssertionError
        If any of the parameters are invalid:
        - All factors must have exactly 2 levels.
        - Number of replicates must be positive.
        - Number of central points must be non-negative.
        - Number of blocks must be positive.
    """
    def __init__(
            self,
            *factors: Factor,
            replicates: int = 1,
            blocks: int | str | List[str] | Literal['highest', 'replica'] = 1,
            central_points: int = 0,
            shuffle: bool = True
            ) -> None:
        super().__init__(
            *factors,
            replicates=replicates,
            blocks=blocks,
            central_points=central_points,
            shuffle=shuffle)
        assert self.is_2k, (
            'All factors must have exactly 2 levels for '
            'FullFactorial2kDesignBuilder.')


def get_default_generators(k: int, p: int) -> list[str]:
    """
    Return standard generators for regular 2-level fractional factorial designs.

    Parameters
    ----------
    k : int
        Total number of factors.
    p : int
        Number of generators (fractionality, so design is 2^(k-p)).

    Returns
    -------
    List[str]
        List of generator strings, e.g. ['C=AB', 'D=AC'].

    Notes
    -----
    These are standard choices for high-resolution designs, suitable for 
    most practical cases. For more, see Montgomery (2017), Box, Hunter 
    & Hunter, or standard DOE tables.
    """
    defaults = {
        (3, 1): ['C=AB'],
        (4, 1): ['D=ABC'],
        (5, 2): ['D=AB', 'E=AC'],
        (6, 3): ['D=AB', 'E=AC', 'F=BC'],
        (7, 3): ['E=ABC', 'F=ABD', 'G=BCD'],
        (8, 4): ['F=ABCD'],
        (9, 4): ['F=ABCD', 'G=ABCE'],
        (10, 5): ['F=ABCDE', 'G=ABCDF'],
        (11, 5): ['F=ABCDE', 'G=ABCDF', 'H=ABCEG'],
        (12, 6): ['F=ABCDEF', 'G=ABCDEFG'],
        (13, 6): ['F=ABCDEF', 'G=ABCDEFG', 'H=ABCDEG'],
        (14, 7): ['F=ABCDEFGH', 'G=ABCDEFGI'],
        (15, 7): ['F=ABCDEFGH', 'G=ABCDEFGHI', 'H=ABCDEFGJ'],
        (16, 8): ['F=ABCDEFGHIJ', 'G=ABCDEFGHIK'],
        (17, 8): ['F=ABCDEFGHIJ', 'G=ABCDEFGHIK', 'H=ABCDEFGHIJ'],
        (18, 9): ['F=ABCDEFGHIJK', 'G=ABCDEFGHIJL'],
        (19, 9): ['F=ABCDEFGHIJK', 'G=ABCDEFGHIJL', 'H=ABCDEFGHIJK'],
    }
    key = (k, p)
    assert key in defaults, (
        f'No default generators for k={k}, p={p}. '
        'Please consult a DOE reference.')

    return defaults[key]


class FractionalFactorialDesignBuilder(BaseDesignBuilder):
    """
    Builder for 2-level fractional factorial designs with optional 
    foldover.


    This class supports the construction of regular 2-level fractional 
    factorial designs using generator strings (e.g., 'C=AB'), and can 
    optionally add a foldover to the design.

    Default generators for common designs (see also 
    get_default_generators):

    | Design      | Basic Factors | Generators                | Example Generator Strings   |
    |-------------|---------------|---------------------------|-----------------------------|
    | 2^(3-1)     | A, B          | C = AB                    | ['C=AB']                    |
    | 2^(4-1)     | A, B, C       | D = ABC                   | ['D=ABC']                   |
    | 2^(5-2)     | A, B, C       | D = AB, E = AC            | ['D=AB', 'E=AC']            |
    | 2^(6-3)     | A, B, C       | D = AB, E = AC, F = BC    | ['D=AB', 'E=AC', 'F=BC']    |
    | 2^(7-3)     | A, B, C, D    | E = ABC, F = ABD, G = BCD | ['E=ABC', 'F=ABD', 'G=BCD'] |


    Parameters
    ----------
    factors : Iterable[Factor]
        Factors defining the design space. All factors must have exactly 
        2 levels.
    generators : List[str]
        List of generator strings that define how dependent factors are 
        constructed from basic factors. For example, if you have 4 
        factors A, B, C, and D, you can set generators to 
        ['C=AB', 'D=BC']. This means that factor C is defined as the 
        product of factors A and B (C = A * B), and factor D as the 
        product of B and C (D = B * C). The order of the generator 
        strings should match the order of the dependent factors in the 
        factors list. For common designs, see the table above or use 
        get_default_generators(k, p).
    fold : bool | str, optional
        Whether to add a foldover. If True, a foldover will be added for 
        all factors. If a string is provided, it should be the name of 
        the factor to fold over. Default is False (no foldover).
    replicates : int, optional
        Number of replicates. Must be positive, by default 1.
    central_points : int, optional
        Number of central points to be added to each block. These are
        used to test linear effects. Must be non-negative. If set to 0,
        no central points are added. by default 0.
    blocks : int, optional
        Number of blocks. Must be positive, by default 1.
    shuffle : bool, optional
        Whether to shuffle the design, by default True.

    Examples
    --------
    Create a 2^(3-1) fractional factorial design (3 factors, 1 generator):

    ```python
    from daspi.doe.build import Factor, FractionalFactorialDesignBuilder

    A = Factor('A', (-1, 1))
    B = Factor('B', (-1, 1))
    C = Factor('C', (-1, 1))
    builder = FractionalFactorialDesignBuilder(A, B, C, generators=['C=AB'])
    df = builder.build_design(corrected=False)
    print(df)
    ```

    ```console
       std_order  run_order  central_point  replica  block  A  B  C
    0          0          0              1        1      1 -1 -1  1
    1          1          1              1        1      1 -1  1 -1
    2          2          2              1        1      1  1 -1 -1
    3          3          3              1        1      1  1  1  1
    ```

    Create the same design with foldover (doubles the number of runs, reverses A):

    ```python
    builder = FractionalFactorialDesignBuilder(
        A, B, C, generators=['C=AB'], fold=True)
    df = builder.build_design(corrected=False)
    print(df)
    ```

    ```console
       std_order  run_order  central_point  replica  block  A  B  C
    0          0          0              1        1      1 -1 -1  1
    1          1          1              1        1      1 -1  1 -1
    2          2          2              1        1      1  1 -1 -1
    3          3          3              1        1      1  1  1  1
    4          4          4              1        1      1  1 -1 -1
    5          5          5              1        1      1  1  1  1
    6          6          6              1        1      1 -1 -1  1
    7          7          7              1        1      1 -1  1 -1
    ```

    Raises
    ------
    AssertionError
        If any of the parameters are invalid:
        - All factors must have exactly 2 levels.
        - Number of replicates must be positive.
        - Number of central points must be non-negative.
        - Number of blocks must be positive.
        - Generators must be provided as a list of strings.
        - Generators must match the number of dependent factors.
        - Each generator string must be in the format 'C=AB' where C is 
          the dependent factor and AB are the independent factors.
        - Each dependent factor must be defined in the generators.
        - The first basic factor must be reversed in sign for foldover.

    Notes
    -----
    In fractional factorial designs, only a subset of all possible 
    factor combinations is run, which leads to confounding (aliasing) 
    between main effects and interactions. Foldover is a technique to 
    resolve some of these confoundings by running an additional set of 
    experiments where the sign of one or more basic factors is reversed. 
    This effectively doubles the number of runs and allows for the 
    separation of certain effects that were aliased in the original 
    fraction. Foldover is especially useful for identifying main effects 
    that may be confounded with two-factor interactions in the initial 
    design.
    Key points:
        - Foldover is meaningful only for regular 2-level fractional
          factorial designs, not for full factorials, non-regular
          designs, or designs with more than two levels per factor.
        - For full factorial designs, foldover is unnecessary because
          there is no aliasing to resolve.
        - For non-regular or mixed-level designs, the concept of
          foldover does not apply in the classical sense.
    """
    def __init__(
            self,
            *factors: Factor,
            generators: List[str],
            fold: bool | str = False,
            replicates: int = 1,
            central_points: int = 0,
            blocks: int = 1,
            shuffle: bool = True
            ) -> None:
        self.generators = generators
        self.fold = fold
        super().__init__(
            *factors,
            replicates=replicates,
            central_points=central_points,
            blocks=blocks,
            shuffle=shuffle)
        assert self.is_2k, (
            'All factors must have exactly 2 levels for '
            'FractionalFactorialDesignBuilder.')
        for i, sub in enumerate(self.factor_names):
            for j, name in enumerate(self.factor_names):
                if i == j:
                    continue
                assert sub not in name, (
                    f'Factor name "{sub}" should not be a substring of '
                    f'other factor names, found in "{name}". '
                    'This can lead to confusion in generator definitions. '
                    'Consider renaming factors to avoid such conflicts.')

    def _build_corrected_base_design(self) -> DataFrame:
        """Generate the integer-coded design matrix for fractional 
        factorial design.

        - Constructs a full factorial for the basic (independent) 
          factors.
        - Adds generated columns as defined by the generator strings 
          (e.g., 'C=AB' is interpreted as C = A*B).

        Returns
        -------
        DataFrame
            Design matrix with integer codes (-1, 1) for each factor. 
            Each row is an experimental run.
        """

        n_basic = len(self.factors) - len(self.generators)
        basic_factors = self.factors[:n_basic]
        basic_names = [f.name for f in basic_factors]
        df_design = pd.DataFrame(
            product(*[f.corrected_levels for f in basic_factors]),
            columns=basic_names)

        for generator in self.generators:
            lh_side, rh_side = generator.split('=')
            interaction_names = [n for n in basic_names if n in rh_side]
            assert len(lh_side) == 1, (
                f'Generator "{generator}" must have a single dependent factor '
                f'on the left side of "=", got {lh_side}')
            assert interaction_names, (
                f'Generator "{generator}" does not match any basic factors: '
                f'{basic_names}')
            generated = df_design[interaction_names].prod(axis=1)
            df_design[lh_side] = -generated if '-' in rh_side else generated

        return df_design