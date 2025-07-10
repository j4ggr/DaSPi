import warnings

import numpy as np
import pandas as pd

from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from typing import Iterable

from abc import ABC
from abc import abstractmethod
from itertools import product
from pandas.core.frame import DataFrame

from .._typing import LevelType

from ..constants import DOE


__all__ = [
    'Factor',
    'BaseDesignBuilder',
    'FullFactorialDesignBuilder',
    'FullFactorial2kDesignBuilder',]


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
    
    name: str
    """Name of the factor."""
    levels: Tuple[LevelType, ...]
    """Levels of the factor (numeric or categorical)."""
    central_point: LevelType | None = None
    """Optional central point level."""
    is_categorical: bool = False
    """Whether the factor is categorical."""
    code_level_map: Dict[float | int, LevelType]
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

        self.name = name

        has_string = any(isinstance(level, str) for level in levels)
        if has_string and not is_categorical:
            warnings.warn(
                f'Found string level(s) in factor {name}. '
                'Assuming factor is categorical.')
            is_categorical = True
        self.is_categorical = is_categorical
        
        self.levels = tuple(levels) if is_categorical else tuple(sorted(levels))
        
        if self.is_categorical:
            self.central_point = None
        else:
            self.central_point = (self.levels[0] + self.levels[-1]) / 2 # type: ignore
    
    @property
    def n_levels(self) -> int:
        """Return number of levels (read-only)."""
        return self._n_levels
    
    @property
    def has_central(self) -> bool:
        """Return whether there is a central point (read-only)."""
        return self.central_point is not None


class BaseDesignBuilder(ABC):
    """Abstract base class for DOE builders.
    
    Parameters
    ----------
    factors : Iterable[Factor]
        Factors defining the design space.
    replicates : int, optional
        Number of replicates. Must be positive, by default 1.
    central_points : int, optional
        Number of central points. Must be non-negative, by default 0.
    blocks : int | str | List[str] | Literal['highest', 'replica'], optional
        Number of blocks. Must be positive, by default 1.
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
    shuffle : bool
    """Whether to shuffle the design."""

    def __init__(
            self, 
            *factors: Factor,
            replicates: int = 1,
            central_points: int = 0,
            blocks: int | str | List[str] | Literal['highest', 'replica'] = 1,
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
            assert any(f.has_central for f in self.factors), (
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
        if blocks in {'highest', 'replica'}:
            self._blocks = blocks # type: ignore
        
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
    def _set_code_level_map(
            factors: Tuple[Factor, ...],
            codes: List[List[float | int]],
            ) -> None:
        """Set code_level_map for each factor based on the codes.

        Parameters
        ----------
        factors : Tuple[Factor, ...]
            Factors to set code_level_map for.
        codes : List[List[float | int]]
            List of codes for each factor.
        
        Raises
        ------
        AssertionError
            If the length of codes does not match the number of factors.
        """
        assert len(codes) == len(factors), (
            'Codes list must match number of factors.')
        for factor, code in zip(factors, codes):
            factor.code_level_map = dict(zip(code, factor.levels))

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
            -If any factor does not have a code_level_map attribute.
            -If the design matrix columns do not match factor names.
        """
        assert all(hasattr(f, 'code_level_map') for f in factors), (
            'All factors must have a code_level_map attribute for decoding.')
        assert all(f.name in df_design.columns for f in factors), (
            'Design matrix columns must match factor names.')

        df_design = df_design.copy()
        for factor in factors:
            codes = df_design[factor.name].astype(int)
            df_design.loc[:, factor.name] = [factor.code_level_map[c] for c in codes]
        return df_design

    @abstractmethod
    def _build_coded_design(self) -> DataFrame:
        """
        Abstract method to generate the integer-coded design matrix.

        This method must be implemented by subclasses to generate the
        design matrix with integer indices representing factor levels.
        It should return a DataFrame where each column corresponds to a
        factor, and each row represents an experimental run. The values
        in the DataFrame are indices into the corresponding factor's
        levels tuple.

        Implement also setting the code_level_map attribute for each 
        factor.

        Returns
        -------
        DataFrame
            Design matrix with integer indices representing factor 
            levels. Each column corresponds to a factor, each row to an 
            experimental run. Values are indices into the corresponding 
            factor's levels tuple.
        """
        pass

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
        if not self.replicates > 1:
            return df_design
        
        df_design = pd.concat([df_design] * self.replicates, ignore_index=True)
        return df_design

    def _add_central_points(self, df_design: DataFrame) -> DataFrame:
        """Add central points to the design matrix if specified.
        If central points are specified, they are added to the design
        matrix as additional rows with the central point level for each
        factor.
        
        Parameters
        ----------
        df_design : DataFrame
            Design matrix to add central points to.
        
        Returns
        -------
        DataFrame
            Design matrix with central points added.
        """
        if not self.central_points > 0:
            return df_design

        codes = []
        for factor in self.factors:
            if factor.has_central and factor.central_point is not None:
                codes.append([DOE.CENTRAL_CODED_VALUE])
                factor.code_level_map |= {
                    DOE.CENTRAL_CODED_VALUE: factor.central_point}
            else:
                codes.append(list(factor.code_level_map.keys()))
        
        df_central = pd.DataFrame(product(*codes), columns=self.factor_names)
        for _ in range(self.central_points):
            _df_central = df_central.copy()
            _df_central[DOE.CENTRAL_POINT] = 0
            df_design = pd.concat([df_design, _df_central], ignore_index=True)
        return df_design

    def _shuffle_design(self, df_design: DataFrame) -> DataFrame:
        """
        Shuffle the design matrix if shuffle is True.
        
        Parameters
        ----------
        df_design : DataFrame
            Design matrix to shuffle.

        Returns
        -------
        DataFrame
            Shuffled design matrix.
        """
        if not self.shuffle:
            return df_design
        
        df_design = df_design.sample(frac=1, ignore_index=True)
        return df_design
    
    def _count_replicates(self, df_design: DataFrame) -> DataFrame:
        """Count the number of replicates for each factor combination.

        This method adds a new column to the design matrix that counts
        the number of replicates for each combination of factor levels.

        Parameters
        ----------
        df_design : DataFrame
            Design matrix to count replicates for.
        
        Returns
        -------
        DataFrame
            Design matrix with an additional column for the replicate 
            count.
        """
        columns = [DOE.CENTRAL_POINT] + self.factor_names
        df_design[DOE.REPLICA] = df_design.groupby(columns).cumcount() + 1
        return df_design
    
    def _add_block_column(self, df_design: DataFrame) -> DataFrame:
        """Add a block column to the design matrix based on the blocks 
        option.

        - If blocks is an int > 1: assign blocks evenly 
          (not statistically confounded).
        - If blocks is an Iterable[str]: assign blocks by confounding 
          with the specified interaction (statistically correct).
        - If blocks is 'highest': assign blocks by confounding with the 
          highest-order interaction (all factors).
        - If blocks is 'replica': assign blocks based on the replicate 
          number.
        - If blocks == 1: assign all runs to block 1.

        Returns
        -------
        DataFrame
            Design matrix with an additional column for the block 
            assignments.
        """
        blocks = self.blocks
        if blocks == 1:
            df_design[DOE.BLOCK] = 1
        
        elif isinstance(blocks, int):
            n = len(df_design)
            block_sizes = [
                n // blocks + (1 if x < n % blocks else 0)
                for x in range(blocks)]
            block_indices = np.concatenate(
                [[i] * size for i, size in enumerate(block_sizes, start=1)])
            df_design[DOE.BLOCK] = block_indices
        
        elif blocks == 'replica':
            df_design[DOE.BLOCK] = df_design[DOE.REPLICA]
        
        elif blocks == 'highest' or isinstance(blocks, (str, list)):
            factors = self.factor_names if blocks == 'highest' else list(blocks)
            block_values = np.prod([df_design[f] for f in factors], axis=0)
            unique_values = np.unique(block_values)
            block_map = {v: i for i, v in enumerate(unique_values, start=1)}
            df_design[DOE.BLOCK] = [block_map[v] for v in block_values]

        else:
            raise ValueError(f'Invalid blocks option: {self.blocks}')

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
          with the highest-order interaction (all factors).
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
        df_design = self._build_coded_design()
        df_design = self._replicate_design(df_design)
        df_design[DOE.CENTRAL_POINT] = 1
        df_design = self._add_central_points(df_design)
        df_design[DOE.STD_ORDER] = np.arange(len(df_design))
        df_design = self._shuffle_design(df_design)
        df_design = self._count_replicates(df_design)
        df_design = self._add_block_column(df_design)
        df_design[DOE.RUN_ORDER] = np.arange(len(df_design))
        df_design = df_design[self.columns]
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
    blocks : int, optional
        Number of blocks. Must be positive, by default 1.
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
    two levels, and 3 replicates:
    
    ```python
    import daspi as dsp
    import numpy as np

    np.random.seed(42)  # optional for reproducibility

    factor_a = dsp.Factor('A', (1, 2))
    factor_b = dsp.Factor('B', (10, 20))
    builder = dsp.FullFactorialDesignBuilder(factor_a, factor_b, replicates=3)
    df = builder.build_design(corrected=True)
    print(df)
    ```

    ```console
        std_order  run_order  central_point  replica  block  A  B
    0          10          0              1        1      1  2  1
    1           9          1              1        1      1  1  2
    2           0          2              1        1      1  1  1
    3           8          3              1        2      1  1  1
    4           5          4              1        2      1  1  2
    5           2          5              1        2      1  2  1
    6           1          6              1        3      1  1  2
    7          11          7              1        1      1  2  2
    8           4          8              1        3      1  1  1
    9           7          9              1        2      1  2  2
    10          3         10              1        3      1  2  2
    11          6         11              1        3      1  2  1
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
            blocks: int = 1,
            shuffle: bool = True
            ) -> None:
        super().__init__(
            *factors,
            replicates=replicates,
            central_points=0,
            blocks=blocks,
            shuffle=shuffle)

    def _build_coded_design(self) -> DataFrame:
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
        codes = [list(np.linspace(-1, 1, n)) for n in self.level_counts]
        self._set_code_level_map(self.factors, codes)

        df_design = pd.DataFrame(
            product(*codes), columns=self.factor_names)
        
        return df_design


class FullFactorial2kDesignBuilder(BaseDesignBuilder):
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
    central_points : int, optional
        Number of central points. Must be non-negative, by default 0.
    blocks : int, optional
        Number of blocks. Must be positive, by default 1.
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
    factors, each with two levels, and 3 replicates and 3 central points
    to test linear effects:
    
    ```python
    import daspi as dsp
    import numpy as np

    np.random.seed(42) # optional for reproducibility

    factor_a = dsp.Factor('A', (0, 1))
    factor_b = dsp.Factor('B', (0, 1))
    builder = dsp.FullFactorial2kDesignBuilder(
        factor_a, factor_b, replicates=3, central_points=3)
    df = builder.build_design(corrected=True)
    print(df)
    ```

    ```console
        std_order  run_order  central_point  replica  block  A  B
    0           9          0              1        1      1 -1  1
    1          11          1              1        1      1  1  1
    2           0          2              1        1      1 -1 -1
    3          13          3              0        1      1  0  0
    4           5          4              1        2      1 -1  1
    5           8          5              1        2      1 -1 -1
    6           2          6              1        1      1  1 -1
    7           1          7              1        3      1 -1  1
    8          14          8              0        2      1  0  0
    9           4          9              1        3      1 -1 -1
    10          7         10              1        2      1  1  1
    11         10         11              1        2      1  1 -1
    12         12         12              0        3      1  0  0
    13          3         13              1        3      1  1  1
    14          6         14              1        3      1  1 -1
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
            central_points: int = 0,
            blocks: int = 1,
            shuffle: bool = True
            ) -> None:
        assert all(f.n_levels == 2 for f in factors), (
            'All factors must have exactly 2 levels for '
            'FullFactorial2kDesignBuilder.')
        super().__init__(
            *factors,
            replicates=replicates,
            central_points=central_points,
            blocks=blocks,
            shuffle=shuffle)

    def _build_coded_design(self) -> DataFrame:
        """Generate the integer-coded design matrix for full factorial 
        design with 2-level factors.

        Returns
        -------
        DataFrame
            Design matrix with integer indices representing factor 
            levels. Each column corresponds to a factor, each row to an 
            experimental run. Values are indices into the corresponding 
            factor's levels tuple.
        """
        codes = [[-1, 1] for _ in self.factors]
        self._set_code_level_map(self.factors, codes)

        df_design = pd.DataFrame(
            product(*codes), columns=self.factor_names)

        return df_design


class FractionalFactorialDesignBuilder(BaseDesignBuilder):
    """
    Builder for 2-level fractional factorial designs with optional 
    foldover.

    This class supports the construction of regular 2-level fractional 
    factorial designs using generator strings (e.g., "C=AB"), and can 
    optionally add a foldover to the design.

    Parameters
    ----------
    factors : Iterable[Factor]
        Factors defining the design space. All factors must have exactly 
        2 levels.
    generators : List[str]
        List of generator strings that define how dependent factors are constructed from basic factors.
        For example, if you have 4 factors A, B, C, and D, you can set generators to ["C=AB", "D=BC"].
        This means that factor C is defined as the product of factors A and B (C = A * B), and factor D as the product of B and C (D = B * C).
        The order of the generator strings should match the order of the dependent factors in the factors list.
    fold : bool, optional
        Whether to add a foldover (default False).
    replicates : int, optional
        Number of replicates. Must be positive, by default 1.
    central_points : int, optional
        Number of central points. Must be non-negative, by default 0.
    blocks : int, optional
        Number of blocks. Must be positive, by default 1.
    shuffle : bool, optional
        Whether to shuffle the design, by default True.

    Examples
    --------
    Create a 2^(3-1) fractional factorial design (3 factors, 1 generator):

    ```python
    from daspi.doe.build import Factor, FractionalFactorialDesignBuilder

    fA = Factor('A', (-1, 1))
    fB = Factor('B', (-1, 1))
    fC = Factor('C', (-1, 1))
    builder = FractionalFactorialDesignBuilder(fA, fB, fC, generators=["C=AB"])
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
    builder = FractionalFactorialDesignBuilder(fA, fB, fC, generators=["C=AB"], fold=True)
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
        - Each generator string must be in the format "C=AB" where C is 
          the dependent factor and AB are the independent factors.
        - Each dependent factor must be defined in the generators.
        - The first basic factor must be reversed in sign for foldover.

    Notes
    -----
    Foldover theory (summary):
        In fractional factorial designs, only a subset of all possible 
        factor combinations is run, which leads to confounding 
        (aliasing) between main effects and interactions. Foldover is a
        technique to resolve some of these confoundings by running an 
        additional set of experiments where the sign of one or more 
        basic factors is reversed. This effectively doubles the number
        of runs and allows for the separation of certain effects that 
        were aliased in the original fraction. Foldover is especially 
        useful for identifying main effects that may be confounded
        with two-factor interactions in the initial design.
    """
    def __init__(
            self,
            *factors: Factor,
            generators: List[str],
            fold: bool = False,
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

    def _build_coded_design(self) -> DataFrame:
        """Generate the integer-coded design matrix for fractional 
        factorial design.

        - Constructs a full factorial for the basic (independent) 
          factors.
        - Adds generated columns as defined by the generator strings 
          (e.g., "C=AB" means C = A*B).
        - If foldover is enabled, appends a second set of runs with the 
          first basic factor reversed in sign.

        Returns
        -------
        DataFrame
            Design matrix with integer codes (-1, 1) for each factor. 
            Each row is an experimental run.
        """
        # Basic factors (independent):
        n_basic = len(self.factors) - len(self.generators)
        basic_names = [f.name for f in self.factors[:n_basic]]
        gen_names = [f.name for f in self.factors[n_basic:]]
        codes = [[-1, 1] for _ in range(n_basic)]
        self._set_code_level_map(self.factors[:n_basic], codes)

        # Build base design (full factorial for basic factors)
        base = pd.DataFrame(product(*codes), columns=basic_names)

        # Add generated columns
        for gen, gen_str in zip(gen_names, self.generators):
            # e.g., "C=AB" means C = A*B
            rhs = gen_str.split('=')[1]
            val = np.ones(len(base), dtype=int)
            for c in rhs:
                val *= base[c]
            base[gen] = val
        # Set code_level_map for generated factors
        for i, gen in enumerate(gen_names):
            self.factors[n_basic + i].code_level_map = {-1: self.factors[n_basic + i].levels[0], 1: self.factors[n_basic + i].levels[1]}

        design = base.copy()

        # Foldover: reverse sign of first basic factor and append
        if self.fold:
            folded = base.copy()
            first = basic_names[0]
            folded[first] = -folded[first]
            design = pd.concat([design, folded], ignore_index=True)

        return design