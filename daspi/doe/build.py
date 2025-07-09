import warnings

import numpy as np
import pandas as pd

from typing import List
from typing import Dict
from typing import Tuple

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
    'FullFactorial2nDesignBuilder',]


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
    code_level_map: Dict[int, LevelType]
    """Mapping from integer codes to their corresponding factor levels.
    This is used for integer-coded designs where levels are represented
    by integers. Set this from outside the class, e.g. in a design 
    builder."""
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
    blocks : int, optional
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

    factors : Tuple[Factor, ...]
    """Factors defining the design space."""
    replicates : int
    """Number of replicates."""
    central_points : int
    """Number of central points."""
    blocks : int
    """Number of blocks."""
    shuffle : bool
    """Whether to shuffle the design."""
    n_factors : int
    """Number of factors."""
    level_counts : Tuple[int, ...]
    """Number of levels for each factor."""

    def __init__(
            self, 
            *factors: Factor,
            replicates: int = 1,
            central_points: int = 0,
            blocks: int = 1,
            shuffle: bool = True,
            ) -> None:
        assert factors, 'At least one factor is required.'
        assert all(isinstance(f, Factor) for f in factors), (
            'All factors must be instances of Factor class.')
        assert not any(f.name in self.standard_columns for f in factors), (
            'Factor names must not conflict with standard columns in the '
            f'design matrix. Standard columns are: {self.standard_columns}')
        assert replicates > 0, 'Number of replicates must be positive.'
        assert central_points >= 0, 'Number of central points must be non-negative.'
        assert not central_points or any(f.has_central for f in factors), (
            'At least one factor must provide a central point, '
            'or set central_points to 0.')
        assert blocks > 0, 'Number of blocks must be positive.'

        self.factors = factors
        self.replicates = replicates
        self.central_points = central_points
        self.blocks = blocks
        self.shuffle = shuffle
        self.n_factors = len(self.factors)
        self.level_counts = tuple(f.n_levels for f in self.factors)
        
        assert self.n_factors == len(set(self.factor_names)), (
            'Factor names must be unique.')

    @property
    def factor_names(self) -> Tuple[str, ...]:
        """Tuple of factor names."""
        return tuple(f.name for f in self.factors)
    
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
        return self.standard_columns + list(self.factor_names)

    @staticmethod
    def _set_code_level_map(
            factors: Tuple[Factor, ...],
            codes: List[List[int]],
            ) -> None:
        """Set code_level_map for each factor based on the codes.

        Parameters
        ----------
        factors : Tuple[Factor, ...]
            Factors to set code_level_map for.
        codes : List[List[int]]
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
        
        replicates = []
        for i in range(self.replicates):
            df_replica = df_design.copy()
            df_replica[DOE.REPLICA] = i + 1
            replicates.append(df_replica)
        df_design = pd.concat(replicates, ignore_index=True)
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
        for i in range(self.central_points):
            _df_central = df_central.copy()
            _df_central[DOE.CENTRAL_POINT] = 0
            _df_central[DOE.REPLICA] = i + 1
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

    def build_design(self, corrected: bool = True) -> DataFrame:
        """Generate the design matrix with original factor values.

        This method builds the design matrix by generating the
        corrected (integer-coded) design, replicating it according to
        the replicates parameter, shuffling it if specified, and 
        adding central points if specified.
        
        Parameters
        ----------
        corrected : bool, optional
            Whether to return the design matrix with integer codes
            (default True). If False, returns the design matrix with
            actual factor values.
        
        Returns
        -------
        DataFrame
            Design matrix with original factor values if corrected is
            False, otherwise with integer codes representing factor
            levels.
        """
        df_design = pd.DataFrame()
        for block in range(1, self.blocks + 1):
            _dm = self._build_coded_design()
            _dm[DOE.REPLICA] = 1
            _dm = self._replicate_design(_dm)
            _dm[DOE.CENTRAL_POINT] = 1
            _dm = self._add_central_points(_dm)
            _dm[DOE.STD_ORDER] = np.arange(len(_dm)) * block
            _dm = self._shuffle_design(_dm)
            _dm[DOE.BLOCK] = block
            df_design = pd.concat([df_design, _dm], ignore_index=True)
        
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

    np.random.seed(42)

    factor_a = dsp.Factor('A', (1, 2))
    factor_b = dsp.Factor('B', (10, 20))
    builder = dsp.FullFactorialDesignBuilder(factor_a, factor_b, replicates=3)
    df = builder.build_design(corrected=True)
    print(df)
    ```

    ```console
        std_order  run_order  central_point  replica  block  A  B
    0          10          0              1        3      1  2  1
    1           9          1              1        3      1  1  2
    2           0          2              1        1      1  1  1
    3           8          3              1        3      1  1  1
    4           5          4              1        2      1  1  2
    5           2          5              1        1      1  2  1
    6           1          6              1        1      1  1  2
    7          11          7              1        3      1  2  2
    8           4          8              1        2      1  1  1
    9           7          9              1        2      1  2  2
    10          3         10              1        1      1  2  2
    11          6         11              1        2      1  2  1
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
        codes = [list(range(1, n + 1)) for n in self.level_counts]
        self._set_code_level_map(self.factors, codes)

        df_design = pd.DataFrame(
            product(*codes), columns=self.factor_names)
        
        return df_design


class FullFactorial2nDesignBuilder(BaseDesignBuilder):
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
            'FullFactorial2nDesignBuilder.')
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