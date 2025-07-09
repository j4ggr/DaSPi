import sys
import pytest

from pathlib import Path
from pandas.core.frame import DataFrame

sys.path.append(str(Path(__file__).parent.resolve()))

from daspi.anova.convert import *

valid_data_dir = Path(__file__).parent/'data'


class TestGetTermName:

    def test_no_interaction(self) -> None:
        assert get_term_name('x1[T.b]') == 'x1'
        assert get_term_name('x1[T.b-mittel]') == 'x1'
        assert get_term_name('x10[T.2.2]') == 'x10'

    def test_with_interaction(self) -> None:
        assert get_term_name('x1[T.b]:x2[T.2]') == 'x1:x2'
        assert get_term_name('x1[T.b-mittel]:x10[T.2.2]') == 'x1:x10'

    def test_multiple_interactions(self) -> None:
        assert get_term_name('x1[T.b]:x2[T.2]:x3[T.True]') == 'x1:x2:x3'
        assert get_term_name('x1[T.b-mittel]:x10[T.2.2]:x3[T.True]') == 'x1:x10:x3'

    def test_no_encoding(self) -> None:
        term_name = get_term_name('Category')
        assert term_name == 'Category'

    def test_empty_string(self) -> None:
        term_name = get_term_name('')
        assert term_name == ''

    def test_invalid_encoding(self) -> None:
        term_name = get_term_name('InvalidEncoding')
        assert term_name == 'InvalidEncoding'


class TestFramesToHTML:
    dfs = [
        DataFrame({'A': [1, 2], 'B': [3, 4]}),
        DataFrame({'C': [5, 6], 'D': [7, 8]})]
    captions = ['Table 1', 'Table 2']

    def test_basics(self) -> None:
        html = frames_to_html(self.dfs, self.captions)
        assert isinstance(html, str)
        assert 'Table 1' in html
        assert 'Table 2' in html
        assert '>A</th>' in html
        assert '>B</th>' in html
        assert '>C</th>' in html
        assert '>D</th>' in html

    def test_empty_dfs(self) -> None:
        html = frames_to_html([], [])
        assert html == ''

    def test_mismatched_lengths(self) -> None:
        dfs = [DataFrame({'A': [1, 2]})]
        assert bool(frames_to_html(dfs, self.captions))
        
        with pytest.raises(AssertionError, match='There must be at most as many captions as DataFrames.'):
            html = frames_to_html(self.dfs, [self.captions[0]])

        with pytest.raises(AssertionError, match='There must be at most as many captions as DataFrames.'):
            html = frames_to_html(dfs, [])

