import sys
import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytest import approx
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from matplotlib.patches import Patch

sys.path.append(Path(__file__).parent.resolve()) # type: ignore

from daspi import LINE
from daspi import COLOR
from daspi import CATEGORY
from daspi import Dodger
from daspi import HueLabel
from daspi import SizeLabel
from daspi import StripeLine
from daspi import StripeSpan
from daspi import ShapeLabel
from daspi.plotlib import Stripe


class TestCategoryLabel:
    colors = HueLabel(('alpha', 'beta', 'gamma', 'delta'), CATEGORY.PALETTE)
    markers = ShapeLabel(('foo', 'bar', 'bazz'), CATEGORY.MARKERS)
    sizes = SizeLabel(1.5, 3.5, CATEGORY.N_SIZE_BINS)

    def test_str(self) -> None:
        assert str(self.colors) == 'HueLabel'
        assert str(self.markers) == 'ShapeLabel'
        assert str(self.sizes) == 'SizeLabel'

    def test_errrors(self) -> None:
        with pytest.raises(AssertionError) as err:
            self.colors['psi']
        assert "Can't get category for label" in str(err.value)
        
        n = self.colors.n_allowed
        with pytest.raises(AssertionError) as err:
            HueLabel([i for i in range(n+1)], CATEGORY.PALETTE)
        assert str(err.value) == (
            f'HueLabel can handle {n} categories, got {n+1}')

        with pytest.raises(AssertionError) as err:
            ShapeLabel(('foo', 'foo', 'bar', 'bazz'), CATEGORY.MARKERS)
        assert str(err.value) == (
            'Labels occur more than once, only unique labels are allowed')

    def test_handles_labels(self) -> None:
        handles, labels = self.colors.handles_labels()
        assert len(self.colors.colors) == self.colors.n_used
        assert len(handles) == len(labels)
        
        handles, labels = self.markers.handles_labels()
        assert len(self.markers.markers) == self.markers.n_used
        assert len(handles) == len(labels)
        
        handles, labels = self.sizes.handles_labels()
        assert len(self.sizes.categories) == self.sizes.n_used
        assert len(handles) == len(labels)
        assert len(handles) == CATEGORY.N_SIZE_BINS

    def test_getitem(self) -> None:
        assert self.colors['alpha'] == CATEGORY.PALETTE[0]
        assert self.colors['beta'] == CATEGORY.PALETTE[1]
        assert self.colors['gamma'] == CATEGORY.PALETTE[2]
        assert self.colors['delta'] == CATEGORY.PALETTE[3]

        assert self.markers['foo'] == CATEGORY.MARKERS[0]
        assert self.markers['bar'] == CATEGORY.MARKERS[1]
        assert self.markers['bazz'] == CATEGORY.MARKERS[2]
    
    def test_sizes(self) -> None:
        s_min, s_max = CATEGORY.MARKERSIZE_LIMITS
        assert self.sizes[1.5] == s_min**2
        assert self.sizes[3.5] == s_max**2


        values = np.linspace(
            self.sizes._min, self.sizes._max, CATEGORY.N_SIZE_BINS)
        sizes = self.sizes(values)
        assert np.array_equal(sizes, np.square(self.sizes.categories))


class TestDodger:

    def test_getitem(self) -> None:
        tick_labels = ('a', 'b', 'c', 'd')
        dodge = Dodger(('r', ), tick_labels=tick_labels)
        assert dodge['r'] == 0

        dodge = Dodger(('r', 'g'), tick_labels=tick_labels)
        assert dodge['r'] == approx(-0.2)
        assert dodge['g'] == approx(0.2)

        dodge = Dodger(('r', 'g', 'b', 'y'), tick_labels=tick_labels)
        assert dodge['r'] == approx(-0.3)
        assert dodge['g'] == approx(-0.1)
        assert dodge['b'] == approx(0.1)
        assert dodge['y'] == approx(0.3)
        
        dodge = Dodger((), tick_labels=tick_labels)
        assert bool(dodge.dodge) == False
        assert bool(dodge['r']) == dodge._default
        
        dodge = Dodger((), tick_labels=())
        assert bool(dodge.dodge) == False
        assert bool(dodge['r']) == dodge._default

    def test_call(self) -> None:
        tick_labels = ('a', 'b', 'c', 'd')
        categories = ('r', 'g', 'b', 'y')
        dodge = Dodger(categories=categories, tick_labels=tick_labels)
        old = pd.Series(tick_labels)
        assert dodge(old, 'r').values == approx(dodge.ticks - 0.3)
        assert dodge(old, 'g').values == approx(dodge.ticks - 0.1)
        assert dodge(old, 'b').values == approx(dodge.ticks + 0.1)
        assert dodge(old, 'y').values == approx(dodge.ticks + 0.3)



class TestStripe:
    
    class ConcreteStripe(Stripe):
        @property
        def handle(self) -> Patch:
            return Patch()
            
        def __call__(self, ax, **kwds) -> None:
            pass

    @pytest.fixture
    def stripe(self) -> ConcreteStripe:
        return self.ConcreteStripe(
            label='test',
            orientation='horizontal', 
            position=1.0,
            width=0.5
        )

    def test_init_defaults(self, stripe: ConcreteStripe) -> None:
        assert stripe.color == COLOR.STRIPE
        assert stripe.lower_limit == 0.0
        assert stripe.upper_limit == 1.0
        assert stripe.zorder == 0.7
        assert stripe.show_position == False

    def test_decimals_property(self) -> None:
        stripe = self.ConcreteStripe('test', 0.1, 1.0, 'vertical')
        assert stripe.decimals == 4
        
        stripe = self.ConcreteStripe('test', 3.0, 1.0, 'vertical')
        assert stripe.decimals == 3
        
        stripe = self.ConcreteStripe('test', 25.0, 1.0, 'vertical')
        assert stripe.decimals == 2
        
        stripe = self.ConcreteStripe('test', 500.0, 1.0, 'vertical')
        assert stripe.decimals == 1
        
        stripe = self.ConcreteStripe('test', 10000.0, 1.0, 'vertical')
        assert stripe.decimals == 0

    def test_determine_decimals(self) -> None:
        assert Stripe.determine_decimals(0.1) == 4
        assert Stripe.determine_decimals(2.5) == 3
        assert Stripe.determine_decimals(25.0) == 2
        assert Stripe.determine_decimals(500.0) == 1
        assert Stripe.determine_decimals(10000.0) == 0

    def test_label_with_position(self) -> None:
        stripe = self.ConcreteStripe(
            'test', 1.234, 1.0, 'horizontal', show_position=True)
        assert stripe.label == '$test=1.234$'
        
        stripe = self.ConcreteStripe('test', 10000, 1.0, 'horizontal', show_position=True)
        assert stripe.label == '$test=10000$'

    def test_label_without_position(self, stripe: ConcreteStripe) -> None:
        assert stripe.label == '$test$'

    def test_identity(self, stripe: ConcreteStripe) -> None:
        expected = f'$test$_{COLOR.STRIPE}'
        assert stripe.identity == expected


class TestStripeLine:

    @pytest.fixture
    def stripe(self) -> StripeLine:
        return StripeLine(
            label='test',
            orientation='horizontal',
            position=1.0,
            width=2.0
        )
    
    @pytest.fixture
    def ax(self) -> Axes:
        return plt.subplots(1, 1)[1]

    def test_init_defaults(self, stripe: StripeLine) -> None:
        assert stripe.color == COLOR.STRIPE
        assert stripe.lower_limit == 0.0
        assert stripe.upper_limit == 1.0
        assert stripe.zorder == 0.7
        assert stripe.show_position == False
        assert stripe.linestyle == LINE.DASHED

    def test_init_custom_values(self) -> None:
        stripe = StripeLine(
            label='custom',
            orientation='vertical',
            position=2.0,
            color='red',
            lower_limit=0.5,
            upper_limit=1.5,
            zorder=1.0,
            show_position=True,
            linestyle='solid',
            width=3.0
        )
        assert stripe.color == 'red'
        assert stripe.lower_limit == 0.5
        assert stripe.upper_limit == 1.5
        assert stripe.zorder == 1.0
        assert stripe.show_position == True
        assert stripe.linestyle == 'solid'
        assert stripe.width == 3.0
    
    def test_init_priorities(self) -> None:
        stripe = StripeLine(
            label='custom',
            orientation='vertical',
            position=2.0,
            width=1,
            color='red',
            linestyle='solid',)
        assert stripe.width == 1
        assert stripe.color == 'red'
        assert stripe.linestyle == 'solid'
        
        stripe = StripeLine(
            label='custom',
            orientation='vertical',
            position=2.0,
            width=1,
            linewidth=2,
            color='red',
            linestyle='solid',)
        assert stripe.width == 2
        assert stripe.color == 'red'
        assert stripe.linestyle == 'solid'
        stripe = StripeLine(
            label='custom',
            orientation='vertical',
            position=2.0,
            width=1,
            linewidth=2,
            lw=3,
            color='red',
            c='blue',
            linestyle='solid',
            ls='dashed')
        assert stripe.width == 3
        assert stripe.color == 'blue'
        assert stripe.linestyle == 'dashed'

    def test_init_with_kwds(self) -> None:
        stripe = StripeLine(
            label='kwds',
            orientation='horizontal',
            position=1.0,
            ls='dotted',
            lw=4.0,
            c='blue'
        )
        assert stripe.linestyle == 'dotted'
        assert stripe.width == 4.0
        assert stripe.color == 'blue'
    
    def test_call_idempotent(self, stripe: StripeLine, ax: Axes) -> None:
        assert not ax in stripe._axes
        n_axes = len(stripe._axes)
        stripe(ax)
        n_lines = len(ax.lines)
        assert ax in stripe._axes
        assert len(stripe._axes) == n_axes + 1
        stripe(ax)
        assert len(ax.lines) == n_lines
        assert len(stripe._axes) == n_axes + 1

    def test_call_horizontal(self, stripe: StripeLine, ax: Axes) -> None:
        stripe(ax)
        assert ax in stripe._axes
        assert len(ax.lines) == 1
        line = ax.lines[0]
        assert line.get_ydata()[0] == stripe.position # type: ignore
        assert line.get_color() == stripe.color
        assert line.get_linewidth() == stripe.width
        assert line.get_zorder() == stripe.zorder

    def test_call_vertical(self, ax: Axes) -> None:
        stripe = StripeLine(
            label='vertical',
            orientation='vertical',
            position=2.0,
            color='red')
        assert not ax in stripe._axes
        n_lines = len(ax.lines)
        stripe(ax)
        assert len(ax.lines) == n_lines + 1
        assert ax in stripe._axes
        line = ax.lines[-1]
        assert line.get_xdata()[0] == stripe.position # type: ignore
        assert line.get_color() == stripe.color
        assert line.get_linewidth() == stripe.width
        assert line.get_zorder() == stripe.zorder

    def test_handle_properties(self, stripe: StripeLine) -> None:
        handle = stripe.handle
        assert isinstance(handle, Line2D)
        assert handle.get_color() == COLOR.STRIPE
        assert handle.get_linewidth() == stripe.width
        assert handle.get_linestyle() == '--'
        assert handle.get_markersize() == 0
        assert len(handle.get_xdata()) == 0 # type: ignore
        assert len(handle.get_ydata()) == 0 # type: ignore

        stripe = StripeLine(
            label='test',
            orientation='horizontal', 
            position=1.0,
            width=3.0,
            color='red',
            linestyle=':')
        handle = stripe.handle
        assert handle.get_color() == 'red'
        assert handle.get_linewidth() == 3.0
        assert handle.get_linestyle() == ':'



class TestStripeSpan:

    @pytest.fixture
    def ax(self) -> Axes:
        return plt.subplots(1, 1)[1]

    def test_init_with_position_width(self) -> None:
        stripe = StripeSpan(
            label='test',
            orientation='horizontal',
            position=1.0,
            width=2.0)
        assert stripe.lower_position == 0.0
        assert stripe.upper_position == 2.0
        assert stripe.position == 1.0
        assert stripe.width == 2.0
        assert stripe.color == COLOR.STRIPE
        assert stripe.alpha == COLOR.CI_ALPHA
        assert stripe.lower_limit == 0.0
        assert stripe.upper_limit == 1.0
        assert stripe.zorder == 0.7
        assert stripe.border_linewidth == 0

    def test_init_with_lower_upper_position(self) -> None:
        stripe = StripeSpan(
            label='test',
            orientation='vertical',
            lower_position=0.0,
            upper_position=2.0)
        assert stripe.lower_position == 0.0
        assert stripe.upper_position == 2.0
        assert stripe.position == 1.0
        assert stripe.width == 2.0

    def test_init_invalid_parameters(self) -> None:
        with pytest.raises(AssertionError):
            StripeSpan(
                label='test',
                orientation='horizontal',
                position=1.0,
                width=2.0,
                lower_position=0.0,
                upper_position=2.0)

        with pytest.raises(AssertionError):
            StripeSpan(
                label='test',
                orientation='horizontal')
            
        with pytest.raises(AssertionError):
            StripeSpan(
                label='test',
                orientation='horizontal',
                position=1.0,
                lower_position=0.0,
                upper_position=2.0)

        with pytest.raises(AssertionError):
            StripeSpan(
                label='test',
                orientation='horizontal',
                position=1.0,
                width=2.0,
                upper_position=2.0)

    def test_handle_property(self) -> None:
        stripe = StripeSpan(
            label='test',
            orientation='horizontal',
            position=1.0,
            width=2.0)
        handle = stripe.handle
        assert isinstance(handle, Patch)
        assert handle.get_alpha() == COLOR.CI_ALPHA
        assert to_hex(handle.get_facecolor()) == COLOR.STRIPE.lower()
        
        stripe = StripeSpan(
            label='test',
            orientation='horizontal',
            position=1.0,
            width=2.0,
            color=COLOR.PALETTE[0],
            alpha=0.123)
        handle = stripe.handle
        assert isinstance(handle, Patch)
        assert handle.get_alpha() == 0.123
        assert to_hex(handle.get_facecolor()) == COLOR.PALETTE[0].lower()

    def test_call_horizontal(self, ax: Axes) -> None:
        stripe = StripeSpan(
            label='test',
            orientation='horizontal',
            position=1.0,
            width=2.0,
            border_linewidth=1.0)
        stripe(ax)
        assert ax in stripe._axes
        assert len(ax.patches) == 1
        patch = ax.patches[0]
        assert patch.get_alpha() == COLOR.CI_ALPHA
        assert patch.get_linewidth() == 1.0
        assert patch.get_zorder() == 0.7

    def test_call_vertical(self, ax: Axes) -> None:
        stripe = StripeSpan(
            label='test',
            orientation='vertical',
            position=1.0,
            width=2.0)
        stripe(ax)
        assert ax in stripe._axes
        assert len(ax.patches) == 1

    def test_call_idempotent(self, ax: Axes) -> None:
        stripe = StripeSpan(
            label='test',
            orientation='horizontal',
            position=1.0,
            width=2.0)
        stripe(ax)
        n_patches = len(ax.patches)
        stripe(ax)
        assert len(ax.patches) == n_patches

    def test_custom_styling(self, ax: Axes) -> None:
        stripe = StripeSpan(
            label='test',
            orientation='horizontal',
            position=1.0,
            width=2.0,
            color='red',
            alpha=0.3,
            zorder=2.0,
            border_linewidth=2.0)
        stripe(ax)
        patch = ax.patches[0]
        assert patch.get_facecolor()[:3] == (1.0, 0.0, 0.0)
        assert patch.get_alpha() == 0.3
        assert patch.get_zorder() == 2.0
        assert patch.get_linewidth() == 2.0
