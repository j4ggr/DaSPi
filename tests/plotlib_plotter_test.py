import sys
import pytest

import matplotlib
matplotlib.use('Agg')

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
from daspi import DEFAULT
from daspi import CATEGORY
from daspi import PLOTTER
from daspi.plotlib.plotter import *


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
        assert to_hex(handle.get_facecolor()) == to_hex(COLOR.PALETTE[0]).lower()

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


class TestGaussianKDEContourUnivariate:
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for testing."""
        np.random.seed(42)
        data = {
            'category': ['A'] * 30 + ['B'] * 30,
            'value': np.concatenate([
                np.random.normal(10, 2, 30),
                np.random.normal(15, 2, 30)
            ])
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def ax(self) -> Axes:
        return plt.subplots(1, 1)[1]

    def test_init_defaults(self, sample_data: pd.DataFrame) -> None:
        """Test initialization with default parameters."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category'
        )
        assert plotter.fill == True
        assert plotter.n_points == DEFAULT.KD_SEQUENCE_LEN
        assert plotter.width == CATEGORY.FEATURE_SPACE
        assert plotter.shape == (DEFAULT.KD_SEQUENCE_LEN, DEFAULT.KD_SEQUENCE_LEN)
        assert plotter.target == 'value'
        assert plotter.feature == 'category'
        assert plotter.target_on_y == True

    def test_init_custom_parameters(self, sample_data: pd.DataFrame) -> None:
        """Test initialization with custom parameters."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category',
            width=0.5,
            fill=False,
            fade_outers=False,
            n_points=50,
            target_on_y=False,
            color='red'
        )
        assert plotter.fill == False
        assert plotter.n_points == 50
        assert plotter.width == 0.5
        assert plotter.shape == (50, 50)
        assert plotter.target_on_y == False
        assert plotter.color == 'red'

    def test_colormap_with_fill(self, sample_data: pd.DataFrame) -> None:
        """Test colormap creation when fill=True."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category',
            fill=True,
            color='blue'
        )
        assert plotter.cmap is not None
        # The colormap should have transparent at one end and the color at the other
        assert plotter.cmap(0.0)[3] == 0.0  # transparent
        # The other end should be the specified color (blue)

    def test_colormap_with_fade_outers(self, sample_data: pd.DataFrame) -> None:
        """Test colormap creation when fill=False and fade_outers=True."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category',
            fill=False,
            fade_outers=True,
            color='green'
        )
        assert plotter.cmap is not None
        # When fade_outers=True, colormap should fade from transparent to color
        assert plotter.cmap(0.0)[3] == 0.0  # transparent at start

    def test_colormap_without_fade(self, sample_data: pd.DataFrame) -> None:
        """Test colormap creation when fill=False and fade_outers=False."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category',
            fill=False,
            fade_outers=False,
            color='red'
        )
        assert plotter.cmap is not None

    def test_kw_default_property(self, sample_data: pd.DataFrame) -> None:
        """Test that kw_default returns correct keyword arguments."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category'
        )
        kwds = plotter.kw_default
        assert 'cmap' in kwds
        assert kwds['cmap'] == plotter.cmap

    def test_transform_method(self, sample_data: pd.DataFrame) -> None:
        """Test the transform method creates correct data structure."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category',
            n_points=20
        )
        
        # Get data for one category
        category_data = sample_data[sample_data['category'] == 'A']
        feature_data = 0.5  # arbitrary feature position
        
        result = plotter.transform(feature_data, category_data['value'])
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert 'category' in result.columns
        assert 'value' in result.columns
        assert PLOTTER.TRANSFORMED_FEATURE in result.columns
        assert PLOTTER.F_BASE_NAME in result.columns
        
        # Check result dimensions - should be n_points^2 for 2D KDE
        assert len(result) == 400  # 20 * 20
        
        # Check F_BASE_NAME is consistent
        assert (result[PLOTTER.F_BASE_NAME] == feature_data).all()

    def test_call_with_fill(self, sample_data: pd.DataFrame, ax: Axes) -> None:
        """Test plotting with fill=True creates contourf."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category',
            fill=True,
            n_points=20,
            ax=ax
        )
        
        initial_collections = len(ax.collections)
        plotter()
        
        # contourf adds collections to the axes
        assert len(ax.collections) > initial_collections

    def test_call_without_fill(self, sample_data: pd.DataFrame, ax: Axes) -> None:
        """Test plotting with fill=False creates contour lines."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category',
            fill=False,
            n_points=20,
            ax=ax
        )
        
        initial_collections = len(ax.collections)
        plotter()
        
        # contour adds collections to the axes
        assert len(ax.collections) > initial_collections

    def test_target_on_y_true(self, sample_data: pd.DataFrame, ax: Axes) -> None:
        """Test that target_on_y=True correctly sets up axes."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category',
            target_on_y=True,
            n_points=20,
            ax=ax
        )
        plotter()
        
        # When target_on_y=True, feature is on x-axis
        assert plotter.x.name == 'category'
        assert plotter.y.name == 'value'

    def test_target_on_y_false(self, sample_data: pd.DataFrame, ax: Axes) -> None:
        """Test that target_on_y=False correctly sets up axes."""
        plotter = GaussianKDEContourUnivariate(
            source=sample_data,
            target='value',
            feature='category',
            target_on_y=False,
            n_points=20,
            ax=ax
        )
        plotter()
        
        # When target_on_y=False, target is on x-axis
        assert plotter.x.name == 'value'
        assert plotter.y.name == 'category'

    def test_multiple_features(self, ax: Axes) -> None:
        """Test with multiple feature categories."""
        np.random.seed(42)
        data = pd.DataFrame({
            'category': ['A'] * 20 + ['B'] * 20 + ['C'] * 20,
            'value': np.concatenate([
                np.random.normal(10, 2, 20),
                np.random.normal(15, 2, 20),
                np.random.normal(12, 2, 20)
            ])
        })
        
        plotter = GaussianKDEContourUnivariate(
            source=data,
            target='value',
            feature='category',
            n_points=20,
            ax=ax
        )
        
        # Should process all three categories
        plotter()
        assert len(ax.collections) > 0

    def test_skip_na_handling(self, ax: Axes) -> None:
        """Test that skip_na parameter is properly handled."""
        np.random.seed(42)
        data = pd.DataFrame({
            'category': ['A'] * 20 + ['B'] * 20,
            'value': list(np.random.normal(10, 2, 15)) + [np.nan] * 5 + 
                     list(np.random.normal(15, 2, 20))
        })
        
        # Test with skip_na='any'
        plotter = GaussianKDEContourUnivariate(
            source=data,
            target='value',
            feature='category',
            skip_na='any',
            n_points=20,
            ax=ax
        )
        
        # Should successfully plot despite NaN values
        plotter()
        assert len(ax.collections) > 0


class TestDensityCurve:

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        np.random.seed(0)
        return pd.DataFrame({'value': np.random.normal(5, 1, 100)})

    @pytest.fixture
    def ax(self) -> Axes:
        return plt.subplots(1, 1)[1]

    def test_estimate_not_implemented(
            self, sample_data: pd.DataFrame) -> None:
        with pytest.raises(NotImplementedError):
            DensityCurve(source=sample_data, target='value')

    def test_constant_curve_default_height(
            self, sample_data: pd.DataFrame) -> None:
        curve = GaussianKDE(
            source=sample_data, target='value', target_on_y=False)
        sequence, estimation = curve._constant_curve(2.0, 7.0)
        assert sequence == [7.0]
        assert estimation == [curve.stretch + 2.0]

    def test_constant_curve_with_height(
            self, sample_data: pd.DataFrame) -> None:
        curve = GaussianKDE(
            source=sample_data, target='value', height=3.0,
            target_on_y=False)
        sequence, estimation = curve._constant_curve(1.0, 4.0)
        assert sequence == [4.0]
        assert estimation == [1.0 + 3.0]

    def test_transform_constant_data(self) -> None:
        df = pd.DataFrame({'value': [5.0] * 20})
        curve = GaussianKDE(
            source=df, target='value', target_on_y=False)
        data = curve.transform(0.0, df['value'])
        assert len(data) == 1
        assert data[curve.target].iloc[0] == 5.0

    def test_estimate_cumulative_numeric_integration(
            self, sample_data: pd.DataFrame) -> None:
        curve = GaussianKDE(
            source=sample_data, target='value', target_on_y=False)
        feature_data = 0.0
        _, cumulated = curve.estimate_cumulative(
            feature_data, sample_data['value'])
        assert cumulated[0] == approx(feature_data, abs=1e-6)
        assert cumulated[-1] == approx(
            feature_data + curve.stretch, abs=1e-6)
        assert np.all(np.diff(cumulated) >= -1e-9)

    def test_estimate_cumulative_with_height(
            self, sample_data: pd.DataFrame) -> None:
        curve = GaussianKDE(
            source=sample_data, target='value', height=4.0,
            target_on_y=False)
        feature_data = 1.0
        _, cumulated = curve.estimate_cumulative(
            feature_data, sample_data['value'])
        assert cumulated[0] == approx(feature_data, abs=1e-6)
        assert cumulated[-1] == approx(feature_data + 4.0, abs=1e-6)

    def test_transform_cumulative_dispatch(
            self, sample_data: pd.DataFrame) -> None:
        curve = GaussianKDE(
            source=sample_data, target='value', target_on_y=False,
            cumulative=True)
        data = curve.transform(0.0, sample_data['value'])
        assert data[curve.feature].min() == approx(0.0, abs=1e-6)
        assert data[curve.feature].max() == approx(curve.stretch, abs=1e-6)

    def test_highlight_quantiles_requires_fill(
            self, sample_data: pd.DataFrame) -> None:
        curve = GaussianKDE(
            source=sample_data, target='value', fill=True,
            agreements=(2, 4, 6))
        assert curve.highlight_quantiles == True
        curve_no_fill = GaussianKDE(
            source=sample_data, target='value', fill=False,
            agreements=(2, 4, 6))
        assert curve_no_fill.highlight_quantiles == False


class TestGaussianKDECumulative:

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        np.random.seed(0)
        return pd.DataFrame({'value': np.random.normal(5, 1, 100)})

    @pytest.fixture
    def ax(self) -> Axes:
        return plt.subplots(1, 1)[1]

    def test_cumulative_flag_default_false(
            self, sample_data: pd.DataFrame) -> None:
        kde = GaussianKDE(source=sample_data, target='value')
        assert kde.cumulative == False

    def test_cumulative_flag_set(self, sample_data: pd.DataFrame) -> None:
        kde = GaussianKDE(
            source=sample_data, target='value', cumulative=True)
        assert kde.cumulative == True

    def test_call_cumulative_produces_monotonic_curve(
            self, sample_data: pd.DataFrame, ax: Axes) -> None:
        kde = GaussianKDE(
            source=sample_data, target='value', target_on_y=False,
            cumulative=True, ax=ax)
        kde()
        values = kde.source[kde.feature].to_numpy()
        assert values.min() == approx(0.0, abs=1e-6)
        assert values.max() == approx(kde.stretch, abs=1e-6)
        assert len(ax.lines) > 0

    def test_call_non_cumulative_produces_bell_shape(
            self, sample_data: pd.DataFrame, ax: Axes) -> None:
        kde = GaussianKDE(
            source=sample_data, target='value', target_on_y=False,
            cumulative=False, ax=ax)
        kde()
        values = kde.source[kde.feature].to_numpy()
        assert values.min() >= 0.0
        assert len(ax.lines) > 0


class TestFittedDensity:

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        np.random.seed(0)
        return pd.DataFrame({'value': np.random.normal(5, 1, 100)})

    @pytest.fixture
    def ax(self) -> Axes:
        return plt.subplots(1, 1)[1]

    def test_init_defaults(self, sample_data: pd.DataFrame) -> None:
        fd = FittedDensity(source=sample_data, target='value')
        assert fd.cumulative == False
        assert fd.strategy == 'data'
        assert fd.margin == 0
        assert fd.fill == True

    def test_init_custom_parameters(self, sample_data: pd.DataFrame) -> None:
        fd = FittedDensity(
            source=sample_data, target='value', strategy='norm',
            cumulative=True, fill=False, height=0.25)
        assert fd.strategy == 'norm'
        assert fd.cumulative == True
        assert fd.fill == False
        assert fd.height == 0.25

    def test_estimate_returns_pdf(self, sample_data: pd.DataFrame) -> None:
        fd = FittedDensity(
            source=sample_data, target='value', strategy='norm',
            target_on_y=False)
        sequence, estimation = fd.estimate(0.0, sample_data['value'])
        assert len(sequence) == fd.n_points
        assert len(estimation) == fd.n_points
        assert np.all(estimation >= 0)

    def test_estimate_cumulative_returns_cdf(
            self, sample_data: pd.DataFrame) -> None:
        fd = FittedDensity(
            source=sample_data, target='value', strategy='norm',
            target_on_y=False)
        _, estimation = fd.estimate_cumulative(
            0.0, sample_data['value'])
        assert np.all(np.diff(estimation) >= -1e-9)
        assert estimation.min() >= -1e-6
        assert estimation.max() <= fd.stretch + 1e-6

    def test_call_pdf(self, sample_data: pd.DataFrame, ax: Axes) -> None:
        fd = FittedDensity(
            source=sample_data, target='value', target_on_y=False,
            strategy='norm', ax=ax)
        fd()
        assert len(ax.lines) > 0

    def test_call_cumulative(
            self, sample_data: pd.DataFrame, ax: Axes) -> None:
        fd = FittedDensity(
            source=sample_data, target='value', target_on_y=False,
            strategy='norm', cumulative=True, ax=ax)
        fd()
        assert len(ax.lines) > 0
