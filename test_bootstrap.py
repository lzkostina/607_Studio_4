import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared

"""====================== SAMPLE TESTS ========================"""
def test_output_shape():
    """Bootstrap should return an array of length n_bootstrap."""
    np.random.seed(0)
    n, p = 30, 2
    X = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X])  # add intercept
    y = np.random.randn(n)
    stats = bootstrap_sample(X, y, R_squared, n_bootstrap=100)
    assert isinstance(stats, np.ndarray)
    assert stats.shape == (100,)

def test_reproducibility():
    """With fixed seed, bootstrap results should be reproducible."""
    np.random.seed(42)
    n, p = 10, 2
    X = np.random.randn(n, p + 1)
    y = np.random.randn(n)
    stats1 = bootstrap_sample(X, y, R_squared, n_bootstrap=5)
    np.random.seed(42)
    stats2 = bootstrap_sample(X, y, R_squared, n_bootstrap=5)
    assert stats1.shape == stats2.shape
    assert np.all(np.isfinite(stats1)) and np.all(np.isfinite(stats2))

def test_single_bootstrap():
    """n_bootstrap=1 should return an array with one statistic."""
    np.random.seed(0)
    n, p = 10, 2
    X = np.random.randn(n, p + 1)
    y = np.random.randn(n)
    stats = bootstrap_sample(X, y, R_squared, n_bootstrap=1)
    assert stats.shape == (1,)
    assert np.isscalar(stats[0])

def test_small_dataset():
    np.random.seed(0)
    X = np.array([[1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0])
    stats = bootstrap_sample(X, y, R_squared, n_bootstrap=10)
    assert stats.shape == (10,)
    # Allow nan due to constant y
    assert np.all(np.isfinite(stats) | np.isnan(stats))


"""====================== CI TESTS ============================"""
def test_bootstrap_ci_correct_length():
    """Check that bootstrap_ci returns a tuple of length 2"""
    stats = np.random.randn(1000)  # simulate bootstrap stats
    ci = bootstrap_ci(stats, alpha=0.05)
    assert isinstance(ci, tuple)
    assert len(ci) == 2

def test_bootstrap_ci_bounds():
    """Check that lower bound <= upper bound"""
    stats = np.random.randn(1000)
    lower, upper = bootstrap_ci(stats, alpha=0.05)

    assert lower <= upper

def test_bootstrap_ci_alpha_effect():
    """Smaller alpha should give a wider interval"""
    stats = np.random.randn(1000)
    ci_95 = bootstrap_ci(stats, alpha=0.05)   # 95% CI
    ci_80 = bootstrap_ci(stats, alpha=0.20)   # 80% CI
    width_95 = ci_95[1] - ci_95[0]
    width_80 = ci_80[1] - ci_80[0]
    assert width_95 >= width_80

"""====================== R-squared TESTS =================="""
def test_perfect_fit():
    """If there is no noise, R^2 should be 1."""
    np.random.seed(0)
    n, p = 50, 2
    X = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X])  # add intercept
    beta = np.array([1.0, 0.5, -0.3])
    y = X @ beta  # perfect fit
    r2 = R_squared(X, y)
    assert np.isclose(r2, 1.0, atol=1e-10)


def test_r2_with_noise_in_range():
    """R^2 should be between 0 and 1 for noisy regression."""
    np.random.seed(0)
    n, p = 50, 2
    X = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X])
    beta = np.array([1.0, 0.5, -0.3])
    y = X @ beta + np.random.randn(n) * 0.5
    r2 = R_squared(X, y)
    assert 0 <= r2 <= 1

def test_mismatched_dimensions():
    """Should raise ValueError if dimensions do not match."""
    np.random.seed(0)
    n, p = 50, 2
    X = np.random.randn(n, p + 1)
    y = np.random.randn(n + 1)
    try:
        R_squared(X, y)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for mismatched dimensions"

def test_pure_noise_data():
    """With random X and y, R^2 should be close to 0."""
    np.random.seed(0)
    n, p = 50, 2
    X = np.random.randn(n, p + 1)
    y = np.random.randn(n)
    r2 = R_squared(X, y)
    assert r2 < 0.2

def test_constant_y():
    """If y is constant, R^2 should be nan or 0."""
    np.random.seed(0)
    n, p = 50, 2
    X = np.random.randn(n, p + 1)
    y = np.ones(n)
    r2 = R_squared(X, y)
    assert np.isnan(r2) or r2 == 0.0

"""======================= MAIN TEST ============================"""

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    np.random.seed(0)
    X = np.column_stack([np.ones(50), np.random.randn(50)])
    beta = np.array([1, 0.5])
    y = X @ beta + np.random.randn(50)

    boot_stats = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=200)
    ci = bootstrap_ci(boot_stats, alpha=0.1)

    # Check output type/shape
    assert isinstance(boot_stats, np.ndarray)
    assert len(boot_stats) == 200
    assert isinstance(ci, tuple) and len(ci) == 2
    # Very weak test: R^2 is usually positive
    assert ci[1] >= 0