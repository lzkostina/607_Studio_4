import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared
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