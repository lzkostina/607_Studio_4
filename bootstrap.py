import numpy as np
"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""


def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    Raises
    ------
    TypeError
        If X is not numeric
        If y is not numeric
        If compute_stat is not a callable
    ValueError
        If n_bootstrap is not a positive integer
        If X.shape[0] != len(y)

    """
    if not callable(compute_stat):
        raise TypeError("compute_stat must be callable")

    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")

    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X must be numeric!")

    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.number):
        raise TypeError("y must be numeric!")

    if X.shape[0] != len(y):
        raise ValueError("Number of rows in X must match length of y!")

    n = len(y)

    stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        stats[b] = compute_stat(X[idx], y[idx])
    return stats

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    Raises
    ------
    ValueError
        If bootstrap_stats is empty
        If alpha is out of [0,1] range
    """
    if len(bootstrap_stats) == 0:
        raise ValueError("bootstrap_stats must not be empty")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    bootstrap_stats = np.asarray(bootstrap_stats)

    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper

def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
    TypeError
        If X is not numeric
        If y is not numeric
    """
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X must be numeric!")

    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.number):
        raise TypeError("y must be numeric!")

    if X.shape[0] != len(y):
        raise ValueError("Number of rows in X must match length of y!")

    # OLS estimate beta = (X^T X)^(-1) X^T y
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta_hat

    # residual sum of squares and total sum of squares
    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)

    if tss == 0:   # y is constant
        return np.nan

    return 1 - rss / tss