=====
Usage
=====

The basic use of Newton-Raphson Logistic Regression::

    import numpy as np
    from sklearn.datasets import make_classification
    from newton_raphson import logistic_regression

    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=5)

    # Perform logistic regression
    res = logistic_regression(X, y)

    # Print results
    res.summary()
