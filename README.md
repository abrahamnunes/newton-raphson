
# Newton-Raphson Logistic Regression

![https://zenodo.org/badge/latestdoi/127929528](https://zenodo.org/badge/127929528.svg) ![https://travis-ci.org/abrahamnunes/newton_raphson](https://img.shields.io/travis/abrahamnunes/newton_raphson.svg) ![https://newton-raphson.readthedocs.io/en/latest/?badge=latest](https://readthedocs.org/projects/newton-raphson/badge/?version=latest)


A simple and fast Python 3+ implementation of logistic regression for association analyses using the Newton-Raphson method.

- Free software: MIT license
- Documentation: https://newton-raphson.readthedocs.io.

## Installation

```  
pip install git+https://github.com/abrahamnunes/newton-raphson
```

## Basic use case

``` python
import numpy as np
from sklearn.datasets import make_classification
from newton_raphson import logistic_regression

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=5)

# Perform logistic regression
res = logistic_regression(X, y)

# Print results
res.summary()
```


## To-Do


- [ ] Add unit tests for Hessian scaling factor
- [ ] Add unit test for Hessian conditioning
- [ ] Add capability to monitor each optimization iteration step

## How to Cite

If you use Newton-Raphson in your work, we would very much appreciate the citation, which can be done as follows:

- Abraham Nunes (2018). Newton-Raphson Logistic Regression. Zenodo. http://doi.org/10.5281/zenodo.1211725

