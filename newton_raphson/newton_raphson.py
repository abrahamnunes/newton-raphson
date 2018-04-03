# -*- coding: utf-8 -*-
#
#   NEWTON_RAPHSON
#       An implementation of the Newton-Raphson algorithm for logistic
#       regression, where it is used for tests of association, rather than
#       classification. For an implementation suitable better for
#       classification, we suggest using scikit-learn instead.
#
#   (c)2018 Abraham Nunes MD MBA, Dalhousie University, Halifax, Nova Scotia
# ==============================================================================
import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import pinv as inverse
from numpy.linalg import norm

rng = np.random.RandomState(32435)

class LogisticRegressionResult(object):
    """
    Object storing the results of logistic regression. Also computes statistics.

    Parameters
    ----------
    coef : ndarray(k)
        Model coefficients for k parameters
    nll  : float
        Negative log-likelihood
    feval : int
        Number of function evaluations done during optimization
    grad : ndarray(k)
        Gradient vector at termination
    hess : ndarray((k, k))
        Hessian at termination
    cov : ndarray((k, k))
        Covariance matrix (inverse of Hessian) at termination
    se : ndarray(k)
        Standard errors of the estimates at termination
    n : int
        Number of samples used to fit the model
    k : int
        Number of parameters in the model
    flg : str
        Flag indicating the stopping criteria met at termination:
            '100': Maximum iterations
            '010': Iterate change tolerance `norm(xnew-xold, ord=2)`
            '001': Gradient norm (L2)
    tolx : float
        Minimally acceptable change in iterate values
    toliter : int
        Maximum number of allowed iterations
    tolgrad : float
        Minimally acceptable gradient magnitude at termination
    deviance : float
        2*nll
    aic : float
        Akaike information criterion
    bic : float
        Bayesian information criterion
    p_model : float
        P-value of the model
    p_coefs : float
        P-value of the coefficients

    Methods
    -------
    compute_statistics(self)
        Computes the model and coefficient statistics
    compute_hessian_conditioning(self)
        Computes condition number of hessian matrix
    make_summary_tables(self)
        Generates tables for summarization of the analysis statistics
    summary(self)
        Prints statistical summary tables

    """
    def __init__(self, coef=None, nll=None, feval=None, grad=None, hess=None, cov=None, se=None, n=None, k=None, flg=None, tolx=None, toliter=None, tolgrad=None):
        self.coef     = coef
        self.nll      = nll
        self.feval    = feval
        self.grad     = grad
        self.hess     = hess
        self.cov      = cov
        self.se       = se
        self.n        = n
        self.k        = k
        self.flg      = flg
        self.tolx     = tolx
        self.toliter  = toliter
        self.tolgrad  = tolgrad
        self.compute_statistics()
        self.compute_hessian_conditioning()
        self.make_summary_tables()

    def compute_statistics(self):
        self.deviance  = 2*self.nll
        self.aic       = 2*self.k + self.deviance
        self.bic       = self.k*np.log(self.n) + self.deviance
        self.p_model   = 1-stats.chi2.cdf(self.deviance, 1)
        self.tstat     = self.coef/self.se
        self.p_coefs   = 1-stats.chi2.cdf(self.tstat**2, 1)

    def compute_hessian_conditioning(self):
        s = np.linalg.svd(self.hess, compute_uv=False)
        self.condition = np.abs(np.max(s)/np.min(s))

    def make_summary_tables(self, significance_threshold=0.05):
        self.coefs_summary = pd.DataFrame({
            'Parameter' : ['Param %s' %i for i in range(self.k)],
            'Estimate'  : self.coef,
            'Odds-Ratio': np.exp(self.coef),
            'SE'        : self.se,
            'p-value'   : self.p_coefs
        })
        self.model_summary = pd.DataFrame({
            'Deviance'  : [self.deviance],
            'p-Value'   : [self.p_model],
            'AIC'       : [self.aic],
            'BIC'       : [self.bic]
        })

    def summary(self):
        print('================= Coefficient Fit Statistics =================')
        print(self.coefs_summary[['Parameter',
                                  'Estimate',
                                  'Odds-Ratio',
                                  'SE',
                                  'p-value']])
        print('=================    Model Fit Statistics    =================')
        print(self.model_summary[['Deviance', 'p-value', 'AIC', 'BIC']])

# ==============================================================================
#   Functions to be used in logistic_regression object. Written here to make
#       LR look more like pseudocode
# ==============================================================================
def add_intercept_column_to_feature_matrix(X):
    """ Adds bias term to leftmost column """
    return np.hstack((np.ones((X.shape[0], 1)), X))

def initialize_parameters(n):
    return np.zeros(n)

def f_logistic(X, params):
    """ The logistic function """
    return 1/(1+np.exp(-np.einsum('ij,j->i', X, params)))

def negative_log_likelihood(y, yhat):
    """ Loss function for logistic regression """
    return -np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat))

def compute_hessian(X, XT, y, yhat):
    """
    Computes the Hessian matrix at the current iteration

    Parameters
    ----------
    X : ndarray((nsamples, nfeatures))
        The independent variables. Features must be column vectors.
    XT : ndarray((nfeatures, nsamples))
         Precomputed transpose of X
    y : ndarray(nsamples)
        The response variables
    yhat : ndarray(nsamples)
        Estimates of the response variable under the current parameterization

    Returns
    -------
    H : ndarray((nparams, nparams))
        Hessian matrix
    s : ndarray(nparams)
        Scaling factors for the Hessian matrix

    Notes
    -----
    This computation is `X.T@np.diag(s)@X`.
    """
    s    = np.einsum('i,ij->j', yhat-y, X)
    H    = (XT*np.tile(yhat*(1-yhat), [X.shape[1], 1]))@X
    return H, s

def convergence_test(k,dx,J,glen,toliter,tolx,tolgrad,verbose):
    """ Checks for satisfaction of convergence criteria """
    # Print current iteration statistics if indicated
    if verbose > 0:
        print('Iter %s | J=%s | L2(dw)=%s | L2(g) = %s' %((k,J,dx,glen)))

    hit_toliter = np.greater(k, toliter)
    hit_tolx    = np.less(dx, tolx)
    hit_tolg    = np.less(glen, tolgrad)
    if hit_toliter or hit_tolx or hit_tolg:
        msg = ''.join(str(int(j)) for j in [hit_toliter,hit_tolx,hit_tolg])
        if verbose == 1:
            print('Newton-Raphson Terminated with message %s' %msg)
        done = True
    else:
        done = False
        msg  = 'Not terminated'
    return done, msg

# ==============================================================================
#   Main Newton-Raphson logistic regression function
# ==============================================================================

def logistic_regression(X, y, intercept=True, toliter=10000, tolx=1e-12,
                        tolgrad=1e-12, verbose=0):
    """
    Runs logistic regression model

    Parameters
    ----------
    X : ndarray((nsamples, nfeatures))
        The independent variables. Features must be column vectors.
    y : ndarray(nsamples)
        The response variables
    intercept : bool
        Whether to add an intercept (bias) term
    toliter : int (default=10000)
        Maximum number of iterations allowed
    tolx : float (default=1e-12)
        Minimum l2-norm of the change in parameter estimates across iterations
    tolgrad : float (default=1e-12)
        Minimum l2-norm of the gradient at each iteration.
    verbose : {0, 1, 2, 3}
        Verbosity level.
            0 = No output
            1 = Termination message only
            2 = Termination message and output for each iteration
            3 = Messages for 1 & 2, and printout of summary statistics

    Returns
    -------
    LogisticRegressionResult

    Notes
    -----
    While the gradient is not used for the optimization itself, we include it in order to potentially use it as a stopping criterion.
    """

    X  = add_intercept_column_to_feature_matrix(X) if intercept is True else X
    XT = X.T # Precompute transpose(X) for efficiency
    x  = initialize_parameters(n=X.shape[1])
    k  = 0; done = False
    while not done:
        oldx = x
        yhat = f_logistic(X, x)
        J    = negative_log_likelihood(y, yhat)
        H, s = compute_hessian(X, XT, y, yhat)
        C    = inverse(H)
        x    = x - C@s
        k    += 1
        dx   = norm(x-oldx)**2  # Change in iterate
        g    = XT@(y-yhat)      # Gradient at current point
        glen = norm(g)**2       # Length of gradient vector at current point
        done, msg = convergence_test(k,dx,J,glen,toliter,tolx,tolgrad,verbose)

    # Save results
    res = LogisticRegressionResult(
            coef=x,
            nll =J,
            feval=k,
            grad=g,
            hess=H,
            cov=C,
            se=np.sqrt(np.diag(C)),
            n=X.shape[0],
            k=X.shape[1],
            flg=msg,
            tolx=tolx,
            toliter=toliter,
            tolgrad=tolgrad)

    if verbose == 3:
        print('\n\n')
        res.summary()

    return res
