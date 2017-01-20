
################################
# Defines useful functions for Bayesian statistics using CH_to_EKM
################################
################################
# Use at top: cython: profile=True

from functools import partial, lru_cache
from itertools import product
from math import pi
import numpy as np
import vegas
cimport numpy as np
# from sympy.functions.special.delta_functions import Heaviside
cimport vegas


################################
# Helpful equations
################################

def n_c_val(k, zero_list):
    """Return number of nonzero coefficients up to and including order k.

    zero_list contains all the orders at which the coefficient is zero."""
    n_c = k + 1
    for zero_order in zero_list:
        if zero_order <= k:
            n_c -= 1
    return n_c


def cbark(*c_tuple):
    """Return \bar c_{(k)} as defined in CH_to_EKM."""
    # abs_list = tuple(map(np.absolute, *c_tuple))
    # print(*c_tuple)
    abs_matrix = np.absolute(c_tuple)
    # print(abs_matrix)
    return np.max(abs_matrix, axis=0)


def find_insignificant_x(f, double small_val=0.01, double epsilon=0.05):
    # Assume f is largest at x=0, is even, and is monotonically decreasing
    # as |x| gets large
    max_f = f(0)
    trial_x = 1
    ratio = f(trial_x) / max_f
    found_upper_bound = False
    found_lower_bound = False
    while (not found_lower_bound) or (not found_upper_bound):
        ratio = f(trial_x) / max_f
        if ratio > small_val:
            lower_bound = trial_x
            trial_x *= 2
            found_lower_bound = True
        else:
            upper_bound = trial_x
            trial_x *= 0.5
            found_upper_bound = True

    trial_x = (upper_bound + lower_bound)/2
    while abs(small_val - f(trial_x) / max_f)/small_val > epsilon:
        ratio = f(trial_x) / max_f
        if ratio > small_val:
            lower_bound = trial_x
        else:
            upper_bound = trial_x
        trial_x = (upper_bound + lower_bound)/2

    return trial_x


def find_dimensionless_dob_limit(func, x_mode, dob,
                                 delta_x=1e-4, epsilon=1e-10):
    d = x_mode
    integral = 0
    fx_old = func(d)
    while True:
        fx_next, fx_prev = func(d+delta_x), fx_old
        trapezoid = (fx_next + fx_prev)/2 * delta_x

        # Check if we've overshot the integral
        if integral+trapezoid >= dob/2:
            try:
                rel_diff = np.absolute(delta_x/d)
            except ZeroDivisionError:
                rel_diff = epsilon + 1

            # print(delta_x, d, rel_diff, integral)
            # Either accept answer or increase precision + try again
            if rel_diff < epsilon:
                return d
            delta_x /= 10
            continue  # From the top!

        # Increment if we did not overshoot
        d += delta_x
        fx_old = fx_next
        integral += trapezoid


def Heaviside(np.ndarray x):
    return 0.5 * (np.sign(x) + 1)


################################
# Prior / Likelihood Functions
################################


def uniform_cn_likelihood(cn, np.ndarray cbar):
    return Heaviside(cbar - np.absolute(cn)) / (2 * cbar)


def gaussian_cn_likelihood(cn, np.ndarray cbar):
    return np.exp(-cn**2/(2 * cbar**2)) / (np.sqrt(2 * pi) * cbar)


def uniform_log_cbar_prior(
        np.ndarray cbar, double cbar_lower, double cbar_upper):
    val = 1/(cbar * np.log(cbar_upper/cbar_lower))
    return val * Heaviside(cbar - cbar_lower) * Heaviside(cbar_upper - cbar)


def gaussian_log_cbar_prior(np.ndarray cbar, double sigma):
    return np.exp(-(np.log(cbar))**2/(2 * sigma**2)) / \
        (np.sqrt(2*pi) * cbar * sigma)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Old stuff specifically for prior set A
def prior_cn_A(cn, cbar):
    return Heaviside(cbar - abs(cn)) / (2 * cbar)


def prior_cbar_A(cbar, cbar_lower, cbar_upper):
    val = 1/(cbar * np.log(cbar_upper/cbar_lower))
    return val * Heaviside(cbar - cbar_lower) * Heaviside(cbar_upper - cbar)


def Delta_k_posterior_A1(Q, k, n_c, cbar_lower, cbar_upper, *c_tuple):
    c_bar_k = cbark(*c_tuple[:k+1])
    val = 1/(2 * Q**(k+1)) * n_c/(n_c+1) / \
        (c_bar_k**(-n_c) - cbar_upper**(-n_c))

    def A1_post(Delta_k):
        c_kplus1 = Delta_k/Q**(k+1)
        cutoff = Heaviside(cbar_upper - cbark(*c_tuple[:k+1], c_kplus1))
        if abs(Delta_k) <= c_bar_k * Q**(k+1):
            return (c_bar_k**(-n_c-1) - cbar_upper**(-n_c-1)) * val * cutoff
        else:
            return ((Q**(k+1)/np.absolute(Delta_k))**(n_c+1) -
                    cbar_upper**(-n_c-1)) * val * cutoff

    return A1_post


def Delta_k_posterior_A1eps(Q, k, n_c, *c_tuple):
    c_bar_k = cbark(*c_tuple[:k+1])
    val = 1/(2 * c_bar_k * Q**(k+1)) * n_c/(n_c+1)

    def A1eps(Delta_k):
        if abs(Delta_k) <= c_bar_k * Q**(k+1):
            return val
        else:
            return val * (c_bar_k * Q**(k+1) / np.absolute(Delta_k))**(n_c+1)

    return A1eps


def dkp_A_eps(Q, k, n_c, p, c_bar_k):
    factor = c_bar_k * Q**(k+1)
    if p <= n_c/(n_c + 1):
        factor *= (n_c + 1)/n_c * p
    else:
        factor *= (1/((n_c + 1)*(1 - p)))**(1/n_c)
    return factor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


################################
# General posteriors
################################


def likelihood_hash(prior_set):
    """Return pr(c_n|cbar)"""
    if prior_set == "A" or prior_set == "B":
        return uniform_cn_likelihood

    if prior_set == "C":
        return gaussian_cn_likelihood


def cbar_prior_hash(prior_set, cbar_lower=None, cbar_upper=None, sigma=None):
    """Return pr(cbar)"""
    if prior_set == "A" or prior_set == "C":
        return partial(uniform_log_cbar_prior,
                       cbar_lower=cbar_lower,
                       cbar_upper=cbar_upper)

    if prior_set == "B":
        return partial(gaussian_log_cbar_prior,
                       sigma=sigma)


cdef class Delta_k_integrand(vegas.BatchIntegrand):
    """Return the numerator or denominator integrand for pr(Delta_k|{c_i})."""

    cdef int is_denominator
    cdef double Delta_k, Q, cbar_lower, cbar_upper, sigma
    cdef str prior_set
    cdef int k, n_c, h
    cdef np.ndarray coeffs

    def __init__(self, is_denominator, Delta_k, prior_set, Q, k, nc, h, coeffs,
                 cbar_lower=0.0, cbar_upper=0.0, sigma=0.0):
        self.is_denominator = is_denominator
        self.Delta_k = Delta_k
        self.prior_set = prior_set
        self.Q = Q
        self.k = k
        self.n_c = nc
        self.h = h
        self.coeffs = coeffs
        self.cbar_lower = cbar_lower
        self.cbar_upper = cbar_upper
        self.sigma = sigma

    def __call__(self, c_vars):
        """c_vars = [cbar, c_{k+2}/cbar, c_{k+3}/cbar, ..., c_h/cbar]"""

        cdef np.ndarray cbar = c_vars[:, 0]
        cdef np.ndarray u_list, u
        func = self.basic_integrand(np.array([cbar]).T)

        if self.is_denominator:
            return func

        # The remainder of expansion beyond c_{k+1} Q^{k+1}
        cdef int parallel_len = cbar.shape[0]
        cdef np.ndarray pr_ckplus1_sum = np.zeros(parallel_len)
        cdef np.ndarray remaind = np.zeros(parallel_len)
        cdef int m

        likelihood = likelihood_hash(self.prior_set)

        if len(c_vars[0]) > 1:  # If more variables than cbar is given
            u_list = c_vars.T[1:]
            for m, u in enumerate(u_list):
                # Transformed variables so must multiply u by cbar
                # Must multiply integrand by cbar because of variable transform
                func *= cbar * likelihood(u * cbar, cbar)

            # To optimize c_m integrals, force them to be from 0 to infinity
            for signs in product([-1, 1], repeat=len(u_list)):
                remaind = np.zeros(parallel_len)
                for m, u in enumerate(u_list):
                    # After dividing by Q^(k+1)
                    remaind += (u * signs[m] * cbar) * self.Q**(m + 1)
                pr_ckplus1_sum += likelihood(
                    self.Delta_k/self.Q**(self.k+1) - remaind, cbar)
        else:
            pr_ckplus1_sum = likelihood(self.Delta_k/self.Q**(self.k+1), cbar)

        # Utilize symmetry of integrand so that integrals are from 0 to infty
        # i.e. \int_{-\infty}^{\infty} = \int_{0}^{\infty} + \int_{-\infty}^0
        # Then make changes of variables to combine into \int_0^\infty
        # pr(c_{k+1} = ... | cbar) is the only factor affected by switch.
        # So sum the permutation of all terms and multiply by rest of integrand
        func *= pr_ckplus1_sum

        return func

    def basic_integrand(self, np.ndarray cbar):
        cdef double ci
        cdef np.ndarray func = cbar_prior_hash(
            self.prior_set, cbar_lower=self.cbar_lower,
            cbar_upper=self.cbar_upper, sigma=self.sigma)(cbar[:, 0])
        for ci in self.coeffs[:self.n_c]:
            # print(ci.shape[0], func.shape[0], cbar[:, 0].shape[0], likelihood_hash(self.prior_set)(ci, cbar[:, 0]).shape[0])
            func *= likelihood_hash(self.prior_set)(ci, cbar[:, 0])

        return func


cdef class Delta_k_posterior:
    """Return the posterior pr(Delta_k|{c_i}) as a function of Delta_k."""

    cdef double Q, cbar_lower, cbar_upper, sigma
    cdef double cbar_lower_limit, cbar_upper_limit, cm_upper_limit
    cdef str prior_set
    cdef int k, n_c, h
    cdef np.ndarray coeffs
    cdef list denom_limit_list, numer_limit_list

    def __init__(self, prior_set, Q, k, nc, h, coeffs,
                 cbar_lower=0.0, cbar_upper=0.0, sigma=0.0):
        self.prior_set = prior_set
        self.Q = Q
        self.k = k
        self.n_c = nc
        self.h = h
        self.coeffs = np.array(coeffs)
        self.cbar_lower = cbar_lower
        self.cbar_upper = cbar_upper
        self.sigma = sigma

        # Determine optimal limits of integration for prior sets
        if prior_set == "A":
            self.cbar_lower_limit = max(np.max(coeffs), cbar_lower)
            self.cbar_upper_limit = cbar_upper
            self.cm_upper_limit = 1
        if prior_set == "B":
            self.cbar_lower_limit = np.max(coeffs)
            self.cbar_upper_limit = np.exp(5 * sigma)
            self.cm_upper_limit = 1
        if prior_set == "C":
            self.cbar_lower_limit = cbar_lower
            sum_sq = 0
            for cn in coeffs[:nc]:
                sum_sq += cn**2
            self.cbar_upper_limit = min(cbar_upper, 4 * sum_sq)
            self.cm_upper_limit = 4

        self.denom_limit_list = [[self.cbar_lower_limit,
                                  self.cbar_upper_limit]]
        self.numer_limit_list = self.denom_limit_list + \
            (h-1) * [[0, self.cm_upper_limit]]

    def __call__(self, Delta_k):
        return self.main_func(Delta_k)

    @lru_cache(maxsize=256)
    def main_func(self, Delta_k):
        denom_f = Delta_k_integrand(
            is_denominator=True, Delta_k=Delta_k, prior_set=self.prior_set,
            Q=self.Q, k=self.k, nc=self.n_c, h=self.h, coeffs=self.coeffs,
            cbar_lower=self.cbar_lower, cbar_upper=self.cbar_upper,
            sigma=self.sigma)

        numer_f = Delta_k_integrand(
            is_denominator=False, Delta_k=Delta_k, prior_set=self.prior_set,
            Q=self.Q, k=self.k, nc=self.n_c, h=self.h, coeffs=self.coeffs,
            cbar_lower=self.cbar_lower, cbar_upper=self.cbar_upper,
            sigma=self.sigma)

        denom_integ = vegas.Integrator(self.denom_limit_list)
        denom_integ(denom_f, nitn=10, neval=1e4)  # Ready the grid
        denom_result = denom_integ(denom_f, nitn=10, neval=1e4)


        numer_integ = vegas.Integrator(self.numer_limit_list)
        numer_integ(numer_f, nitn=10, neval=1e4)  # Ready the grid
        numer_result = numer_integ(numer_f, nitn=10, neval=1e4)


        return (numer_result/(denom_result * self.Q**(self.k+1))).mean
