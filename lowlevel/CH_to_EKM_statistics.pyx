# cython: overflowcheck=True
################################
# Defines useful functions for Bayesian statistics using CH_to_EKM
################################
################################
# Use at top: cython: profile=True

from functools import partial, lru_cache
from itertools import product
from math import pi
from gmpy2 import mpz, mpfr  # For precision
from scipy.integrate import quad
from scipy.special import gammainc as spgammainc
from scipy.special import gamma as spgamma
import mpmath
from mpmath import gammainc
import cython
import numpy as np
import vegas
cimport numpy as np
# from sympy.functions.special.delta_functions import Heaviside
cimport vegas
import warnings

mpmath.mp.dps = 15

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
    cdef double max_f = f(0)
    cdef double trial_x = 0.1
    cdef double ratio
    # ratio = f(trial_x) / max_f
    found_upper_bound = False
    found_lower_bound = False
    while (not found_lower_bound) or (not found_upper_bound):
        try:
            ratio = f(trial_x) / max_f
        except ZeroDivisionError:
            print("First pass")
            continue  # Probably a random MC mistake. Try again.
        # print(trial_x)
        if ratio > small_val:
            lower_bound = trial_x
            trial_x *= 5
            found_lower_bound = True
        else:
            upper_bound = trial_x
            trial_x *= 0.2
            found_upper_bound = True
    # print("--------")
    attempts = 0
    trial_x = (upper_bound + lower_bound)/2
    while abs(small_val - f(trial_x) / max_f)/small_val > epsilon:
        try:
            ratio = f(trial_x) / max_f
        except ZeroDivisionError:
            print("Second pass")
            continue  # Probably a random MC mistake. Try again.
        # print(trial_x)
        if ratio > small_val:
            lower_bound = trial_x
        else:
            upper_bound = trial_x
        trial_x = (upper_bound + lower_bound)/2
        attempts += 1
        if attempts > 100:
            break

    return trial_x


def find_dimensionless_dob_limit(func, x_mode, dob,
                                 delta_x=1e-4, epsilon=1e-10):
    # warnings.simplefilter("always")
    cdef double d = x_mode
    cdef double integral = 0.0
    cdef double fx_old = func(d)
    cdef int counter = 0
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
        counter += 1
        if counter > 1e5:
            # print("find_dimensionless_dob_limit gave up")
            warnings.warn("find_dimensionless_dob_limit gave up", RuntimeWarning)
            return d


def trapezoid_integ_rule(func, lower_limit, upper_limit, N=500):
    h = (upper_limit - lower_limit)/N
    integral = 0
    for i in range(N):
        integral += h/2 * (func(lower_limit + h*(i+1)) + func(lower_limit + h*i))
    return integral



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
    return val * Heaviside(np.array(cbar - cbar_lower)) * Heaviside(np.array(cbar_upper - cbar))


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
    cdef double cbar_val

    def __init__(self, is_denominator, Delta_k, prior_set, Q, k, nc, h, coeffs,
                 cbar_lower=0.0, cbar_upper=0.0, sigma=0.0, cbar_val=-1.0):
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
        self.cbar_val = cbar_val

    def __call__(self, c_vars):
        """c_vars = [cbar, c_{k+2}/cbar, c_{k+3}/cbar, ..., c_h/cbar], unless
        cbar_val != -1.0, in which case cbar is left out."""

        cdef np.ndarray cbar

        if self.cbar_val == -1.0:
            cbar = c_vars[:, 0]
        else:
            cbar = np.array([[self.cbar_val]])

        if self.prior_set == "D":
            return self.Gauss_integrand(cbar)

        func = self.basic_integrand(np.array([cbar]).T)
        if self.is_denominator:
            return func

        cdef np.ndarray u_list, u

        # The remainder of expansion beyond c_{k+1} Q^{k+1}
        cdef int parallel_len = cbar.shape[0]
        cdef np.ndarray pr_ckplus1_sum = np.zeros(parallel_len)
        cdef np.ndarray remaind = np.zeros(parallel_len)
        cdef int m

        likelihood = likelihood_hash(self.prior_set)

        if len(c_vars[0]) > 1:  # If more variables than cbar is given
            if self.cbar_val == -1.0:
                u_list = c_vars.T[1:]
            else:
                u_list = c_vars.T
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
            # print("ci:", ci, "\n",
            #       "func:", func, "\n",
            #       "cbar:", cbar)
            func *= likelihood_hash(self.prior_set)(ci, cbar[:, 0])

        # print("ffunc:", func)
        return func

    def Gauss_integrand(self, np.ndarray x):
        # Still implementing!
        cdef double q = self.Q**(self.k+1) * np.sqrt((1-self.Q**(2*self.h))/(1-self.Q))
        cdef double gammaSq = sum([cn**2 for cn in self.coeffs])
        cdef double integrand = x[:, 0]**(self.n_c-1) * np.exp(-np.log(x[:, 0])**2 / (2*self.sigma**2)) * np.exp(-gammaSq * x[:, 0]**2 / 2)

        if not self.is_denominator:
            integrand *= x[:, 0] * np.exp(-self.Delta_k**2 * x[:, 0]**2 / (2 * q**2)) / (np.sqrt(2*pi) * q)

        return integrand


cdef class Delta_k_posterior:
    """Return the posterior pr(Delta_k|{c_i}) as a function of Delta_k."""

    cdef double Q, cbar_lower, cbar_upper, sigma, gam
    cdef double cbar_lower_limit, cbar_upper_limit, cm_upper_limit
    cdef str prior_set
    cdef int k, n_c, h, cbar_delta_fn
    cdef np.ndarray coeffs
    cdef list cbar_limit_list, cm_limit_list

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
        self.cbar_delta_fn = False

        gamSq = 0
        for cn in self.coeffs[:self.n_c]:
            gamSq += cn**2
        self.gam = np.sqrt(gamSq)

        # Determine optimal limits of integration for prior sets
        try:
            max_coeff = np.max(np.absolute(coeffs[:nc]))
        except ValueError:
            max_coeff = 0
        if prior_set == "A":
            # Make the integral saturate near c_> without causing issues
            # if any of the c's are too large.
            if max_coeff >= cbar_upper:
                for index, c in enumerate(self.coeffs[:nc]):
                    if c > cbar_upper:
                        self.coeffs[index] = cbar_upper
                try:
                    max_coeff = np.max(np.absolute(self.coeffs[:nc]))
                except ValueError:
                    max_coeff = 0
                self.cbar_delta_fn = True

            self.cbar_lower_limit = max(max_coeff, cbar_lower)
            self.cbar_upper_limit = cbar_upper
            self.cm_upper_limit = 1
        if prior_set == "B":
            self.cbar_lower_limit = max_coeff
            self.cbar_upper_limit = np.exp(5 * sigma)
            self.cm_upper_limit = 1
        if prior_set == "C":
            self.cbar_lower_limit = cbar_lower
            # c_> must be big enough to make exp(cn^2/2cbar) small
            # but also keep 1/cbar small...
            # but also not go beyond the prescribed cbar_> parameter...
            self.cbar_upper_limit = min(cbar_upper, max(5 * self.gam**2, 2))
            self.cm_upper_limit = 4

        self.cbar_limit_list = [[self.cbar_lower_limit,
                                  self.cbar_upper_limit]]
        self.cm_limit_list = (h-1) * [[0, self.cm_upper_limit]]

    def __call__(self, Delta_k):
        return self.main_func(Delta_k)

    @lru_cache(maxsize=256)
    def main_func(self, Delta_k):

        if self.prior_set == "C":
            if self.Q == 1.0:
                q = np.sqrt(self.h)
            else:
                q = np.sqrt(self.Q**(2*self.k+2) * (1.0-self.Q**(2*self.h))/(1.0-self.Q**2))

            if len(self.coeffs) == 0:
                pr_density = 1/(np.sqrt(2*pi*self.h) * np.log(self.cbar_upper/self.cbar_lower))
                pr_density /= abs(Delta_k)
                pr_density *= (spgammainc(0.5, Delta_k**2/(2.0*self.cbar_lower**2))
                               - spgammainc(0.5, Delta_k**2/(2.0*self.cbar_upper**2))
                               ) * spgamma(0.5)
                return pr_density

            z = self.gam**2 + Delta_k**2/q**2
            pr_density = (self.gam**2/z)**(0.5*(1.0+self.n_c)) / (np.sqrt(pi) * q * self.gam)

            if self.cbar_lower**2 < z < self.cbar_upper**2:
                # Low precision
                gamma_func = mpmath.fp.gammainc
            else:
                # High precision (just in case)
                gamma_func = gammainc

            if z > 5*self.cbar_upper**2:
                # High precision (slow)
                pr_density *= gamma_func(0.5*(1.0+self.n_c),
                                         z/(2.0*self.cbar_upper**2),
                                         z/(2.0*self.cbar_lower**2)
                                         )
                pr_density /= gamma_func(0.5*self.n_c,
                                         self.gam**2/(2.0*self.cbar_upper**2),
                                         self.gam**2/(2.0*self.cbar_lower**2)
                                         )
                return float(pr_density)
            else:
                # Low precision (very fast)
                pr_density *= (spgammainc(0.5*(1.0+self.n_c), z/(2.0*self.cbar_lower**2))
                               - spgammainc(0.5*(1.0+self.n_c), z/(2.0*self.cbar_upper**2))
                               ) * spgamma(0.5*(1.0+self.n_c))
                
                pr_density /= (spgammainc(0.5*self.n_c, self.gam**2/(2.0*self.cbar_lower**2))
                               - spgammainc(0.5*self.n_c, self.gam**2/(2.0*self.cbar_upper**2))
                               ) * spgamma(0.5*self.n_c)
                return pr_density

            

        elif self.cbar_delta_fn:
            denom_f = Delta_k_integrand(
                is_denominator=True, Delta_k=Delta_k, prior_set=self.prior_set,
                Q=self.Q, k=self.k, nc=self.n_c, h=self.h, coeffs=self.coeffs,
                cbar_lower=self.cbar_lower, cbar_upper=self.cbar_upper,
                sigma=self.sigma, cbar_val=self.cbar_upper)

            numer_f = Delta_k_integrand(
                is_denominator=False, Delta_k=Delta_k, prior_set=self.prior_set,
                Q=self.Q, k=self.k, nc=self.n_c, h=self.h, coeffs=self.coeffs,
                cbar_lower=self.cbar_lower, cbar_upper=self.cbar_upper,
                sigma=self.sigma, cbar_val=self.cbar_upper)

            denom_result = denom_f([[self.cbar_upper]])

            if self.h == 1:
                numer_result = numer_f([[self.cbar_upper]])
            else:
                numer_integ = vegas.Integrator(self.cm_limit_list)
                numer_integ(numer_f, nitn=10, neval=1e4)  # Ready the grid
                numer_result = numer_integ(numer_f, nitn=10, neval=1e4)
            return numer_result/(denom_result * self.Q**(self.k+1))
        else:
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

            denom_integ = vegas.Integrator(self.cbar_limit_list)
            denom_integ(denom_f, nitn=10, neval=1e4)  # Ready the grid
            denom_result = denom_integ(denom_f, nitn=10, neval=1e4)

            numer_integ = vegas.Integrator(self.cbar_limit_list + self.cm_limit_list)
            numer_integ(numer_f, nitn=10, neval=1e4)  # Ready the grid
            numer_result = numer_integ(numer_f, nitn=10, neval=1e4)

        try:
            final_result = (numer_result/(denom_result * self.Q**(self.k+1))).mean
        except ZeroDivisionError:
            print("numer:", numer_result.mean, "denom:", denom_result.mean)
            raise

        return final_result

        # print(Delta_k, numer_result, denom_result, self.Q**(self.k+1))
        



cdef class Lambda_b_pdf:

    cdef double cbar_lower, cbar_upper, sigma
    cdef double lambda_lower, lambda_upper, lambda_sigma, lambda_mu
    cdef double cbar_lower_limit, cbar_upper_limit, norm
    cdef str prior_set, lambda_prior
    cdef int k
    cdef np.ndarray b_coeffs
    cdef int include_lambda_prior
    # cdef list denom_limit_list, numer_limit_list

    def __init__(self, prior_set, k, b_coeffs, lambda_prior,
                 lambda_lower=0.0, lambda_upper=0.0, lambda_mu=0.0, lambda_sigma=0.0,
                 cbar_lower=0.0, cbar_upper=0.0, sigma=0.0, include_lambda_prior=True):
        self.prior_set = prior_set
        self.k = k
        # self.h = h
        self.b_coeffs = np.array(b_coeffs)
        self.lambda_prior = lambda_prior
        self.lambda_lower = lambda_lower
        self.lambda_upper = lambda_upper
        self.lambda_mu = lambda_mu
        self.lambda_sigma = lambda_sigma
        self.cbar_lower = cbar_lower
        self.cbar_upper = cbar_upper
        self.sigma = sigma
        self.include_lambda_prior = include_lambda_prior
        # self.coeff_row = 0
        if lambda_prior == "u":
            # self.norm = 1/quad(self.Lb_unnormalized_pdf, lambda_lower, lambda_upper)[0]
            self.norm = 1.0
        elif lambda_prior == "g":
            # self.norm = 1/quad(self.Lb_unnormalized_pdf, 1, 1500)[0]
            self.norm = 1.0
        # self.norm = 1/trapezoid_integ_rule(self.Lb_unnormalized_pdf, lambda_lower, lambda_upper)
        # Lb_integ = vegas.Integrator([[self.lambda_lower, self.lambda_upper]])
        # Lb_pdf = partial(self.Lb_unnormalized_pdf)
        # print("made it?", self.Lb_unnormalized_pdf)
        # # Lb_integ(self.Lb_unnormalized_pdf(), nitn=10, neval=1e4)
        # Lb_integ(Lb_pdf, nitn=10, neval=1e4)
        # print("here?")
        # self.norm = Lb_integ(self.Lb_unnormalized_pdf, nitn=10, neval=1e4).mean

    def __call__(self, Lambda_b):
        # return float(self.Lb_unnormalized_pdf(Lambda_b) * self.norm)
        return self.Lb_unnormalized_pdf(Lambda_b)

    def cbar_integrand(self, cbar, Lambda_b):
        cbar_prior = cbar_prior_hash(self.prior_set, self.cbar_lower, self.cbar_upper, self.sigma)
        likelihood = likelihood_hash(self.prior_set)
        integrand_factor = cbar_prior(np.array(cbar))
        cdef int n
        cdef double cn

        for n in range(2, self.k+1):
            # integrand_factor *= np.exp(-(self.coeffs[n] * L**n)**2/(2*cbar**2))
            cn = self.b_coeffs[n]
            for j in range(n):
                cn *= Lambda_b
            # print(self.b_coeffs[n] * 600**n)
            integrand_factor *= likelihood(cn, np.array([cbar]))
        return integrand_factor

    def analytic_cbar_factor_prior_C(self, Lambda_b):
        cdef double x = 0
        cdef double cn = 0
        cdef int j = 0
        for n in range(2, self.k+1):
            cn = self.b_coeffs[n]
            for j in range(n):
                cn *= Lambda_b
            x = x + cn**2
        return x**((1-self.k) * 0.5)

    def analytic_cbar_factor_prior_A(self, Lambda_b):
        cdef double cn
        cn_list = []
        for n in range(2, self.k+1):
            cn = self.b_coeffs[n]
            for j in range(n):
                cn *= Lambda_b
            cn_list.append(abs(cn))
        return max(cn_list)**(1.0 - self.k)

    def lambda_prior_func(self, Lambda_b):
        if self.lambda_prior == "u":
            if Lambda_b < self.lambda_lower or Lambda_b > self.lambda_upper:
                return 0
            else:
                return 1/Lambda_b
        elif self.lambda_prior == "uu":
            if Lambda_b < self.lambda_lower or Lambda_b > self.lambda_upper:
                return 0
            else:
                return 1
        elif self.lambda_prior == "g":
            return np.exp(-(Lambda_b-self.lambda_mu)**2/(2*self.lambda_sigma**2))

    def Lb_unnormalized_pdf(self, Lambda_b):

        # cdef long long factor, value
        cdef int i = 0

        # factor = mpfr(float(Lambda_b))**( (self.k**2 + self.k - 4) / 2 )
        if self.cbar_lower <= 0.001 and self.cbar_upper >= 1000 and self.prior_set == "C":
            value = self.analytic_cbar_factor_prior_C(Lambda_b)
        elif self.cbar_lower <= 0.001 and self.cbar_upper >= 1000 and self.prior_set == "A":
            value = self.analytic_cbar_factor_prior_A(Lambda_b)
        else:
            cbar_func = partial(self.cbar_integrand, Lambda_b=Lambda_b)
            value = quad(cbar_func, self.cbar_lower, self.cbar_upper)[0]
        for i in range((self.k**2 + self.k - 2) / 2):
            value = value * Lambda_b

        if self.include_lambda_prior:
            value = value * self.lambda_prior_func(Lambda_b)
        # value *= Lambda_b**((self.k**2 + self.k - 4) / 2)
        # value = factor * trapezoid_integ_rule(cbar_func, self.cbar_lower, self.cbar_upper)
        # cbar_integ = vegas.Integrator([[self.cbar_lower, self.cbar_upper]])
        # cbar_integ(cbar_func, nitn=10, neval=1e4)
        # value = (factor * cbar_integ(cbar_func, nitn=10, neval=1e4)).mean

        # print(self.b_coeffs)
        # print(self.cbar_lower, self.cbar_upper, Lambda_b)
        # print(self.k, quad(cbar_func, self.cbar_lower, self.cbar_upper)[0], quad(cbar_func, self.cbar_lower, self.cbar_upper)[1], factor, value)
        # return np.prod(value)
        return value
