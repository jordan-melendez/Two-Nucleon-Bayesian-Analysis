################################
# Defines useful functions for Bayesian statistics using CH_to_EKM
################################
################################

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from sympy.functions.special.delta_functions import Heaviside
import vegas


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
    abs_list = tuple(map(abs, c_tuple))
    return max(abs_list)


################################
# Prior set A
################################

def prior_cn_A(cn, cbar):
    return Heaviside(cbar - abs(cn)) / (2 * cbar)


def prior_cbar_A(cbar, cbar_lower, cbar_upper):
    val = 1/(cbar * np.log(cbar_upper/cbar_lower))
    return val * Heaviside(cbar - cbar_lower) * Heaviside(cbar_upper - cbar)


def Delta_k_posterior_A1(Q, k, n_c, cbar_lower, cbar_upper, *c_tuple):
    val = 1/(2 * Q**(k+1)) * (n_c/(n_c+1)) * \
        Heaviside(cbar_upper - cbark(k+1, *c_tuple)) / \
        (cbark(k, *c_tuple)**(-n_c) - cbar_upper**(-n_c))

    def A1_post(Delta_k):
        if abs(Delta_k) <= cbark(k, *c_tuple) * Q**(k+1):
            return (cbark(k, *c_tuple)**(-n_c-1) - cbar_upper**(-n_c-1)) * val
        else:
            return ((Q**(k+1)/abs(Delta_k))**(n_c+1) -
                    cbar_upper**(-n_c-1)) * val

    return A1_post


def dkp_A_eps(Q, k, n_c, p, c_bar_k):
    factor = c_bar_k * Q**(k+1)
    if p <= n_c/(n_c + 1):
        factor *= (n_c + 1)/n_c * p
    else:
        factor *= (1/((n_c + 1)*(1 - p)))**(1/n_c)
    return factor

################################
# Prior set B
################################


################################
# Prior set C
################################


################################
# General posteriors
################################

def Delta_k_posterior(cn_prior, cbar_prior, h, Q, k, n_c, *c_tuple):
    """Return a pdf that takes \Delta_k as an argument.

    Arguments
    ---------
    cn_prior   = function
                 pr(c_n|cbar), which takes (c_n, cbar) as args
    cbar_prior = function
                 pr(cbar), which only takes cbar as an arg. Can use 'partial'
                 to take a function of more arguments to turn it only
                 into one of cbar.
    h          = int
                 Equals k_max - k and is the number of orders after k kept to
                 estimate error.
    Q          = float
                 p/\Lambda_b
    k          = int
                 The order to which the sum is calculated.
    n_c        = int
                 The number of nonzero coefficients <= k.
                 Equals k+1 if all nonzero.
    c_tuple    = tuple of floats
                 (c_0, c_1, ..., c_k)
    """

    def total_integral(Delta_kh):

        def numerator_integrand(c_list):
            """Return numerator integrand of Eq. (30).

            c_list = float list
                     [cbar, c_{k+2}, ..., c_{k_max}] which range from
                     [[0, \infty], [-\infty, \infty], ..., [-\infty, \infty]]
            """
            integrand = cn_prior(Delta_kh/Q**(k+1), c_list[0]) * \
                cbar_prior(c_list[0])

            for cn in c_tuple:
                integrand *= cn_prior(cn, c_list[0])

            for cm in c_list[1:]:
                integrand *= cn_prior(cm, c_list[0])

            return integrand

        def denominator_integrand(cbar):
            """Return denominator integrand of Eq. (30).

            c_list = float list
                     [cbar, c_{k+2}, ..., c_{k_max}] which range from
                     [[0, \infty], [-\infty, \infty], ..., [-\infty, \infty]]
            """
            integrand = cbar_prior(cbar[0])

            for cn in c_tuple:
                integrand *= cn_prior(cn, cbar[0])

            return integrand

        infty = 50
        limit_list = [[0, infty]]
        denom_integ = vegas.Integrator(limit_list)
        denom_result = denom_integ(denominator_integrand, nitn=10, neval=1000).mean

        limit_list = [[0, infty]]
        for i in range(h):
            limit_list.append([-infty, infty])

        numer_integ = vegas.Integrator(limit_list)
        numer_result = numer_integ(numerator_integrand, nitn=10, neval=1000).mean

        # print("numer:", numer_result)
        # print("Q:", Q)
        # print("denom:", denom_result)
        return numer_result/(Q**(k+1) * denom_result)

    return total_integral

################################
# Test stuff
################################

# ctup = 1.0, 1.0, 1.0, 1.0
# From total cross section at E = 50
# ctup = 1.0, -1.431187418674369, 0.150200003435126, -0.208208006179793, \
#     3.313000457353716


# # From C_0-0-0-0, theta = 0
# ctup = 1.0, 0.747916884724334, 3.129626874642131, -1.647209272500563
# cbar_eps = 0.001
# Q = 153/600
# # k = 3
# # n_c = k
# h = 1
# p_decimal = .68
# X0 = 183.6
# z_lst = [1]

# for k in [0, 1, 2, 3, 4, 5]:
#     c_bar_k = cbark(*ctup[:n_c_val(k, z_lst)])
#     dkp = dkp_A_eps(Q, k, n_c_val(k, z_lst), p_decimal, c_bar_k)
#     print("k:", k, "d_k:", dkp, "Delta_k:", dkp * X0)



# partial_prior_cbar_A = partial(prior_cbar_A, cbar_lower=cbar_eps,
#                                cbar_upper=1/cbar_eps)
# dk_post = Delta_k_posterior_A1(Q, k, k+1, cbar_eps, 1/cbar_eps, *ctup)
# dkh_post = Delta_k_posterior(prior_cn_A, partial_prior_cbar_A,
#                              h, Q, k, n_c, *ctup)
# print(dk_post(0))
# print(dkh_post(0))
# x_max = .05
# x_step = 0.01
# delta_domain = np.arange(-x_max, x_max, x_step)
# delta_k1_range = [dk_post(d) for d in delta_domain]
# delta_kh_range = [dkh_post(d) for d in delta_domain]
# plt.plot(delta_domain, delta_k1_range, 'r--',
#          delta_domain, delta_kh_range, 'g^')
# plt.show()
