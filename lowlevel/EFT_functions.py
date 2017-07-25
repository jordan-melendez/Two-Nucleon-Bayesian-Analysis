# Functions relating to the EFT expansion of observables.

import sys
import os
sys.path.append(os.path.expanduser(
    '~/Dropbox/Bayesian_codes/observable_analysis_Jordan'))
import numpy as np
from numpy import array, vectorize, zeros
from src.lowlevel.filenames import observable_filename, dob_filename, npwa_filename
from src.lowlevel.datafile import DataFile
from src.lowlevel.kinematics import E_to_p, p_to_E

def order_to_power(order):
    """Relates the order (LO, NLO, N2LO, etc.) to the power of Q in chiral EFT."""

    power = order
    if order > 0:
        power += 1
    return power


def Q_approx(p, Lambda_b, single_expansion=False):
    if single_expansion:
        m_pi = 0
    else:
        m_pi = 138  # Set to 0 to just return p/Lambda_b
    # Interpolate to smooth the transition from m_pi to p
    n = 8
    q = (m_pi**n + p**n) / (m_pi**(n-1) + p**(n-1)) / Lambda_b
    return q

# A vectorized version of Q that can accept an array of momenta
Q_ratio = vectorize(Q_approx, excluded=['Lambda_b'])


# e = 200
# print(Q_approx(E_to_p(e, "np"), 600, single_expansion=False))
# p = 138
# print(p_to_E(p, "np"))

def get_average_scale(indep_var_list, func_list):
    """Return the average magnitude of a function over a range of values.
    """
    integral = 0
    for i in range(1, len(indep_var_list)):
        integral += abs(func_list[i]) * (indep_var_list[i] - indep_var_list[i-1])
    indep_var_max, indep_var_min = np.max(indep_var_list), np.min(indep_var_list)
    return integral/(indep_var_max - indep_var_min)


def get_X_ref(observable_dir, observable, indep_var, indep_var_list, param_var,
              param, hash_val, convention, Lambda_b, *X_orders, X_NPWA_list=None):
    """uhhhhhh.."""
    order_dict = {0: "LO", 1: "NLO", 2: "N2LO", 3: "N3LO", 4: "N4LO"}
    if hash_val == "ave":
        if indep_var == "theta":
            E_lab = param
        else:
            E_lab = indep_var_list

        X_ref = abs(X_orders[0])
        for i in range(1, len(X_orders)):
            X_ref += abs(X_orders[i] - X_orders[i-1]) / \
                Q_ratio(E_to_p(E_lab, "np"), Lambda_b)**(i+1)
        return X_ref
    else:
        hash_order = int(hash_val[0])
        hash_order_str = order_dict[hash_order]
        X_list = X_orders[hash_order]
        hash_type = hash_val[1:]

    if hash_val[1:] == "dsigma":
        hash_order = int(hash_val[0])
        hash_order_str = order_dict[hash_order]
        X_list = X_orders[hash_order]
        if observable == ['t', 't', 't', 't']:
            return X_list
        elif observable == ['0', '0', '0', '0']:
            return X_list
        else:
            return np.ones(len(X_list))

    X_ref = []

    if hash_type == "L":  # Local
        X_ref = X_list

    if hash_type == "T":  # theta average
        if indep_var == "theta":
            X_ref = get_average_scale(indep_var_list, X_list) * \
                np.ones(len(X_list))
        else:
            for E in indep_var_list:
                files = load_observable_files(
                    observable_dir, observable, "theta", ivar_start=0,
                    ivar_stop=181, ivar_step=1, param_var="energy", param=E,
                    orders=[hash_order_str], convention=convention)

                data = load_observable_data(*files)

                X_ref.append(get_average_scale(list(range(181)), data[1]))

    if hash_type == "E":  # Energy average
        if indep_var == "energy":
            X_ref = get_average_scale(indep_var_list, X_list) * \
                np.ones(len(X_list))
        else:
            for theta in indep_var_list:
                files = load_observable_files(
                    observable_dir, observable, "energy", ivar_start=0,
                    ivar_stop=351, ivar_step=1, param_var="theta", param=theta,
                    orders=[hash_order_str], convention=convention)

                data = load_observable_data(*files)

                X_ref.append(get_average_scale(list(range(351)), data[1]))

    return X_ref

    # if hash_val == 0:
    #     X_ref = X_0_list

    # if hash_val == 1:
    #     if indep_var == "theta":
    #         X_ref = get_average_scale(indep_var_list, X_0_list) * \
    #             np.ones(len(X_0_list))
    #     else:
    #         for E in indep_var_list:
    #             files = load_observable_files(
    #                 observable_dir, observable, "theta", ivar_start=0,
    #                 ivar_stop=181, ivar_step=1, param_var="energy", param=E,
    #                 orders=["LO"], convention=convention)

    #             data = load_observable_data(*files)

    #             X_ref.append(get_average_scale(list(range(181)), data[1]))

    # if hash_val == 2:
    #     if indep_var == "energy":
    #         X_ref = get_average_scale(indep_var_list, X_0_list) * \
    #             np.ones(len(X_0_list))
    #     else:
    #         for theta in indep_var_list:
    #             files = load_observable_files(
    #                 observable_dir, observable, "energy", ivar_start=0,
    #                 ivar_stop=351, ivar_step=1, param_var="theta", param=theta,
    #                 orders=["LO"], convention=convention)

    #             data = load_observable_data(*files)

    #             X_ref.append(get_average_scale(list(range(351)), data[1]))

    # if hash_val == 3:
    #     X_ref = X_orders[4]

    # if hash_val == 4:
    #     if indep_var == "theta":
    #         X_ref = get_average_scale(indep_var_list, X_orders[4]) * \
    #             np.ones(len(X_orders[4]))
    #     else:
    #         for E in indep_var_list:
    #             files = load_observable_files(
    #                 observable_dir, observable, "theta", ivar_start=0,
    #                 ivar_stop=181, ivar_step=1, param_var="energy", param=E,
    #                 orders=["N4LO"], convention=convention)

    #             data = load_observable_data(*files)

    #             X_ref.append(get_average_scale(list(range(181)), data[1]))

    # if hash_val == 5:
    #     if indep_var == "energy":
    #         X_ref = get_average_scale(indep_var_list, X_orders[4]) * \
    #             np.ones(len(X_orders[4]))
    #     else:
    #         for theta in indep_var_list:
    #             files = load_observable_files(
    #                 observable_dir, observable, "energy", ivar_start=0,
    #                 ivar_stop=351, ivar_step=1, param_var="theta", param=theta,
    #                 orders=["N4LO"], convention=convention)

    #             data = load_observable_data(*files)

    #             X_ref.append(get_average_scale(list(range(351)), data[1]))

    # return X_ref




XLO = object()
def coeffs(Q, *X_orders, X_ref=XLO):
    """Return coefficient tuples given Q ratio (array) and X observable (array).

    Parameters
    ----------
    Q       : float or ndarray
              The expansion parameter.
    X_orders: float or ndarray
              The value of observable X at LO, NLO, ... in increasing order.
    X0      : float or ndarray
              The scaling factor of the EFT expansion.

    Returns
    -------
    c_array : ndarray
              Coefficient array of [cLO, cNLO, cN2LO, ...],
              where cLO, etc. could be an arrays
    """
    if X_ref is XLO:
        X_ref = X_orders[0]

    c_list = []
    for order in range(len(X_orders)):
        if order == 0:
            c_list.append(X_orders[0]/X_ref)
        else:
            coeff = (X_orders[order] - X_orders[order-1]) / \
                (X_ref * Q**order_to_power(order))
            c_list.append(coeff)
    return tuple(c_list)


def load_observable_files(observable_dir, observable, indep_var, ivar_start,
                          ivar_stop, ivar_step, param_var, param, orders, convention):
    obs_files = []
    for order in orders:
        name = observable_filename(
            observable, indep_var, ivar_start, ivar_stop,
            ivar_step, param_var, param, order, convention)
        obs_files.append(DataFile().read(os.path.join(observable_dir, name)))

    return tuple(obs_files)


def load_observable_data(*observable_files):
    """Return a tuple of lists: independent variable, LO, NLO, N2LO, etc. observables.

    Parameters
    ----------
    observable_files: DataFile
                      A tuple of DataFiles
    """
    obs_files = []
    param = array(observable_files[0][0])
    observables = [param]
    for observable_file in observable_files:
        observables.append(array(observable_file[1]))

    return tuple(observables)


def find_percent_success(
    error_band_dir, observable_list, theta_grid, energy_grid, order_list,
    ignore_orders, Lambda_b, lambda_mult, X_ref_hash, p_decimal, prior_set,
    h, convention, indep_var_list=None, cbar_lower=None, cbar_upper=None,
    sigma=None, potential_info=None
        ):
    theta_start = 0
    theta_stop = 181
    theta_step = 1
    energy_start = 1
    energy_stop = 351
    energy_step = 1
    # N = len(theta_grid) * len(energy_grid)
    N = 0
    n_successes = 0
    npwa_dir = "../npwa_data/"
    npwa_theta_start = 0
    npwa_theta_stop = 180
    npwa_theta_step = 1
    npwa_energy_start = 1
    npwa_energy_stop = 351
    npwa_energy_step = 1
    for observable in observable_list:
        # for i in range(len(order_list)-h):
        if h == 1:  # Compare to subsequent order
            rng = len(order_list) - 1
        elif h > 1:  # Compare to npwa
            rng = len(order_list)
        for i in range(rng):
            j = i + h
            if observable == ['t', 't', 't', 't']:
                given_file_name = dob_filename(
                    obs_indices=observable, indep_var="energy", ivar_start=energy_start,
                    ivar_stop=energy_stop, ivar_step=energy_step, param_var="theta", param=0,
                    order=order_list[i], ignore_orders=ignore_orders, Lambda_b=Lambda_b,
                    lambda_mult=lambda_mult, X_ref_hash=X_ref_hash,
                    p_decimal=p_decimal, prior_str=prior_set, h=h,
                    convention=convention, indep_var_list=indep_var_list,
                    cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma,
                    potential_info=potential_info)
                given_file = DataFile().read(os.path.join(error_band_dir, given_file_name))
                if h == 1:
                    test_file_name = dob_filename(
                        obs_indices=observable, indep_var="energy", ivar_start=energy_start,
                        ivar_stop=energy_stop, ivar_step=energy_step, param_var="theta", param=0,
                        order=order_list[j], ignore_orders=ignore_orders, Lambda_b=Lambda_b,
                        lambda_mult=lambda_mult, X_ref_hash=X_ref_hash,
                        p_decimal=p_decimal, prior_str=prior_set, h=h,
                        convention=convention, indep_var_list=indep_var_list,
                        cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma,
                        potential_info=potential_info)
                    test_file = DataFile().read(os.path.join(error_band_dir, test_file_name))
                elif h > 1:
                    test_file_name = npwa_filename(observable, param_name="", param_val=None)
                    test_file = DataFile().read(os.path.join(npwa_dir, test_file_name))

                for energy in energy_grid:
                    N += 1
                    row = None
                    if indep_var_list is None:
                        row = int((energy - energy_start)/energy_step)
                    else:
                        # print(indep_var_list)
                        for index, ivar in enumerate(indep_var_list):
                            if ivar == energy:
                                row = index
                    npwa_row = int((energy - npwa_energy_start)/npwa_energy_step)
                    upper_bound = given_file[row, 3]
                    lower_bound = given_file[row, 2]
                    if h == 1:
                        test_observable = test_file[row, 1]
                    elif h > 1:
                        test_observable = test_file[npwa_row, 1]
                    if lower_bound <= test_observable <= upper_bound:
                        n_successes += 1
            else:
                for energy in energy_grid:
                    given_file_name = dob_filename(
                        obs_indices=observable, indep_var="theta", ivar_start=theta_start,
                        ivar_stop=theta_stop, ivar_step=theta_step, param_var="energy", param=energy,
                        order=order_list[i], ignore_orders=ignore_orders, Lambda_b=Lambda_b,
                        lambda_mult=lambda_mult, X_ref_hash=X_ref_hash,
                        p_decimal=p_decimal, prior_str=prior_set, h=h,
                        convention=convention, indep_var_list=indep_var_list,
                        cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma,
                        potential_info=potential_info)
                    given_file = DataFile().read(os.path.join(error_band_dir, given_file_name))
                    if h == 1:
                        test_file_name = dob_filename(
                            obs_indices=observable, indep_var="theta", ivar_start=theta_start,
                            ivar_stop=theta_stop, ivar_step=theta_step, param_var="energy", param=energy,
                            order=order_list[j], ignore_orders=ignore_orders,
                            Lambda_b=Lambda_b, lambda_mult=lambda_mult, X_ref_hash=X_ref_hash,
                            p_decimal=p_decimal, prior_str=prior_set, h=h,
                            convention=convention, indep_var_list=indep_var_list,
                            cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma,
                            potential_info=potential_info)
                        test_file = DataFile().read(os.path.join(error_band_dir, test_file_name))
                    elif h > 1:
                        test_file_name = npwa_filename(observable, param_name="energy", param_val=energy)
                        test_file = DataFile().read(os.path.join(npwa_dir, test_file_name))

                    for theta in theta_grid:
                        N += 1
                        row = None
                        if indep_var_list is None:
                            row = int((theta - theta_start)/theta_step)
                        else:
                            for index, ivar in enumerate(indep_var_list):
                                if ivar == theta:
                                    row = index
                        npwa_row = int((theta - npwa_theta_start)/npwa_theta_step)
                        upper_bound = given_file[row, 3]
                        lower_bound = given_file[row, 2]
                        if h == 1:
                            test_observable = test_file[row, 1]
                        elif h > 1:
                            test_observable = test_file[npwa_row, 1]
                        if lower_bound <= test_observable <= upper_bound:
                            n_successes += 1

    return n_successes/N, N


# def test():
#     return 1, 2, 3

# a, b, c = test()

# print(a)
# print(b)

# n_vals = [0, 1, 2, 3, 4]
# Q = array([i for i in range(1, 5)])
# X0 = array([i for i in range(2, 6)])
# X1 = array([2*i for i in range(1, 5)])
# X2 = array([3*i for i in range(1, 5)])
# X3 = array([4*i for i in range(1, 5)])
# X4 = array([5*i for i in range(1, 5)])
# Q = 10
# X0 = 1
# X1 = 2
# X2 = 3
# X3 = 4
# X4 = 5
# cLO, cNLO, cN2LO, cN3LO, cN4LO = coeffs(Q, X0, X1, X2, X3, X4)
# print(cLO, cNLO, cN2LO, cN3LO, cN4LO)
