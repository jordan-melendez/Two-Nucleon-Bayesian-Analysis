# Functions to generate (what I hope will be) a consistent set of filenames.

import os
import re


###############################################################################

def get_potential_file_info(directory, convention):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(convention + "_phases.txt"):
                info_str = re.search("kvnn(.*)_phases", file)
                return "kvnn" + info_str.group(1)


def phase_filename(potential_file_name, convention):
    phase_file_name = re.sub(r".out", r"_phases.txt", potential_file_name)
    if potential_file_name == phase_file_name:
        print("The potential file name did not have the correct form.")
        return None
    return phase_file_name


def potential_filename(phase_file_name, convention):
    name_str = "_" + convention + r"_phases.txt"
    potential_file_name = re.sub(name_str, r".out", phase_file_name)
    if potential_file_name == phase_file_name:
        print("The phase file name did not have the correct form.")
        return None
    return potential_file_name


def mesh_filename(phase_file_name, convention):
    name_str = "_" + convention + r"_phases.txt"
    mesh_file_name = re.sub(name_str, r"_mesh.out", phase_file_name)
    mesh_file_name = re.sub(r"^vnn_", r"vsrg_", mesh_file_name)
    if mesh_file_name == phase_file_name:
        print("The phase file name did not have the correct form.")
        return None
    return mesh_file_name

###############################################################################


def observable_filename(obs_indices, indep_var, ivar_start, ivar_stop,
                        ivar_step, param_var, param, order, convention,
                        potential_info=None):
    """Return a standard filename for observable files based on parameters.

    Parameters
    ----------
    obs_indices = list
                  [p, q, i, k] specifies the spin observable.
    indep_var   = str
                  the name of the independent variable: ["theta", "energy"].
    ivar_start  = int
                  The start of the range of independent variables in the file.
    ivar_stop   = int
                  The end of the range of independent variables in the file.
    ivar_step   = int
                  The step size of the range of indep_var.
    param_var   = str
                  One of ["theta", "energy"] that is NOT indep_var.
    param       = float
                  value of param_var.
    order       = str
                  One of ["LO", "NLO", "N2LO", "N3LO", "N4LO"].
    """

    if potential_info is None:
        if order == "LOp":
            potential_info = get_potential_file_info("LO", convention)
        else:
            potential_info = get_potential_file_info(order, convention)

    if obs_indices == ['t', 't', 't', 't']:
        name = "C_t-t-t-t_vs_" + str(indep_var) + \
               "-" + str(ivar_start) + "-" + str(ivar_stop) + "-" + \
               str(ivar_step) + "_" + str(potential_info) + "_" + str(order) + ".dat"
    else:
        name = "C_" + obs_indices[0] + "-" + obs_indices[1] + "-" + \
               obs_indices[2] + "-" + obs_indices[3] + "_vs_" + indep_var + \
               "-" + str(ivar_start) + "-" + str(ivar_stop) + "-" + \
               str(ivar_step) + "_" + param_var + "-" + str(param) + \
               "_" + str(potential_info) + "_" + str(order) + ".dat"
    return name


def coeff_filename(obs_indices, indep_var, ivar_start, ivar_stop,
                   ivar_step, param_var, param, order,
                   Lambda_b, lambda_mult, X_ref_hash, convention,
                   potential_info=None):
    obs_filename = observable_filename(
        obs_indices, indep_var, ivar_start, ivar_stop,
        ivar_step, param_var, param, order, convention, potential_info)
    obs_split = re.split("_(\w{0,2}?LOp{0,1})[.]dat", obs_filename)
    name = "coeffs_" + obs_split[0] + "_Lambdab-" + str(Lambda_b) + "_lambda-mult"
    try:
        for lamb in lambda_mult:
            name = name + "-" + str(lamb)
    except TypeError:
        name = name + "-" + str(lambda_mult)
    name = name + "_Xref-" + str(X_ref_hash) + "_" + obs_split[1] + ".dat"
    return name


def dob_filename(obs_indices, indep_var, ivar_start, ivar_stop,
                 ivar_step, param_var, param, order,
                 Lambda_b, lambda_mult, X_ref_hash,
                 p_decimal, prior_str, h, convention,
                 cbar_lower=None, cbar_upper=None, sigma=None,
                 potential_info=None):
    """Return a standard filename for DOB files based on parameters.

    Parameters
    ----------
    p_percent            = int
                           The percent probability corresponding to the DOB interval.
    coefficient_filename = str
                           The filename of the observable coefficients for extracting DOBs
    """
    obs_filename = observable_filename(
        obs_indices, indep_var, ivar_start, ivar_stop,
        ivar_step, param_var, param, order, convention, potential_info)
    obs_split = re.split("_(\w{0,2}?LOp{0,1})[.]dat", obs_filename)
    lamb_mult_str = ""
    try:
        for lamb in lambda_mult:
            lamb_mult_str = lamb_mult_str + "-" + str(lamb)
    except TypeError:
        lamb_mult_str = lamb_mult_str + "-" + str(lambda_mult)
    name = """DOB_{name}_Lambdab-{Lambda_b}_lambda-mult{lambda_mult}_Xref-{X_ref_hash}_p-{p_decimal}_prior-{prior_str}_h-{h}_cbarl-{cbar_lower}_cbaru-{cbar_upper}_sigma-{sigma}_{order}.dat""".format(p_decimal=p_decimal, prior_str=prior_str, h=h, cbar_lower=cbar_lower,  cbar_upper=cbar_upper, sigma=sigma, name=obs_split[0], order=order, Lambda_b=Lambda_b, lambda_mult=lamb_mult_str, X_ref_hash=X_ref_hash)
    return name


def npwa_filename(observable, param_name, param_val):
    name = "npwa_C_"
    for obs_index in observable:
        name = name + obs_index + "-"
    name = name[:-1]

    if observable != ['t', 't', 't', 't']:
        name = name + "_" + param_name + "-" + str(param_val)

    name = name + ".dat"
    return name


def plot_obs_error_bands_filename(
        obs_indices, indep_var, ivar_start, ivar_stop, ivar_step,
        param_var, param, order_list, Lambda_b, lambda_mult, X_ref_hash, p_decimal_list,
        prior_str, h, convention, cbar_lower=None, cbar_upper=None, sigma=None,
        potential_info=None):

    p_str_list = ""
    for p in p_decimal_list:
        p_str_list = p_str_list + str(p) + "-"
    p_str_list = p_str_list[:-1]

    dob_fn = dob_filename(
        obs_indices, indep_var, ivar_start, ivar_stop,
        ivar_step, param_var, param, order_list[-1],
        Lambda_b, lambda_mult, X_ref_hash,
        p_str_list, prior_str, h, convention,
        cbar_lower, cbar_upper, sigma,
        potential_info)

    name = re.sub("DOB", "plot", dob_fn)
    name = re.split("_(\w{0,2}?LOp{0,1})", name)[0]

    name = name + "_orders"

    for order in order_list:
        name = name + "-" + order

    return name + ".pdf"


def subplot_obs_error_bands_filename(
        obs_indices, indep_var, ivar_start, ivar_stop, ivar_step,
        param_var, param, order_list, Lambda_b, lambda_mult, X_ref_hash, p_decimal_list,
        prior_str, h, convention, cbar_lower=None, cbar_upper=None, sigma=None,
        potential_info=None):
    plot_name = plot_obs_error_bands_filename(
        obs_indices, indep_var, ivar_start, ivar_stop, ivar_step,
        param_var, param, order_list, Lambda_b, lambda_mult, X_ref_hash,
        p_decimal_list, prior_str, h, convention, cbar_lower, cbar_upper,
        sigma, potential_info)
    name = re.sub("plot", "subplots", plot_name)
    return name


def plot_coeff_error_bands_filename(
        obs_indices, indep_var, ivar_start, ivar_stop,
        ivar_step, param_var, param, order_list, Lambda_b, lambda_mult,
        X_ref_hash, convention, potential_info=None):

    name = "plot_" + coeff_filename(
            obs_indices, indep_var, ivar_start, ivar_stop,
            ivar_step, param_var, param, order_list[-1],
            Lambda_b, lambda_mult, X_ref_hash, convention, potential_info)

    name = re.split("_(\w{0,2}?LOp{0,1})", name)[0]

    name = name + "_orders"

    for order in order_list:
        name = name + "-" + order

    return name + ".pdf"


def plot_consistency_filename(
        obs_indices_list, p_start, p_stop, p_step,
        order_list, Lambda_b, lambda_mult_list,
        X_ref_hash, prior_str, h, convention, combine_obs,
        theta_start=None, theta_stop=None, theta_step=None,
        energy_start=None, energy_stop=None, energy_step=None,
        theta_grid=None, energy_grid=None,
        cbar_lower=None, cbar_upper=None, sigma=None, potential_info=None,
        separate_orders=False
        ):
    placeholder = "xxxxxxxxxxx"
    # name = "consistency_" + coeff_filename(
    #         obs_indices_list[0], indep_var="percent", ivar_start=p_start, ivar_stop=p_stop,
    #         ivar_step=p_step, param_var=placeholder, param=0, order=order_list[-1],
    #         Lambda_b=Lambda_b, lambda_mult=lambda_mult_list, X_ref_hash=X_ref_hash,
    #         convention=convention, potential_info=potential_info)
    name = dob_filename(
        obs_indices_list[0], indep_var="perc", ivar_start=p_start, ivar_stop=p_stop,
        ivar_step=p_step, param_var=placeholder, param=0, order=order_list[-1],
        Lambda_b=Lambda_b, lambda_mult=lambda_mult_list, X_ref_hash=X_ref_hash,
        p_decimal=placeholder, prior_str=prior_str, h=h, convention=convention,
        cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma,
        potential_info=potential_info)

    name = re.sub("DOB", "cons", name)

    if separate_orders:
        order_str = "sep"
    else:
        order_str = "com"
    for j in range(len(order_list)-1):
        order_str = order_str + "-" + order_list[j]
    name = re.sub(order_list[-1], order_str, name)

    found_placeholder = re.search(placeholder + "-0_", name)
    if found_placeholder is not None:  # If the observable isnt ['t','t','t',t']
        name_parts = re.split(placeholder + "-0_", name)
        name = name_parts[0] + name_parts[1]

    name_parts = re.split("_p-" + placeholder, name)
    name = name_parts[0] + name_parts[1]

    obs_str = "_"
    for obs_indices in obs_indices_list:
        obs_str = obs_str + indices_to_short_observable_name(obs_indices) + "-"

    obs_str = obs_str[:-1] + "_"

    name = re.sub("_C_\w{1,3}-\w{1,3}-\w{1,3}-\w{1,3}_", obs_str, name)

    name_parts = re.split("(.dat)", name)

    # name = """{prefix}theta-{theta_start}-{theta_stop}-{theta_step}_energy-{e_start}-{e_stop}-{e_step}{suffix}""".format(prefix=name_parts[0], theta_start=theta_start, theta_stop=theta_stop, theta_step=theta_step, e_start=e_start, e_stop=e_stop, e_step=e_step, suffix=name_parts[1])

    if theta_grid is not None:
        name = name_parts[0] + "_theta-vals"
        for theta in theta_grid:
            name = name + "-" + str(theta)
    else:
        name = name_parts[0] + "theta-range-" + str(theta_start) + "-" + str(theta_stop) + str(theta_step)

    if energy_grid is not None:
        name = name + "_energy-vals"
        for energy in energy_grid:
            name = name + "-" + str(energy)
    else:
        name = name + "energy-range-" + str(energy_start) + "-" + str(energy_stop) + str(energy_step)

    name = name + name_parts[1]

    name = re.split(".dat", name)[0]
    name = name + ".pdf"

    return name


def subplot_6_obs_error_bands_filename(
        obs_indices_list, indep_var, ivar_start, ivar_stop, ivar_step,
        param_var, param, order_list, Lambda_b, lambda_mult, X_ref_hash, p_decimal_list,
        prior_str, h, convention, cbar_lower=None, cbar_upper=None, sigma=None,
        potential_info=None):
    plot_name = plot_obs_error_bands_filename(
        obs_indices_list[0], indep_var, ivar_start, ivar_stop, ivar_step,
        param_var, param, order_list, Lambda_b, lambda_mult, X_ref_hash, p_decimal_list,
        prior_str, h, convention, cbar_lower, cbar_upper, sigma,
        potential_info)
    name = re.sub("plot", "subplots", plot_name)

    obs_str = "_"
    for obs_indices in obs_indices_list:
        obs_str = obs_str + indices_to_short_observable_name(obs_indices) + "-"

    obs_str = obs_str[:-1] + "_"

    name = re.sub("_C_\w{1,3}-\w{1,3}-\w{1,3}-\w{1,3}_", obs_str, name)
    return name


def subplot_6_coefficients_filename(
        obs_indices_list, indep_var, ivar_start, ivar_stop,
        ivar_step, param_var, param, order_list, Lambda_b, lambda_mult, X_ref_hash,
        convention, potential_info=None):
    plot_name = plot_coeff_error_bands_filename(
        obs_indices_list[0], indep_var, ivar_start, ivar_stop, ivar_step,
        param_var, param, order_list, Lambda_b, lambda_mult, X_ref_hash, convention,
        potential_info)
    name = re.sub("plot", "subplots", plot_name)

    obs_str = "_"
    for obs_indices in obs_indices_list:
        obs_str = obs_str + indices_to_short_observable_name(obs_indices) + "-"

    obs_str = obs_str[:-1] + "_"

    name = re.sub("_C_\w{1,3}-\w{1,3}-\w{1,3}-\w{1,3}_", obs_str, name)
    return name


def indices_to_observable_name(observable_index_list):
    if observable_index_list == ['t', 't', 't', 't']:
        return r'$\sigma$'

    if observable_index_list == ['0', '0', '0', '0']:
        return r'$d\sigma/d\Omega$'

    if observable_index_list == ['0', '0', 'n', '0']:
        return r'$A_y$'

    if observable_index_list == ['n', '0', 'n', '0']:
        return r'$D$'

    if observable_index_list == ['sp', '0', 'k', '0']:
        return r'$A$'

    if observable_index_list == ['0', '0', 's', 's']:
        return r'$A_{xx}$'

    if observable_index_list == ['0', '0', 'n', 'n']:
        return r'$A_{yy}$'


def indices_to_short_observable_name(observable_index_list):
    if observable_index_list == ['t', 't', 't', 't']:
        return r'sig'

    if observable_index_list == ['0', '0', '0', '0']:
        return r'dsdO'

    if observable_index_list == ['0', '0', 'n', '0']:
        return r'Ay'

    if observable_index_list == ['n', '0', 'n', '0']:
        return r'D'

    if observable_index_list == ['sp', '0', 'k', '0']:
        return r'A'

    if observable_index_list == ['0', '0', 's', 's']:
        return r'Axx'

    if observable_index_list == ['0', '0', 'n', 'n']:
        return r'Ayy'

# obs_indices = ['0', 'x', '0', '0']
# indep_var = "theta"
# ivar_start = 0
# ivar_stop = 181
# ivar_step = 1
# param_var = 'energy'
# param = 0
# order = "LO"
# potential_info = None
# Lambda_b = 650
# X_ref_hash = "xxx"
# cbar_lower = 0.001
# cbar_upper = 1/cbar_lower
# sigma = 0
# p_decimal = .68
# prior_str = "B"
# order_list = ["LO", "NLO", "N2LO", "N3LO", "N4LO"]
# p_decimal_list = [0.68, 0.95]

# obsf = observable_filename(
#         obs_indices, indep_var, ivar_start, ivar_stop, ivar_step, param_var,
#         param, order, potential_info)
# cf = coeff_filename(
#     obs_indices, indep_var, ivar_start, ivar_stop,
#     ivar_step, param_var, param, order,
#     Lambda_b, X_ref_hash, potential_info)
# dobf = dob_filename(
#         obs_indices, indep_var, ivar_start, ivar_stop,
#         ivar_step, param_var, param, order,
#         Lambda_b, X_ref_hash,
#         p_decimal, prior_str,
#         cbar_lower, cbar_upper, sigma,
#         potential_info)
# plot_obsf = plot_obs_error_bands_filename(
#         obs_indices, indep_var, ivar_start, ivar_stop, ivar_step,
#         param_var, param, order_list, Lambda_b, X_ref_hash, p_decimal_list,
#         prior_str, cbar_lower, cbar_upper, sigma,
#         potential_info)
# plot_cfn = plot_coeff_error_bands_filename(
#         obs_indices, indep_var, ivar_start, ivar_stop,
#         ivar_step, param_var, param, order_list, Lambda_b, X_ref_hash,
#         potential_info)

# print(obsf)
# print(cf)
# print(dobf)
# print(plot_obsf)
# print(plot_cfn)
