###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
###############################################################################
# Create file with observable and DOB error bands
###############################################################################

import argparse
import sys
import os
from math import nan
from numpy import absolute
from scipy.interpolate import interp1d
from scipy.integrate import quad
from src.lowlevel.observables import *
import src.lowlevel.datafile
from src.lowlevel.kinematics import E_to_p
from src.lowlevel.EFT_functions import order_to_power, Q_ratio, coeffs, load_observable_files, load_observable_data, get_X_ref
from src.lowlevel.filenames import observable_filename, coeff_filename, dob_filename, Lambda_pdf_filename
from src.lowlevel.CH_to_EKM_statistics import cbark, dkp_A_eps, n_c_val, Delta_k_posterior, find_insignificant_x, find_dimensionless_dob_limit, Lambda_b_pdf, trapezoid_integ_rule


def main(
         coeff_dir,
         output_dir,
         # indep_var,
         # ivar_start,
         # ivar_stop,
         # ivar_step,
         theta_list,
         energy_list,
         # orders,
         ignore_orders,
         observable_list,
         interaction,
         X_ref_hash,
         prior_set,
         cbar_lower,
         cbar_upper,
         sigma,
         Lambda_prior,
         Lambda_lower,
         Lambda_upper,
         Lambda_mu,
         Lambda_sigma,
         convention):

    # all_orders_list = ["LOp", "LO", "NLO", "N2LO", "N3LO", "N4LO"]
    std_orders_list = ["LO", "NLO", "N2LO", "N3LO", "N4LO"]
    # order_list = std_orders_list[:orders+1]
    k_dict = {"LOp": 0, "LO": 1, "NLO": 2, "N2LO": 3, "N3LO": 4, "N4LO": 5}
    order_dict = {0: "LOp", 1: "LO", 2: "NLO", 3: "N2LO", 4: "N3LO", 5: "N4LO"}
    # k_list = [k_dict[order] for order in orders
    #           if k_dict[order] not in ignore_orders]

    k_list = []
    k = 0
    while k in ignore_orders:
        k += 1
    while k <= k_dict["N4LO"]:
        while k+1 in ignore_orders:
            k += 1
        k_list.append(k)
        k += 1

    # order_list = [order_dict[k] for k in k_list]
    # if indep_var == "theta":
    #     param_var = "energy"
    #     # theta_rad = ivar * deg_to_rad
    #     # E = param
    #     ivar_units = "(deg)"
    #     param_units = "(MeV)"
    # else:
    #     param_var = "theta"
    #     # theta_rad = param * deg_to_rad
    #     # E = ivar
    #     ivar_units = "(MeV)"
    #     param_units = "(deg)"
    # ivar_list = list(range(ivar_start, ivar_stop, ivar_step))

    std_e_start = 1
    std_e_end = 351
    std_e_step = 1
    std_t_start = 0
    std_t_end = 181
    std_t_step = 1

    theta_temp = theta_list
    # k_list = [5]
    # Each k will be in a separate plot
    for k in k_list:
        print(order_dict[k])
        # But everything else will be combined
        b_coeff_array = []

        for observable in observable_list:
            if observable == ['t', 't', 't', 't']:
                theta_list = [0]
            else:
                theta_list = theta_temp

            for theta in theta_list:

                filename = coeff_filename(
                    obs_indices=observable,
                    indep_var="energy",
                    ivar_start=std_e_start,
                    ivar_stop=std_e_end,
                    ivar_step=std_e_step,
                    param_var="theta",
                    param=theta,
                    order="N4LO",
                    Lambda_b=1,
                    lambda_mult=1.0,
                    X_ref_hash=X_ref_hash,
                    convention=convention,
                    potential_info=None)

                coeff_file = DataFile().read(os.path.join(coeff_dir, filename))

                # Lambda_filename = Lambda_pdf_filename(
                #     obs_indices_list=observable_list,
                #     theta_list=theta_list,
                #     energy_list=energy_list,
                #     order=order_dict[k],
                #     ignore_orders=ignore_orders,
                #     X_ref_hash=X_ref_hash,
                #     prior_str=prior_set,
                #     convention=convention,
                #     cbar_lower=cbar_lower,
                #     cbar_upper=cbar_upper,
                #     sigma=sigma,
                #     Lambda_lower=Lambda_lower,
                #     Lambda_upper=Lambda_upper,
                #     potential_info=None)
                # print(Lambda_filename)
                # print(
                #     "obs_indices=", observable,
                #     "theta_list=", theta_list,
                #     "energy_list=", energy_list,
                #     "order=", order_dict[k],
                #     "ignore_orders=", ignore_orders,
                #     "X_ref_hash=", X_ref_hash,
                #     "prior_str=", prior_set,
                #     "convention=", convention,
                #     "cbar_lower=", cbar_lower,
                #     "cbar_upper=", cbar_upper,
                #     "sigma=", sigma,
                #     "Lambda_lower=", Lambda_lower,
                #     "Lambda_upper=", Lambda_upper,
                #     "potential_info=", None)

                for energy in energy_list:

                    energy_index = int((energy - std_e_start)/std_e_step)

                    temp_coeff_list = coeff_file[energy_index, 2:]
                    temp_coeff_list.insert(1, 0.0)

                    b_coeff_array.append(temp_coeff_list)

        if Lambda_prior == "u" or Lambda_prior == "uu":
            Lambda_lower_range = Lambda_lower
            Lambda_upper_range = Lambda_upper
        elif Lambda_prior == "g":
            Lambda_lower_range = 1
            Lambda_upper_range = Lambda_mu + 3*Lambda_sigma
        Lambda_domain = np.arange(Lambda_lower_range, Lambda_upper_range+1)
        Lambda_list = array([1.0 for Lb in Lambda_domain])
        for coeffs in b_coeff_array:
            print(coeffs)
            # print(
            #     prior_set, k, coeffs, Lambda_prior, Lambda_lower, Lambda_upper,
            #     Lambda_mu, Lambda_sigma, cbar_lower, cbar_upper, sigma)
            Lambda_b_posterior = Lambda_b_pdf(
                prior_set, k, coeffs, Lambda_prior, Lambda_lower, Lambda_upper,
                Lambda_mu, Lambda_sigma, cbar_lower, cbar_upper, sigma,
                include_lambda_prior=False)
            # Lambda_b_posterior(500)
            Lambda_list *= array([Lambda_b_posterior(Lb) for Lb in Lambda_domain])
            Lambda_list /= max(Lambda_list)

        Lambda_list *= array([Lambda_b_posterior.lambda_prior_func(Lb) for Lb in Lambda_domain])
        # print(Lambda_list)
        Lamb_pdf = interp1d(Lambda_domain, Lambda_list, kind="linear")
        # norm = 1/quad(Lamb_pdf, Lambda_lower_range, Lambda_upper_range)[0]
        norm = 1/trapezoid_integ_rule(Lamb_pdf, Lambda_lower_range,
                                      Lambda_upper_range,
                                      N=int(Lambda_upper_range-Lambda_lower_range))
        Lambda_list *= norm

        Lambda_file = DataFile().write(("Lambda_b (MeV)", Lambda_domain), ("pdf (1/MeV)", Lambda_list))
        Lambda_filename = Lambda_pdf_filename(
            obs_indices_list=observable_list,
            theta_list=theta_list,
            energy_list=energy_list,
            order=order_dict[k],
            ignore_orders=ignore_orders,
            X_ref_hash=X_ref_hash,
            prior_str=prior_set,
            convention=convention,
            cbar_lower=cbar_lower,
            cbar_upper=cbar_upper,
            sigma=sigma,
            Lambda_prior=Lambda_prior,
            Lambda_lower=Lambda_lower,
            Lambda_upper=Lambda_upper,
            Lambda_mu=Lambda_mu,
            Lambda_sigma=Lambda_sigma,
            potential_info=None)
        Lambda_file.export_to_file(
            os.path.join(output_dir, Lambda_filename), is_scientific=True
            )

if __name__ == "__main__":
    ###########################################
    # Start args for running from command line
    ###########################################
    # For help:
    # >> python get_coefficients.py -h
    parser = argparse.ArgumentParser(
        description="Executable script to extract np observable coefficients."
        )
    parser.add_argument(
        "coeff_dir",
        help="The directory in which the coefficients are stored.")
    parser.add_argument(
        "output_dir",
        help="The relative path where output files will be stored")
    parser.add_argument(
        "interaction",
        help="The type of scattering interaction.",
        choices=["nn", "pp", "np"])
    parser.add_argument(
        "prior_set",
        help="The string corresponding to a given prior set.",
        choices=["A", "B", "C"])
    parser.add_argument(
        "cbar_lower",
        help="Lower bound for cbar on sets A and C.",
        type=float)
    parser.add_argument(
        "cbar_upper",
        help="Upper bound for cbar on sets A and C.",
        type=float)
    parser.add_argument(
        "sigma",
        help="Standard deviation for cbar on set B.",
        type=float)
    parser.add_argument(
        "Lambda_prior_set",
        help="The string corresponding to a given Lambda_b prior set.",
        choices=["u", "uu", "g"])
    parser.add_argument(
        "Lambda_lower",
        help="For Lambda prior set u: Lower bound for Lambda_b.",
        type=float)
    parser.add_argument(
        "Lambda_upper",
        help="For Lambda prior set u: Upper bound for Lambda_b.",
        type=float)
    parser.add_argument(
        "Lambda_mu",
        help="For Lambda prior set g: mean of Lambda_b.",
        type=float)
    parser.add_argument(
        "Lambda_sigma",
        help="For Lambda prior set g: standard deviation of Lambda_b.",
        type=float)
    theta_group = parser.add_mutually_exclusive_group(required=True)
    theta_group.add_argument(
        "--theta_range", "--trange",
        type=int, nargs=3,
        metavar=("start", "stop", "step"),
        # required=True,
        help="Cycle theta through [start, stop) in increments of step.")
    theta_group.add_argument(
        "--theta_values", "--tvals",
        help="The values of the indep var at which to find error bands.",
        type=int, nargs="+")
    energy_group = parser.add_mutually_exclusive_group(required=True)
    energy_group.add_argument(
        "--energy_values", "--evals",
        type=int, nargs="+",
        help="""The values of energy to use in Lambda_b pdf.""")
    energy_group.add_argument(
        "--energy_range",
        type=int, nargs=3,
        metavar=("start", "stop", "step"),
        help="Cycle energy through [start, stop) in increments of step to use in pdf.")
    # parser.add_argument(
    #     "--orders",
    #     help="The order up to (and including) which to extract coefficients.",
    #     required=True, type=int,
    #     choices=[0, 1, 2, 3, 4])
    # parser.add_argument(
    #     "--orders",
    #     help="The orders at which to calculate DoBs.",
    #     nargs="+", required=True,
    #     choices=["NLO", "N2LO", "N3LO", "N4LO"])
    parser.add_argument(
        "--ignore_orders",
        help="The kth orders (Q^k) to ignore when calculating DoBs.",
        nargs="+", type=int,
        choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument(
        "--observables",
        metavar="p,q,i,k",
        nargs="+", required=True,
        help="The observables C_{pqik} to calculate.",
        type=lambda s: s.split(","))
    # p_decimals_group = parser.add_mutually_exclusive_group(required=True)
    # p_decimals_group.add_argument(
    #     "--p_decimals",
    #     help="The DOB percent divided by 100.",
    #     type=float, nargs="+")
    # p_decimals_group.add_argument(
    #     "--p_range",
    #     type=int, nargs=3,
    #     metavar=("p_start", "p_stop", "p_step"),
    #     help="Cycle p (%) through [p_start, p_stop) in increments of p_step."
    #     )
    parser.add_argument(
        "--X_ref_hash",
        required=True,
        help="""The way X_ref should be calculated.
            """,
        type=str
        )
    parser.add_argument(
        "--convention",
        required=True,
        help="The Stapp or Blatt phase convention.",
        choices=["stapp", "blatt"])

    args = parser.parse_args()
    arg_dict = vars(args)
    print(arg_dict)

    if arg_dict["energy_range"] is not None:
        e0 = arg_dict["energy_range"][0]
        ef = arg_dict["energy_range"][1]
        es = arg_dict["energy_range"][2]
        energy_lst = [i for i in range(e0, ef, es)]
    else:
        energy_lst = arg_dict["energy_values"]

    if arg_dict["theta_range"] is not None:
        t0 = arg_dict["theta_range"][0]
        tf = arg_dict["theta_range"][1]
        ts = arg_dict["theta_range"][2]
        theta_lst = [i for i in range(t0, tf, ts)]
    else:
        theta_lst = arg_dict["theta_values"]

    if arg_dict["prior_set"] == "B":
        cup = 0
        clow = 0
        sigma = arg_dict["sigma"]
    else:
        sigma = 0
        cup = arg_dict["cbar_upper"]
        clow = arg_dict["cbar_lower"]

    if arg_dict["ignore_orders"] is None:
        ignore_orders = []
    else:
        ignore_orders = arg_dict["ignore_orders"]

    if arg_dict["Lambda_prior_set"] == "u" or arg_dict["Lambda_prior_set"] == "uu":
        Lmu = 0
        Lsig = 0
        Ll = arg_dict["Lambda_lower"]
        Lu = arg_dict["Lambda_upper"]
    elif arg_dict["Lambda_prior_set"] == "g":
        Lmu = arg_dict["Lambda_mu"]
        Lsig = arg_dict["Lambda_sigma"]
        Ll = 0
        Lu = 0

    # Lmu = arg_dict["Lambda_mu"]
    # Lsig = arg_dict["Lambda_sigma"]
    # Ll = arg_dict["Lambda_lower"]
    # Lu = arg_dict["Lambda_upper"]

    main(
        coeff_dir=arg_dict["coeff_dir"],
        output_dir=arg_dict["output_dir"],
        # indep_var=arg_dict["indep_var"],
        # ivar_start=arg_dict["indep_var_range"][0],
        # ivar_stop=arg_dict["indep_var_range"][1],
        # ivar_step=arg_dict["indep_var_range"][2],
        theta_list=theta_lst,
        energy_list=energy_lst,
        # orders=arg_dict["orders"],
        ignore_orders=ignore_orders,
        observable_list=arg_dict["observables"],
        interaction=arg_dict["interaction"],
        # p_decimal_list=p_grid,
        X_ref_hash=arg_dict["X_ref_hash"],
        prior_set=arg_dict["prior_set"],
        cbar_lower=clow,
        cbar_upper=cup,
        sigma=sigma,
        Lambda_prior=arg_dict["Lambda_prior_set"],
        Lambda_lower=Ll,
        Lambda_upper=Lu,
        Lambda_mu=Lmu,
        Lambda_sigma=Lsig,
        convention=arg_dict["convention"])
