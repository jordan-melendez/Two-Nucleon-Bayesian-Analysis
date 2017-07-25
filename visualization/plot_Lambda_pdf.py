###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
###############################################################################
# 
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
from src.lowlevel.filenames import observable_filename, coeff_filename, dob_filename, Lambda_pdf_filename, plot_Lambda_pdf_filename
from src.lowlevel.CH_to_EKM_statistics import cbark, dkp_A_eps, n_c_val, Delta_k_posterior, find_insignificant_x, find_dimensionless_dob_limit, Lambda_b_pdf
from matplotlib import rcParams, rc
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt

rcParams['ps.distiller.res'] = 60000
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def main(
         data_dir,
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
         Lambda_lower,
         Lambda_upper,
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

    # std_e_start = 1
    # std_e_end = 351
    # std_e_step = 1
    # std_t_start = 0
    # std_t_end = 181
    # std_t_step = 1

    fig = plt.figure(figsize=(3.4, 3.4))
    ax = fig.add_subplot(1, 1, 1)

    # Lambda_domain = np.arange(0, 1600)
    # plt.xlim([0, 1600])

    # Each k will be in a separate plot
    for k in k_list:
        print(order_dict[k])
        # But everything else will be combined

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
            Lambda_lower=Lambda_lower,
            Lambda_upper=Lambda_upper,
            potential_info=None)

        Lambda_file = DataFile().read(os.path.join(data_dir, Lambda_filename))
        Lambda_vals = Lambda_file[0]
        Lambda_pdf = Lambda_file[1]

        plt.xlim([0, 1600])
        # major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 1600, 300)
        minor_ticks = np.arange(0, 1600, 100)

        ax.set_xlabel(r"$\Lambda_b$\,(MeV)", fontsize=10)

        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(major_ticks)
        plt.plot(Lambda_vals, Lambda_pdf, '-', color="purple", label=r"pr$(\Lambda_b|b_2,\dots,b_k)$")

        Lambda_func = interp1d(Lambda_vals, Lambda_pdf)
        fill_color = "blue"
        lower95 = find_dimensionless_dob_limit(
            Lambda_func, x_mode=Lambda_lower, dob=2*.025, delta_x=10)
        upper95 = find_dimensionless_dob_limit(
            Lambda_func, x_mode=Lambda_lower, dob=2*.975, delta_x=10)
        dob95 = np.arange(lower95, upper95)
        label95 = r"95\%: $[{:.0f},{:.0f}]$\,MeV".format(lower95, upper95)
        # fill95 = '#7daaf2'
        fill95 = '#73C6B6'
        plt.fill_between(dob95, 0, Lambda_func(dob95),
                        facecolor=fill95, color=fill95, alpha=.3, label=label95)
        lower68 = find_dimensionless_dob_limit(
            Lambda_func, x_mode=Lambda_lower, dob=2*.16, delta_x=10)
        upper68 = find_dimensionless_dob_limit(
            Lambda_func, x_mode=Lambda_lower, dob=2*.84, delta_x=10)
        dob68 = np.arange(lower68, upper68)
        label68 = r"68\%: $[{:.0f},{:.0f}]$\,MeV".format(lower68, upper68)
        fill68 = '#68a0f9'
        fill68 = '#45B39D'
        ax.fill_between(dob68, 0, Lambda_func(dob68),
                        facecolor=fill68, color=fill68, alpha=1, label=label68)
        median = find_dimensionless_dob_limit(
            Lambda_func, x_mode=Lambda_lower, dob=2*.5, delta_x=10)
        labelmedian = r"Median = {:4.0f}\,MeV".format(median)
        ax.vlines(median, [0], Lambda_func(median), color="#16A085", label=labelmedian)
        plt.axvline(x=Lambda_lower, color="orange", ls="--")
        plt.axvline(x=Lambda_upper, color="orange", ls="--")
        # plt.legend(fontsize=10)

        ax.legend(fontsize=8)

        plot_name = plot_Lambda_pdf_filename(
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
            Lambda_lower=Lambda_lower,
            Lambda_upper=Lambda_upper,
            potential_info=None)
        plt.draw()
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")

        plt.cla()

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
        "data_dir",
        help="The directory in which the Lambda_b pdf data are stored.")
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
        "Lambda_lower",
        help="Lower bound for Lambda_b.",
        type=float)
    parser.add_argument(
        "Lambda_upper",
        help="Upper bound for Lambda_b.",
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

    main(
        data_dir=arg_dict["data_dir"],
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
        Lambda_lower=arg_dict["Lambda_lower"],
        Lambda_upper=arg_dict["Lambda_upper"],
        convention=arg_dict["convention"])
