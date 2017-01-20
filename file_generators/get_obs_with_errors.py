###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
# Created: Oct-9-2016
# Revised:
#
# Revision Log:
#
###############################################################################
#
#
###############################################################################
# Create files for plotting observables with associated error bands
###############################################################################

import argparse
import sys
import os
from math import nan
from data.observables import *
import datafile
# import CH_to_EKM_statistics as stat


# def coeff_file_name(Lambda_b, obs_filename):
#     name = "Coeffs_Lambda-" + str(Lambda_b) + "_" + obs_filename
#     return name


# def dob_file_name(p_decimal, Lambda_b, obs_filename):
#     """Return a standard filename for DOB files based on parameters.

#     Parameters
#     ----------
#     p_percent    = int
#                    The percent probability corresponding to the DOB interval.
#     obs_filename = str
#                    The filename of the observable for extracting DOBs
#     """
#     name = obs_filename.split("Coeffs_")[0]
#     name = "DOB_" + str(p_decimal) + "_" + name
#     return name


# def X0_dob_file_name(p_decimal_list, Lambda_b, obs_filename):
#     decimals = ""
#     for p in p_decimal_list:
#         decimals = decimals + str(p) + "-"
#     return "X0_DOB_" + decimals[:-1] + "_Lambda-" + str(Lambda_b) + "_" + obs_filename


# def Q(p, Lambda_b):
#     m_pi = 138  # Set to 0 to just return p/Lambda_b
#     return max(p, m_pi)/Lambda_b


def main(
         observ_dir,
         dk_dir,
         output_dir,
         indep_var,
         ivar_start,
         ivar_stop,
         ivar_step,
         param_list,
         observable_list,
         Lambda_b,
         p_decimal_list,
         k_list,
         # zero_list
         ):

    # This list is specific to chiral EFT
    # what should I do about it later?
    all_orders_list = ["LOp", "LO", "NLO", "N2LO", "N3LO", "N4LO"]
    order_list = [all_orders_list[k] for k in k_list]
    std_orders = ["LO", "LO", "NLO", "N2LO", "N3LO", "N4LO"]

    if indep_var == "theta":
        param_var = "energy"
        # theta_rad = ivar * deg_to_rad
        # E = param
        ivar_units = "(deg)"
        param_units = "(MeV)"
    else:
        param_var = "theta"
        # theta_rad = param * deg_to_rad
        # E = ivar
        ivar_units = "(MeV)"
        param_units = "(deg)"

    ivar_list = list(range(ivar_start, ivar_stop, ivar_step))
    for observable in observable_list:
        for param in param_list:
            obs_filename = "total_cross_section.dat"
            filename_for_coeffs = "c_n.txt"
            LO_filename = observable_filename(observable, indep_var,
                                              ivar_start, ivar_stop,
                                              ivar_step, param_var,
                                              param, "LO")
            # LO_filename = "A_total_cross_section_LO.txt"
            # N4LO_filename = "A_total_cross_section_N4LO.txt"
            LO_file = DataFile().read(os.path.join(observ_dir, LO_filename))

            N4LO_filename = observable_filename(observable, indep_var,
                                                ivar_start, ivar_stop,
                                                ivar_step, param_var,
                                                param, "N4LO")

            headers = [indep_var + " " + ivar_units, LO_filename[:9]]
            # headers = [indep_var + " " + ivar_units, "sigma (mb)"]
            for p in p_decimal_list:
                headers.append(str(p))

            for index, k in enumerate(k_list):
                k_order_filename = observable_filename(
                    observable, indep_var, ivar_start, ivar_stop, ivar_step,
                    param_var, param, std_orders[k]
                    )
                # k_order_filename = "A_total_cross_section_" + str(std_orders[k]) + ".txt"
                k_observ_file = DataFile().read(
                    os.path.join(observ_dir, k_order_filename)
                    )
                columns = [k_observ_file[i]
                           for i in range(len(k_observ_file.data[0]))]
                error_bands = [[] for p in p_decimal_list]
                for i, p in enumerate(p_decimal_list):
                    dk_filename = dob_file_name(p, Lambda_b, N4LO_filename)
                    dk_file = DataFile().read(os.path.join(dk_dir,
                                                           dk_filename))
                    columns.append(
                        [abs(X0 * dk) for X0, dk in zip(LO_file[1], dk_file[k+1])]
                        )
                final_file = DataFile().write(*zip(headers, columns))
                # export_k_order_filename = "A_total_cross_section_" + str(all_orders_list[k]) + ".txt"
                export_k_order_filename = observable_filename(
                    observable, indep_var, ivar_start, ivar_stop, ivar_step,
                    param_var, param, all_orders_list[k]
                    )
                final_file.export_to_file(
                    X0_dob_file_name(p_decimal_list, Lambda_b,
                                     export_k_order_filename)
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
        "observ_dir",
        help="The directory in which the observable files are stored.")
    parser.add_argument(
        "dk_dir",
        help="The directory in which the d_k DOB files are stored.")
    parser.add_argument(
        "output_dir",
        help="The relative path where output files will be stored")
    parser.add_argument(
        "indep_var",
        help="The variable ([deg] or [MeV]) that varies in a given data file.",
        choices=["theta", "energy"])
    parser.add_argument(
        "Lambda_b",
        help="The breakdown scale of the EFT, given in MeV.",
        type=int)
    parser.add_argument(
        "--p_decimal_list",
        help="The DOB percent values divided by 100.",
        type=float, nargs="+",
        required=True)
    parser.add_argument(
        "--k_list",
        help="The truncation orders at which to extract coefficients.",
        type=int, nargs="+",
        required=True)
    parser.add_argument(
        "--zero_list",
        help="The orders at which the coefficients are identically zero.",
        type=int, nargs="+",
        required=True)
    # parser.add_argument(
    #     "n_c",
    #     help="The number of nonzero coefficients.",
    #     type=int)
    parser.add_argument(
        "--indep_var_range", "--irange",
        type=int, nargs=3,
        metavar=("start", "stop", "step"),
        required=True,
        help="Cycle indep_var through [start, stop) in increments of step.")
    param_group = parser.add_mutually_exclusive_group(required=True)
    param_group.add_argument(
        "--param_values", "--pvals",
        type=int, nargs="+",
        help="""The value at which to hold the remaining variable
                (theta [deg] or energy [MeV]) in a given file.""")
    param_group.add_argument(
        "--param_range", "--prange",
        type=int, nargs=3,
        metavar=("start", "stop", "step"),
        help="Cycle param_value through [start, stop) in increments of step.")
    # parser.add_argument(
    #     "--orders",
    #     help="The order up to (and including) which to extract coefficients.",
    #     required=True, type=int,
    #     choices=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--observables",
        metavar="p,q,i,k",
        nargs="+", required=True,
        help="The observables C_{pqik} to calculate.",
        type=lambda s: s.split(","))

    args = parser.parse_args()
    arg_dict = vars(args)
    print(arg_dict)

    if arg_dict["param_range"] is not None:
        p0 = arg_dict["param_range"][0]
        pf = arg_dict["param_range"][1]
        ps = arg_dict["param_range"][2]
        param_lst = [i for i in range(p0, pf, ps)]
    else:
        param_lst = arg_dict["param_values"]

    main(
        observ_dir=arg_dict["observ_dir"],
        dk_dir=arg_dict["dk_dir"],
        output_dir=arg_dict["output_dir"],
        indep_var=arg_dict["indep_var"],
        ivar_start=arg_dict["indep_var_range"][0],
        ivar_stop=arg_dict["indep_var_range"][1],
        ivar_step=arg_dict["indep_var_range"][2],
        param_list=param_lst,
        observable_list=arg_dict["observables"],
        Lambda_b=arg_dict["Lambda_b"],
        p_decimal_list=arg_dict["p_decimal_list"],
        k_list=arg_dict["k_list"],
        # zero_list=arg_dict["zero_list"]
        )
