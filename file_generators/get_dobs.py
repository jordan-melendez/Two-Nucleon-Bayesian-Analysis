###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
# Created: Oct-6-2016
# Revised:
#
# Revision Log:
#
###############################################################################
#
#
###############################################################################
# Find extract DOBs from coefficient files
###############################################################################

import argparse
import sys
import os
from math import nan
from data.observables import *
import datafile
import CH_to_EKM_statistics as stat


def main(
         coeff_dir,
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
         zero_list,
         interaction):

    # This list is specific to chiral EFT
    # what should I do about it later?
    all_orders_list = ["LOp", "LO", "NLO", "N2LO", "N3LO", "N4LO"]
    order_list = [all_orders_list[k] for k in k_list]

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
    for p_decimal in p_decimal_list:
        for observable in observable_list:
            for param in param_list:
                # obs_filename = "total_cross_section.dat"
                # filename_for_coeffs = "c_n.txt"
                obs_filename = observable_filename(observable, indep_var,
                                                   ivar_start, ivar_stop,
                                                   ivar_step, param_var,
                                                   param, order_list[-1])
                filename_for_coeffs = coeff_file_name(Lambda_b,
                                                      obs_filename)
                d_sub_k_matrix = []
                with open(os.path.join(coeff_dir, filename_for_coeffs)) as f:
                    for line in f:
                        if line[0] != "#":
                            # Assumes coefficients start in the third column
                            # i.e. only one column for parameter
                            coeff_list = list(map(float, line.split()[2:]))
                            d_sub_k_line = [int(line.split()[0])]

                            if indep_var == "theta":
                                E = param
                                theta = float(line.split()[0])
                            else:
                                E = float(line.split()[0])
                                theta = param

                            p = E_to_p(E, interaction)
                            for index, k in enumerate(k_list):
                                # Right now assuming k=1 term is zero
                                # relax this later
                                c_bar_k = stat.cbark(
                                    *coeff_list[:stat.n_c_val(k, zero_list)]
                                    )
                                # print(c_bar_k, Q(p, Lambda_b), k)
                                d_sub_k_line.append(
                                    stat.dkp_A_eps(
                                        Q(p, Lambda_b), k,
                                        stat.n_c_val(k, zero_list),
                                        p_decimal, c_bar_k
                                        )
                                    )
                            d_sub_k_matrix.append(d_sub_k_line)
                # print(d_sub_k_matrix)

                dob_name = os.path.join(
                    output_dir,
                    dob_file_name(p_decimal, Lambda_b, obs_filename)
                    )
                dob_file = DataFile()
                dob_file.data = d_sub_k_matrix
                secs = [indep_var + " " + ivar_units, *[str(k) for k in k_list]]
                dob_file.sections = secs

                dob_file.export_to_file(dob_name)

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
        help="The directory in which the observables are stored.")
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
        "interaction",
        help="The type of scattering interaction.",
        choices=["nn", "pp", "np"])
    parser.add_argument(
        "--p_decimals",
        help="The DOB percent divided by 100.",
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
        coeff_dir=arg_dict["coeff_dir"],
        output_dir=arg_dict["output_dir"],
        indep_var=arg_dict["indep_var"],
        ivar_start=arg_dict["indep_var_range"][0],
        ivar_stop=arg_dict["indep_var_range"][1],
        ivar_step=arg_dict["indep_var_range"][2],
        param_list=param_lst,
        # orders=arg_dict["orders"],
        observable_list=arg_dict["observables"],
        Lambda_b=arg_dict["Lambda_b"],
        p_decimal_list=arg_dict["p_decimals"],
        k_list=arg_dict["k_list"],
        zero_list=arg_dict["zero_list"],
        interaction=arg_dict["interaction"])
