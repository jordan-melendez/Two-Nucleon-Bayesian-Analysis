###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
# Created: Aug-15-2016
# Revised:
#
# Revision Log:
#
###############################################################################
#
#
###############################################################################
# Find c_n coefficients based on order-by-order data
###############################################################################

import argparse
import sys
import os
from math import nan
from numpy import ones
from src.lowlevel.observables import *
import src.lowlevel.datafile
from src.lowlevel.kinematics import E_to_p
from src.lowlevel.EFT_functions import order_to_power, Q_ratio, coeffs, load_observable_files, load_observable_data, get_X_ref
from src.lowlevel.filenames import observable_filename, coeff_filename


def main(
         observable_dir,
         output_dir,
         indep_var,
         ivar_start,
         ivar_stop,
         ivar_step,
         param_list,
         orders,
         observable_list,
         Lambda_b,
         lambda_mult,
         interaction,
         X_ref_hash,
         convention):

    all_orders_list = ["LO", "NLO", "N2LO", "N3LO", "N4LO"]
    order_list = all_orders_list[:orders+1]
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

            files = load_observable_files(
                observable_dir, observable, indep_var, ivar_start,
                ivar_stop, ivar_step, param_var, param, order_list, convention)

            data = load_observable_data(*files)
            ind_variables = data[0]
            observable_arrays = data[1:]
            # Make a general function to get any X_ref.
            # X_ref = ones(len(observable_arrays[0])) * \
            #     get_natural_scale(ind_variables, observable_arrays[0])
            X_ref = get_X_ref(
                observable_dir, observable, indep_var, ivar_list,
                param_var, param, X_ref_hash, convention, lambda_mult*Lambda_b,
                *observable_arrays, X_NPWA_list=None)

            if indep_var == "theta":
                Q_val = Q_ratio(E_to_p(param, interaction), lambda_mult*Lambda_b)
            else:
                Q_val = Q_ratio(E_to_p(ind_variables, interaction), lambda_mult*Lambda_b)

            c_tuple = coeffs(Q_val, *observable_arrays, X_ref=X_ref)

            order_tuples = map(lambda ord: (order_list[ord], c_tuple[ord]),
                               range(len(order_list)))
            c_file = DataFile().write(
                (indep_var + " " + ivar_units, ind_variables),
                ("X_ref",  X_ref),
                *order_tuples)

            # obs_file = observable_filename(
            #     observable, indep_var, ivar_start, ivar_stop, ivar_step,
            #     param_var, param, order_list[-1])
            # final_file = coeff_filename(Lambda_b, obs_file, X_ref_hash)
            final_file = coeff_filename(
                observable, indep_var, ivar_start, ivar_stop,
                ivar_step, param_var, param, order_list[-1],
                Lambda_b, lambda_mult, X_ref_hash, convention,
                potential_info=None)
            c_file.export_to_file(os.path.join(output_dir, final_file))


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
        "observable_dir",
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
        "lambda_mult",
        help="The lambda value that multiplies Lambda_b.",
        type=float)
    parser.add_argument(
        "interaction",
        help="The type of scattering interaction.",
        choices=["nn", "pp", "np"])
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
    parser.add_argument(
        "--orders",
        help="The order up to (and including) which to extract coefficients.",
        required=True, type=int,
        choices=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--observables",
        metavar="p,q,i,k",
        nargs="+", required=True,
        help="The observables C_{pqik} to calculate.",
        type=lambda s: s.split(","))
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

    if arg_dict["param_range"] is not None:
        p0 = arg_dict["param_range"][0]
        pf = arg_dict["param_range"][1]
        ps = arg_dict["param_range"][2]
        param_lst = [i for i in range(p0, pf, ps)]
    else:
        param_lst = arg_dict["param_values"]

    main(
        observable_dir=arg_dict["observable_dir"],
        output_dir=arg_dict["output_dir"],
        indep_var=arg_dict["indep_var"],
        ivar_start=arg_dict["indep_var_range"][0],
        ivar_stop=arg_dict["indep_var_range"][1],
        ivar_step=arg_dict["indep_var_range"][2],
        param_list=param_lst,
        orders=arg_dict["orders"],
        observable_list=arg_dict["observables"],
        Lambda_b=arg_dict["Lambda_b"],
        lambda_mult=arg_dict["lambda_mult"],
        interaction=arg_dict["interaction"],
        X_ref_hash=arg_dict["X_ref_hash"],
        convention=arg_dict["convention"])
