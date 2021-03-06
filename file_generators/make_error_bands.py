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
from src.lowlevel.observables import *
import src.lowlevel.datafile
from src.lowlevel.kinematics import E_to_p
from src.lowlevel.EFT_functions import order_to_power, Q_ratio, coeffs, load_observable_files, load_observable_data, get_X_ref
from src.lowlevel.filenames import observable_filename, coeff_filename, dob_filename
from src.lowlevel.CH_to_EKM_statistics import cbark, dkp_A_eps, n_c_val, Delta_k_posterior, find_insignificant_x, find_dimensionless_dob_limit


def main(
         observable_dir,
         output_dir,
         indep_var,
         ivar_start,
         ivar_stop,
         ivar_step,
         indep_var_list,
         param_list,
         orders,
         ignore_orders,
         observable_list,
         Lambda_b,
         lambda_mult,
         p_decimal_list,
         interaction,
         X_ref_hash,
         prior_set,
         h,
         cbar_lower,
         cbar_upper,
         sigma,
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
    while k <= k_dict[orders[-1]]:
        while k+1 in ignore_orders:
            k += 1
        k_list.append(k)
        k += 1

    # order_list = [order_dict[k] for k in k_list]
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
                ivar_stop, ivar_step, param_var, param, std_orders_list, convention)

            data = load_observable_data(*files)
            ind_variables = data[0]
            observable_arrays = list(data[1:])
            # X_ref = get_natural_scale(ind_variables, observable_arrays[1])
            X_ref = get_X_ref(
                observable_dir, observable, indep_var, ivar_list,
                param_var, param, X_ref_hash, convention, lambda_mult*Lambda_b,
                *observable_arrays, X_NPWA_list=None)

            if indep_var_list is not None:
                indep_var_list_indices = [int((ivar-ivar_start)/ivar_step)
                                          for ivar in indep_var_list]
                obs_temp = [[] for obs in observable_arrays]
                for index, obs in enumerate(observable_arrays):
                    obs_temp[index] = array([obs[i] for i in indep_var_list_indices])
                observable_arrays = list(obs_temp)
                ind_variables = array(indep_var_list)
                X_ref = array([X_ref[i] for i in indep_var_list_indices])

            if indep_var == "theta":
                Q_val = Q_ratio(E_to_p(param, interaction), lambda_mult*Lambda_b)
            else:
                Q_val = Q_ratio(E_to_p(ind_variables, interaction), lambda_mult*Lambda_b)

            # Extract coeffs under assumption that Q^1 isn't in observable_arrays
            c_tuple_lists = list(coeffs(Q_val, *observable_arrays, X_ref=X_ref))
            # Make LO and LOp the same, since Q^1 doesn't contribute
            observable_arrays = [observable_arrays[0]] + observable_arrays
            c_tuple_lists.insert(1, zeros(len(ind_variables)))

            # Now remove orders we don't care about
            # Yeah, this will remove what was just added if 1 is ignored.
            for ig_ord in reversed(ignore_orders):
                c_tuple_lists.pop(ig_ord)
                observable_arrays.pop(ig_ord)

            # if indep_var_list is not None:
            #     indep_var_list_indices = [int((ivar-ivar_start)/ivar_step)
            #                               for ivar in indep_var_list]
            #     c_temp_tuple = [[] for c in c_tuple_lists]
            #     for index, c in enumerate(c_tuple_lists):
            #         c_temp_tuple[index] = [c[i] for i in indep_var_list_indices]
            #     c_tuple_lists = tuple(c_temp_tuple)
            #     ind_variables = indep_var_list

            # print(c_tuple[0][199], c_tuple[1][199], c_tuple[2][199], c_tuple[3][199], c_tuple[4][199])
            # print(cbark(*c_tuple))

            # If these conditions are met, a formula can be used instead of
            # VEGAS to find DoBs
            is_easy_dob = prior_set == "A" and h == 1 and \
                cbar_lower <= 0.001 and cbar_upper >= 1000

            if not is_easy_dob:
                print(observable, prior_set, param)
                pass

            for k in k_list:
                order = order_dict[k]
                nc = n_c_val(k, ignore_orders)
                c_bar_k = cbark(*c_tuple_lists[:nc])

                if is_easy_dob:
                    for p in p_decimal_list:
                        dk = dkp_A_eps(Q_val, k, nc, p, c_bar_k)

                        dob_name = dob_filename(
                            observable, indep_var, ivar_start, ivar_stop,
                            ivar_step, param_var, param, order, ignore_orders,
                            Lambda_b, lambda_mult, X_ref_hash,
                            p, prior_set, h, convention,
                            indep_var_list, cbar_lower, cbar_upper, sigma,
                            potential_info=None)
                        dob_file = DataFile().write(
                            (indep_var + " " + ivar_units, ind_variables),
                            ("Observable", observable_arrays[nc-1]),
                            ("Lower Bound", observable_arrays[nc-1] - absolute(X_ref*dk)),
                            ("Upper Bound", observable_arrays[nc-1] + absolute(X_ref*dk)))
                        dob_file.export_to_file(os.path.join(output_dir, dob_name))
                else:
                    dk = [[] for p in p_decimal_list]
                    for c_index, c_tuple in enumerate(zip(*c_tuple_lists)):
                        try:
                            qq = Q_val[c_index]
                        except IndexError:
                            qq = Q_val
                        # print(prior_set, observable, param, k,
                        #       array(c_tuple[:nc]), qq)
                        pr_deltak = Delta_k_posterior(
                            prior_set=prior_set, Q=qq, k=k, nc=nc,
                            h=h, coeffs=array(c_tuple), cbar_lower=cbar_lower,
                            cbar_upper=cbar_upper, sigma=sigma)
                        try:
                            # x_crit = find_insignificant_x(pr_deltak)
                            # print(qq, c_bar_k)
                            c_crit = c_bar_k[c_index]
                            if prior_set == "B":
                                if c_crit < np.exp(sigma)/10:
                                    c_crit = np.exp(sigma)/10
                                elif c_crit > np.exp(sigma)*10:
                                    c_crit = np.exp(sigma)*10
                            else:
                                if c_crit < cbar_lower:
                                    c_crit = cbar_lower
                                elif c_crit > cbar_upper:
                                    c_crit = cbar_upper

                            x_crit = 2 * c_crit * qq**(k+1)
                        except ZeroDivisionError:
                            print("x_crit could not be found!\n",
                                  "Prior:", prior_set, "\n",
                                  observable, "\n",
                                  "Param:", param, "\n",
                                  "k:", k, "\n",
                                  "n_c:", nc, "\n",
                                  "clist:", array(c_tuple), "\n",
                                  "Q:", qq
                                  )
                            print("x_crit:", x_crit)

                        for index, p in enumerate(p_decimal_list):
                            # try:
                            #     # print(x_crit/100, p)
                            #     dob = find_dimensionless_dob_limit(pr_deltak, x_mode=0, delta_x=x_crit/100, dob=p)
                            # except RuntimeWarning:
                            #     print(observable, prior_set, param, k, nc, array(c_tuple), qq)
                            #     print(x_crit/100, p)
                                # dob = find_dimensionless_dob_limit(pr_deltak, x_mode=0, delta_x=x_crit/100, dob=p)

                            dob = find_dimensionless_dob_limit(pr_deltak, x_mode=0, delta_x=x_crit/100, dob=p)
                            dk[index].append(dob)

                    for ind, p in enumerate(p_decimal_list):
                        dob_name = dob_filename(
                            observable, indep_var, ivar_start, ivar_stop,
                            ivar_step, param_var, param, order, ignore_orders,
                            Lambda_b, lambda_mult, X_ref_hash,
                            p, prior_set, h, convention,
                            indep_var_list, cbar_lower, cbar_upper, sigma,
                            potential_info=None)
                        dob_file = DataFile().write(
                            (indep_var + " " + ivar_units, ind_variables),
                            ("Observable",
                             observable_arrays[nc-1]),
                            ("Lower Bound",
                             observable_arrays[nc-1] - absolute(X_ref*dk[ind])
                             ),
                            ("Upper Bound",
                             observable_arrays[nc-1] + absolute(X_ref*dk[ind])
                             )
                        )
                        dob_file.export_to_file(
                            os.path.join(output_dir, dob_name)
                        )
                    # print(dk[199])
                    # input("stop.")
                    # obs_name = observable_filename(
                    #     observable, indep_var, ivar_start, ivar_stop,
                    #     ivar_step, param_var, param, all_orders_list[k])
                    

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
        "prior_set",
        help="The string corresponding to a given prior set.",
        choices=["A", "B", "C"])
    parser.add_argument(
        "h",
        help="The number of coefficients that contribute to \Delta_k.",
        type=int)
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
        "--indep_var_range", "--irange",
        type=int, nargs=3,
        metavar=("start", "stop", "step"),
        required=True,
        help="Cycle indep_var through [start, stop) in increments of step. If indep_var_list is given, this is still used to find the correct observable files.")
    parser.add_argument(
        "--indep_var_list",
        help="The values of the indep var at which to find error bands. If none, use the indep_var range",
        type=int, nargs="+")
    param_group = parser.add_mutually_exclusive_group(required=True)
    param_group.add_argument(
        "--param_values", "--pvals",
        type=int, nargs="+",
        help="""The value at which to hold the remaining variable
                (theta [deg] or energy [MeV]) in a given file.""")
    param_group.add_argument(
        "--param_range",
        type=int, nargs=3,
        metavar=("start", "stop", "step"),
        help="Cycle param_value through [start, stop) in increments of step.")
    # parser.add_argument(
    #     "--orders",
    #     help="The order up to (and including) which to extract coefficients.",
    #     required=True, type=int,
    #     choices=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--orders",
        help="The orders at which to calculate DoBs.",
        nargs="+", required=True,
        choices=["LO", "NLO", "N2LO", "N3LO", "N4LO"])
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
    p_decimals_group = parser.add_mutually_exclusive_group(required=True)
    p_decimals_group.add_argument(
        "--p_decimals",
        help="The DOB percent divided by 100.",
        type=float, nargs="+")
    p_decimals_group.add_argument(
        "--p_range",
        type=int, nargs=3,
        metavar=("p_start", "p_stop", "p_step"),
        help="Cycle p (%) through [p_start, p_stop) in increments of p_step."
        )
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

    if arg_dict["p_range"] is not None:
        p0 = arg_dict["p_range"][0]
        pf = arg_dict["p_range"][1]
        ps = arg_dict["p_range"][2]
        p_grid = [p/100 for p in range(p0, pf, ps)]
    else:
        p_grid = arg_dict["p_decimals"]

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
        observable_dir=arg_dict["observable_dir"],
        output_dir=arg_dict["output_dir"],
        indep_var=arg_dict["indep_var"],
        ivar_start=arg_dict["indep_var_range"][0],
        ivar_stop=arg_dict["indep_var_range"][1],
        ivar_step=arg_dict["indep_var_range"][2],
        indep_var_list=arg_dict["indep_var_list"],
        param_list=param_lst,
        orders=arg_dict["orders"],
        ignore_orders=ignore_orders,
        observable_list=arg_dict["observables"],
        Lambda_b=arg_dict["Lambda_b"],
        lambda_mult=arg_dict["lambda_mult"],
        interaction=arg_dict["interaction"],
        p_decimal_list=p_grid,
        X_ref_hash=arg_dict["X_ref_hash"],
        prior_set=arg_dict["prior_set"],
        h=arg_dict["h"],
        cbar_lower=clow,
        cbar_upper=cup,
        sigma=sigma,
        convention=arg_dict["convention"])
