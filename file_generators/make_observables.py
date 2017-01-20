###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
# Created: Aug-31-2016
# Revised:
#
# Revision Log:
#
###############################################################################
#
#
###############################################################################
# Gather phases and generate observables.
###############################################################################

import argparse
import sys
import os
from numpy import array
import time
# sys.path.append(os.path.expanduser(
#     '~/Dropbox/Bayesian_codes/observable_analysis_Jordan/src'))
# sys.path.append(os.path.expanduser(
#     '~/Dropbox/Bayesian_codes/observable_analysis_Jordan/src/data'))
from src.lowlevel.observables import *
from src.lowlevel.filenames import *
from src.lowlevel.kinematics import *
import src.lowlevel.datafile

# indep_var = "theta"


def main(output_dir, indep_var, ivar_start, ivar_stop, ivar_step,
         interaction, param_list, phase_dirs, order_list, observable_list,
         convention):

    deg_to_rad = pi/180
    ivar_vals = [k for k in range(ivar_start, ivar_stop, ivar_step)]

    # kvnn_dict = {
    #     "LO": "41",
    #     "NLO": "46",
    #     "N2LO": "51",
    #     "N3LO": "56",
    #     "N4LO": "61"
    # }

    if indep_var == "theta":
        E_list = param_list
    else:
        E_list = ivar_vals

    for phase_dir, order in zip(phase_dirs, order_list):
        # print("\n" + "Args:", p, q, i, k)
        print("\n" + order)
        print("-----")
        delta, epsilon = make_phase_dicts(phase_dir, convention)
        potential_info = get_potential_file_info(phase_dir, convention)
        S_mat = make_S_matrix(delta, epsilon, convention)
        J_set = {tup[1] for tup in S_mat.keys()}
        J_array = array(list(J_set), dtype=int)

        # First handle total cross section
        if ["t", "t", "t", "t"] in observable_list:
            sig_dict = sigma_textbook(delta, epsilon, interaction, convention)
            # E_list = list({tup[0] for tup in delta.keys()})
            sig_list = [sig_dict[E] for E in E_list]
            sig_file = DataFile()
            sig_file.write(("energy (MeV)", E_list),
                           ("total sigma (mb)", sig_list))
            deltaE = E_list[1] - E_list[0]
            sig_name = observable_filename(
                ["t", "t", "t", "t"], "energy", E_list[0], E_list[-1]+deltaE,
                deltaE, "theta", 0, order, potential_info)
            full_file_name = os.path.join(output_dir, sig_name)
            sig_file.export_to_file(full_file_name)

        # Now spin observables
        for i, param in enumerate(param_list):
            print(param)
            # init_time = time.time()
            observ_files = [DataFile() for j in range(len(observable_list))]
            observ_list = [[] for j in range(len(observable_list))]
            for ivar in ivar_vals:
                if indep_var == "theta":
                    param_var = "energy"
                    theta_rad = ivar * deg_to_rad
                    E = param
                    ivar_units = "(deg)"
                    param_units = "(MeV)"
                else:
                    param_var = "theta"
                    theta_rad = param * deg_to_rad
                    E = ivar
                    ivar_units = "(MeV)"
                    param_units = "(deg)"
                phi = 0
                p_rel = E_to_p(E, interaction)
                M_st = make_M_singlet_triplet_matrix(
                    S_mat, interaction, J_array, E, theta_rad, phi)
                M_uncoup = make_M_uncoupled_matrix(M_st)
                diff_sigma = observable_C_tensor(
                    M_uncoup,
                    vec_lookup("0", theta_rad, p_rel, interaction),
                    vec_lookup("0", theta_rad, p_rel, interaction),
                    vec_lookup("0", theta_rad, p_rel, interaction),
                    vec_lookup("0", theta_rad, p_rel, interaction))

                for j, observable_indices in enumerate(observable_list):
                    if observable_indices == ["0", "0", "0", "0"]:
                        observ_list[j].append(diff_sigma)
                    elif observable_indices != ["t", "t", "t", "t"]:
                        observ = observable_C_tensor(
                            M_uncoup,
                            vec_lookup(observable_indices[0], theta_rad, p_rel, interaction),
                            vec_lookup(observable_indices[1], theta_rad, p_rel, interaction),
                            vec_lookup(observable_indices[2], theta_rad, p_rel, interaction),
                            vec_lookup(observable_indices[3], theta_rad, p_rel, interaction))
                        # print(diff_sigma, observ)
                        if diff_sigma != 0:
                            coeff = observ/diff_sigma
                        else:
                            coeff = 0
                        observ_list[j].append(coeff)

            # print("Elapsed time:", time.time() - init_time)

            for j, obs_ind in enumerate(observable_list):
                if obs_ind != ['t', 't', 't', 't']:
                    observ_file = DataFile()
                    observ_name = "C_" + obs_ind[0] + "-" + obs_ind[1] + "-" + \
                        obs_ind[2] + "-" + obs_ind[3]
                    observ_file.write((indep_var+" "+ivar_units, ivar_vals),
                                      (observ_name, observ_list[j]))
                    file_name = observable_filename(
                        obs_ind, indep_var, ivar_start, ivar_stop,
                        ivar_step, param_var, param, order, potential_info)
                    full_file_name = os.path.join(output_dir, file_name)
                    observ_file.export_to_file(full_file_name)


if __name__ == "__main__":
    ###########################################
    # Start args for running from command line
    ###########################################
    # For help:
    # >> python make_obserables.py -h
    parser = argparse.ArgumentParser(
        description="Executable script to make np spin observables."
        )
    parser.add_argument(
        "output_dir",
        help="The relative path where output files will be stored")
    parser.add_argument(
        "indep_var",
        help="The variable ([deg] or [MeV]) that varies in a given data file.",
        choices=["theta", "energy"])
    parser.add_argument(
        "interaction",
        help="The interaction type.",
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
        "--phase_dirs",
        help="The directories in which the phases are stored (by order).",
        nargs="+", required=True)
    parser.add_argument(
        "--orders",
        help="The orders at which to calculate observables.",
        nargs="+", required=True,
        choices=["LO", "NLO", "N2LO", "N3LO", "N4LO"])
    parser.add_argument(
        "--observables",
        metavar="p,q,i,k",
        nargs="+", required=True,
        help="The observables C_{pqik} to calculate.",
        type=lambda s: s.split(","))
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
        output_dir=arg_dict["output_dir"],
        indep_var=arg_dict["indep_var"],
        ivar_start=arg_dict["indep_var_range"][0],
        ivar_stop=arg_dict["indep_var_range"][1],
        ivar_step=arg_dict["indep_var_range"][2],
        interaction=arg_dict["interaction"],
        param_list=param_lst,
        phase_dirs=arg_dict["phase_dirs"],
        order_list=arg_dict["orders"],
        observable_list=arg_dict["observables"],
        convention=arg_dict["convention"])
