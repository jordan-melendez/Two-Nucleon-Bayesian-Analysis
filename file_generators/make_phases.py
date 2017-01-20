###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
# Created: Jul-28-2016
# Revised:
#
# Revision Log:
#
###############################################################################
#
#
###############################################################################
# Numerically solves Lippman-Schwinger equation for phase shifts
###############################################################################

import argparse
from numpy import reshape
from numpy import array
# from datafile import DataFile
import os
import sys
from src.lowlevel import filenames, calc_phase_shifts
from src.lowlevel.datafile import DataFile

# Old code for when I read in some parameter data from Config files.
# class MyConfigParser(ConfigParser):
#     def get_list(self, section, option):
#         value = self.get(section, option)
#         return list(filter(None, (x.strip() for x in value.splitlines())))

#     def get_list_int(self, section, option):
#         return [int(x) for x in self.get_list(section, option)]


def main(filename, phase_dir, potential_dir,
         mesh_dir, interaction, k_max, lab_energies, convention):
    """Organize input data and call calc_phase_shifts().
    """

    # config = MyConfigParser()
    # config.read(config_file)
    # hbar2_per_m = config.get("parameters", "hbar2_per_m")
    # k_max = float(config.get("parameters", "k_max"))
    # lab_energies = array(config.get_list_int("parameters", "energy_list"))
    # interaction = config.get("parameters", "interaction")
    # k_max = float(config.get("parameters", "k_max"))
    # lab_energies = array(config.get_list_int("parameters", "energy_list"))

    filename = os.path.basename(filename)
    mesh_file = os.path.join(mesh_dir, filenames.mesh_filename(filename, convention))
    mesh = DataFile().read(mesh_file)
    points = array(mesh[0])
    weights = array(mesh[1])
    num_pts = len(points)

    potential_file = os.path.join(
        potential_dir, filenames.potential_filename(filename, convention))
    potential = DataFile().read(potential_file)

    # Coupled files have 4 columns of potential data,
    # while uncoupled only have 1.
    # The DataFile class counts the columns as its "length".
    if len(potential) == 3:
        is_coupled = False
    elif len(potential) == 6:
        is_coupled = True
    else:
        return "Potential file structure not compatible."

    phase_file = DataFile()
    if is_coupled:
        # Transpose so k' is along rows and k along columns?
        # Should only matter for V_mat_12 and V_mat_21.
        # I should figure the transpose issues out, but this seems to work.
        v_11_mat = reshape(array(potential[2]), (num_pts, num_pts))
        v_12_mat = reshape(array(potential[3]), (num_pts, num_pts))
        v_21_mat = reshape(array(potential[4]), (num_pts, num_pts))
        v_22_mat = reshape(array(potential[5]), (num_pts, num_pts))
        v = v_11_mat, v_12_mat, v_21_mat, v_22_mat
        solution = calc_phase_shifts.calc_phase_shifts(
            points, weights, k_max, lab_energies, v, interaction, is_coupled,
            convention
            )
        phase_file.write(("E (MeV)", lab_energies),
                         ("delta_a_bar (deg)", solution[0]),
                         ("delta_b_bar (deg)", solution[1]),
                         ("epsilon_bar (deg)", solution[2]),
                         )
    else:
        v = reshape(array(potential[2]), (num_pts, num_pts))
        solution = calc_phase_shifts.calc_phase_shifts(
            points, weights, k_max, lab_energies, v, interaction, is_coupled,
            convention
            )
        phase_file.write(("E (MeV)", lab_energies),
                         ("delta (deg)", solution),
                         )

    phase_file_name = os.path.join(phase_dir, filename)
    phase_file.export_to_file(phase_file_name)

    ##############
    # End Main()
    ##############


# For running as an executable.
if __name__ == '__main__':
    # Set up args for running from terminal
    parser = argparse.ArgumentParser(
        description="""Make phase files from potential using
        Gaussian quadrature with parameters in CONFIG"""
        )
    parser.add_argument(
        "filename",
        help="The filename for which to make phases.")
    parser.add_argument(
        "phase_dir",
        help="Directory where phase files are placed.")
    parser.add_argument(
        "potential_dir",
        help="Directory of potential on a momentum mesh for the interaction.")
    parser.add_argument(
        "momentum_mesh_dir",
        help="""Directory of file containing momentum points and weights for
        Gaussian quadrature""")
    parser.add_argument(
        "interaction",
        help="The interaction type: {'pp', 'nn', 'np'}.",
        choices=["nn", "pp", "np"])
    parser.add_argument(
        "k_max",
        help="The maximum k value on the momentum mesh",
        type=float)
    param_group = parser.add_mutually_exclusive_group(required=True)
    param_group.add_argument(
        "--energy_range", "--erange",
        type=int, nargs=3,
        metavar=("start", "stop", "step"),
        help="The energy range [start, stop) in increments of step.")
    param_group.add_argument(
        "--energy_values", "--evals",
        type=int, nargs="+",
        help="""The list of energies.""")
    parser.add_argument(
        "--convention",
        required=True,
        help="The Stapp or Blatt phase convention.",
        choices=["stapp", "blatt"])
    # parser.add_argument(
    #     "config_file",
    #     help="Contains parameters of particular interaction.")

    args = parser.parse_args()
    arg_dict = vars(args)

    if arg_dict["energy_range"] is not None: 
        e0 = arg_dict["energy_range"][0]
        ef = arg_dict["energy_range"][1]
        es = arg_dict["energy_range"][2]
        e_lst = [i for i in range(e0, ef, es)]
    else:
        e_lst = arg_dict["energy_values"]

    # Where the work is done
    main(args.filename, args.phase_dir, args.potential_dir,
         args.momentum_mesh_dir, args.interaction, args.k_max, e_lst, args.convention)
