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
# Make config files for storing parameters of the potential and momentum mesh
# used in the partial wave analysis.
###############################################################################

import argparse
import os

###########################################
# Start args for running from command line
###########################################
# For help:
# >> python make_configs.py -h 
parser = argparse.ArgumentParser(
    description="Write parameters to a CONFIG.ini file"
    )
parser.add_argument(
    "directory",
    help="The directory in which CONFIG will be made.")
parser.add_argument(
    "interaction",
    help="The type of interaction or potential model")
# parser.add_argument(
#     "hbar2_per_m",
#     type=float,
#     help="The value of hbar squared over reduced mass for the system")
parser.add_argument(
    "k_max",
    type=float,
    help="Maximum momentum value of the quadrature mesh.")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-r", "-energy_range",
    type=int, nargs=3,
    metavar=("e_min", "e_max", "e_step"),
    help="Evaluate collision in range [e_min, e_max) separated by e_step.")
group.add_argument(
    "-l", "-energy_list",
    type=int, nargs='*',
    metavar="energy",
    help="Evaluate collision at all energies provided.")
###########################################
# End args for running from command line
###########################################


def write_line(f, param, **parameters):
    f.write(param + " = " + str(parameters[param]) + "\n")


def make_config(config_file, **parameters):
    with open(config_file, "w+") as f:
        f.write("\n[parameters]\n")
        write_line(f, "interaction", **parameters)
        # write_line(f, "hbar2_per_m", **parameters)
        write_line(f, "k_max", **parameters)
        f.write("energy_list =\n")
        for e in parameters["energy_list"]:
            f.write("\t" + str(e) + "\n")

# Each case must be hardcoded for now.

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dict = vars(args)
    file_name = "CONFIG.ini"
    full_file = os.path.join(arg_dict["directory"], file_name)
    if arg_dict["r"] is not None:
        arg_dict["energy_list"] = range(*arg_dict["r"])
    elif arg_dict["l"] is not None:
        arg_dict["energy_list"] = arg_dict["l"]
    else:
        arg_dict["energy_list"] = []
    arg_dict.pop('r', None)
    arg_dict.pop('l', None)
    make_config(full_file, **arg_dict)
