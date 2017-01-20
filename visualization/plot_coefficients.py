import argparse
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
from src.lowlevel.filenames import *
from src.lowlevel.datafile import DataFile
from matplotlib import rc
from matplotlib import rcParams

rcParams['ps.distiller.res'] = 60000
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def main(
         coeff_dir,
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

    color_dict = {
        "LOp": plt.get_cmap("Greys"),
        "LO": plt.get_cmap("Purples"),
        "NLO": plt.get_cmap("Oranges"),
        "N2LO": plt.get_cmap("Greens"),
        "N3LO": plt.get_cmap("Blues"),
        "N4LO": plt.get_cmap("Reds")
    }

    fill_transparency = 1
    x = np.arange(ivar_start, ivar_stop, ivar_step)

    fig = plt.figure(figsize=(3.4, 3.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_zorder(50)

    if indep_var == "theta":
        param_var = "energy"
        indep_var_label = r"$\theta$ (deg)"
        param_var_label = r"$E_{\mathrm{lab}}"
        param_var_units = r"$ MeV"
    else:
        param_var = "theta"
        indep_var_label = r"$E$ (MeV)"
        param_var_label = r"$\theta"
        param_var_units = r"^\circ$"

    for observable in observable_list:
        for param in param_list:
            ax.set_xlabel(indep_var_label)
            # ax.set_ylabel('')

            # Create the description box
            # text_str = r"$C_{" + observable[0] + observable[1] + \
            #     observable[2] + observable[3] + r"}$" + ", "  # + "\n"
            text_str = indices_to_observable_name(observable) + ", "
            if observable != ['t', 't', 't', 't']:
                text_str += param_var_label + r" = " + str(param) + param_var_units + ", "  # + "\n"
            text_str += r"$\Lambda_b = " + str(Lambda_b*lambda_mult) + r"$\,MeV"
            # ax.text(.5, .96, text_str,
            #         horizontalalignment='center',
            #         verticalalignment='top',
            #         multialignment='center',
            #         transform=ax.transAxes,
            #         bbox=dict(facecolor='white', alpha=1, boxstyle='square', pad=.5),
            #         zorder=20)
            plt.title(text_str, fontsize=10)
            # legend_patches = []

            # First get global min/max of all orders
            for i, order in enumerate(orders):
                # obs_name = observable_filename(
                #     observable, indep_var, ivar_start, ivar_stop,
                #     ivar_step, param_var, param, orders[-1])
                # coeff_name = coeff_filename(Lambda_b, obs_name, X_ref_hash)
                coeff_name = coeff_filename(
                    observable, indep_var, ivar_start, ivar_stop,
                    ivar_step, param_var, param, orders[-1],
                    Lambda_b, lambda_mult, X_ref_hash, convention, potential_info=None)
                coeff_file = DataFile().read(os.path.join(coeff_dir, coeff_name))

                if i == 0:
                    coeff = coeff_file[2]
                    coeff_min = np.nanmin(coeff)
                    coeff_max = np.nanmax(coeff)
                else:
                    old_coeff = coeff
                    coeff = coeff_file[2+i]
                    # Probably the worst way to do this.
                    coeff_min = min(np.nanmin(np.minimum(old_coeff, coeff)), coeff_min)
                    coeff_max = max(np.nanmax(np.maximum(old_coeff, coeff)), coeff_max)

                # Plot the lines
                ax.plot(x, coeff, color=color_dict[order](.6),
                        linewidth=1, label=order, zorder=i)

            # Decide the padding above/below the lines
            # This weights values far from 0 more heavily.
            ymin = coeff_min - .25 * abs(coeff_min)
            ymax = coeff_max + .25 * abs(coeff_max)
            ax.set_ylim([ymin, ymax])

            # Use block patches instead of lines
            # Use innermost "dark" color of bands for legend
            # legend_patches.append(
            #     mp.patches.Patch(
            #         color=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
            #         label=order,
            #     ))
            ax.legend(loc="best", fontsize=10)

            # Squeeze and save it
            plt.tight_layout()
            plot_name = plot_coeff_error_bands_filename(
                observable, indep_var, ivar_start, ivar_stop,
                ivar_step, param_var, param, orders, Lambda_b, lambda_mult,
                X_ref_hash, convention, potential_info=None)
            fig.savefig(os.path.join(output_dir, plot_name))

            # Clear the axes for the next observable/parameter.
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
        "coeff_dir",
        help="The directory in which the coefficients are stored.")
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
        help="The orders to show on the plots.",
        type=str, nargs="+",
        required=True, choices=["LO", "NLO", "N2LO", "N3LO", "N4LO"])
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
        coeff_dir=arg_dict["coeff_dir"],
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
