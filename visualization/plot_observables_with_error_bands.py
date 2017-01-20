
import argparse
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
from src.lowlevel.filenames import *
from src.lowlevel.datafile import DataFile
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch, HandlerLine2D
from matplotlib.patches import Rectangle

rcParams['ps.distiller.res'] = 60000
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


class HandlerSquare(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = xdescent + 0.5 * (width - height), ydescent
        p = mpatches.Rectangle(xy=center, width=1.5*height,
                               height=height, angle=0.0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def main(
        error_band_dir,
        output_dir,
        indep_var,
        ivar_start,
        ivar_stop,
        ivar_step,
        param_list,
        observable_list,
        Lambda_b,
        lambda_mult,
        p_decimal_list,
        orders,
        interaction,
        X_ref_hash,
        prior_set,
        h,
        cbar_lower,
        cbar_upper,
        sigma,
        convention
        ):
    """A description. It plots stuff."""

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
    ax.set_zorder(15)
    ax.set_axisbelow(False)

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
            if observable == ["t", "t", "t", "t"] or \
                (observable == ["0", "0", "0", "0"] and indep_var == "energy"):
                ax.set_yscale("log", nonposy='clip')

            ax.set_xlabel(indep_var_label)
            # ax.set_ylabel('')

            # Create the description box
            # text_str = r"$C_{" + observable[0] + observable[1] + \
            #     observable[2] + observable[3] + r"}$" + ", "  # + "\n"
            text_str = indices_to_observable_name(observable) + ", "
            if observable != ['t', 't', 't', 't']:
                text_str += param_var_label + r" = " + str(param) + param_var_units + ", "  # + "\n"
            text_str += r"$\Lambda_b = " + str(lambda_mult*Lambda_b) + r"$ MeV"
            # ax.text(.5, .96, text_str,
            #         horizontalalignment='center',
            #         verticalalignment='top',
            #         multialignment='center',
            #         transform=ax.transAxes,
            #         bbox=dict(facecolor='white', alpha=1, boxstyle='square', pad=.5),
            #         zorder=20)
            plt.title(text_str, fontsize=10)
            legend_patches = []

            try:
                npwa_name = npwa_filename(observable, param_var, param)
                npwa_file = DataFile().read(os.path.join("../npwa_data/", npwa_name))
                npwa_plot, = ax.plot(npwa_file[0], npwa_file[1],
                                     color="black", linewidth=2,
                                     label="NPWA", zorder=10,
                                     linestyle="-.")
            except FileNotFoundError:
                npwa_plot = None

            # First get global min/max of all orders
            for i, order in enumerate(orders):
                # obs_name = observable_filename(
                #     observable, indep_var, ivar_start, ivar_stop,
                #     ivar_step, param_var, param, order)
                # dob_name = dob_filename(p_decimal_list[0], Lambda_b, obs_name)
                dob_name = dob_filename(
                    observable, indep_var, ivar_start, ivar_stop,
                    ivar_step, param_var, param, order,
                    Lambda_b, lambda_mult, X_ref_hash,
                    p_decimal_list[0], prior_set, h, convention,
                    cbar_lower, cbar_upper, sigma,
                    potential_info=None)
                dob_file = DataFile().read(os.path.join(error_band_dir, dob_name))
                if i == 0:
                    obs = dob_file[1]
                    obs_min = np.min(obs)
                    obs_max = np.max(obs)
                else:
                    old_obs = obs
                    obs = dob_file[1]
                    # Probably the worst way to do this.
                    obs_min = min(np.min(np.minimum(old_obs, obs)), obs_min)
                    obs_max = max(np.max(np.maximum(old_obs, obs)), obs_max)

            # Decide the padding above/below the lines
            # This weights values far from 0 more heavily.
            ymin = obs_min - .25 * abs(obs_min)
            ymax = obs_max + .25 * abs(obs_max)
            ax.set_ylim([ymin, ymax])
            ax.set_xlim([ivar_start, ivar_stop-1])

            # Start layering the plots
            for i, order in enumerate(orders):
                # obs_name = observable_filename(
                #     observable, indep_var, ivar_start, ivar_stop,
                #     ivar_step, param_var, param, order)
                # dob_name = dob_filename(p_decimal_list[0], Lambda_b, obs_name)
                dob_name = dob_filename(
                    observable, indep_var, ivar_start, ivar_stop,
                    ivar_step, param_var, param, order,
                    Lambda_b, lambda_mult, X_ref_hash,
                    p_decimal_list[0], prior_set, h, convention,
                    cbar_lower, cbar_upper, sigma,
                    potential_info=None)
                dob_file = DataFile().read(os.path.join(error_band_dir, dob_name))

                # Plot the lines
                obs = dob_file[1]
                ax.plot(x, obs, color=color_dict[order](.99), zorder=i)

                # Plot the error bands
                for band_num, p in enumerate(sorted(p_decimal_list, reverse=True)):
                    # dob_name = dob_filename(p, Lambda_b, obs_name)
                    dob_name = dob_filename(
                        observable, indep_var, ivar_start, ivar_stop,
                        ivar_step, param_var, param, order,
                        Lambda_b, lambda_mult, X_ref_hash,
                        p, prior_set, h, convention,
                        cbar_lower, cbar_upper, sigma,
                        potential_info=None)
                    dob_file = DataFile().read(os.path.join(error_band_dir, dob_name))
                    obs_lower = dob_file[2]
                    obs_upper = dob_file[3]
                    ax.fill_between(
                        x, obs_lower, obs_upper,
                        facecolor=color_dict[order](
                            (band_num + 1) / (len(p_decimal_list) + 1)
                            ),
                        color=color_dict[order](
                            (band_num + 1) / (len(p_decimal_list) + 1)
                            ),
                        alpha=fill_transparency, interpolate=True, zorder=i)

                # Use block patches instead of lines
                # Use innermost "dark" color of bands for legend
                # legend_patches.append(
                #     mp.patches.Patch(
                #         color=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
                #         label=order,
                #     ))
                legend_patches.append(
                    mpatches.Rectangle(
                        (1, 1), 0.25, 0.25,
                        # color=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
                        edgecolor=color_dict[order](.9),
                        facecolor=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
                        label=order,
                        linewidth=1
                    ))

                if npwa_plot is None:
                    my_handles = legend_patches
                    handler_dict = dict(zip(my_handles, [HandlerSquare() for i in legend_patches]))
                else:
                    my_handles = [npwa_plot, *legend_patches]
                    squares = [HandlerSquare() for i in legend_patches]
                    line = HandlerLine2D(marker_pad=1, numpoints=None)
                    handler_dict = dict(zip(my_handles, [line] + squares))

                ax.legend(loc="best", handles=my_handles,
                          handler_map=handler_dict,
                          handletextpad=.7,
                          handlelength=.6,
                          fontsize=10)

                # Squeeze and save it
                plt.tight_layout()
                # plot_name = plot_obs_error_bands_filename(
                #         observable, indep_var, ivar_start, ivar_stop,
                #         ivar_step, param_var, param, orders[:i+1],
                #         Lambda_b, p_decimal_list)
                plot_name = plot_obs_error_bands_filename(
                    observable, indep_var, ivar_start, ivar_stop, ivar_step,
                    param_var, param, orders[:i+1], Lambda_b, lambda_mult,
                    X_ref_hash, p_decimal_list,
                    prior_set, h, convention, cbar_lower, cbar_upper, sigma,
                    potential_info=None)
                fig.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")

            # Clear the axes for the next observable/parameter.
            plt.cla()

if __name__ == "__main__":
    ###########################################
    # Start args for running from command line
    ###########################################
    # For help:
    # >> python plot_observables_with_error_bands.py -h
    parser = argparse.ArgumentParser(
        description="Executable script to extract np observable coefficients."
        )
    parser.add_argument(
        "error_band_dir",
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
        "--p_decimals",
        help="The DOB percent divided by 100.",
        type=float, nargs="+",
        required=True)
    parser.add_argument(
        "--orders",
        help="The orders to show on the plots.",
        type=str, nargs="+",
        required=True, choices=["LOp", "LO", "NLO", "N2LO", "N3LO", "N4LO"])
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

    if arg_dict["prior_set"] == "B":
        cup = 0
        clow = 0
        sigma = arg_dict["sigma"]
    else:
        sigma = 0
        cup = arg_dict["cbar_upper"]
        clow = arg_dict["cbar_lower"]

    main(
        error_band_dir=arg_dict["error_band_dir"],
        output_dir=arg_dict["output_dir"],
        indep_var=arg_dict["indep_var"],
        ivar_start=arg_dict["indep_var_range"][0],
        ivar_stop=arg_dict["indep_var_range"][1],
        ivar_step=arg_dict["indep_var_range"][2],
        param_list=param_lst,
        observable_list=arg_dict["observables"],
        Lambda_b=arg_dict["Lambda_b"],
        lambda_mult=arg_dict["lambda_mult"],
        p_decimal_list=arg_dict["p_decimals"],
        orders=arg_dict["orders"],
        interaction=arg_dict["interaction"],
        X_ref_hash=arg_dict["X_ref_hash"],
        prior_set=arg_dict["prior_set"],
        h=arg_dict["h"],
        cbar_lower=clow,
        cbar_upper=cup,
        sigma=sigma,
        convention=arg_dict["convention"])
