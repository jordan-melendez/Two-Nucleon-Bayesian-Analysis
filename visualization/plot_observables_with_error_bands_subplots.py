
import argparse
import matplotlib as mp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from src.lowlevel.filenames import *
from src.lowlevel.datafile import DataFile
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Rectangle


rcParams['ps.distiller.res'] = 60000
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
mp.rcParams["axes.formatter.useoffset"] = False


class HandlerSquare(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = xdescent + 0.5 * (width - height), ydescent
        p = mpatches.Rectangle(xy=center, width=height,
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


    # widths = [1 for i in range(len(orders) - 1)]
    # widths.append(len(orders)+2)
    # fig, ax = plt.subplots(nrows=1, ncols=len(orders),
    #                        sharex=True, sharey=True,
    #                        gridspec_kw={'width_ratios': widths})
    # ax = fig.add_subplot(1, 1, 1)
    # ax[-1].set_zorder(15)
    # ax[-1].set_axisbelow(False)

    if indep_var == "theta":
        param_var = "energy"
        indep_var_label = r"$\theta$ (deg)"
        param_var_label = r"$E_{\mathrm{lab}}"
        param_var_units = r"$\,MeV"
    else:
        param_var = "theta"
        indep_var_label = r"$E$ (MeV)"
        param_var_label = r"$\theta"
        param_var_units = r"^\circ$"

    # plot_type = "slivers"
    plot_type = "boxes"
    # =========================================================================
    # 2x2 box by 1 big box
    if plot_type == "boxes":
        aspect_width = 8   # integer
        aspect_height = 3  # integer
        aspect_ratio = aspect_width/aspect_height
        paper_width = 7    # inches
        fig = plt.figure(figsize=(paper_width, paper_width/aspect_ratio))
        if aspect_width > aspect_height:
            gs = gridspec.GridSpec(4*aspect_height, 4*aspect_width)
        width_spacing = 0.4
        height_spacing = 0.4
        gs.update(left=0.00, right=1,
                  wspace=width_spacing, hspace=height_spacing)
        main_ax = plt.subplot(gs[:, 2*aspect_width:])
        ax = [plt.subplot(gs[0:2*aspect_height, 0:aspect_width], sharex=main_ax, sharey=main_ax),
              plt.subplot(gs[0:2*aspect_height, aspect_width:2*aspect_width], sharex=main_ax, sharey=main_ax),
              plt.subplot(gs[2*aspect_height:4*aspect_height, 0:aspect_width], sharex=main_ax, sharey=main_ax),
              plt.subplot(gs[2*aspect_height:4*aspect_height, aspect_width:2*aspect_width], sharex=main_ax, sharey=main_ax),
              main_ax]
    # =========================================================================

    # =========================================================================
    # 1x4 slivers by 1 big box (original figsize 12, 4)
    if plot_type == "slivers":
        fig = plt.figure(figsize=(12, 3))
        gs = gridspec.GridSpec(1, 6)
        width_spacing = 0.05
        gs.update(left=0.05, right=0.95, wspace=width_spacing, hspace=0.03)
        main_ax = plt.subplot(gs[4:])
        ax = [plt.subplot(gs[0], sharex=main_ax, sharey=main_ax),
              plt.subplot(gs[1], sharex=main_ax, sharey=main_ax),
              plt.subplot(gs[2], sharex=main_ax, sharey=main_ax),
              plt.subplot(gs[3], sharex=main_ax, sharey=main_ax),
              main_ax]
    # =========================================================================

    for observable in observable_list:

        for param in param_list:

            if observable == ["t", "t", "t", "t"] or \
                    (observable == ["0", "0", "0", "0"] and indep_var == "energy"):
                for axis in ax:
                    axis.set_yscale("log", nonposy='clip')
                    # axis.ticklabel_format(style="plain")
            else:
                for axis in ax:
                    axis.set_yscale("linear")

            plt.setp(ax[1].get_yticklabels(), visible=False)
            plt.setp(ax[3].get_yticklabels(), visible=False)
            if plot_type == "slivers":
                # For 1x4 slivers:
                plt.setp(ax[2].get_yticklabels(), visible=False)

            ax[-1].yaxis.tick_right()
            ax[-1].yaxis.set_ticks_position('both')
            ax[-1].yaxis.set_label_position("right")

            # Make small plots have no x tick labels
            plt.setp([a.get_xticklabels() for a in ax[:-1]], visible=False)

            # for axis in ax:
            #     axis.set_aspect('equal')
            ax[-1].set_xlabel(indep_var_label)
            # ax.set_ylabel('')

            # Create the description box
            # text_str = r"$C_{" + observable[0] + observable[1] + \
            #     observable[2] + observable[3] + r"}$" + "\n"
            text_str = indices_to_observable_name(observable) + ", "
            if observable != ['t', 't', 't', 't']:
                text_str += param_var_label + r" = " + str(param) + param_var_units + "\n"
            text_str += r"$\Lambda_b = " + str(lambda_mult*Lambda_b) + r"$\,MeV"
            # ax[-1].text(.95, .95, text_str,
            #             horizontalalignment='right',
            #             verticalalignment='top',
            #             multialignment='center',
            #             transform=ax[-1].transAxes,
            #             bbox=dict(facecolor='white', alpha=1, boxstyle='round', pad=.3),
            #             zorder=20)
            legend_patches = []

            try:
                npwa_name = npwa_filename(observable, param_var, param)
                npwa_file = DataFile().read(os.path.join("../npwa_data/", npwa_name))
                for sub_plt in range(len(orders)):
                    npwa_plot, = ax[sub_plt].plot(npwa_file[0], npwa_file[1],
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
            ax[-1].set_ylim([ymin, ymax])

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
                for sub_plt in range(i, len(orders)):
                    # zorder must be <= 2 or else it will cover axes and ticks
                    ax[sub_plt].plot(x, obs, color=color_dict[order](.99),
                                     zorder=i/5)

                # Plot the error bands
                    for band_num, p in enumerate(sorted(p_decimal_list,
                                                        reverse=True)):
                        # dob_name = dob_filename(p, Lambda_b, obs_name)
                        dob_name = dob_filename(
                            observable, indep_var, ivar_start, ivar_stop,
                            ivar_step, param_var, param, order,
                            Lambda_b, lambda_mult, X_ref_hash,
                            p, prior_set, h, convention,
                            cbar_lower, cbar_upper, sigma,
                            potential_info=None)
                        dob_file = DataFile().read(os.path.join(error_band_dir,
                                                                dob_name))
                        obs_lower = dob_file[2]
                        obs_upper = dob_file[3]
                        ax[sub_plt].fill_between(
                            x, obs_lower, obs_upper,
                            facecolor=color_dict[order](
                                (band_num + 1) / (len(p_decimal_list) + 1)
                                ),
                            color=color_dict[order](
                                (band_num + 1) / (len(p_decimal_list) + 1)
                                ),
                            alpha=fill_transparency, interpolate=True,
                            zorder=i/5)

                # Use block patches instead of lines
                # Use innermost "dark" color of bands for legend
                legend_patches.append(
                    mpatches.Rectangle(
                        (0.5, 0.5), 0.25, 0.25,
                        # color=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
                        edgecolor=color_dict[order](.9),
                        facecolor=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
                        label=order,
                        linewidth=1
                    ))

            handler_dict = dict(zip(legend_patches, [HandlerSquare() for i in legend_patches]))
            # Legend stuff
            if npwa_plot is None:
                my_handles = legend_patches
            else:
                my_handles = [npwa_plot, *legend_patches]

            # Legend on main plot
            # ax[-1].legend(loc="best", handles=my_handles)
            extra = Rectangle((0, 0), .1, .1, fc="w", fill=False, edgecolor='none', linewidth=0)
            # leg = ax[-1].legend([extra], [], title=text_str, loc="best", handlelength=0, handletextpad=0, fancybox=True)
            leg = ax[-1].legend([extra], [text_str], loc="best", handlelength=0, handletextpad=0, fancybox=True, prop={'size': 10})
            # plt.setp(ax[-1].get_legend_handles_labels()[1], multialignment='center')

            if plot_type == "slivers":
                # Legend below small plots for sliver plot
                ax[0].legend(bbox_to_anchor=(0., -.1 - width_spacing, 4+3*width_spacing, .1),
                             loc=3, ncol=6, mode="expand", borderaxespad=0.,
                             handles=my_handles)

            if plot_type == "boxes":
                # Legend below small plots for box plot
                ax[2].legend(bbox_to_anchor=(-.0, -4*aspect_height - height_spacing/8, 2+width_spacing/12, 4*aspect_height),
                             loc=1, ncol=6, mode="expand", borderaxespad=0.,
                             handles=my_handles, prop={'size': 10},
                             handletextpad=-.1,
                             handler_map=handler_dict,
                             handlelength=1.5)

            # leg = plt.gca().get_legend()

            # Spacing between subplots
            # fig.subplots_adjust(hspace=0, wspace=0)

            # Unnecessary if the fill has a zorder <= 2.
            # for axis in ax:
            #     for k, spine in axis.spines.items():  #ax.spines is a dictionary
            #         spine.set_zorder(10)

            # Squeeze and save it
            # plt.tight_layout()
            # plt.axis('scaled', 'datalim')
            # plot_name = plot_obs_error_bands_filename(
            #         observable, indep_var, ivar_start, ivar_stop,
            #         ivar_step, param_var, param, orders[:i+1],
            #         Lambda_b, p_decimal_list)
            plot_name = subplot_obs_error_bands_filename(
                    observable, indep_var, ivar_start, ivar_stop, ivar_step,
                    param_var, param, orders, Lambda_b, lambda_mult,
                    X_ref_hash, p_decimal_list,
                    prior_set, h, convention, cbar_lower, cbar_upper, sigma,
                    potential_info=None)
            plt.draw()
            plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")

            # Clear the axes for the next observable/parameter.
            [axis.cla() for axis in ax]

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
