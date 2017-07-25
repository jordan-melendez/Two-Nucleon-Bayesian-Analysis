
import argparse
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from src.lowlevel.filenames import *
from src.lowlevel.datafile import DataFile
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch, HandlerLine2D
from matplotlib.patches import Rectangle
from subprocess import call
from matplotlib.lines import Line2D

rcParams['ps.distiller.res'] = 60000
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
params = {
          # 'backend': 'ps',
          'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{siunitx}'],
          # 'axes.labelsize': 10,
          # 'text.fontsize': 10,
          # 'legend.fontsize': 8,
          # 'xtick.labelsize': 10,
          # 'ytick.labelsize': 10,
          'text.usetex': True,
          # 'figure.figsize': fig_size,
          'axes.unicode_minus': True
          }
rcParams.update(params)


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
        ignore_orders,
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
    hatch_list = ["", "x"]

    orders_name_dict = {
        "NLO": r"NLO",
        "N2LO": r"N$^2$LO",
        "N3LO": r"N$^3$LO",
        "N4LO": r"N$^4$LO"
    }

    fill_transparency = 1
    if indep_var == "theta":
        x = list(range(1, 180))
    else:
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
        indep_var_label = r"$E_{\mathrm{lab}}$ (MeV)"
        param_var_label = r"$\theta"
        param_var_units = r"^\circ$"

    for observable in observable_list:
        for param in param_list:
            # if observable == ["t", "t", "t", "t"] or \
            #         (observable == ["0", "0", "0", "0"] and indep_var == "energy"):
            #     ax.set_yscale("log", nonposy='clip')

            ax.set_xlabel(indep_var_label)
            # ax.set_ylabel('')

            if indep_var == "theta":
                major_tick_spacing = 60
                minor_tick_spacing = 20
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(major_tick_spacing))
                ax.xaxis.set_minor_locator(
                    ticker.MultipleLocator(minor_tick_spacing))

            # horizontal line
            ax.axhline(0, linestyle='--', color='k', linewidth=.5)

            # Create the description box
            # text_str = r"$C_{" + observable[0] + observable[1] + \
            #     observable[2] + observable[3] + r"}$" + ", "  # + "\n"
            text_str = indices_to_residual_name(observable)
            if observable == ['t', 't', 't', 't']:
                text_str += r" (mb)"
            elif observable == ['0', '0', '0', '0']:
                text_str += r" (mb/sr)"

            # Probably don't include this extra info. Leave for caption.
            #
            # text_str += ", "
            # if observable != ['t', 't', 't', 't']:
            #     text_str += ", " + param_var_label + r" = " + str(param) + param_var_units + ", "  # + "\n"
            # text_str += r"$\Lambda_b = " + str(lambda_mult*Lambda_b) + r"$ MeV"

            # Don't put in a text box
            #
            # ax.text(.5, .96, text_str,
            #         horizontalalignment='center',
            #         verticalalignment='top',
            #         multialignment='center',
            #         transform=ax.transAxes,
            #         bbox=dict(facecolor='white', alpha=1, boxstyle='square', pad=.5),
            #         zorder=20)

            # Don't put in the title
            # plt.title(text_str, fontsize=10)

            # Instead use y axis
            ax.set_ylabel(text_str, fontsize=10)
            legend_patches = []

            try:
                npwa_name = npwa_filename(observable, param_var, param)
                npwa_file = DataFile().read(os.path.join("../npwa_data/", npwa_name))
                npwa_ivars = npwa_file[0]
                npwa_data = npwa_file[1]
                if indep_var == "theta":
                    npwa_dict = dict(zip(npwa_ivars, npwa_data))
                    npwa_data = [npwa_dict[i] for i in range(1, 180)]
                npwa_data = array(npwa_data)
                # for sub_plt in range(len(orders)):
                #     npwa_plot, = ax[sub_plt].plot(npwa_file[0], npwa_file[1],
                #                                   color="black", linewidth=2,
                #                                   label="NPWA", zorder=10,
                #                                   linestyle="-.")
                npwa_plot = None
            except FileNotFoundError:
                npwa_plot = None

            label_list = []
            # First get global min/max of all orders
            for i, order in enumerate(orders):
                # obs_name = observable_filename(
                #     observable, indep_var, ivar_start, ivar_stop,
                #     ivar_step, param_var, param, order)
                # dob_name = dob_filename(p_decimal_list[0], Lambda_b, obs_name)
                dob_name = dob_filename(
                        observable, indep_var, ivar_start, ivar_stop,
                        ivar_step, param_var, param, order, ignore_orders,
                        Lambda_b, lambda_mult, X_ref_hash,
                        p_decimal_list[0], prior_set, h, convention, None,
                        cbar_lower, cbar_upper, sigma,
                        potential_info=None)
                dob_file = DataFile().read(os.path.join(error_band_dir, dob_name))
                dob_dict = dict(zip(dob_file[0], dob_file[1]))
                if indep_var == "theta":
                    if observable == ['0', '0', 'n', '0']:
                        npwa_d = [npwa_dict[i] for i in range(20, 180)]
                        dob_data = [dob_dict[i] for i in range(20, 180)]
                    else:
                        npwa_d = npwa_data
                        dob_data = [dob_dict[i] for i in range(1, 180)]
                if indep_var == "energy":
                    npwa_d = npwa_data
                    dob_data = [dob_dict[i] for i in range(1, 351)]
                if i == 0:
                    res = (array(dob_data) - npwa_d)
                    res_min = np.min(res)
                    res_max = np.max(res)
                else:
                    old_res = res
                    res = (array(dob_data) - npwa_d)
                    # Probably the worst way to do this.
                    res_min = min(np.min(np.minimum(old_res, res)), res_min)
                    res_max = max(np.max(np.maximum(old_res, res)), res_max)

            # Decide the padding above/below the lines
            # This weights values far from 0 more heavily.
            ymin = res_min - .25 * abs(res_min)
            ymax = res_max + .25 * abs(res_max)
            ax.set_ylim([ymin, ymax])
            if indep_var == "theta":
                ax.set_xlim([0, 180])
            else:
                ax.set_xlim([0, 350])
            # ax.set_xlim([ivar_start, ivar_stop-1])

            # Start layering the plots
            for i, order in enumerate(orders):
                # obs_name = observable_filename(
                #     observable, indep_var, ivar_start, ivar_stop,
                #     ivar_step, param_var, param, order)
                # dob_name = dob_filename(p_decimal_list[0], Lambda_b, obs_name)
                dob_name = dob_filename(
                    observable, indep_var, ivar_start, ivar_stop,
                    ivar_step, param_var, param, order, ignore_orders,
                    Lambda_b, lambda_mult, X_ref_hash,
                    p_decimal_list[0], prior_set, h, convention, None,
                    cbar_lower, cbar_upper, sigma,
                    potential_info=None)
                dob_file = DataFile().read(os.path.join(error_band_dir, dob_name))

                # Plot the lines
                dob_dict = dict(zip(dob_file[0], dob_file[1]))
                if indep_var == "theta":
                    dob_data = [dob_dict[i] for i in range(1, 180)]
                if indep_var == "energy":
                    dob_data = [dob_dict[i] for i in range(1, 351)]

                res = (array(dob_data) - npwa_data)
                # print(npwa_data[0], npwa_data[1])
                ax.plot(x, res, color=color_dict[order](.99), zorder=i/5)

                # Plot the error bands
                for band_num, p in enumerate(sorted(p_decimal_list, reverse=True)):
                    # dob_name = dob_filename(p, Lambda_b, obs_name)
                    dob_name = dob_filename(
                        observable, indep_var, ivar_start, ivar_stop,
                        ivar_step, param_var, param, order, ignore_orders,
                        Lambda_b, lambda_mult, X_ref_hash,
                        p, prior_set, h, convention, None,
                        cbar_lower, cbar_upper, sigma,
                        potential_info=None)
                    dob_file = DataFile().read(os.path.join(error_band_dir, dob_name))
                    if indep_var == "theta":
                        dob_dict_lower = dict(zip(dob_file[0], dob_file[2]))
                        dob_data_lower = [dob_dict_lower[i] for i in range(1, 180)]

                        dob_dict_upper = dict(zip(dob_file[0], dob_file[3]))
                        dob_data_upper = [dob_dict_upper[i] for i in range(1, 180)]
                    if indep_var == "energy":
                        dob_dict_lower = dict(zip(dob_file[0], dob_file[2]))
                        dob_data_lower = [dob_dict_lower[i] for i in range(1, 351)]

                        dob_dict_upper = dict(zip(dob_file[0], dob_file[3]))
                        dob_data_upper = [dob_dict_upper[i] for i in range(1, 351)]
                    res_lower = (array(dob_data_lower) - npwa_data)
                    res_upper = (array(dob_data_upper) - npwa_data)
                    ax.fill_between(
                        x, res_lower, res_upper,
                        facecolor=color_dict[order](
                            (band_num + 1) / (len(p_decimal_list) + 1)
                            ),
                        color=color_dict[order](
                            (band_num + 1) / (len(p_decimal_list) + 1)
                            ),
                        # color="black",
                        hatch=hatch_list[band_num],
                        alpha=fill_transparency, interpolate=True, zorder=i/5)

                # Use block patches instead of lines
                # Use innermost "dark" color of bands for legend
                # legend_patches.append(
                #     mp.patches.Patch(
                #         color=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
                #         label=order,
                #     ))
                # legend_patches.append(
                #     mpatches.Rectangle(
                #         (1, 1), 0.25, 0.25,
                #         # color=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
                #         edgecolor=color_dict[order](.9),
                #         facecolor=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
                #         label=order,
                #         linewidth=1
                #     ))
                light_rec = mpatches.Rectangle(
                        (0.5, 0.5), 1, 1,
                        edgecolor=color_dict[order](.99),
                        facecolor=color_dict[order](1 / (len(p_decimal_list) + 1)),
                        label=orders_name_dict[order],
                        linewidth=.4
                    )
                dark_rec = Line2D(
                        (0, 1), (0, 0),
                        # edgecolor=color_dict[order](.9),
                        # facecolor=color_dict[order](2 / (len(p_decimal_list) + 1)),
                        color=color_dict[order](2 / (len(p_decimal_list) + 1)),
                        # label=orders_name_dict[order],
                        # linewidth=3
                        linewidth=2
                    )
                thin_rec = Line2D(
                        (0, 1), (0, 0),
                        # edgecolor=color_dict[order](.9),
                        # facecolor=color_dict[order](2 / (len(p_decimal_list) + 1)),
                        color=color_dict[order](.99),
                        # label=orders_name_dict[order],
                        # linewidth=.6
                        linewidth=0
                    )
                legend_patches.append(
                        (light_rec, dark_rec, thin_rec)
                    )

                handler_dict = {}
                label_list.append(orders_name_dict[order])
                for patch_tuple in legend_patches:
                    # handler_dict[patch_tuple[0]] = None
                    handler_dict[patch_tuple[1]] = HandlerLine2D(marker_pad=.15)
                    handler_dict[patch_tuple[2]] = HandlerLine2D(marker_pad=.06)
                if npwa_plot is None:
                    my_handles = legend_patches
                    # handler_dict = dict(zip(my_handles, [HandlerSquare() for i in legend_patches]))
                    my_labels = label_list
                else:
                    my_handles = [npwa_plot, *legend_patches]
                    # squares = [HandlerSquare() for i in legend_patches]
                    # line = HandlerLine2D(marker_pad=1, numpoints=None)
                    # handler_dict = dict(zip(my_handles, [line] + squares))
                    my_labels = ["NPWA", *label_list]

                # ax.legend(loc="best", handles=my_handles, labels=my_labels,
                #           handler_map=handler_dict,
                #           handletextpad=.7,
                #           handlelength=.6,
                #           fontsize=10)
                ax.legend(
                    my_handles, my_labels,
                    bbox_to_anchor=(0., 1.02, 1., .102),
                    loc=3, ncol=6, mode="expand", borderaxespad=0.,
                    prop={'size': 8},
                    handletextpad=.5,
                    handler_map=handler_dict,
                    # handler_map={dark_rec: HandlerLine2D(marker_pad = 0)},
                    handlelength=1.5,
                    borderpad=.45
                    )

                # Squeeze and save it
                plt.tight_layout()
                # plot_name = plot_obs_error_bands_filename(
                #         observable, indep_var, ivar_start, ivar_stop,
                #         ivar_step, param_var, param, orders[:i+1],
                #         Lambda_b, p_decimal_list)
                plot_name = plot_res_error_bands_filename(
                    observable, indep_var, ivar_start, ivar_stop, ivar_step,
                    param_var, param, orders[:i+1], ignore_orders, Lambda_b,
                    lambda_mult, X_ref_hash, p_decimal_list,
                    prior_set, h, convention, None, cbar_lower, cbar_upper,
                    sigma, potential_info=None)
                fig.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")

                # To fix the blurriness issue when pdf is put into a LaTeX doc.
                # Generate .eps file and then change to .pdf
                # I don't know the root problem.
                call(["epstopdf", os.path.join(output_dir, plot_name)])
                call(["rm", os.path.join(output_dir, plot_name)])

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
        "--ignore_orders",
        help="The kth orders (Q^k) to ignore when calculating DoBs.",
        nargs="+", type=int,
        choices=[0, 1, 2, 3, 4, 5])
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

    if arg_dict["ignore_orders"] is None:
        ignore_orders = []
    else:
        ignore_orders = arg_dict["ignore_orders"]

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
        ignore_orders=ignore_orders,
        interaction=arg_dict["interaction"],
        X_ref_hash=arg_dict["X_ref_hash"],
        prior_set=arg_dict["prior_set"],
        h=arg_dict["h"],
        cbar_lower=clow,
        cbar_upper=cup,
        sigma=sigma,
        convention=arg_dict["convention"])
