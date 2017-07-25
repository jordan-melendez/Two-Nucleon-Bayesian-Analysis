import argparse
import matplotlib as mp
# mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import re
from src.lowlevel.filenames import *
from src.lowlevel.datafile import DataFile
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import matplotlib.colors as colors
from mpl_toolkits.axes_grid.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from subprocess import call
from src.lowlevel.CH_to_EKM_statistics import n_c_val, Delta_k_posterior
from scipy.interpolate import interp2d
import matplotlib.gridspec as gridspec

# rcParams['ps.distiller.res'] = 6000
# rcParams["ps.usedistiller"] = 'xpdf'
# rcParams["text.dvipnghack"] = True
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})

# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rcParams['text.usetex'] = True
# rc('text', usetex=True)


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
         convention,
         prior_set,
         cbar_less,
         cbar_greater):

    color_dict = {
        "LOp": plt.get_cmap("Greys"),
        "LO": plt.get_cmap("Purples"),
        "NLO": plt.get_cmap("Oranges"),
        "N2LO": plt.get_cmap("Greens"),
        "N3LO": plt.get_cmap("Blues"),
        "N4LO": plt.get_cmap("Reds")
    }
    color_list = [
        plt.get_cmap("Oranges")(0.4),
        plt.get_cmap("Greens")(0.6),
        plt.get_cmap("Blues")(0.6),
        plt.get_cmap("Reds")(0.6)
        ]
    linestyle_list = [
        "-",                # Solid
        (1, (1, 2, 6, 2)),  # My dashed-dot
        # ":",                # Dotted
        (1, (1.2, 1.3)),    # My dotted
        (1, (3, 2))         # My dashed
    ]
    linewidth_list = [
        .9,
        1,
        1.2,
        1
    ]

    orders_name_dict = {
        "NLO": r"NLO",
        "N2LO": r"N$^2$LO",
        "N3LO": r"N$^3$LO",
        "N4LO": r"N$^4$LO",
        "None": ""
    }

    R_dict = {
        0: "0.8",
        1: "0.9",
        2: "1.0",
        3: "1.1",
        4: "1.2",
        # EM Potentials:
        5: "450",
        6: "500",
        7: "550"
        }

    Lambdab_dict = {
        0: "600",
        1: "600",
        2: "600",
        3: "500",
        4: "400"
        }

    make_inset = False
    make_side_plots = False

    fill_transparency = 1
    x = np.arange(ivar_start, ivar_stop, ivar_step)

    order_dict = {"LO": 2, "NLO": 3, "N2LO": 4, "N3LO": 5, "N4LO": 6, "None": 0}

    # The order used to find the coeff file
    # file_order = orders[-1]
    file_order = "N4LO"

    is_posterior_plotted = prior_set is not None and \
        cbar_less is not None and cbar_greater is not None

    # fig = plt.figure(figsize=(3.4, 3.4))
    # ax = fig.add_subplot(1, 1, 1)
    if is_posterior_plotted and make_side_plots:
        # fig, (ax1, ax, ax2) = plt.subplots(1, 3, sharey=True, figsize=(8, 3.4))
        # Plot figure with subplots of different sizes
        fig = plt.figure(1)
        # set up subplot grid
        gs = gridspec.GridSpec(1, 4)
        ax = plt.subplot(gs[0, 1:-1])
        ax1 = plt.subplot(gs[0, :1], sharey=ax)
        ax2 = plt.subplot(gs[0, -1], sharey=ax)
    else:
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
    # ax.set_zorder(50)

    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    # left, bottom, width, height = [0.25, 0.2, 0.3, 0.5]
    # ax2 = fig.add_axes([left, bottom, width, height])

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
            ax.set_xlabel(indep_var_label, fontsize=10)
            if is_posterior_plotted and make_side_plots:
                ax1.set_ylabel(r"$c_n$", fontsize=10)
            else:
                ax.set_ylabel(r"$c_n$", fontsize=10)
            # ax.set_ylabel('')

            # Create the description box
            # text_str = r"$C_{" + observable[0] + observable[1] + \
            #     observable[2] + observable[3] + r"}$" + ", "  # + "\n"
            text_str = indices_to_observable_name(observable)
            if observable == ['t', 't', 't', 't']:
                # text_str = r"$R=1.2$\,fm,\\$\Lambda_b=600$\,MeV"
                coeff_file_name = coeff_filename(
                    observable, indep_var, ivar_start, ivar_stop,
                    ivar_step, param_var, param, file_order,
                    Lambda_b, lambda_mult, X_ref_hash, convention, potential_info=None)
                R_str = re.search("kvnn_(.*?)_", coeff_file_name)
                R_int = int(R_str.group(1))
                if R_int < 70:
                    R_hash = int(R_str.group(1)) % 5
                elif 70 <= R_int < 75:
                    R_hash = 5
                elif 75 <= R_int < 80:
                    R_hash = 6
                elif 80 <= R_int < 85:
                    R_hash = 7
                R_value = R_dict[R_hash]
                if R_int < 70:
                    text_str = r"$R=" + str(R_value) + r"$\,fm"
                else:
                    text_str = r"$\Lambda=" + str(R_value) + r"$\,MeV"
            # if observable != ['t', 't', 't', 't']:
            #     text_str += ", " + param_var_label + r" = " + str(param) + param_var_units # + "\n"
            # text_str += r", $\Lambda_b = " + str(Lambda_b*lambda_mult) + r"$\,MeV"
            # ax.text(.5, .96, text_str,
            #         horizontalalignment='center',
            #         verticalalignment='top',
            #         multialignment='center',
            #         transform=ax.transAxes,
            #         bbox=dict(facecolor='white', alpha=1, boxstyle='square', pad=.5),
            #         zorder=20)
            # plt.title(text_str, fontsize=10)
            # ax.set_ylabel(text_str, fontsize=10)
            # legend_patches = []

            if indep_var == "energy":
                major_ticks = np.arange(0, 351, 100)
                # minor_ticks = np.arange(50, 351, 100)
                x_minor_locator = AutoMinorLocator(n=2)
                xmin = 0
                xmax = 350
            elif indep_var == "theta":
                major_ticks = np.arange(0, 181, 60)
                # minor_ticks = np.arange(30, 181, 60)
                x_minor_locator = AutoMinorLocator(n=3)
                xmin = 0
                xmax = 180
            # ax.set_xticks(minor_ticks, minor=True)
            ax.set_xticks(major_ticks)
            ax.xaxis.set_minor_locator(x_minor_locator)

            y_major_locator = MaxNLocator(axis='y', nbins=6)
            y_minor_locator = AutoMinorLocator(n=2)
            ax.yaxis.set_major_locator(y_major_locator)
            ax.yaxis.set_minor_locator(y_minor_locator)

            if make_inset:
                # ax2.set_xlim([xmin, xmax])
                # ax2.set_ylim([ymin, ymax])
                # # Plot the 0-line
                # ax2.plot(
                #         x, np.zeros(len(x)), color=mp.rcParams['xtick.color'],
                #         linewidth=mp.rcParams['xtick.major.width'],
                #         zorder=0, ls="--")

                # axins = inset_axes(
                #     ax,
                #     width="50%", # width = 30% of parent_bbox
                #     height=.8, # height : 1 inch
                #     loc=8)

                # These are in unitless percentages of the figure size. (0,0 is bottom left)
                left, bottom, width, height = [0.27, 0.23, 0.65, 0.23]
                axins = fig.add_axes([left, bottom, width, height])

                # axins = zoomed_inset_axes(
                #                ax,
                #                2,  # zoom = 0.5
                #                loc=3)

                axins.plot(x, np.zeros(len(x)), color=mp.rcParams['xtick.color'],
                    linewidth=mp.rcParams['xtick.major.width'],
                    zorder=0, ls="--")

                # sub region of the original image
                # x1, x2, y1, y2 = 0, 350, -0.2, 0.6
                x1, x2, y1, y2 = 0, 350, -0.5, .75
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                # plt.xticks(visible=False)
                # axins.set_xticks(minor_ticks, minor=True)
                axins.xaxis.set_minor_locator(x_minor_locator)
                axins.set_xticks(major_ticks)
                # axins.set_yticks([-0.2, 0.2, 0.6], minor=True)
                # axins.set_yticks([0, 0.4])
                # axins.set_yticks([-0., 0.2, 0.6], minor=True)
                yins_minor_locator = AutoMinorLocator(n=2)
                axins.yaxis.set_minor_locator(yins_minor_locator)
                axins.set_yticks([-.5, 0., .5])
                axins.yaxis.set_tick_params(labelsize=8, zorder=10)
                axins.xaxis.set_tick_params(labelsize=8)
                # mp.axis.Tick.set_zorder(axins.yaxis, 50)
                # axins.yaxis.get_ticklines().set_zorder(10)
                # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
                # plt.draw()
                # plt.show()

            # Hold all coefficients for use in posterior density, if plotted
            coeff_matrix = [[] for i in orders]

            coeff_min = -10
            coeff_max = 10

            # First get global min/max of all orders
            for i, order in enumerate(orders):
                # obs_name = observable_filename(
                #     observable, indep_var, ivar_start, ivar_stop,
                #     ivar_step, param_var, param, orders[-1])
                # coeff_name = coeff_filename(Lambda_b, obs_name, X_ref_hash)
                coeff_name = coeff_filename(
                    observable, indep_var, ivar_start, ivar_stop,
                    ivar_step, param_var, param, file_order,
                    Lambda_b, lambda_mult, X_ref_hash, convention, potential_info=None)
                coeff_file = DataFile().read(os.path.join(coeff_dir, coeff_name))

                if i == 0:
                    coeff = coeff_file[order_dict[order]]
                    coeff_min = np.nanmin(coeff)
                    coeff_max = np.nanmax(coeff)
                else:
                    old_coeff = coeff
                    coeff = coeff_file[order_dict[order]]
                    # Probably the worst way to do this.
                    coeff_min = min(np.nanmin(np.minimum(old_coeff, coeff)), coeff_min)
                    coeff_max = max(np.nanmax(np.maximum(old_coeff, coeff)), coeff_max)

                coeff_matrix[i] = coeff

                # Plot the lines
                ax.plot(x, coeff,
                        # color=color_dict[order](.7),
                        color=color_list[i],
                        linewidth=linewidth_list[i],
                        label=orders_name_dict[order],
                        zorder=i/10-1,
                        ls=linestyle_list[i])

                if make_inset:
                    axins.plot(
                        x, coeff,
                        # color=color_dict[order](.7),
                        color=color_list[i],
                        linewidth=linewidth_list[i],
                        label=orders_name_dict[order],
                        zorder=i/10-1,
                        ls=linestyle_list[i]
                        )

            # Decide the padding above/below the lines
            # This weights values far from 0 more heavily.
            # ymin = coeff_min - .25 * abs(coeff_min)
            # ymax = coeff_max + .25 * abs(coeff_max)
            # ymin = coeff_min
            # ymax = coeff_max
            ymin = coeff_min - .1 * max(abs(coeff_min), abs(coeff_max))
            ymax = coeff_max + .1 * max(abs(coeff_min), abs(coeff_max))

            # Round the min/max appropriately
            try:
                ymin_magnitude = min(int(math.floor(math.log10(abs(ymin)))), 0)
                # ymin = round(ymin, -ymin_magnitude)
                ymin = np.sign(ymin) * math.ceil(abs(ymin) / 10**ymin_magnitude) * 10**ymin_magnitude
            except ValueError:
                pass

            try:
                ymax_magnitude = min(int(math.floor(math.log10(abs(ymax)))), 0)
                # ymax = round(ymax, -ymax_magnitude)
                ymax = np.sign(ymax) * math.ceil(abs(ymax) / 10**ymax_magnitude) * 10**ymax_magnitude
            except ValueError:
                pass

            if ymin == 0 and ymax == 0:
                ymin = -1
                ymax = 1

            if observable == ['t', 't', 't', 't']:
                ymin = -4
                ymax = 5
            ax.set_ylim([ymin, ymax])
            ax.set_xlim([xmin, xmax])

            # Plot the 0-line
            ax.plot(x, np.zeros(len(x)), color=mp.rcParams['xtick.color'],
                    linewidth=mp.rcParams['xtick.major.width'],
                    zorder=-2, ls="--")

            # -----------------------
            # Posterior density stuff
            # -----------------------

            if is_posterior_plotted:

                # Order matrix such that rows -> energies, and cols -> orders
                coeff_matrix = np.array(coeff_matrix).T

                # Number of posterior evaluations (vertically)
                num_y = 200
                # delta_y = (ymax-ymin)/num_y

                unknown_coeff_range = np.linspace(ymin, ymax, num=num_y)

                posterior_matrix = []

                if orders == []:
                    k = 0
                    nc = 0
                    coeff_post = Delta_k_posterior(
                        prior_set=prior_set,
                        Q=1,
                        k=k,
                        nc=nc,
                        h=1,
                        coeffs=[],
                        cbar_lower=cbar_less,
                        cbar_upper=cbar_greater
                        )
                    post_list = [coeff_post(c) for c in unknown_coeff_range]
                    # print(post_list)
                    projected_title = r"pr($c_i$)"
                    for i in range(len(x)):
                        posterior_matrix.append(post_list)
                else:
                    k = order_dict[orders[-1]] - 1
                    nc = n_c_val(k, [0, 1])
                    projected_title = r"pr($c_i | "
                    for order in orders:
                        projected_title += r"c_" + str(order_dict[order] - 1) + r","
                    projected_title = projected_title[:-1]
                    projected_title += r"$)"

                for coeffs in coeff_matrix:
                    # The posterior for an unknown coefficient is the same
                    # as the Delta_k posterior, but with Q = 1 and h = 1.
                    coeff_post = Delta_k_posterior(
                        prior_set=prior_set,
                        Q=1,
                        k=k,
                        nc=nc,
                        h=1,
                        coeffs=coeffs,
                        cbar_lower=cbar_less,
                        cbar_upper=cbar_greater
                        )

                    posterior_matrix.append(
                        [coeff_post(c) for c in unknown_coeff_range]
                        )

                if indep_var == "theta":
                    x1 = 60
                    x2 = 120
                    indep_var_point1 = r"$\theta = " + str(x1) + r"$"
                    indep_var_point2 = r"$\theta = " + str(x2) + r"$"
                    indep_var_units = r" (deg)"
                else:
                    x1 = 50
                    x2 = 200
                    indep_var_point1 = r"$E_{\mathrm{lab}} = " + str(x1) + r"$"
                    indep_var_point2 = r"$E_{\mathrm{lab}} = " + str(x2) + r"$"
                    indep_var_units = r" (MeV)"

                # projected_title1 = projected_title + " at " + indep_var_point1
                # projected_title2 = projected_title + " at " + indep_var_point2

                x1_index, x2_index = np.argmin(np.abs(x - x1)), np.argmin(np.abs(x - x2))

                post1 = np.array(posterior_matrix[x1_index])
                post2 = np.array(posterior_matrix[x2_index])

                post1_color = "purple"
                post2_color = "purple"

                max_post = max(np.max(post1), np.max(post2))
                # min_post = min(np.min(post1), np.min(post2))
                min_post = 0
                max_post += .1*(max_post-min_post)
                # min_post -= .1*(max_post-min_post)

                if make_side_plots:
                    ax1.plot(-post1, unknown_coeff_range, c=post1_color)
                    ax1.set_title(projected_title, fontsize=10)
                    ax2.plot(post2, unknown_coeff_range, c=post1_color)
                    ax2.set_title(projected_title, fontsize=10)

                    ax1.set_xlim([-max_post, -min_post])
                    ax1.set_xlabel(indep_var_point1 + indep_var_units, fontsize=10)
                    ax2.set_xlim([min_post, max_post])
                    ax2.set_xlabel(indep_var_point2 + indep_var_units, fontsize=10)
                    ax1.tick_params(labelbottom='off')
                    ax2.tick_params(labelbottom='off', labelleft='off')
                    ax.tick_params(labelleft='off')

                # For plotting, row index corresponds to y and col index to x:
                posterior_matrix = np.array(posterior_matrix).T

                # Make smooth
                interp_post = interp2d(x, unknown_coeff_range,
                                       posterior_matrix, kind='linear')
                num_x = 200
                xnew = np.linspace(x[0], x[-1], num=num_x)
                posterior_matrix = interp_post(xnew, unknown_coeff_range)

                z_max = np.abs(posterior_matrix).max()
                z_min = z_max/100000
                ax.pcolormesh(
                    xnew, unknown_coeff_range, posterior_matrix,
                    cmap="inferno",
                    # vmin=z_min, vmax=z_max,
                    norm=colors.LogNorm(vmin=z_min, vmax=z_max),
                    edgecolors='face',
                    zorder=-10)

                if make_side_plots:
                    # ax.axvline(x=x1, ymin=0, ymax=0.05, c="r",)
                    # ax.axvline(x=x1, ymin=0.95, ymax=1, c="r",)
                    # ax.axvline(x=x2, ymin=0.95, ymax=1, c="r",)
                    # ax.axvline(x=x2, ymin=0, ymax=0.05, c="r",)
                    arrow_kwargs = {
                        "head_width": 10,
                        "head_length": 10,
                        "length_includes_head": True,
                        # "fc": 'purple',
                        # "ec": 'purple',
                        "width": 10
                    }
                    arrow_dy = 0.05
                    ax.arrow(x1, ymin, 0, arrow_dy, fc=post1_color, ec=post1_color, **arrow_kwargs)
                    ax.arrow(x2, ymin, 0, arrow_dy, fc=post2_color, ec=post2_color, **arrow_kwargs)
                    ax.arrow(x1, ymax, 0, -arrow_dy, fc=post1_color, ec=post1_color, **arrow_kwargs)
                    ax.arrow(x2, ymax, 0, -arrow_dy, fc=post2_color, ec=post2_color, **arrow_kwargs)
                # ax.imshow(
                #     posterior_matrix,
                #     cmap="Greys",
                #     interpolation="spline16",
                #     extent=[x[0], x[-1], ymin, ymax],
                #     # vmin=z_min, vmax=z_max,
                #     norm=colors.LogNorm(vmin=z_min, vmax=z_max),
                #     # edgecolors='none',
                #     zorder=-10)

            # -----------------------
            # Final touches
            # -----------------------

            # print(ax.get_tick_params)
            # print(mp.rcParams['xtick.major.width'])

            # Use block patches instead of lines
            # Use innermost "dark" color of bands for legend
            # legend_patches.append(
            #     mp.patches.Patch(
            #         color=color_dict[order](len(p_decimal_list) / (len(p_decimal_list) + 1)),
            #         label=order,
            #     ))
            # if len(orders) != 0:
            if is_posterior_plotted:
                if orders == []:
                    r = mp.patches.Rectangle(
                        (0, 0), 1, 1, fill=False, edgecolor='none',
                        visible=False)
                    leg = ax.legend(
                          [r], [""],
                          fontsize=8,
                          bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                          mode="expand",
                          borderaxespad=0.,
                          handlelength=2.15, handletextpad=.5,
                          )
                    leg.get_frame().set_edgecolor('w')
                else:
                    ax.legend(
                          fontsize=8, ncol=len(orders),
                          bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                          mode="expand",
                          borderaxespad=0.,
                          handlelength=2.15, handletextpad=.5)
            else:
                ax.legend(loc="best", fontsize=8,
                          handlelength=2.15, handletextpad=.5).set_title(text_str, prop={'size': 8})
            # ax.legend().set_title(text_str, prop={'size': 8})

            # Squeeze and save it
            fig.set_tight_layout(True)
            # plt.tight_layout()
            plot_name = plot_coeff_error_bands_filename(
                observable, indep_var, ivar_start, ivar_stop,
                ivar_step, param_var, param, orders, Lambda_b, lambda_mult,
                X_ref_hash, convention, prior_set=prior_set,
                cbar_lower=cbar_less, cbar_upper=cbar_greater,
                potential_info=None)
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
    # >> python plot_coefficients.py -h
    parser = argparse.ArgumentParser(
        description="Executable script to plot np observable coefficients."
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
        required=True, choices=["None", "LO", "NLO", "N2LO", "N3LO", "N4LO"])
    parser.add_argument(
        "--observables",
        metavar="p,q,i,k",
        nargs="+", required=True,
        help="The observables C_{pqik} to calculate.",
        type=lambda s: s.split(","))
    parser.add_argument(
        "--X_ref_hash",
        required=True,
        help="""The way X_ref should be calculated. In general, use "0dsigma".
            """,
        type=str
        )
    parser.add_argument(
        "--convention",
        required=True,
        help="The Stapp or Blatt phase convention.",
        choices=["stapp", "blatt"])
    parser.add_argument(
        "--posterior_shading_args",
        type=str, nargs=3,
        metavar=("prior_set", "cbar_less", "cbar_greater"),
        # required=True,
        help="""Adds pcolor shading overlay of posterior pr(c_i|c_2,...,c_k),
            which shows prediction of future coefficients based on known ones.
            Currently supports prior sets A and C. Set each variable to 'None'
            to ignore this, or remove arguments altogether.""")

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

    # print(arg_dict["posterior_shading_args"])
    if arg_dict["posterior_shading_args"] is not None:
        try:
            prior_set = arg_dict["posterior_shading_args"][0]
            cbar_less = float(arg_dict["posterior_shading_args"][1])
            cbar_greater = float(arg_dict["posterior_shading_args"][2])
        except ValueError:
            prior_set = None
            cbar_less = None
            cbar_greater = None
    else:
        prior_set = None
        cbar_less = None
        cbar_greater = None

    if arg_dict["orders"] == ["None"]:
        orders = []
    else:
        orders = arg_dict["orders"]

    main(
        coeff_dir=arg_dict["coeff_dir"],
        output_dir=arg_dict["output_dir"],
        indep_var=arg_dict["indep_var"],
        ivar_start=arg_dict["indep_var_range"][0],
        ivar_stop=arg_dict["indep_var_range"][1],
        ivar_step=arg_dict["indep_var_range"][2],
        param_list=param_lst,
        orders=orders,
        observable_list=arg_dict["observables"],
        Lambda_b=arg_dict["Lambda_b"],
        lambda_mult=arg_dict["lambda_mult"],
        interaction=arg_dict["interaction"],
        X_ref_hash=arg_dict["X_ref_hash"],
        convention=arg_dict["convention"],
        prior_set=prior_set,
        cbar_less=cbar_less,
        cbar_greater=cbar_greater
        )
