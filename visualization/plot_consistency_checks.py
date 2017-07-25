

import sys
import os
sys.path.append(os.path.expanduser(
    '~/Dropbox/Bayesian_codes/observable_analysis_Jordan'))

import argparse
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
from src.lowlevel.filenames import *
from src.lowlevel.datafile import DataFile
from src.lowlevel.EFT_functions import find_percent_success
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch, HandlerLine2D
from matplotlib.patches import Rectangle
# from scipy.stats import binom
from scipy.special import gamma
from src.lowlevel.CH_to_EKM_statistics import find_dimensionless_dob_limit
from scipy.stats._distn_infrastructure import (
        rv_discrete, rv_continuous, _lazywhere, _ncx2_pdf, _ncx2_cdf, get_distribution_names)
from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh
from scipy.special import entr, gammaln as gamln
from scipy import special
from scipy.stats import beta
from scipy.optimize import fmin
from matplotlib.ticker import AutoMinorLocator
from subprocess import call

# rcParams['ps.distiller.res'] = 60000
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# Change scipy's discrete binom class to a continous one
class binom_gen(rv_continuous):
    """A binomial continous random variable.
    %(before_notes)s
    Notes
    -----
    The probability mass function for `binom` is::
       binom.pmf(k) = choose(n, k) * p**k * (1-p)**(n-k)
    for ``k`` in ``{0, 1,..., n}``.
    `binom` takes ``n`` and ``p`` as shape parameters.
    %(after_notes)s
    %(example)s
    """
    def _rvs(self, n, p):
        return self._random_state.binomial(n, p, self._size)

    def _argcheck(self, n, p):
        self.b = n
        return (n >= 0) & (p >= 0) & (p <= 1)

    def _logpmf(self, x, n, p):
        # k = floor(x)
        k = x
        combiln = (gamln(n+1) - (gamln(k+1) + gamln(n-k+1)))
        return combiln + special.xlogy(k, p) + special.xlog1py(n-k, -p)

    def _pmf(self, x, n, p):
        return exp(self._logpmf(x, n, p))

    def _cdf(self, x, n, p):
        # k = floor(x)
        k = x
        vals = special.bdtr(k, n, p)
        return vals

    def _sf(self, x, n, p):
        # k = floor(x)
        k = x
        return special.bdtrc(k, n, p)

    def _ppf(self, q, n, p):
        # vals = ceil(special.bdtrik(q, n, p))
        vals = special.bdtrik(q, n, p)
        # vals1 = np.maximum(vals - 1, 0)
        vals1 = np.maximum(vals, 0)
        # temp = special.bdtr(vals1, n, p)
        # return np.where(temp >= q, vals1, vals)
        return np.where(vals1 >= vals, vals1, vals)

    def _stats(self, n, p, moments='mv'):
        q = 1.0 - p
        mu = n * p
        var = n * p * q
        g1, g2 = None, None
        if 's' in moments:
            g1 = (q - p) / sqrt(var)
        if 'k' in moments:
            g2 = (1.0 - 6*p*q) / var
        return mu, var, g1, g2

    def _entropy(self, n, p):
        k = np.r_[0:n + 1]
        vals = self._pmf(k, n, p)
        return np.sum(entr(vals), axis=0)
binom = binom_gen(name='binom')


class HandlerSquare(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = xdescent + 0.5 * (width - height), ydescent
        p = mpatches.Rectangle(xy=center, width=1.5*height,
                               height=height, angle=0.0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

orders_name_dict = {
        "NLO": r"NLO",
        "N2LO": r"N$^2$LO",
        "N3LO": r"N$^3$LO",
        "N4LO": r"N$^4$LO"
    }


def HDIofICDF(dist_name, credMass=0.95, **args):
    # freeze distribution with given arguments
    distri = dist_name(**args)
    # initial guess for HDIlowTailPr
    incredMass = 1.0 - credMass

    def intervalWidth(lowTailPr):
        return distri.ppf(credMass + lowTailPr) - distri.ppf(lowTailPr)

    # find lowTailPr that minimizes intervalWidth
    HDIlowTailPr = fmin(intervalWidth, incredMass, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return distri.ppf([HDIlowTailPr, credMass + HDIlowTailPr])


def main(
        error_band_dir,
        output_dir,
        theta_grid,
        energy_grid,
        theta_range,
        energy_range,
        observable_list,
        Lambda_b,
        lambda_mult_list,
        p_decimal_range,
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
        convention,
        combine_obs,
        separate_orders
        ):
    """A description. It plots stuff."""

    if theta_range is not None:
        t0 = theta_range[0]
        tf = theta_range[1]
        ts = theta_range[2]
        theta_values = [i for i in range(t0, tf, ts)]
    else:
        t0 = None
        tf = None
        ts = None
        theta_values = theta_grid

    if energy_range is not None:
        e0 = energy_range[0]
        ef = energy_range[1]
        es = energy_range[2]
        energy_values = [i for i in range(e0, ef, es)]
    else:
        e0 = None
        ef = None
        es = None
        energy_values = energy_grid

    if p_decimal_range is not None:
        p0 = p_decimal_range[0]
        pf = p_decimal_range[1]
        ps = p_decimal_range[2]
        p_values = [p/100 for p in range(p0, pf, ps)]
    else:
        p0 = None
        pf = None
        ps = None
        p_values = p_decimal_list

    if ['t', 't', 't', 't'] in observable_list:
        indep_var_list = [50, 96, 143, 200, 250, 300]
        if not set(energy_values) <= set(indep_var_list):
            indep_var_list = [
                20, 40, 60, 80, 100,
                120, 140, 160, 180, 200,
                220, 240, 260, 280, 300,
                320, 340]
    else:
        # indep_var_list = [20, 60, 90, 120, 160]
        indep_var_list = [20, 40, 60, 80, 100, 120, 140, 160]

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

    gray68 = plt.get_cmap("Greys")(.402)
    gray95 = plt.get_cmap("Greys")(.2)

    marker_list = ['s', 'D', 'o', '>', '^', '<']

    # lambda_color_list = [
    #     plt.get_cmap("Blues")(0.5),
    #     plt.get_cmap("Yellows")(0.5),
    #     plt.get_cmap("Reds")(0.5),
    # ]
    lambda_color_list = [
        "lightcoral",
        "gold",
        "cornflowerblue",
    ]

    lambda_marker_list = ['v', 'o', '^', 's', 'D', '>']

    fill_transparency = 1
    x = np.arange(0, 101, 20)
    y = np.arange(0, 101, 20)

    # fig = plt.figure(figsize=(3.4, 3.4))
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_zorder(15)
    ax.set_axisbelow(False)

    # if indep_var == "theta":
    #     param_var = "energy"
    #     indep_var_label = r"$\theta$ (deg)"
    #     param_var_label = r"$E_{\mathrm{lab}}"
    #     param_var_units = r"$ MeV"
    # else:
    #     param_var = "theta"
    #     indep_var_label = r"$E$ (MeV)"
    #     param_var_label = r"$\theta"
    #     param_var_units = r"^\circ$"

    # ax.set_ylabel('')

    # Create the description box
    # text_str = r"$C_{" + observable[0] + observable[1] + \
    #     observable[2] + observable[3] + r"}$" + ", "  # + "\n"

    # text_str = indices_to_observable_name(observable) + ", "
    # if observable != ['t', 't', 't', 't']:
    #     text_str += param_var_label + r" = " + str(param) + param_var_units + ", "  # + "\n"
    # text_str += r"$\Lambda_b = " + str(Lambda_b) + r"$ MeV"

    # ax.text(.5, .96, text_str,
    #         horizontalalignment='center',
    #         verticalalignment='top',
    #         multialignment='center',
    #         transform=ax.transAxes,
    #         bbox=dict(facecolor='white', alpha=1, boxstyle='square', pad=.5),
    #         zorder=20)
    plot_title = "Consistency Plot"
    legend_title = ""
    order_str = ""
    legend_patches = []

    p_percent_list = [100*i for i in p_values]

    if separate_orders is True:
        # for j in range(len(orders)-1):
        #     order_str = order_str + orders[j] + "-"
        # order_str = order_str[:-1]

        # if observable_list == [['t', 't', 't', 't']]:
        #     indep_var_list = energy_values
        # else:
        #     indep_var_list = theta_values

        if observable_list == [
                ['0', '0', 'n', '0'],
                ['n', '0', 'n', '0'],
                ['sp', '0', 'k', '0'],
                ['0', '0', 's', 's'],
                ['0', '0', 'n', 'n']
                ]:
            legend_title = r"$X_{pqik}$"
        elif observable_list == [
                ['0', '0', '0', '0'],
                ['0', '0', 'n', '0'],
                ['n', '0', 'n', '0'],
                ['sp', '0', 'k', '0'],
                ['0', '0', 's', 's'],
                ['0', '0', 'n', 'n']
                ]:
            legend_title = r"$d\sigma/d\Omega,X_{pqik}$"
        else:
            for observable in observable_list:
                legend_title = legend_title + indices_to_observable_name(observable) +  ", "
            legend_title = legend_title[:-2]

        for observable in observable_list:
            plot_title = plot_title + ", " + indices_to_observable_name(observable)

        for lambda_mult in lambda_mult_list:
            plot_title = plot_title + r", $\lambda =" + str(lambda_mult) + r"$"
            legend_title = legend_title + r", $\lambda =" + str(lambda_mult) + r"$"
            if h == 1:
                for j in range(len(orders)-h):
                    consistency_list = []
                    for i, p_decimal in enumerate(p_values):
                        succ_perc, N = find_percent_success(
                            error_band_dir, observable_list, theta_values, energy_values,
                            orders[j:j+h+1], ignore_orders, Lambda_b, lambda_mult,
                            X_ref_hash, p_decimal, prior_set, h, convention,
                            indep_var_list,
                            cbar_lower, cbar_upper, sigma,
                            potential_info=None)
                        consistency_list.append(100*succ_perc)
                    plot_label = orders_name_dict[orders[j]]  # + r" $\lambda =" + str(lambda_mult) + r"$"
                    ax.plot(p_percent_list, consistency_list,
                            label=plot_label, marker=marker_list[j],
                            color=color_list[j])
            elif h > 1:
                for j in range(len(orders)):
                    consistency_list = []
                    for i, p_decimal in enumerate(p_values):
                        succ_perc, N = find_percent_success(
                            error_band_dir, observable_list, theta_values, energy_values,
                            orders[j:j+1], ignore_orders, Lambda_b, lambda_mult,
                            X_ref_hash, p_decimal, prior_set, h, convention,
                            indep_var_list,
                            cbar_lower, cbar_upper, sigma,
                            potential_info=None)
                        consistency_list.append(100*succ_perc)
                    plot_label = orders_name_dict[orders[j]]  # + r" $\lambda =" + str(lambda_mult) + r"$"
                    ax.plot(p_percent_list, consistency_list,
                            label=plot_label, marker=marker_list[j],
                            color=color_list[j])

    elif combine_obs is True:
        # order_str = orders[-1]
        # if observable_list == [['t', 't', 't', 't']]:
        #     indep_var_list = energy_values
        # else:
        #     indep_var_list = theta_values

        if observable_list == [
                ['0', '0', 'n', '0'],
                ['n', '0', 'n', '0'],
                ['sp', '0', 'k', '0'],
                ['0', '0', 's', 's'],
                ['0', '0', 'n', 'n']
                ]:
            legend_title = r"$X_{pqik}$"
        elif observable_list == [
                ['0', '0', '0', '0'],
                ['0', '0', 'n', '0'],
                ['n', '0', 'n', '0'],
                ['sp', '0', 'k', '0'],
                ['0', '0', 's', 's'],
                ['0', '0', 'n', 'n']
                ]:
            legend_title = r"$d\sigma/d\Omega, X_{pqik}$"
        else:
            for observable in observable_list:
                legend_title = legend_title + indices_to_observable_name(observable) + ", "
            legend_title = legend_title[:-2]

        for observable in observable_list:
            plot_title = plot_title + ", " + indices_to_observable_name(observable)
        for j, lambda_mult in enumerate(lambda_mult_list):
            consistency_list = []
            for i, p_decimal in enumerate(p_values):
                succ_perc, N = find_percent_success(
                    error_band_dir, observable_list, theta_values, energy_values,
                    orders, ignore_orders, Lambda_b, lambda_mult,
                    X_ref_hash, p_decimal, prior_set, h, convention,
                    indep_var_list,
                    cbar_lower, cbar_upper, sigma,
                    potential_info=None)
                consistency_list.append(100*succ_perc)
            ax.plot(p_percent_list, consistency_list,
                    label=r"$\lambda =" + str(lambda_mult) + r"$",
                    marker=lambda_marker_list[j],
                    color=lambda_color_list[j])

    else:
        # order_str = orders[-1]
        for lambda_mult in lambda_mult_list:
            plot_title = plot_title + r", $\lambda =" + str(lambda_mult) + r"$"
            legend_title = legend_title + r", $\lambda =" + str(lambda_mult) + r"$"
            for j, observable in enumerate(observable_list):
                # if observable == ['t', 't', 't', 't']:
                #     indep_var_list = energy_values
                # else:
                #     indep_var_list = theta_values
                consistency_list = []
                for i, p_decimal in enumerate(p_values):
                    succ_perc, N = find_percent_success(
                        error_band_dir, [observable], theta_values, energy_values,
                        orders, ignore_orders, Lambda_b, lambda_mult,
                        X_ref_hash, p_decimal, prior_set, h, convention,
                        indep_var_list,
                        cbar_lower, cbar_upper, sigma,
                        potential_info=None)
                    consistency_list.append(100*succ_perc)
                plot_label = indices_to_observable_name(observable)  # + r" $\lambda =" + str(lambda_mult) + r"$"
                ax.plot(p_percent_list, consistency_list,
                        label=plot_label, marker=marker_list[j],
                        color=color_list[j])

    binom_68_u = []
    binom_68_l = []
    binom_95_u = []
    binom_95_l = []
    binom_CI_domain = [.1] + list(range(1, 101, 1))
    for p_decimal in binom_CI_domain:
        # interval_68 = binom.interval(.68, N, p_decimal/100, loc=0)
        # interval_68_perc = 100/N*interval_68[0], 100/N*interval_68[1]

        # binom_68_u.append(interval_68_perc[0])
        # binom_68_l.append(interval_68_perc[1])

        # interval_95 = binom.interval(.95, N, p_decimal/100, loc=0)
        # interval_95_perc = 100/N*interval_95[0], 100/N*interval_95[1]

        # binom_95_u.append(interval_95_perc[0])
        # binom_95_l.append(interval_95_perc[1])

        n = p_decimal*N/100
        beta_args = {"a": n+1, "b": N-n+1, "loc": 0, "scale": 1}

        # Equal-tail interval
        # interval_68 = beta.interval(.68, n+1, N-n+1, loc=0)
        # Highest density interval
        interval_68 = HDIofICDF(beta, credMass=0.68, **beta_args)
        # Convert to percent
        interval_68_perc = 100*interval_68[0], 100*interval_68[1]

        binom_68_u.append(interval_68_perc[0])
        binom_68_l.append(interval_68_perc[1])

        # Equal-tail interval
        # interval_95 = beta.interval(.95, n+1, N-n+1, loc=0)
        # Highest density interval
        interval_95 = HDIofICDF(beta, credMass=0.95, **beta_args)
        # Convert to percent
        interval_95_perc = 100*interval_95[0], 100*interval_95[1]

        binom_95_u.append(interval_95_perc[0])
        binom_95_l.append(interval_95_perc[1])

    ax.plot(x, x, color="black", zorder=0)

    # ax.plot(binom_CI_domain, binom_95_u, color="black", ls="--")
    # ax.plot(binom_CI_domain, binom_95_l, color="black", ls="--")
    # ax.plot(binom_95_u, binom_CI_domain, color="black", ls="--", zorder=0)
    # ax.plot(binom_95_l, binom_CI_domain, color="black", ls="--", zorder=0)
    ax.fill_betweenx(
        binom_CI_domain, binom_95_u, binom_95_l,
        facecolor=gray95,
        color=gray95,
        # alpha=.3,
        zorder=0)

    # ax.plot(binom_CI_domain, binom_68_u, color="black", ls="--")
    # ax.plot(binom_CI_domain, binom_68_l, color="black", ls="--")
    # ax.plot(binom_68_u, binom_CI_domain, color="black", ls="--", zorder=0)
    # ax.plot(binom_68_l, binom_CI_domain, color="black", ls="--", zorder=0)
    ax.fill_betweenx(
        binom_CI_domain, binom_68_u, binom_68_l,
        facecolor=gray68,
        color=gray68,
        # alpha=.4,
        zorder=0)
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

    # my_handles = legend_patches
    # handler_dict = dict(zip(my_handles, [HandlerSquare() for i in legend_patches]))

    # ax.legend(loc="best", handles=my_handles,
    #           handler_map=handler_dict,
    #           handletextpad=.7,
    #           handlelength=.6,
    #           fontsize=10)

    # plt.title(plot_title, fontsize=10)
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 100])
    ax.legend(loc="best", fontsize=8).set_title(legend_title, prop={'size': 8})
    ax.set_xlabel(r"DoB (\%)")
    ax.set_ylabel(r"Success Rate (\%), $N = " + str(N) + r"$")

    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)

    # Squeeze and save it
    plt.tight_layout()
    # plot_name = plot_obs_error_bands_filename(
    #         observable, indep_var, ivar_start, ivar_stop,
    #         ivar_step, param_var, param, orders[:i+1],
    #         Lambda_b, p_decimal_list)
    # plot_name = plot_obs_error_bands_filename(
    #     observable, indep_var, ivar_start, ivar_stop, ivar_step,
    #     param_var, param, orders[:i+1], Lambda_b, X_ref_hash, p_decimal_list,
    #     prior_set, h, convention, cbar_lower, cbar_upper, sigma,
    #     potential_info=None)
    # plot_name = "hello.pdf"

    plot_name = plot_consistency_filename(
        observable_list, p0, pf, ps,
        orders, ignore_orders, Lambda_b, lambda_mult_list,
        X_ref_hash, prior_set, h,
        convention, combine_obs,
        theta_start=t0, theta_stop=tf, theta_step=ts,
        energy_start=e0, energy_stop=ef, energy_step=es,
        theta_grid=theta_grid, energy_grid=energy_grid,
        indep_var_list=indep_var_list,
        cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma,
        potential_info=None, separate_orders=separate_orders
        )

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
        "Lambda_b",
        help="The breakdown scale of the EFT, given in MeV.",
        type=int)
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
        "--orders",
        help="The orders to show on the plots.",
        type=str, nargs="+",
        required=True, choices=["LOp", "LO", "NLO", "N2LO", "N3LO", "N4LO"])
    parser.add_argument(
        "--ignore_orders",
        help="The kth orders (Q^k) to ignore when calculating DoBs.",
        nargs="+", type=int,
        choices=[0, 1, 2, 3, 4, 5])
    theta_group = parser.add_mutually_exclusive_group(required=True)
    theta_group.add_argument(
        "--theta_values", "--tvals",
        type=int, nargs="+",
        help="""The theta values at which to examine consistency.""")
    theta_group.add_argument(
        "--theta_range", "--trange",
        type=int, nargs=3,
        metavar=("t_start", "t_stop", "t_step"),
        help="Cycle theta_value through [t_start, t_stop) in increments of t_step.")
    energy_group = parser.add_mutually_exclusive_group(required=True)
    energy_group.add_argument(
        "--energy_values", "--evals",
        type=int, nargs="+",
        help="""The energy values at which to examine consistency.""")
    energy_group.add_argument(
        "--energy_range", "--erange",
        type=int, nargs=3,
        metavar=("e_start", "e_stop", "e_step"),
        help="Cycle energy_value through [e_start, e_stop) in increments of e_step.")
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
    parser.add_argument(
        "--combine_obs",
        required=True,
        help="Choose whether separate observables should be combined in one consistency check",
        choices=["True", "False"])
    parser.add_argument(
        "--lambda_mult_list",
        nargs="+", required=True,
        help="The list of lambda values that multiplie Lambda_b.",
        type=float)
    parser.add_argument(
        "--separate_orders",
        required=True,
        help="Choose whether orders should be separated for comparison",
        choices=["True", "False"])

    args = parser.parse_args()
    arg_dict = vars(args)
    print(arg_dict)

    if arg_dict["prior_set"] == "B":
        cup = 0
        clow = 0
        sigma = arg_dict["sigma"]
    else:
        sigma = 0
        cup = arg_dict["cbar_upper"]
        clow = arg_dict["cbar_lower"]

    if arg_dict["combine_obs"] == "True":
        combine_obs = True
    else:
        combine_obs = False

    if arg_dict["separate_orders"] == "True":
        separate_orders = True
    else:
        separate_orders = False

    if arg_dict["ignore_orders"] is None:
        ignore_orders = []
    else:
        ignore_orders = arg_dict["ignore_orders"]

    main(
        error_band_dir=arg_dict["error_band_dir"],
        output_dir=arg_dict["output_dir"],
        theta_grid=arg_dict["theta_values"],
        energy_grid=arg_dict["energy_values"],
        theta_range=arg_dict["theta_range"],
        energy_range=arg_dict["energy_range"],
        observable_list=arg_dict["observables"],
        Lambda_b=arg_dict["Lambda_b"],
        lambda_mult_list=arg_dict["lambda_mult_list"],
        p_decimal_range=arg_dict["p_range"],
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
        convention=arg_dict["convention"],
        combine_obs=combine_obs,
        separate_orders=separate_orders)
