#######################
# Plots pr(Lambda_b|{b_i})
#######################

import argparse
import colorsys
import matplotlib as mpl
from matplotlib import rcParams, rc
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy import array, arange, vectorize
from scipy.interpolate import interp1d
from scipy.integrate import quad
import seaborn as sns
import seaborn.utils as utils
from seaborn.palettes import color_palette, husl_palette, light_palette
import pandas as pd
# import numpy as np
import os
import sys
import time
sys.path.append(os.path.expanduser(
    '~/Dropbox/Bayesian_codes/observable_analysis_Jordan/src'))
sys.path.append(os.path.expanduser(
    '~/Dropbox/Bayesian_codes/observable_analysis_Jordan/src/lowlevel'))
# from src.lowlevel.CH_to_EKM_statistics import *
from datafile import *
from CH_to_EKM_statistics import *
from violin_plot_functions import *
from filenames import plot_Lambda_violin_pdf_filename, Lambda_pdf_filename
from subprocess import call


rcParams['ps.distiller.res'] = 60000
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
linewidth = mpl.rcParams["lines.linewidth"]

sns.set_style(
    "whitegrid",
    {
        'font.family': [u'serif'],
        'font.serif': [u'Computer Modern'],
        'font.size': 10,
        'ytick.minor.color': 'gray',
        'ytick.direction': u'in',
        'ytick.minor.size': 7.0
    }
)


def main(
         data_dir,
         output_dir,
         theta_list,
         energy_list,
         orders,
         ignore_orders,
         observable_sets,
         interaction,
         X_ref_hash,
         prior_set,
         cbar_lower,
         cbar_upper,
         sigma,
         Lambda_prior,
         Lambda_lower,
         Lambda_upper,
         Lambda_mu,
         Lambda_sigma,
         convention,
         category,
         orient,
         hue,
         inner,
         split,
         onesided,
         palette,
         scale,
         Lmin,
         Lmax,
         Lstep):

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    observable_dict = {
        "sigma": [['t', 't', 't', 't']],
        "dsdO": [['0', '0', '0', '0']],
        "Ay": [['0', '0', 'n', '0']],
        "D": [['n', '0', 'n', '0']],
        "A": [['sp', '0', 'k', '0']],
        "Axx": [['0', '0', 's', 's']],
        "Ayy": [['0', '0', 'n', 'n']],
        "spins": [
            ['0', '0', 'n', '0'],
            ['n', '0', 'n', '0'],
            ['sp', '0', 'k', '0'],
            ['0', '0', 's', 's'],
            ['0', '0', 'n', 'n']
            ],
        "allspins": [
            ['0', '0', '0', '0'],
            ['0', '0', 'n', '0'],
            ['n', '0', 'n', '0'],
            ['sp', '0', 'k', '0'],
            ['0', '0', 's', 's'],
            ['0', '0', 'n', 'n']
            ]
    }

    ax.tick_params(axis='y', which='minor', colors='gray')

    ind_var_str = r"$\Lambda_b$\,(MeV)"

    observable_str_dict = {
        "sigma": r"$\sigma$",
        "dsdO": r"$\displaystyle\frac{d\sigma}{d\Omega}$",
        "Ay": r"$A_y$",
        "Axx": r"$A_{xx}$",
        "Ayy": r"$A_{yy}$",
        "spins": r"$X_{pqik}$",
        "allspins": r"$X_{pqik}$"
    }

    orders_name_dict = {
        "NLO": r"NLO",
        "N2LO": r"N$^2$LO",
        "N3LO": r"N$^3$LO",
        "N4LO": r"N$^4$LO"
    }

    df_list = []

    for observable_hash in observable_sets:
        for order in orders:
            if observable_dict[observable_hash] == [['t', 't', 't', 't']]:
                t_vals = [0]
            else:
                t_vals = theta_list
            # print(observable_hash)
            Lambda_filename = Lambda_pdf_filename(
                    obs_indices_list=observable_dict[observable_hash],
                    theta_list=t_vals,
                    energy_list=energy_list,
                    order=order,
                    ignore_orders=ignore_orders,
                    X_ref_hash=X_ref_hash,
                    prior_str=prior_set,
                    convention=convention,
                    cbar_lower=cbar_lower,
                    cbar_upper=cbar_upper,
                    sigma=sigma,
                    Lambda_prior=Lambda_prior,
                    Lambda_lower=Lambda_lower,
                    Lambda_upper=Lambda_upper,
                    Lambda_mu=Lambda_mu,
                    Lambda_sigma=Lambda_sigma,
                    potential_info=None)

            Lambda_file = DataFile().read(os.path.join(data_dir, Lambda_filename))
            Lambda_vals = Lambda_file[0]
            Lambda_pdf = Lambda_file[1]

            df_list.append(pd.DataFrame(
                {
                    ind_var_str: Lambda_vals,
                    "pdf": Lambda_pdf,
                    "Observable": observable_str_dict[observable_hash],
                    "Order": orders_name_dict[order]
                }
            ))

    df = pd.concat(df_list)

    # print("Plot time")
    sns.set_color_codes(palette)
    violinfunctionplot(x=ind_var_str, y="pdf", category=category, data=df,
                       orient=orient, width=.8, hue=hue, inner=inner,
                       split=split,
                       palette=sns.color_palette(["b", "r"]),
                       # color=["b", "r", "b", "r"],
                       scale=scale, ax=ax,
                       onesided=onesided, HDI=True)

    # major_ticks = np.arange(0, 1501, 300)
    major_ticks = np.arange(Lmin, Lmax+1, Lstep)
    # minor_ticks = np.arange(0, 800, 100)

    # ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks)
    # ax.set_xlim([0, 1501])
    ax.set_xlim([Lmin, Lmax+1])

    # print("?")
    plot_name = plot_Lambda_violin_pdf_filename(
        obs_sets=observable_sets, theta_list=theta_list,
        energy_list=energy_list, orders=orders, ignore_orders=ignore_orders,
        X_ref_hash=X_ref_hash, prior_str=prior_set, convention=convention,
        cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma,
        Lambda_prior=Lambda_prior, Lambda_lower=Lambda_lower,
        Lambda_upper=Lambda_upper, Lambda_mu=Lambda_mu,
        Lambda_sigma=Lambda_sigma,
        potential_info=None, category=category, orient=orient,
        hue=hue, inner=inner, split=split, palette=palette, scale=scale)
    # plt.show()
    plt.draw()
    plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")

    # To fix the blurriness issue when pdf is put into a LaTeX doc.
    # Generate .eps file and then change to .pdf
    # I don't know the root problem.
    call(["epstopdf", os.path.join(output_dir, plot_name)])
    call(["rm", os.path.join(output_dir, plot_name)])

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
        "data_dir",
        help="The directory in which the Lambda_b pdf data are stored.")
    parser.add_argument(
        "output_dir",
        help="The relative path where output files will be stored")
    parser.add_argument(
        "interaction",
        help="The type of scattering interaction.",
        choices=["nn", "pp", "np"])
    parser.add_argument(
        "prior_set",
        help="The string corresponding to a given prior set.",
        choices=["A", "B", "C"])
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
        "Lambda_prior_set",
        help="The string corresponding to a given Lambda_b prior set.",
        choices=["u", "uu", "g"])
    parser.add_argument(
        "Lambda_lower",
        help="Lower bound for Lambda_b.",
        type=float)
    parser.add_argument(
        "Lambda_upper",
        help="Upper bound for Lambda_b.",
        type=float)
    parser.add_argument(
        "Lambda_mu",
        help="For Lambda prior set g: mean of Lambda_b.",
        type=float)
    parser.add_argument(
        "Lambda_sigma",
        help="For Lambda prior set g: standard deviation of Lambda_b.",
        type=float)
    theta_group = parser.add_mutually_exclusive_group(required=True)
    theta_group.add_argument(
        "--theta_range", "--trange",
        type=int, nargs=3,
        metavar=("start", "stop", "step"),
        # required=True,
        help="Cycle theta through [start, stop) in increments of step.")
    theta_group.add_argument(
        "--theta_values", "--tvals",
        help="The values of the indep var at which to find error bands.",
        type=int, nargs="+")
    energy_group = parser.add_mutually_exclusive_group(required=True)
    energy_group.add_argument(
        "--energy_values", "--evals",
        type=int, nargs="+",
        help="""The values of energy to use in Lambda_b pdf.""")
    energy_group.add_argument(
        "--energy_range",
        type=int, nargs=3,
        metavar=("start", "stop", "step"),
        help="Cycle energy through [start, stop) in increments of step to use in pdf.")
    # parser.add_argument(
    #     "--orders",
    #     help="The order up to (and including) which to extract coefficients.",
    #     required=True, type=int,
    #     choices=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--orders",
        help="The orders at which to calculate DoBs.",
        nargs="+", required=True,
        choices=["NLO", "N2LO", "N3LO", "N4LO"])
    parser.add_argument(
        "--ignore_orders",
        help="The kth orders (Q^k) to ignore when calculating DoBs.",
        nargs="+", type=int,
        choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument(
        "--observable_sets",
        metavar="obs",
        nargs="+", required=True,
        help="The observables C_{pqik} to calculate.",
        type=str)
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
        "--category",
        required=True,
        help="The variable to use as the category.",
        choices=["Observable", "Order"])
    parser.add_argument(
        "--hue",
        required=True,
        help="The variable to use as the hue (subcategory).",
        choices=["Observable", "Order"])
    parser.add_argument(
        "--orient",
        required=True,
        help="The orientation of the plot.",
        choices=["h", "v"])
    parser.add_argument(
        "--inner",
        # required=True,
        help="The method of showing DoB intervals.",
        choices=["center", "orth"])
    parser.add_argument(
        "--split",
        required=True,
        help="Whether or not to split the (2) subcategories.",
        choices=["True", "False"])
    parser.add_argument(
        "--onesided",
        required=True,
        help="Whether or not to make two upright halves of violins.",
        choices=["True", "False"])
    parser.add_argument(
        "--palette",
        required=True,
        help="The color palette to fill the plots")
    parser.add_argument(
        "--scale",
        required=True,
        help="How to scale the plots.",
        choices=["area", "width"])
    parser.add_argument(
        "--Lmin",
        required=True,
        help="The lower limit on the plotted Lambda domain.",
        type=int)
    parser.add_argument(
        "--Lmax",
        required=True,
        help="The upper limit on the plotted Lambda domain.",
        type=int)
    parser.add_argument(
        "--Lstep",
        required=True,
        help="The increments to draw gridlines on Lambda domain.",
        type=int)

    args = parser.parse_args()
    arg_dict = vars(args)
    print(arg_dict)

    if arg_dict["energy_range"] is not None:
        e0 = arg_dict["energy_range"][0]
        ef = arg_dict["energy_range"][1]
        es = arg_dict["energy_range"][2]
        energy_lst = [i for i in range(e0, ef, es)]
    else:
        energy_lst = arg_dict["energy_values"]

    if arg_dict["theta_range"] is not None:
        t0 = arg_dict["theta_range"][0]
        tf = arg_dict["theta_range"][1]
        ts = arg_dict["theta_range"][2]
        theta_lst = [i for i in range(t0, tf, ts)]
    else:
        theta_lst = arg_dict["theta_values"]

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

    if arg_dict["split"] == "True":
        split = True
    else:
        split = False

    if arg_dict["onesided"] == "True":
        onesided = True
    else:
        onesided = False

    if arg_dict["Lambda_prior_set"] == "u" or arg_dict["Lambda_prior_set"] == "uu":
        Lmu = 0
        Lsig = 0
        Ll = arg_dict["Lambda_lower"]
        Lu = arg_dict["Lambda_upper"]
    elif arg_dict["Lambda_prior_set"] == "g":
        Lmu = arg_dict["Lambda_mu"]
        Lsig = arg_dict["Lambda_sigma"]
        Ll = 0
        Lu = 0

    main(
        data_dir=arg_dict["data_dir"],
        output_dir=arg_dict["output_dir"],
        # indep_var=arg_dict["indep_var"],
        # ivar_start=arg_dict["indep_var_range"][0],
        # ivar_stop=arg_dict["indep_var_range"][1],
        # ivar_step=arg_dict["indep_var_range"][2],
        theta_list=theta_lst,
        energy_list=energy_lst,
        orders=arg_dict["orders"],
        ignore_orders=ignore_orders,
        observable_sets=arg_dict["observable_sets"],
        interaction=arg_dict["interaction"],
        # p_decimal_list=p_grid,
        X_ref_hash=arg_dict["X_ref_hash"],
        prior_set=arg_dict["prior_set"],
        cbar_lower=clow,
        cbar_upper=cup,
        sigma=sigma,
        Lambda_prior=arg_dict["Lambda_prior_set"],
        Lambda_lower=Ll,
        Lambda_upper=Lu,
        Lambda_mu=Lmu,
        Lambda_sigma=Lsig,
        convention=arg_dict["convention"],
        category=arg_dict["category"],
        orient=arg_dict["orient"],
        hue=arg_dict["hue"],
        inner=arg_dict["inner"],
        split=split,
        onesided=onesided,
        palette=arg_dict["palette"],
        scale=arg_dict["scale"],
        Lmin=arg_dict["Lmin"],
        Lmax=arg_dict["Lmax"],
        Lstep=arg_dict["Lstep"])

