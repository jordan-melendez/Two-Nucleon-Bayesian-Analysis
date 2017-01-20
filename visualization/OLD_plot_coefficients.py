#################################################################
# Plot posteriors for Lambda_b from generated output files
#
# !!!!******** IMPORTANT PLOTTING NOTE **********************!!!!
# !!!! Should be from observables at the same momentum/ group!!!!
# !!!! Or else title will be wrong                           !!!!
# !!!!******** IMPORTANT PLOTTING NOTE **********************!!!!
#
#################################################################

import numpy as np
import math as m
import re
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import pylab
from matplotlib.ticker import MaxNLocator
from scipy.integrate import quad
# from num_to_str import number_to_str
# import dobs

tex_dict = {
    # No measurement
    "0": r"0",
    # Beam frame
    "x": r"x",
    "y": r"y",
    "z": r"z",
    # Scattered particle frame
    "m": r"x\prime",
    "n": r"y",
    "ell": r"z\prime",
    # Different names for the same thing
    "xp": r"x\prime",
    "yp": r"y\prime",
    "zp": r"z\prime",
    # Recoil particle frame
    "xpp": r"x\prime\prime",
    "ypp": r"y\prime\prime",
    "zpp": r"z\prime\prime",
}
#################################################################
# Name of coefficient file to be plotted
#################################################################
path_prefix = "./"
path_suffix = "/"
scattered_pol = "0"
recoil_pol = "0"
beam_pol = "y"
target_pol = "y"
indep_var_name = r"\theta"
indep_var_units = "[deg]"
E = 200  # 50, 96, 143, 200

tex_observ_name = r"$C_{" + tex_dict[scattered_pol] + recoil_pol + \
            beam_pol + target_pol + r"}$"


def filename(E):
    name = "C_" + scattered_pol + "-" + recoil_pol + "-" + \
            beam_pol + "-" + target_pol + "_vs_theta_E" + \
            str(E) + "_Lamb600_coefficients" + ".txt"
    return name


# fileNames = [os.path.join(path_prefix, path_suffix, filename(order, E))
#              for order in orders]

description = 'plots'
plotFileName = filename(E).split('.txt')[0] + '_' + description + '.pdf'

labels = [r'LO',
          r'NLO',
          r'N$^2$LO',
          r'N$^3$LO',
          r'N$^4$LO']

xlims = [0, 180]  # Degrees


#################################################################
# Set up plot stuff
#################################################################

colors = ['#00848D', '#50008D', '#8D0900', '#3E8D00', 'y']  # tetradic
font = {
    # 'family' : 'normal',
    # 'weight' : 'normal',
    'size': 16}
mpl.rc('font', **font)
mpl.rc('text', usetex=False)
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',\
#                                        r'\usepackage{color}']
mpl.rcParams['text.latex.unicode'] = True
mpl.rc('axes', linewidth=1)

# Don't want x and y axis tick labels to overlap
pylab.rcParams['xtick.major.pad'] = '10'


# Looks like the standard Latex math font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

# w, h = mpl.figure.figaspect(1.4)
fig = plt.figure(figsize=(6.9, 7.1))
ax = fig.add_subplot(111)

#ax.set_ylabel(tex_observ_name)
ax.set_xlabel(r'$' + indep_var_name + r'$ ' + indep_var_units)

plt_text = tex_observ_name + '\n' + "E = " + str(E)
plt.text(0.6, 0.9, plt_text, ha='left', va='center',
         transform=ax.transAxes)
plt.tight_layout()


ax.set_xlim(xlims)
# Controls how many ticks to label
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.xaxis.set_major_locator(MaxNLocator(6))

#################################################################
# Unpack each data file and plot it on the axes
#################################################################


# with open(file) as f:
#     content = f.readlines()
data = np.loadtxt(filename(E), skiprows=2, unpack=True)
indep_var = data[0, :]
LO = data[1, :]
NLO = data[2, :]
N2LO = data[3, :]
N3LO = data[4, :]
N4LO = data[5, :]
nl = data[0, :].shape[0]

plots = [LO, NLO, N2LO, N3LO, N4LO]

#################################################################
# Plot limits on lambda
#################################################################

ax.axvline(x=indep_var[0], linestyle='-', linewidth=0.8, color='g')
ax.axvline(x=indep_var[nl-1], linestyle='-', linewidth=0.8, color='g')

#################################################################
# Do the plot of data file k
#################################################################

for k, coeff in enumerate(plots):
    ax.plot(indep_var, coeff, linewidth=2,
            color=colors[k], label=labels[k])


# ttl = r"C_{pqik}"

# plt.title(ttl, fontsize=14, loc='center')
plt.legend(loc=0, fontsize=13)
plt.savefig(plotFileName)

#################################################################
# End of file
#################################################################
