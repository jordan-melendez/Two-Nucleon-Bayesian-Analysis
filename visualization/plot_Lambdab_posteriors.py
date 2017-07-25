#######################
# Plots pr(Lambda_b|{b_i})
#######################

from matplotlib import rcParams, rc
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy import array
from scipy.interpolate import interp1d
from scipy.integrate import quad
# import numpy as np
import os
import sys
import time
sys.path.append(os.path.expanduser(
    '~/Dropbox/Bayesian_codes/observable_analysis_Jordan/src'))
sys.path.append(os.path.expanduser(
    '~/Dropbox/Bayesian_codes/observable_analysis_Jordan/src/lowlevel'))
# from src.lowlevel.CH_to_EKM_statistics import *
from CH_to_EKM_statistics import *


t0 = time.time()
#########################
# Plot stuff
#########################

rcParams['ps.distiller.res'] = 60000
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# delta_k1_range = [pr_dk_true(d) for d in delta_domain]
# delta_kh_range = [pr_dk_MC(d) for d in delta_domain]

#########################
# Common plot parameters
#########################

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


# minorLocator = MultipleLocator(.2)

# major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 1600, 300)
minor_ticks = np.arange(0, 1600, 100)

ax.set_xticks(minor_ticks, minor=True)
ax.set_xticks(major_ticks)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)

orig_Lb = 600
barray = array([[1, 1/orig_Lb, 1/orig_Lb**2, 1/orig_Lb**3, 1/orig_Lb**4]]).T
barray = array([[1, 0/orig_Lb, -1.43/orig_Lb**2, 0.16/orig_Lb**3]]).T
barray = array(
    [[1, 1, 1, 1],
     [0, 0, 0, 0],
     [-1.43/orig_Lb**2, -0.92/orig_Lb**2, -0.35/orig_Lb**2, -0.11/orig_Lb**2],
     [0.16/orig_Lb**3, 0.86/orig_Lb**3, 1.21/orig_Lb**3, 1.44/orig_Lb**3],
     [-0.26/orig_Lb**4, -0.61/orig_Lb**4, -0.27/orig_Lb**4, 0.25/orig_Lb**4]
     ]).T
barray = array(
    [[1, 1, 1, 1],
     [0, 0, 0, 0],
     [-1.4323299/orig_Lb**2, -0.9162/orig_Lb**2, -0.35/orig_Lb**2, 0.11/orig_Lb**2],
     [0.1549449/orig_Lb**3, 0.86/orig_Lb**3, 1.21/orig_Lb**3, 1.44/orig_Lb**3],
     [-0.25763/orig_Lb**4, -0.60528/orig_Lb**4, -0.27/orig_Lb**4, 0.25/orig_Lb**4]
     ]).T
barray = array([
 [1, 0, -4.24e-6, 0.763e-9, -1.82e-12, 50.2e-15],
 [1, 0, -2.63e-6, 4.14e-9, -4.86e-12, 14.7e-15],
 [1, 0, -0.975e-6, 5.77e-9, -2.18e-12, 2.53e-15],
 [1, 0, 0.300e-6, 6.66e-9, 2.07e-12, -5.70e-15]
 ])
prior_set = "C"
k = 4
Lambda_lower = 300
Lambda_upper = 1500
cbar_lower = 0.001
cbar_upper = 1000
Lambda_domain = np.arange(0, 1600)

# kbarray = barray[:k+1]
Lambda_list = array([1.0 for Lb in Lambda_domain])
for i in range(len(barray)):
    kbarray = barray[i]
    # print(kbarray)
    Lambda_b_posterior = Lambda_b_pdf(prior_set, k, kbarray, Lambda_lower, Lambda_upper, cbar_lower, cbar_upper)
    Lambda_list *= array([Lambda_b_posterior(Lb) for Lb in Lambda_domain])


Lamb_pdf = interp1d(Lambda_domain, Lambda_list, kind="linear")
norm = 1/quad(Lamb_pdf, Lambda_lower, Lambda_upper)[0]

plt.plot(Lambda_domain, norm * Lambda_list, '-', color="purple", label=r"$\Lambda_b$")

Lambda_func = interp1d(Lambda_domain, norm * Lambda_list)
fill_color = "blue"
lower95 = find_dimensionless_dob_limit(
    Lambda_func, x_mode=Lambda_lower, dob=2*.025, delta_x=10)
upper95 = find_dimensionless_dob_limit(
    Lambda_func, x_mode=Lambda_lower, dob=2*.975, delta_x=10)
dob95 = np.arange(lower95, upper95)
label95 = r"95\% DoB = [{:4.0f},{:4.0f}] MeV".format(lower95, upper95)
# fill95 = '#7daaf2'
fill95 = '#73C6B6'
ax.fill_between(dob95, 0, Lambda_func(dob95),
                facecolor=fill95, color=fill95, alpha=.3, label=label95)
lower68 = find_dimensionless_dob_limit(
    Lambda_func, x_mode=Lambda_lower, dob=2*.16, delta_x=10)
upper68 = find_dimensionless_dob_limit(
    Lambda_func, x_mode=Lambda_lower, dob=2*.84, delta_x=10)
dob68 = np.arange(lower68, upper68)
label68 = r"68\% DoB = [{:4.0f},{:4.0f}] MeV".format(lower68, upper68)
fill68 = '#68a0f9'
fill68 = '#45B39D'
ax.fill_between(dob68, 0, Lambda_func(dob68),
                facecolor=fill68, color=fill68, alpha=1, label=label68)
median = find_dimensionless_dob_limit(
    Lambda_func, x_mode=Lambda_lower, dob=2*.5, delta_x=10)
labelmedian = r"Median = {:4.0f} MeV".format(median)
ax.vlines(median, [0], Lambda_func(median), color="#16A085", label=labelmedian)
plt.axvline(x=Lambda_lower, color="orange", ls="--")
plt.axvline(x=Lambda_upper, color="orange", ls="--")
plt.legend(fontsize=10)


# Plot it!
ax.legend()
print("Time:", time.time() - t0)
plt.show()
