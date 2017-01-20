#######################
# Plots pr(Delta_k|{c_i})
#######################

from matplotlib import rcParams, rc
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy import array
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


minorLocator = MultipleLocator(.2)
# for the minor ticks, use no labels; default NullFormatter
ax.yaxis.set_minor_locator(minorLocator)

# minorLocator = MultipleLocator(.04)
# # for the minor ticks, use no labels; default NullFormatter
# ax.xaxis.set_minor_locator(minorLocator)

x_max = 0.6
x_step = 0.005
delta_domain = np.arange(-x_max, x_max, x_step)
ctup = 1.0174161613422765, -0.28537045814544998, 1.2004338183631127, -0.67167654634822205, 5.1436659342924873
# From total cross section at E = 50
# ctup = 1.0, -1.431187418674369, 0.150200003435126, -0.208208006179793, \
#     3.313000457353716
cbar_eps = 0.001
cbar_lower = cbar_eps
cbar_upper = 1/cbar_eps
sigma = 1.0
Q = .33
k = 3
n_c = n_c_val(k, [1])

text_str = r"$\mathrm{pr}(\Delta_" + str(k) + "|"
for i, c in enumerate(ctup[:k+1]):
    text_str = text_str + "c_" + str(i) + "=" + str(c) + ", "
text_str = text_str[:-2] + r")$"

ax.set_title(text_str)
ax.set_xlabel(r'$\Delta_' + str(k) + r"$")
# ax.axis([-x_max, x_max, 0, 17])


ctup = 1.0174161613422765, -0.28537045814544998, 1.2004338183631127, -0.67167654634822205, 5.1436659342924873
prior_set = "A"
h = 1
# posterior_1 = v_pr_Delta_k(
#     0, prior_set=prior_set, Q=Q, k=k, nc=n_c, h=h,
#     coeffs=ctup, cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma)
pr_dk = Delta_k_posterior(
    prior_set=prior_set, Q=Q, k=k, nc=n_c, h=h,
    coeffs=ctup, cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma)
v_pr_dk = np.vectorize(pr_dk)
print(pr_dk(0))
t1 = time.time() - t0
print("Time:", t1, "seconds")
print(pr_dk(.2))
t2 = time.time() - t1 - t0
print("Time:", t2, "seconds")
print(pr_dk(.3))
t3 = time.time() - t2 - t0
print("Time:", t3, "seconds")

x_crit = find_insignificant_x(pr_dk)
print(x_crit, pr_dk(x_crit)/pr_dk(0))
t4 = time.time() - t3 - t0
print("Time:", t4, "seconds")

delta_domain = np.arange(0, x_crit, x_crit/100)
dk = find_dimensionless_dob_limit(pr_dk, x_mode=0, delta_x=x_crit/100, dob=.68)
dob_range = np.arange(0, dk, x_crit/100)
ax.fill_between(dob_range, 0, v_pr_dk(dob_range),
                facecolor="red", color="red", alpha=.3)
t5 = time.time() - t4 - t0
print("Time p=.68:", t5, "seconds")
dk = find_dimensionless_dob_limit(pr_dk, x_mode=0, delta_x=x_crit/100, dob=.95)
dob_range = np.arange(0, dk, x_crit/100)
ax.fill_between(dob_range, 0, v_pr_dk(dob_range),
                facecolor="red", color="red", alpha=.3)
t6 = time.time() - t5 - t0
print("Time p=.95:", t6, "seconds")
label = r"$" + prior_set + r"^{(" + str(h) + r")}$"
plt.plot(delta_domain, v_pr_dk(delta_domain), 'r-', label=label)
t7 = time.time() - t6 - t0
print("Time total plot:", t7, "seconds")

# Show the grid spacing with the minor ticks to determine if reasonable
minorLocator = MultipleLocator(x_crit/100)
ax.xaxis.set_minor_locator(minorLocator)





prior_set = "C"
h = 1
# posterior_1 = v_pr_Delta_k(
#     0, prior_set=prior_set, Q=Q, k=k, nc=n_c, h=h,
#     coeffs=ctup, cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma)
pr_dk = Delta_k_posterior(
    prior_set=prior_set, Q=Q, k=k, nc=n_c, h=h,
    coeffs=ctup, cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma)
v_pr_dk = np.vectorize(pr_dk)
print(pr_dk(0))
t1 = time.time() - t0
print("Time:", t1, "seconds")
print(pr_dk(.2))
t2 = time.time() - t1 - t0
print("Time:", t2, "seconds")
print(pr_dk(.3))
t3 = time.time() - t2 - t0
print("Time:", t3, "seconds")

x_crit = find_insignificant_x(pr_dk)
print(x_crit, pr_dk(x_crit)/pr_dk(0))
t4 = time.time() - t3 - t0
print("Time:", t4, "seconds")

delta_domain = np.arange(0, x_crit, x_crit/100)
dk = find_dimensionless_dob_limit(pr_dk, x_mode=0, delta_x=x_crit/100, dob=.15)
dob_range = np.arange(0, dk, x_crit/100)
# ax.fill_between(dob_range, 0, v_pr_dk(dob_range),
#                 facecolor="blue", color="blue", alpha=.3)
ax.vlines([dk], ymin=[0, 0], ymax=v_pr_dk([dk]),
          color="blue", linestyles="--")
t5 = time.time() - t4 - t0
print("Time p=.68:", t5, "seconds")
dk = find_dimensionless_dob_limit(pr_dk, x_mode=0, delta_x=x_crit/100, dob=.95)
dob_range = np.arange(0, dk, x_crit/100)
# ax.fill_between(dob_range, 0, v_pr_dk(dob_range),
#                 facecolor="blue", color="blue", alpha=.3)
ax.vlines([dk], ymin=[0, 0], ymax=v_pr_dk([dk]),
          color="blue", linestyles="--")
t6 = time.time() - t5 - t0
print("Time p=.95:", t6, "seconds")
label = r"$" + prior_set + r"^{(" + str(h) + r")}$"
plt.plot(delta_domain, v_pr_dk(delta_domain), 'b-', label=label)
t7 = time.time() - t6 - t0
print("Time total plot:", t7, "seconds")

#########################
# Plot 1
#########################
# prior_set = "A"
# h = 4
# posterior_1 = v_pr_Delta_k(
#     delta_domain, prior_set=prior_set, Q=Q, k=k, nc=n_c, h=h,
#     coeffs=ctup, cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma)

# post_func_1 = partial(np.interp, xp=delta_domain, fp=posterior_1)
# dk = find_dimensionless_dob_limit(post_func_1, x_mode=0, dob=.68)
# dk_range = np.arange(-dk, dk, x_step)
# ax.fill_between(dk_range, 0, post_func_1(dk_range),
#                 facecolor="red", color="red", alpha=.3)

# dk = find_dimensionless_dob_limit(post_func_1, x_mode=0, dob=.95)
# dk_range = np.arange(-dk, dk, x_step)
# ax.fill_between(dk_range, 0, post_func_1(dk_range),
#                 facecolor="red", color="red", alpha=.3)

# label = r"$" + prior_set + r"^{(" + str(h) + r")}$"
# plt.plot(delta_domain, posterior_1, 'r-', label=label)


#########################
# Plot 2
#########################
# prior_set = "C"
# h = 4
# posterior_2 = v_pr_Delta_k(
#     delta_domain, prior_set=prior_set, Q=Q, k=k, nc=n_c, h=h,
#     coeffs=ctup, cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma)

# post_func_2 = partial(np.interp, xp=delta_domain, fp=posterior_2)
# dk = find_dimensionless_dob_limit(post_func_2, x_mode=0, dob=.68)
# ax.vlines([-dk, dk], ymin=[0, 0], ymax=post_func_2([-dk, dk]),
#           color="blue", linestyles="-.")

# dk = find_dimensionless_dob_limit(post_func_2, x_mode=0, dob=.95)
# ax.vlines([-dk, dk], ymin=[0, 0], ymax=post_func_2([-dk, dk]),
#           color="blue", linestyles="--")


# label = r"$" + prior_set + r"^{(" + str(h) + r")}$"
# plt.plot(delta_domain, posterior_2, 'b-', label=label)


# #########################
# # Plot 3
# #########################
# # prior_set = "C"
# # h = 1
# # posterior_3 = v_pr_Delta_k(
# #     delta_domain, prior_set=prior_set, Q=Q, k=k, nc=n_c, h=h,
# #     coeffs=ctup, cbar_lower=cbar_lower, cbar_upper=cbar_upper, sigma=sigma)

# # post_func_3 = partial(np.interp, xp=delta_domain, fp=posterior_3)
# # dk = find_dimensionless_dob_limit(post_func_3, x_mode=0, dob=.68)
# # ax.vlines([-dk, dk], ymin=[0, 0], ymax=post_func_3([-dk, dk]),
# #           color="green", linestyles="-.")

# # dk = find_dimensionless_dob_limit(post_func_3, x_mode=0, dob=.95)
# # ax.vlines([-dk, dk], ymin=[0, 0], ymax=post_func_3([-dk, dk]),
# #           color="green", linestyles="--")


# # label = r"$" + prior_set + r"^{(" + str(h) + r")}$"
# # plt.plot(delta_domain, posterior_3, 'g-', label=label)

# Plot it!
ax.legend()
print("Time:", time.time() - t0)
plt.show()
# fig.savefig(text_str + ".pdf")
