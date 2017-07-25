import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import theano.tensor as tt
import pymc3 as pm
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from scipy.interpolate import interp1d
import os
import sys
sys.path.append(os.path.expanduser(
    '~/Dropbox/Bayesian_codes/observable_analysis_Jordan/src'))
sys.path.append(os.path.expanduser(
    '~/Dropbox/Bayesian_codes/observable_analysis_Jordan/src/lowlevel'))
from filenames import coeff_filename
from datafile import DataFile
from EFT_functions import Q_ratio
from kinematics import E_to_p
from pymc3.distributions.distribution import Continuous
import mpmath
from math import pi
from matplotlib import rc
from matplotlib import rcParams
from pymc3 import MvStudentT
import theano
from pymc3.distributions.special import gammaln
from theano.tensor.nlinalg import det, matrix_inverse, trace


rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
rcParams['text.usetex'] = True

# For reproducibility
np.random.seed(4)


def rbf_kernel(x, xp, ell):
    return np.exp(-(x-xp)**2 / ell**2)


def sum_sq(lst):
    val = 0
    for item in lst:
        val += item**2
    return val


class MvStudentTRegulated(MvStudentT):
    R"""
    Multivariate Student-T log-likelihood.

    .. math::
        f(\mathbf{x}| \nu,\mu,\Sigma) =
        \frac
            {\Gamma\left[(\nu+p)/2\right]}
            {\Gamma(\nu/2)\nu^{p/2}\pi^{p/2}
             \left|{\Sigma}\right|^{1/2}
             \left[
               1+\frac{1}{\nu}
               ({\mathbf x}-{\mu})^T
               {\Sigma}^{-1}({\mathbf x}-{\mu})
             \right]^{(\nu+p)/2}}


    ========  =============================================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu` if :math:`\nu > 1` else undefined
    Variance  :math:`\frac{\nu}{\mu-2}\Sigma`
                  if :math:`\nu>2` else undefined
    ========  =============================================


    Parameters
    ----------
    nu : int
        Degrees of freedom.
    Sigma : matrix
        Covariance matrix.
    mu : array
        Vector of means.
    """

    def __init__(self, nu, Sigma, mu=None, p=None, cbarless=None, cbargreater=None,
                 *args, **kwargs):
        super(MvStudentT, self).__init__(*args, **kwargs)
        self.nu = nu = tt.as_tensor_variable(nu)
        mu = tt.zeros(Sigma.shape[0]) if mu is None else tt.as_tensor_variable(mu)
        self.Sigma = Sigma = tt.as_tensor_variable(Sigma)

        self.mean = self.median = self.mode = self.mu = mu

        self.cbarless = cbarless
        self.cbargreater = cbargreater
        self.p = p

    def random(self, point=None, size=None):
        chi2 = np.random.chisquare
        mvn = np.random.multivariate_normal

        nu, S, mu = draw_values([self.nu, self.Sigma, self.mu], point=point)

        return (np.sqrt(nu) * (mvn(np.zeros(len(S)), S, size).T /
                               chi2(nu, size))
                ).T + mu

    def logp(self, value):

        S = self.Sigma
        nu = self.nu
        mu = self.mu

        # if self.p is None:
        #     d = S.shape[0]
        # else:
        #     d = self.p
        d = 1
        # if self.p is None:
        #     dd = d
        # else:
        #     dd = self.p

        X = value - mu

        Q = X.dot(matrix_inverse(S)).dot(X.T).sum()
        log_det = tt.log(det(S))
        log_pdf = gammaln((nu + d) / 2.) - 0.5 * \
            (d * tt.log(np.pi * nu) + log_det) - gammaln(nu / 2.)
        log_pdf -= 0.5 * (nu + d) * tt.log(1 + Q / nu)

        return log_pdf

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        nu = dist.nu
        Sigma = dist.Sigma
        cbarless = dist.cbarless
        cbargreater = dist.cbargreater
        return r'${} \sim \text{{MvStudentTRegularized}}(\mathit{{nu}}={}, \mathit{{mu}}={}, \mathit{{Sigma}}={}, \mathit{{\bar c_<}}={}, \mathit{{\bar c_>}}={})$'.format(
                name,
                get_variable_name(nu),
                get_variable_name(mu),
                get_variable_name(Sigma),
                get_variable_name(cbarless),
                get_variable_name(cbargreater))


def postCeps(Delta, qvec, ckvecsq, nc):
    gammasq = qvec**2 * ckvecsq
    val = 1/np.sqrt(pi * gammasq) * mpmath.fp.gamma((1+nc)/2) / mpmath.fp.gamma(nc/2)
    val *= (1 + Delta**2 / gammasq)**(-(1+nc)/2)
    return val

# print(postCeps(0, 1, 1, 1))
n_pts = 10
n_curves = 4
n_samples = 60
n_mcmc_interp = 1200
n_domain_interp = 10 * n_pts
dom_min = 1
dom_max = 350
coarse_domain_pts = np.linspace(dom_min, dom_max, n_pts)
interp_domain_pts = np.linspace(dom_min, dom_max, n_domain_interp)
# Indices of mcmc samples.
# Will interpolate between samples for smooth exploration of posterior.
coarse_pts = np.arange(0, n_samples)
interp_pts = np.linspace(0, n_samples-1, n_mcmc_interp)
# print(coarse_pts)
# print(interp_pts)
cov = np.zeros((n_pts, n_pts))
ell = (dom_max-dom_min)/5
# ell = 40

# Observable stuff
Lambdab = 600
h = 1
k = 5
nc = k - 1
prel_list = E_to_p(interp_domain_pts, 'np')
path = "../../data/vsrg_EKM_kvnn_41_lam12.0_kmax15_kmid3/analysis/obs_coefficients"
file = coeff_filename(
        obs_indices=['t', 't', 't', 't'],
        indep_var='energy',
        ivar_start=1,
        ivar_stop=351,
        ivar_step=1,
        param_var='',
        param=0,
        order="N4LO",
        Lambda_b=Lambdab,
        lambda_mult=1.0,
        X_ref_hash='0dsigma',
        convention='blatt',
        potential_info='kvnn_61_lam12.0_reg_0_3_0_blatt')
full_filename = os.path.join(path, file)
# print(full_filename)
dfile = DataFile().read(full_filename)
indep_var_vals = np.array(dfile[0])
Xref = np.array(dfile[1])
coeffs = np.array(dfile[3:])
# print(coeffs)
cksq_all = np.array([sum_sq(ckvec) for ckvec in coeffs[:, :nc]])
cksq_func = interp1d(indep_var_vals, cksq_all, kind='cubic')
cksq = cksq_func(coarse_domain_pts)
# print(coarse_domain_pts, cksq)
# q = np.sqrt(sum([Q_ratio(prel_list, Lambdab)**(2*n) for n in range(k+1, k+h+1)]))
Q = Q_ratio(prel_list, Lambdab)
q_all = np.ones(len(cksq_all))
q = np.ones(len(cksq))

interp_coeffs = np.zeros((n_domain_interp, k-1))
for j, coeff in enumerate(coeffs.T):
    int_coeff_func = interp1d(indep_var_vals, coeff, kind='cubic')
    interp_coeffs[:, j] = int_coeff_func(interp_domain_pts)

Xref_interp_func = interp1d(indep_var_vals, Xref, kind='cubic')
Xref_interp = Xref_interp_func(interp_domain_pts)
observable = Xref_interp * (1 + sum(interp_coeffs[:, j] * Q**(j+2) for j in range(nc)))

npwa_name = "../../data/npwa_data/npwa_C_t-t-t-t.dat"
npwa_file = DataFile().read(npwa_name)
npwa_indep_var = npwa_file[0]
npwa_data = npwa_file[1]
npwa_interp_func = interp1d(npwa_indep_var, npwa_data, kind='cubic')
npwa_interp = npwa_interp_func(interp_domain_pts)

residual_interp = observable - npwa_interp

curve_names = ["curve{}".format(i) for i in range(n_curves)]

with pm.Model() as model:
    # mu = pm.Normal('mu', mu=0, sd=1)
    # cov = np.array([[1., 0.99], [0.99, 1]])
    nu = nc
    for i, x in enumerate(coarse_domain_pts):
        for j, xp in enumerate(coarse_domain_pts):
            # cov[i, j] = rbf_kernel(x, xp, ell)
            cov[i, j] = q[i] * q[j] * np.sqrt(cksq[i] * cksq[j]) \
                * rbf_kernel(x, xp, ell) \
                / nu  # A guess? Makes wiggles more inline with pdf
    # print(cov)
    mu = np.zeros(n_pts)
    # vals = pm.MvNormal('vals', mu=mu, cov=cov, shape=n_pts)
    # vals = [pm.MvNormal(curve_names[i], mu=mu, cov=cov, shape=n_pts)
    #         for i in range(n_curves)]
    vals = [MvStudentT(curve_names[i], nu=nu, Sigma=cov, mu=mu, shape=n_pts,
                          # testval=np.zeros(n_pts)
                          )
            for i in range(n_curves)]
    step = pm.NUTS()
    # step = pm.Metropolis()
    # step = pm.Slice()
    # step = pm.HamiltonianMC()
    trace = pm.sample(n_samples,
                      # tune=245111,
                      # step=step,
                      # init='advi_map',
                      # n_init=350000,
                      # njobs=5
                      )


# all_data = np.roll(trace['vals'][:], -1, axis=0)
all_data = np.zeros((n_samples, n_curves, n_pts))
for curve_index, name in enumerate(curve_names):
    # print(name, trace[name][:])
    for sample_index in range(n_samples):
        all_data[sample_index, curve_index, :] = trace[name][sample_index, :]

# Move first sample to the end to improve continuity in animation
all_data = np.roll(all_data, -1, axis=0)
# print(all_data)



# interpolate between mcmc sampled points (first layer of interpolation)
interpolated_data_mcmc = np.zeros((n_mcmc_interp, n_curves, n_pts))
for curve_index in range(n_curves):
    for point_index in range(n_pts):
        coarse_sample_data = all_data[:, curve_index, point_index]
        interp_func = interp1d(coarse_pts, coarse_sample_data, kind='cubic')
        fine_sample_data = interp_func(interp_pts)
        interpolated_data_mcmc[:, curve_index, point_index] = fine_sample_data

# interpolate between domain points to hide finite approx (final interpolation)
interpolated_data = np.zeros((n_mcmc_interp, n_curves, n_domain_interp))
for sample_index in range(n_mcmc_interp):
    for curve_index in range(n_curves):
        coarse_dom_data = interpolated_data_mcmc[sample_index, curve_index, :]
        interp_func = interp1d(coarse_domain_pts, coarse_dom_data, kind='cubic')
        fine_dom_data = interp_func(interp_domain_pts)
        interpolated_data[sample_index, curve_index, :] = fine_dom_data

# print(interpolated_data)

uncertainty_matrix = np.zeros((n_mcmc_interp, n_domain_interp))
for sample_index in range(n_mcmc_interp):
    uncertainty_matrix[sample_index, :] = sum(
        [Xref_interp * interpolated_data[sample_index, i, :] * Q**(k+1+i)
         for i in range(n_curves)]
        )
# uncertainty_matrix = sum([interpolated_data[:, i, :] * Q**(k+1+i) for i in range(n_curves)])

curve_display = [r"$c_" + str(k+1+i) + r"$" for i in range(n_curves)]
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
wpad = 0.2
hpad = 2
pad = 2
fig.tight_layout(pad=pad, w_pad=wpad, h_pad=hpad)
fig.subplots_adjust(wspace=wpad)
x_label = r"$E_{\mathrm{lab}}$"
ax2.plot(interp_domain_pts, residual_interp, label=r"N$^4$LO Prediction")
ax2.plot(interp_domain_pts, 1000*np.ones(n_domain_interp),
         color="gray",
         label=r"$+$ unknown $c_i$")  # To add legend for pcolormesh
# ax.set_title(r"Possible Observable Coefficients")
ax.set_xlabel(x_label)
ax.set_ylabel(r"$c_i$")
ax2.set_title(r"Total Cross Section Residual")
ax2.set_xlabel(x_label)
ax2.set_ylabel(r"$\sigma_{\mathrm{res}}$ (mb)")
ymin = -4
ymax = 5
# ymin = np.amin(all_data)
# ymax = np.amax(all_data)
ax.set_ylim(ymin, ymax)
res_ymin = -4
res_ymax = 1
ax2.set_ylim(res_ymin, res_ymax)
line_list = [ax.plot(interp_domain_pts,
                     interpolated_data[0, i, :],
                     label=curve_display[i])[0]
             for i in range(n_curves)]
# ax.legend(loc="upper right")
ax.legend(
    fontsize=10,
    ncol=n_curves,
    bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    mode="expand",
    borderaxespad=0.,
    handlelength=2.15, handletextpad=.5)

Delta_list = np.linspace(ymin, ymax, 100)
post_matrix = np.zeros((len(Delta_list), len(indep_var_vals)))
for i, Delta in enumerate(Delta_list):
    post_matrix[i] = postCeps(Delta, q_all, cksq_all, nc)

z_max = np.abs(post_matrix).max()
z_min = z_max/100000
ax.pcolormesh(
    indep_var_vals, Delta_list, post_matrix,
    cmap="inferno",
    # vmin=z_min, vmax=z_max,
    norm=colors.LogNorm(vmin=z_min, vmax=z_max),
    edgecolors='face',
    zorder=-10)

n_ybins = 100
ybins = np.linspace(res_ymin, res_ymax, n_ybins)
n_xbins = n_domain_interp
band_matrix = np.zeros((n_mcmc_interp, n_ybins, n_xbins))

for i, sample in enumerate(uncertainty_matrix):
    band_matrix[i] += band_matrix[i-1]
    for x_index in range(n_domain_interp):
        unc_val = sample[x_index] + residual_interp[x_index]
        if res_ymin < unc_val < res_ymax:
            y_index = int(np.floor(
                (unc_val - res_ymin
                 ) / (res_ymax - res_ymin) * n_ybins
                ))
            band_matrix[i, y_index, x_index] += 1

ax2.legend(
    fontsize=10,
    loc="lower left",
    )

# quad1 = ax2.pcolormesh(interp_domain_pts, ybins, band_matrix[0],
#                        shading='gouraud',
#                        cmap='Greys',
#                        vmin=0,
#                        # vmax=np.amax(band_matrix),
#                        vmax=4,
#                        # norm=colors.LogNorm(vmin=2, vmax=np.amax(band_matrix)),
#                        edgecolors='face',
#                        # label=r"$+$ unknown $c_i$"
#                        )


# fig.savefig("animated_curves.pdf")

# quad1.set_array([])
quad1 = ax2.pcolormesh(interp_domain_pts, ybins, band_matrix[0],
                       shading='gouraud',
                       cmap='Greys',
                       # vmin=0,
                       # vmax=np.amax(band_matrix),
                       norm=colors.LogNorm(vmin=.5, vmax=np.amax(band_matrix)),
                       edgecolors='face'
                       )


def init():
    for line in line_list:
        line.set_data([], [])
    # artist_list = line_list
    # artist_list.append(quad1)
    quad1.set_array([])
    return tuple((*line_list, quad1))


def data_gen(t=0):
    # print(h)
    # while True:
    while True:
        t += 1
        # t = t % n_samples
        # yield all_data[t, :, :]
        t = t % n_mcmc_interp
        yield interpolated_data[t, :, :], band_matrix[t, :, :]


def run(data):
    # print(data)
    # plt.cla()
    lines, bands = data
    for line_num, line in enumerate(line_list):
        range_pts = lines[line_num, :]
        # interp_range_func = interp1d(domain_pts, range_pts, kind='cubic')
        # range_pts = interp_range_func(interp_domain_pts)
        line.set_data(interp_domain_pts, range_pts)
        # plt.plot(domain_pts, range_pts)
    quad1.set_array(bands.ravel())
    # artist_list = line_list
    # artist_list.append(quad1)
    return tuple((*line_list, quad1))


my_ani = animation.FuncAnimation(fig, run, frames=data_gen,
                                 init_func=init,
                                 interval=10,
                                 repeat_delay=0,
                                 save_count=n_mcmc_interp,
                                 blit=False
                                 )

mywriter = animation.FFMpegWriter(
    fps=60,  # Speed
    # metadata=dict(artist='Jordan Melendez'),  # Me
    bitrate=3500  # Quality
    )
my_ani.save("animated_curves.mov", writer=mywriter)
# 
# Length of animation in seconds is save_count / fps

# plt.show()


# for i in range(n_samples):
#     for j in range(n_curves):
#         range_pts = trace['vals'][i, j, :]
#         plt.plot(domain_pts, range_pts)
#     plt.show()

