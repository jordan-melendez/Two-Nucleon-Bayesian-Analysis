###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
# Created: Aug-10-2016
# Revised:
#
# Revision Log:
#
###############################################################################
#
#
###############################################################################
# Take directory of phase-shift files and consolidate into a dictionary.
###############################################################################

from cmath import exp, sin, cos, pi
from collections import defaultdict
from datafile import DataFile
import os
import re
# from scipy.special import sph_harm
# from sympy.functions.special import spherical_harmonics
import sympy
from sympy.physics.wigner import wigner_3j
from sympy.physics.quantum.cg import CG
from sympy import KroneckerDelta, evalf, N, simplify, Ynm, im
import numpy as np
from numpy import absolute, sqrt, zeros, kron, identity, trace, matrix, array, conjugate
# from mpmath import fsum
import copy
from functools import partial
import time
from itertools import product
from math import fsum


# Uncomment if phase files are in radians:
# ang_conv = 1
# Uncomment if phase files are in degrees:
ang_conv = pi/180


def p_wave_to_JLS(string):
    """Return J, L, S from a nuclear notation string of the form ^{2S+1}Letter_J.
    """
    L_dict = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6,
              "K": 7, "L": 8, "M": 9, "N": 10, "O": 11, "Q": 12}  # The correct way
              # "J": 7, "K": 8, "L": 9, "M": 10, "N": 11, "O": 12}  # The way I'll use for now.
    if len(string) == 3 or len(string) == 4:
        S = (int(string[0]) - 1) // 2
        L = L_dict[string[1]]
        J = int(string[2:])
        return J, L, S
    else:
        return "Something is wrong."


def is_p_wave_coupled(J, L, S):
    if J > 0 and S == 1 and L != J:
        return True
    return False


def make_phase_dicts(directory, partial_wave_list=[]):
    """Return delta(E, J, ell, s) and epsilon(E, J) from directory of phase files.
    """
    delta = defaultdict(float)
    epsilon = defaultdict(float)

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith("_phases.txt"):
                phase_file = DataFile().read(filepath)
                p_wave = re.search(r"_([1-9][A-Z]\d{1,2})_", filepath)
                if p_wave:
                    p_wave_str = p_wave.group(1)
                    J, L, S, = p_wave_to_JLS(p_wave_str)
                    is_coupled = is_p_wave_coupled(J, L, S)
                else:
                    return "There was a problem finding the partial wave."
                if partial_wave_list == [] or p_wave_str in partial_wave_list:
                    for row in phase_file.data:
                        E = row[0]
                        if is_coupled:
                            delta[E, J, J-1, S] = row[1] * ang_conv
                            delta[E, J, J+1, S] = row[2] * ang_conv
                            epsilon[E, J] = row[3] * ang_conv
                        else:
                            delta[E, J, L, S] = row[1] * ang_conv
    return delta, epsilon


def make_S_matrix(delta_dict, epsilon_dict):
    """Return dictionary S[E, J, Lp, Sp, L, S]."""
    S_mat = defaultdict(float)
    E_set = {tup[0] for tup in delta_dict.keys()}
    J_set = {tup[1] for tup in delta_dict.keys()}
    for E in E_set:
        for J in J_set:
            if J == 0:  # Uncoupled section
                L, Lp, S, Sp = 0, 0, 0, 0
                S_mat[E, J, Lp, Sp, L, S] = exp(2j * delta_dict[E, J, L, S])
                L, Lp, S, Sp = 1, 1, 1, 1
                S_mat[E, J, Lp, Sp, L, S] = exp(2j * delta_dict[E, J, L, S])
            else:  # S is now 4-dimensional
                # Uncoupled part
                Lp, L, Sp, S = J, J, 0, 0
                S_mat[E, J, Lp, Sp, L, S] = exp(2j * delta_dict[E, J, L, S])
                Sp, S = 1, 1
                S_mat[E, J, J, Sp, J, S] = exp(2j * delta_dict[E, J, L, S])
                # Coupled part
                Lp, L = J-1, J-1
                S_mat[E, J, Lp, Sp, L, S] = cos(2 * epsilon_dict[E, J]) * \
                    exp(2j * delta_dict[E, J, L, S])
                Lp, L = J+1, J+1
                S_mat[E, J, Lp, Sp, L, S] = cos(2 * epsilon_dict[E, J]) * \
                    exp(2j * delta_dict[E, J, L, S])
                Lp, L = J-1, J+1
                S_mat[E, J, Lp, Sp, L, S] = 1j * sin(2*epsilon_dict[E, J]) * \
                    exp(1j*(delta_dict[E, J, Lp, Sp] + delta_dict[E, J, L, S]))
                S_mat[E, J, L, S, Lp, Sp] = S_mat[E, J, Lp, Sp, L, S]
    return S_mat


def make_M_mat(S_mat, interaction):
    """Return M dictionary of functions of theta, and phi.

    Example:
    M_mat = make_M_mat(S_mat)
    print(M_mat[E, Sp, mp, S, m](theta, phi))
    """

    # This should help when calling the M matrix for multiple angles.
    # The angles will only be evaluated at the last step.
    def M(theta, phi, E, S, m, Sp, mp):
        """Return M mat in millibarns"""
        fm_sq_to_mb = 10
        M_term = 0
        J_set = {tup[1] for tup in S_mat.keys()}
        for J in J_set:
            for L in range(max(0, J-1), J+2):
                for Lp in range(max(0, J-1), J+2):
                    M_term += 1j**(L - Lp) * (2*J+1) * sqrt(2*L+1) * \
                        Ynm(Lp, m-mp, theta, phi) * \
                        wigner_3j(Lp, Sp, J, m-mp, mp, -m) * \
                        wigner_3j(L, S, J, 0, m, -m) * \
                        (S_mat[E, J, Lp, Sp, L, S] -
                            KroneckerDelta(Lp, L)*KroneckerDelta(Sp, S))
        M_term *= sqrt(4*pi)/(2j * E_to_k(E, interaction))*(-1)**(S-Sp)
        M_term *= sqrt(fm_sq_to_mb)  # Convert units
        return N(M_term)

    M_mat = defaultdict(float)
    E_set = {tup[0] for tup in S_mat.keys()}
    for E in E_set:
        for S in range(2):
            for m in range(-S, S+1):
                for Sp in range(2):
                    for mp in range(-Sp, Sp+1):
                        # Partial is needed to avoid problems with 'late
                        # binding' for the M function.
                        M_mat[E, Sp, mp, S, m] = \
                            partial(M, E=E, S=S, m=m, Sp=Sp, mp=mp)
    return M_mat


def make_uncoupled_M(M_mat, E, theta, phi):
    M_unc = zeros((4, 4), dtype=complex)
    m_list = [1/2, -1/2]
    # "product" saves me from 4 nested for loops!
    for (mp1_index, mp1), (mp2_index, mp2), (m1_index, m1), (m2_index, m2) \
            in product(enumerate(m_list), repeat=4):
        row = len(m_list)*mp1_index + mp2_index
        col = len(m_list)*m1_index + m2_index
        the_sum = 0
        for s, sp in product(range(2), repeat=2):
            if isinstance(M_mat[E, sp, mp1 + mp2, s, m1 + m2], float):
                the_sum += M_mat[E, sp, mp1 + mp2, s, m1 + m2]
            else:
                the_sum += simplify(
                    CG(1/2, mp1, 1/2, mp2, sp, mp1 + mp2).doit() *
                    CG(1/2, m1, 1/2, m2, s, m1 + m2).doit() *
                    M_mat[E, sp, mp1 + mp2, s, m1 + m2](theta, phi))
        M_unc[row, col] = the_sum
    return matrix(M_unc)


def observable_C_tensor(M_uncoupled, scattered_spin, recoil_spin,
                        beam_spin, target_spin):
    """Calculate polarization correlation tensor \sigma C_{pqik} in AJP 1978.

    p, q, i, k must be 0, 'x', 'y', or 'z', which correspond to the
    polarization of each particle, where z is their direction of travel
    (different for each particle) and y is the normal vector to the scattering
    plane.
    """

    i2 = identity(2)
    sx = array([[0, 1], [1, 0]])
    sy = array([[0, -1j], [1j, 0]])
    sz = array([[1, 0], [0, -1]])
    sigma_vec = array([sx, sy, sz])

    def mat_vec_dot(mat, vec):
        new_array = zeros(mat[0].shape, dtype=complex)
        for i in range(len(vec)):
            new_array += mat[i] * vec[i]
        return new_array

    if isinstance(scattered_spin, int) and scattered_spin == 0:
        scattered_sigma = i2
    elif isinstance(scattered_spin, np.ndarray):
        scattered_sigma = mat_vec_dot(sigma_vec, scattered_spin)
    else:
        return "There is a problem with the scattered_spin parameter"

    if isinstance(recoil_spin, int) and recoil_spin == 0:
        recoil_sigma = i2
    elif isinstance(recoil_spin, np.ndarray):
        recoil_sigma = mat_vec_dot(sigma_vec, recoil_spin)
    else:
        return "There is a problem with the recoil_spin parameter"

    if isinstance(beam_spin, int) and beam_spin == 0:
        beam_sigma = i2
    elif isinstance(beam_spin, np.ndarray):
        beam_sigma = mat_vec_dot(sigma_vec, beam_spin)
    else:
        return "There is a problem with the beam_spin parameter"

    if isinstance(target_spin, int) and target_spin == 0:
        target_sigma = i2
    elif isinstance(target_spin, np.ndarray):
        target_sigma = mat_vec_dot(sigma_vec, target_spin)
    else:
        return "There is a problem with the target_spin parameter"

    C = 1/4 * trace(kron(scattered_sigma, recoil_sigma) @ M_uncoupled @
                    kron(beam_sigma, target_sigma) @ M_uncoupled.H)
    return C.real


def y_rotation(t):
    """Create an active y rotation matrix."""
    return array([[cos(t), 0, sin(t)],
                  [0, 1, 0],
                  [-sin(t), 0, cos(t)]])


def beam_to_scattered_frame(vec, theta_cm):
    return y_rotation(theta_cm/2) @ vec


def beam_to_recoil_frame(vec, theta_cm):
    return y_rotation(-theta_cm/2) @ vec


def sigma_textbook(delta, epsilon, interaction):
    sigma = {}
    E_set = {tup[0] for tup in delta.keys()}
    J_set = {tup[1] for tup in delta.keys()}
    fm_sq_to_mb = 10
    for E in E_set:
        sigma[E] = 0
        for J in J_set:
            # Because delta and epsilon are zero unless they have been told
            # otherwise, the coupled and uncoupled cases can be combined.
            temp = sin(delta[E, J, J, 0])**2 + sin(delta[E, J, J, 1])**2 + \
                2 * sin(epsilon[E, J])**2 + cos(2*epsilon[E, J]) * \
                (sin(delta[E, J, J-1, 1])**2 + sin(delta[E, J, J+1, 1])**2)
            sigma[E] += (pi/E_to_k(E, interaction)**2 * (2*J + 1) * temp).real * fm_sq_to_mb
    return sigma


def sigma_optical(M_mat, interaction):
    sigma = {}
    E_set = {tup[0] for tup in M_mat.keys()}
    for E in E_set:
        sigma[E] = im(pi/E_to_k(E, interaction) *
                      (2*M_mat[E, 1, 1, 1, 1](0, 0) +
                       M_mat[E, 1, 0, 1, 0](0, 0) +
                       M_mat[E, 0, 0, 0, 0](0, 0)))
    return sigma


def a_saclay(M_mat, E, theta, phi):
    return 1/2 * (M_mat[E, 1, 1, 1, 1](theta, phi) +
                  M_mat[E, 1, 0, 1, 0](theta, phi) -
                  M_mat[E, 1, 1, 1, -1](theta, phi))


def b_saclay(M_mat, E, theta, phi):
    return 1/2 * (M_mat[E, 1, 1, 1, 1](theta, phi) -
                  M_mat[E, 0, 0, 0, 0](theta, phi) +
                  M_mat[E, 1, 1, 1, -1](theta, phi))


def c_saclay(M_mat, E, theta, phi):
    return 1/2 * (M_mat[E, 1, 1, 1, 1](theta, phi) +
                  M_mat[E, 0, 0, 0, 0](theta, phi) +
                  M_mat[E, 1, 1, 1, -1](theta, phi))


def d_saclay(M_mat, E, theta, phi):
    return -1/(sqrt(2) * sin(theta)) * (M_mat[E, 1, 1, 1, 0](theta, phi) +
                                        M_mat[E, 1, 0, 1, 1](theta, phi))


def e_saclay(M_mat, E, theta, phi):
    return 1j/sqrt(2) * (M_mat[E, 1, 1, 1, 0](theta, phi) -
                         M_mat[E, 1, 0, 1, 1](theta, phi))


def differential_cross_section(M_mat, E, theta, phi):
    return 1/2 * (absolute(a_saclay(M_mat, E, theta, phi))**2 +
                  absolute(b_saclay(M_mat, E, theta, phi))**2 +
                  absolute(c_saclay(M_mat, E, theta, phi))**2 +
                  absolute(d_saclay(M_mat, E, theta, phi))**2 +
                  absolute(e_saclay(M_mat, E, theta, phi))**2)


def analyzing_power(M_mat, E, theta, phi):
    return sympy.re(simplify(conjugate(a_saclay(M_mat, E, theta, phi)) *
                    e_saclay(M_mat, E, theta, phi)))


def C_llll_or_C_mmmm(M_mat, E, theta, phi):
    return 1/2 * (absolute(a_saclay(M_mat, E, theta, phi))**2 +
                  absolute(b_saclay(M_mat, E, theta, phi))**2 +
                  absolute(c_saclay(M_mat, E, theta, phi))**2 +
                  absolute(d_saclay(M_mat, E, theta, phi))**2 -
                  absolute(e_saclay(M_mat, E, theta, phi))**2)


def C_nn00_or_C_00nn(M_mat, E, theta, phi):
    return 1/2 * (absolute(a_saclay(M_mat, E, theta, phi))**2 -
                  absolute(b_saclay(M_mat, E, theta, phi))**2 -
                  absolute(c_saclay(M_mat, E, theta, phi))**2 +
                  absolute(d_saclay(M_mat, E, theta, phi))**2 +
                  absolute(e_saclay(M_mat, E, theta, phi))**2)

# \hbar^2 / m (MeV-fm^2)
hbar2_per_m = 41.47105


# Define the rule to go from energy to momentum
def E_to_k(E_lab, interaction):
    """Return k in fm^{-1}.

    Parameters
    ----------
    energy      = float
                  lab energy given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """
    hbarc = 197.33  # Mev-fm
    p_rel = E_to_p(E_lab, interaction)

    return p_rel/hbarc


def E_to_p(E_lab, interaction):
    """Return p in MeV.

    Parameters
    ----------
    energy      = float
                  lab energy given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """

    m_p = 938.27208  # MeV/c^2
    m_n = 939.56541  # MeV/c^2
    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p
    p_rel = sqrt(
        E_lab * m2**2 * (E_lab + 2 * m1) /
        ((m1 + m2)**2 + 2 * m2 * E_lab)
        ).real
    return p_rel


def p_to_E(p_rel, interaction):
    """Return E_lab in MeV.

    Parameters
    ----------
    p_rel       = float
                  relative momentum given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """
    m_p = 938.27208  # MeV/c^2
    m_n = 939.56541  # MeV/c^2
    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p
    E_lab = (2 * p_rel**2 - 2 * m1 * m2 +
             2 * sqrt((m1**2 + p_rel**2) * (m2**2 + p_rel**2))) / (2 * m2)
    return E_lab.real


def k_to_E(k_rel, interaction):
    """Return E_lab in MeV.

    Parameters
    ----------
    k_rel       = float
                  relative momentum given in fm^{-1}.
    interaction = str
                  {"pp", "nn", "np"}
    """
    hbarc = 197.33  # Mev-fm
    E_lab = p_to_E(hbarc * k_rel, interaction)
    return E_lab


def observable_filename(obs_indices, indep_var, ivar_start, ivar_stop,
                        ivar_step, param_var, param, order):
    """Return a standard filename for observable files based on parameters.

    Parameters
    ----------
    obs_indices = list
                  [p, q, i, k] specifies the spin observable.
    indep_var   = str
                  the name of the independent variable: ["theta", "energy"].
    ivar_start  = int
                  The start of the range of independent variables in the file.
    ivar_stop   = int
                  The end of the range of independent variables in the file.
    ivar_step   = int
                  The step size of the range of indep_var.
    param_var   = str
                  One of ["theta", "energy"] that is NOT indep_var.
    param       = float
                  value of param_var.
    order       = str
                  One of ["LO", "NLO", "N2LO", "N3LO", "N4LO"].
    """
    kvnn_dict = {
            "LO": "41",
            "LOp": "41",
            "NLO": "46",
            "N2LO": "51",
            "N3LO": "56",
            "N4LO": "61"
        }
    if obs_indices == ['t', 't', 't', 't']:
        name = "C_t-t-t-t_vs_energy" + \
               "-" + str(ivar_start) + "-" + str(ivar_stop) + "-" + \
               str(ivar_step) + "_" + order + "_vnn_kvnn_" + \
               kvnn_dict[order] + ".dat"
    else:
        name = "C_" + obs_indices[0] + "-" + obs_indices[1] + "-" + \
               obs_indices[2] + "-" + obs_indices[3] + "_vs_" + indep_var + \
               "-" + str(ivar_start) + "-" + str(ivar_stop) + "-" + \
               str(ivar_step) + "_" + param_var + "-" + str(param) + \
               "_" + order + "_vnn_kvnn_" + kvnn_dict[order] + ".dat"
    return name


# ppp = 212
# for inter in ["pp", "nn", "np"]:
#     El = p_to_E(ppp, inter)
#     p2 = E_to_p(El, inter)
#     print(inter, "initial:", ppp, "Energy:", El, "final:", p2)

# eee = 330
# for inter in ["pp", "nn", "np"]:
#     pr = E_to_k(eee, inter)
#     E2 = k_to_E(pr, inter)
#     print(inter, "initial:", eee, "prel:", pr, "final:", E2)

# XLO = "N4LO"
# phase_dir = "../../data/vsrg_EKM_R0p9_kmax15/" + XLO + "/phases/"
# # # # phase_dir = "../../data/p_wave_analysis/vnn/phases"
# delta, epsilon = make_phase_dicts(phase_dir, partial_wave_list=["1S0"])
# sig = sigma_textbook(delta, epsilon)
# E_list = list({tup[0] for tup in delta.keys()})
# sig_list = [sig[E] for E in E_list]
# # print(E_set, sig)
# file = DataFile().write(("Energy (MeV)", E_list), ("sigma (mb)", sig_list))
# file.export_to_file("total_cross_section_" + XLO + "_1S0.dat")

# S_mat = make_S_matrix(delta, epsilon)
# M_mat = make_M_mat(S_mat)
# E = 50
# theta = 100 * ang_conv
# phi = 0
# M_uncoup = make_uncoupled_M(M_mat, E, theta, phi)
# x_hat = array([1, 0, 0])  # In scattering plane
# y_hat = array([0, 1, 0])  # Normal to scattering plane
# z_hat = array([0, 0, 1])  # Direction of beam
# ell = beam_to_scattered_frame(z_hat, theta)
# m = beam_to_scattered_frame(x_hat, theta)
# n = y_hat

# print("m:", m)

# scatt_pol = m
# recoil_pol = 0
# beam_pol = z_hat
# target_pol = 0

# diff_cross = observable_C_tensor(M_uncoup, 0, 0, 0, 0)
# new_obs = observable_C_tensor(
#     M_uncoup, scatt_pol, recoil_pol, beam_pol, target_pol)
# coeff = new_obs/diff_cross
# print("theta    dsigma        sigma X         X")
# print(theta*180/pi, diff_cross, new_obs, coeff)

# hard_obs = C_nn00_or_C_00nn(M_mat, E, theta, phi)
# print(theta*180/pi, diff_cross, hard_obs, hard_obs/diff_cross)
# t0 = time.time()
# diff_orig = differential_cross_section(M_mat, E, theta, phi)
# t1 = time.time()
# print("orig.", diff_orig, t1 - t0)
# M_uncoup = make_uncoupled_M(M_mat, E, theta, phi)
# diff_new = observable_C_tensor(M_uncoup, 0, 0, 0, 0)
# print("new", diff_new, time.time() - t1)


# t0 = time.time()
# print("Making M_mat...")
# M_mat = make_M_mat(S_mat)
# t1 = time.time()
# print("Done! t =", t1 - t0)
# print("Evaluating M_mat...")
# print(M_mat[10, 1, 1, 1, 1](1, 1))
# t2 = time.time()
# print("Done! t =", t2 - t1)
# print("Evaluating M_mat again...")
# print(M_mat[10, 1, 0, 1, 0](1, 1))
# t3 = time.time()
# print("Done! t =", t3 - t2)


# t0 = time.time()
# print("Making M_mat2...")
# M_mat2 = make_M_mat_from_E_list(S_mat, [5, 10])
# t1 = time.time()
# print("Done! t =", t1 - t0)
# print("Evaluating M_mat2...")
# print(M_mat2[10, 1, 1, 1, 1](1, 1))
# t2 = time.time()
# print("Done! t =", t2 - t1)
# print("Evaluating M_mat2 again...")
# print(M_mat2[10, 1, 0, 1, 0](1, 1))
# t3 = time.time()
# print("Done! t =", t3 - t2)

# sigma_t = sigma_textbook(delta, epsilon)

# sigma_op = sigma_optical(M_mat)
# lab_en = [50, 95, 145, 200]
# true_sigma = [167, 78.3, 54.2, 42.6]
# print(50, E_to_k(50))
# print("E", "sigma textbook")
# for i, e in enumerate(lab_en):
#     print(e, sigma_t[e])


# for E in [5, 10, 25, 50, 96, 100, 143, 150, 200]:
#     print(E)
#     print(delta[E, 0, 0, 0], delta[E, 0, 1, 1])
#     for J in range(1, 7):
#         print(delta[E, J, J, 0], delta[E, J, J, 1], delta[E, J, J-1, 1],
#               epsilon[E, J], delta[E, J, J+1, 1])
#     print("\n")
