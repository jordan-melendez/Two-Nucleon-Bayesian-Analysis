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
# Take directory of phase-shift files, constuct M matrix, extract observables.
###############################################################################

from cmath import exp, sin, cos, pi
from collections import defaultdict
from src.lowlevel.datafile import DataFile
from src.lowlevel.kinematics import *
from src.lowlevel.matrix_operations import *
from numpy import absolute, sqrt, conjugate
from numpy import array, identity, kron, matrix, ndarray, trace, zeros
import os
import re
import time


# Uncomment if phase files are in radians:
# ang_conv = 1
# Uncomment if phase files are in degrees:
ang_conv = pi/180


def p_wave_to_JLS(string):
    """Return J, L, S from a nuclear notation string of the form ^{2S+1}Letter_J.
    """
    L_dict = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6,
              "K": 7, "L": 8, "M": 9, "N": 10, "O": 11, "Q": 12}  # The correct way
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


def make_phase_dicts(directory, convention, partial_wave_list=[]):
    """Return delta(E, J, ell, s) and epsilon(E, J) from directory of phase files.
    """
    delta = defaultdict(float)
    epsilon = defaultdict(float)

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith(convention + "_phases.txt"):
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


def make_S_matrix(delta_dict, epsilon_dict, convention="stapp"):
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
                if convention == "stapp":
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
                if convention == "blatt":
                    delta_minus = delta_dict[E, J, L-1, S]
                    delta_plus = delta_dict[E, J, L+1, S]

                    S_mat[E, J, J-1, Sp, J-1, S] = \
                        cos(epsilon_dict[E, J])**2 * exp(2j * delta_minus) + \
                        sin(epsilon_dict[E, J])**2 * exp(2j * delta_plus)

                    S_mat[E, J, J+1, Sp, J+1, S] = \
                        cos(epsilon_dict[E, J])**2 * exp(2j * delta_plus) + \
                        sin(epsilon_dict[E, J])**2 * exp(2j * delta_minus)

                    S_mat[E, J, J-1, Sp, J+1, S] = \
                        1/2 * sin(2*epsilon_dict[E, J]) * \
                        (exp(2j * delta_minus) - exp(2j * delta_plus))

                    S_mat[E, J, J+1, S, J-1, Sp] = S_mat[E, J, J-1, Sp, J+1, S]
    return S_mat


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


# def beam_to_scattered_frame(vec, theta_cm):
#     return n_rotation(theta_cm/2) @ vec


# def beam_to_recoil_frame(vec, theta_cm):
#     return n_rotation(-theta_cm/2) @ vec


def sigma_textbook(delta, epsilon, interaction, convention):
    sigma = {}
    E_set = {tup[0] for tup in delta.keys()}
    J_set = {tup[1] for tup in delta.keys()}
    fm_sq_to_mb = 10
    for E in E_set:
        sigma[E] = 0
        for J in J_set:
            # Because delta and epsilon are zero unless they have been told
            # otherwise, the coupled and uncoupled cases can be combined.
            if convention == "stapp":
                temp = \
                    sin(delta[E, J, J, 0])**2 + sin(delta[E, J, J, 1])**2 + \
                    2 * sin(epsilon[E, J])**2 + cos(2*epsilon[E, J]) * \
                    (sin(delta[E, J, J-1, 1])**2 + sin(delta[E, J, J+1, 1])**2)
            if convention == "blatt":
                temp = \
                    sin(delta[E, J, J, 0])**2 + sin(delta[E, J, J, 1])**2 + \
                    sin(delta[E, J, J-1, 1])**2 + sin(delta[E, J, J+1, 1])**2
            sigma[E] += (pi/E_to_k(E, interaction)**2 * (2*J + 1) * temp).real * fm_sq_to_mb
    return sigma


def sigma_optical(M_mat, interaction):
    sigma = {}
    E_set = {tup[0] for tup in M_mat.keys()}
    for E in E_set:
        sigma[E] = (pi/E_to_k(E, interaction) *
                    (2*M_mat[E, 1, 1, 1, 1](0, 0) +
                    M_mat[E, 1, 0, 1, 0](0, 0) +
                    M_mat[E, 0, 0, 0, 0](0, 0))).imag
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
    return N(conjugate(a_saclay(M_mat, E, theta, phi)) *
             e_saclay(M_mat, E, theta, phi)).real


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


# print(clebsch_matrix)
# interaction = "np"
# XLO = "N4LO"
# phase_dir = "../../data/vsrg_EKM_R0p9_kmax15/" + XLO + "/phases/"
# # # # phase_dir = "../../data/p_wave_analysis/vnn/phases"
# delta, epsilon = make_phase_dicts(phase_dir)
# sig = sigma_textbook(delta, epsilon, interaction)
# # E_list = list({tup[0] for tup in delta.keys()})
# S_mat = make_S_matrix(delta, epsilon)
# J_set = {tup[1] for tup in S_mat.keys()}
# J_array = array(list(J_set), dtype=int)
# E_list = [50]
# # E = 50
# # theta = 100 * ang_conv
# phi = 0

# for E in range(20, 21):
#     for theta in range(11, 12):
#         print("E =", E, "theta =", theta)
#         print(make_M_singlet_triplet_matrix(S_mat, interaction, J_array, E, theta, phi))
#         print(py_make_M_singlet_triplet_matrix(S_mat, interaction, E, theta, phi))
#         print()

# print(my_spherical_harmonics(1, -1, cos(11*pi/180).real))

# start_time = time.time()
# M_mat = make_M_mat(S_mat, E_list, interaction)
# M_uncoup = make_uncoupled_M(M_mat, E, theta, phi)
# print("Working:", M_uncoup)


# clebsch_matrix = make_Clebsch_matrix()
# # print(clebsch_matrix)
# second_time = time.time()
# print(second_time - start_time)
# M_st = make_M_singlet_triplet_matrix(S_mat, interaction, E, theta, phi)
# m_unc_fast = make_M_uncoupled_matrix(M_st, clebsch_matrix)
# print("Testing:", m_unc_fast)
# print(time.time() - second_time)
# print(M_uncoup - m_unc_fast)

# ll, mm, tt, pp, j_1, j_2, j_3, m_1, m_2, m_3 = symbols('ll mm tt pp j_1 j_2 j_3 m_1 m_2 m_3')
# from sympy.abc import x
# my_spherical_harmonics = lambdify((ll, mm, tt, pp), Ynm(ll, mm, tt, pp))
# my_wigner_3j_symbol = lambdify(x, wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3))


# for j1, j2, j3, m1, m2, m3 in product(range(5), repeat=6):
#     print(j1, j2, j3, m1, m2, m3, my_clebsch_gordan(j1, m1, j2, m2, j3, m3))

# for ell, m, theta, phi in product([0, 1, 2], [0, 0, 2], range(0, 30, 1), range(0, 60, 1)):
#     print(ell, m, theta, phi, my_spherical_harmonics(ell, m, theta, phi))#, N(Ynm(ell, m, theta, phi)))
    # N(my_spherical_harmonics(ell, m, theta, phi)), N(Ynm(ell, m, theta, phi))

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
