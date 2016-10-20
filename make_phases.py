###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
# Created: Jul-28-2016
# Revised:
#
# Revision Log:
#
###############################################################################
#
#
###############################################################################
# Numerically solves Lippman-Schwinger equation for phase shifts
###############################################################################

import sys
import time
from configparser import *
import numpy as np
from numpy import linalg, matrix, ones, zeros, identity, linspace, copy, matlib
from numpy import append, reshape, cos, tan, arctan, pi, sin
from numpy import exp, sqrt, array, random
from math import fsum, nan
from scipy.interpolate import griddata, RectBivariateSpline
from datafile import DataFile
from observables import E_to_k, k_to_E
import os
import argparse


class MyConfigParser(ConfigParser):
    def get_list(self, section, option):
        value = self.get(section, option)
        return list(filter(None, (x.strip() for x in value.splitlines())))

    def get_list_int(self, section, option):
        return [int(x) for x in self.get_list(section, option)]


def calc_phase_shifts(k_points, k_weights, k_max, lab_energies, V_matrix,
                      interaction, is_coupled=False):
    """Solve the Lippmann-Schwinger equation and return phase shifts.

    Parameters
    ----------
    momentum_mesh: momentum_mesh
                   Points and weights used for Gaussian quadrature solution.
    lab_energies: numpy array
                  Energies (in MeV) at which to solve the LS equation.
    V_matrix: square numpy array (or tuple of arrays)
              Potential matrices. If coupled, 4 matrices needed in a tuple.
    interaction: str
                 Potential type {"nn", "pp", "np"}
    is_coupled: bool
                Solve a coupled LS equation.
    """

    k_com = E_to_k(lab_energies, interaction)
    N = len(k_points)
    num_E = len(lab_energies)
    deg_to_rad = pi/180
    rad_to_deg = 180/pi

    if is_coupled:
        V_spline = []
        for i in range(4):
            V_spline.append(RectBivariateSpline(k_points, k_points,
                                                V_matrix[i]))
        delta_a_bars = zeros(num_E)
        delta_b_bars = zeros(num_E)
        eps_bars = zeros(num_E)
    else:
        V_spline = RectBivariateSpline(k_points, k_points, V_matrix)
        phase_shifts = zeros(num_E)

    # Now solve the LS equation at each desired energy.
    for index, k_zero in enumerate(k_com):
        k_full = np.hstack((k_points, k_zero))
        col_mesh, row_mesh = np.meshgrid(k_full, k_full)

        # Make V matrix from V_std and correct energy entries from V_interp.
        if is_coupled:
            pot = []
            for i in range(4):
                pot.append(V_spline[i].ev(row_mesh, col_mesh))
            # Make big V matrix out of submatrices
            V_top = np.hstack((pot[0], pot[1]))
            V_bot = np.hstack((pot[2], pot[3]))
            V = np.vstack((V_top, V_bot))
        else:
            V = V_spline.ev(row_mesh, col_mesh)

        # Make denominator vector defined in Landau Eq. (18.19)
        D = [2/pi*(k_weights[i]*k_points[i]**2)/(k_points[i]**2 - k_zero**2)
             for i in range(N)]
        D0 = -2/pi*k_zero**2*sum(k_weights[j]/(k_points[j]**2 - k_zero**2)
                                 for j in range(N))
        # add correction for integration cutoff
        D0 = D0 + (-2/pi)*k_zero**2 * \
            (1/(2*k_zero))*np.log((k_max + k_zero)/(k_max - k_zero))

        D = array(D + [D0])

        # Make F matrix defined in Landau Eq. (18.22).
        # Numpy arrays multiply (*) element-wise, so with repmat we can
        # multiply each column of V with the corresponding entry of D.
        if is_coupled:
            # In the coupled case, the kronicker delta only affects the
            # submatrices along the diagonal of the big F matrix
            F = identity(2*(N+1)) + matlib.repmat(D, 2*(N+1), 2) * V
        else:
            F = identity(N+1) + matlib.repmat(D, N+1, 1) * V

        # Solve for R using Landau Eq. (18.23): F.R = V
        R = linalg.solve(F, V)

        # With the R matrix in hand, solve for the
        #   (1) phase,                       in uncoupled case
        #   (2) phases and mixing parameter, in the coupled case
        if is_coupled:
            # get the elements corresponding to desired energy (or k_zero)
            r11 = R[N, N]
            r12 = R[-1, N]
            r21 = R[N, -1]
            r22 = R[-1, -1]

            eps = .5 * arctan(2*r12/(r11 - r22))
            r_eps = (r11 - r22)/cos(2*eps)
            delta_a = - arctan(.5*k_zero*(r11 + r22 + r_eps))
            delta_b = - arctan(.5*k_zero*(r11 + r22 - r_eps))
            # Bar phase shifts and mixing parameter
            eps_bar = .5*np.arcsin(sin(2*eps)*sin(delta_a - delta_b))
            eps_bars[index] = eps_bar * rad_to_deg
            delta_a_bars[index] = .5 * rad_to_deg * \
                (delta_a + delta_b + np.arcsin(tan(2*eps_bar)/tan(2*eps)))
            delta_b_bars[index] = .5 * rad_to_deg * \
                (delta_a + delta_b - np.arcsin(tan(2*eps_bar)/tan(2*eps)))
        else:
            # Landau Eq. (18.13)
            phase_shifts[index] = rad_to_deg * arctan(- k_zero*R[-1, -1])

    if is_coupled:
        return delta_a_bars, delta_b_bars, eps_bars
    else:
        return phase_shifts


# # Define the rule to go from energy to momentum
# def E_to_k(energy):
#     return sqrt(energy/(2*hbar2_per_m))


# # Define the rule to go from momentum to energy
# def k_to_E(k):
#     return 2 * hbar2_per_m * k**2


# # \hbar^2 / m (MeV-fm^2)
# hbar2_per_m = 41.47105


def main(potential_file, mesh_file, config_file):

    config = MyConfigParser()
    config.read(config_file)
    # hbar2_per_m = config.get("parameters", "hbar2_per_m")
    # k_max = float(config.get("parameters", "k_max"))
    # lab_energies = array(config.get_list_int("parameters", "energy_list"))
    interaction = config.get("parameters", "interaction")
    k_max = float(config.get("parameters", "k_max"))
    lab_energies = array(config.get_list_int("parameters", "energy_list"))

    mesh = DataFile().read(mesh_file)
    points = array(mesh[0])
    weights = array(mesh[1])
    num_pts = len(points)

    potential = DataFile().read(potential_file)

    # Coupled files have 4 columns of potential data,
    # while uncoupled only have 1.
    if len(potential) == 3:
        is_coupled = False
    elif len(potential) == 6:
        is_coupled = True
    else:
        return "Potential file structure not compatible."

    base_pot_file = os.path.splitext(os.path.basename(potential_file))[0]
    p_wave = base_pot_file[4:7]
    phase_file_name = base_pot_file + "_phases.txt"
    phase_file = DataFile()
    if is_coupled:
        # Transpose so k' is along rows and k along columns.
        # Should only matter for V_mat_12 and V_mat_21.
        v_11_mat = reshape(array(potential[2]), (num_pts, num_pts))
        v_12_mat = reshape(array(potential[3]), (num_pts, num_pts))
        v_21_mat = reshape(array(potential[4]), (num_pts, num_pts))
        v_22_mat = reshape(array(potential[5]), (num_pts, num_pts))
        v = v_11_mat, v_12_mat, v_21_mat, v_22_mat
        solution = calc_phase_shifts(
            points, weights, k_max, lab_energies, v, interaction, is_coupled
            )
        phase_file.write(("E (MeV)", lab_energies),
                         ("delta_a_bar (deg)", solution[0]),
                         ("delta_b_bar (deg)", solution[1]),
                         ("epsilon_bar (deg)", solution[2]),
                         )
    else:
        v = reshape(array(potential[2]), (num_pts, num_pts))
        solution = calc_phase_shifts(
            points, weights, k_max, lab_energies, v, interaction, is_coupled
            )
        phase_file.write(("E (MeV)", lab_energies),
                         ("delta (deg)", solution),
                         )

    phase_file.export_to_file(phase_file_name)


if __name__ == '__main__':
    # Set up args for running from terminal
    parser = argparse.ArgumentParser(
        description="""Make phase files from potential using
        Gaussian quadrature with parameters in CONFIG"""
        )
    parser.add_argument(
        "potential_file",
        help="Potential on a momentum mesh for the interaction.")
    parser.add_argument(
        "momentum_mesh_file",
        help="""File containing momentum points and weights for
        Gaussian quadrature""")
    parser.add_argument(
        "config_file",
        help="Contains parameters of particular interaction.")
    args = parser.parse_args()

    # Where the work is done
    main(args.potential_file, args.momentum_mesh_file, args.config_file)
