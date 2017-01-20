# A function that calculates NN phaseshifts from potential files.

import numpy as np
from numpy import array, append, matrix, zeros, identity, matlib, linalg
from numpy import cos, tan, arctan, arctan2, pi, sin, sqrt, sign, exp
from math import fsum
from itertools import product
import scipy
from scipy.interpolate import griddata, RectBivariateSpline
from src.lowlevel.kinematics import E_to_k, k_to_E


def calc_phase_shifts(k_points, k_weights, k_max, lab_energies, V_matrix,
                      interaction, is_coupled=False, convention="stapp"):
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
        D0 = -2/pi * k_zero**2 * fsum(
            k_weights[j]/(k_points[j]**2 - k_zero**2) for j in range(N)
            )
        # add correction for integration cutoff
        D0 = D0 + (-2/pi) * k_zero**2 * \
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
        # R = linalg.lstsq(F, V)[0]

        # With the R matrix in hand, solve for the
        #   (1) phase,                       in uncoupled case
        #   (2) phases and mixing parameter, in the coupled case
        if is_coupled:
            # get the elements corresponding to desired energy (or k_zero)
            r11 = R[N, N]
            r12 = R[-1, N]
            r21 = R[N, -1]
            r22 = R[-1, -1]

            # eps = .5 * arctan(2*r12/(r11 - r22))
            eps = .5 * arctan2(2*r12, r11 - r22)

            # if eps > pi/4:
            #     eps -= pi/2
            # if eps <= -pi/4:
            #     eps += pi/2
            r_eps = (r11 - r22)/cos(2*eps)
            # r_eps = sqrt(4 * r12**2 + (r11 - r22)**2)
            delta_a = - arctan(.5*k_zero*(r11 + r22 + r_eps))
            delta_b = - arctan(.5*k_zero*(r11 + r22 - r_eps))

            # See http://journals.aps.org/prc/pdf/10.1103/PhysRevC.35.869
            # for info on how to restrict the values of these phases

            # This difference is like a ``polar angle''
            # Must restrict it accordingly
            while delta_a - delta_b <= 0:
                delta_a += pi
            while delta_a - delta_b > pi/2:
                delta_b += pi

            # while delta_a <= -pi/2:
            #     delta_a += pi
            # while delta_a > pi:
            #     delta_a -= pi
            # while delta_b <= -pi:
            #     delta_b += pi
            # while delta_b > pi/2:
            #     delta_b -= pi

            # while (delta_a - delta_b < 0 or delta_a - delta_b > pi) or (delta_a + delta_b < -pi or delta_a + delta_b > pi):
            #     while delta_a - delta_b < 0:
            #         if delta_a < 0:
            #             delta_a += pi
            #         elif delta_b > 0:
            #             delta_b -= pi

            #     while delta_a - delta_b > pi:
            #         if delta_a > pi/2:
            #             delta_a -= pi
            #         elif delta_b < -pi/2:
            #             delta_b += pi

            #     while delta_a + delta_b < -pi:
            #         if delta_a < 0:
            #             delta_a += pi
            #         elif delta_b < -pi/2:
            #             delta_b += pi

            #     while delta_a + delta_b > pi:
            #         if delta_a > pi/2:
            #             delta_a -= pi
            #         elif delta_b > 0:
            #             delta_b -= pi

            #     print(delta_a, delta_b, delta_a - delta_b, delta_a + delta_b)

            # s = delta_a + delta_b
            # d = delta_a - delta_b

            # while (d < 0 or d > pi) or (s < -pi or s > pi):
            #     while s < -pi and d < 0:
            #         delta_a += pi
            #         s = delta_a + delta_b
            #         d = delta_a - delta_b

            #     while s < -pi and d > pi:
            #         delta_b += pi
            #         s = delta_a + delta_b
            #         d = delta_a - delta_b

            #     while s > pi and d < 0:
            #         delta_b -= pi
            #         s = delta_a + delta_b
            #         d = delta_a - delta_b

            #     while s > pi and d > pi:
            #         delta_a -= pi
            #         s = delta_a + delta_b
            #         d = delta_a - delta_b

            #     if d < 0 and s < 0:
            #         delta_a += pi
            #         s = delta_a + delta_b
            #         d = delta_a - delta_b

            #     print(delta_a, delta_b, s, d)

            # for m, n in product(range(-3, 3), range(-3, 3)):
            #     da_temp = delta_a + m * pi
            #     db_temp = delta_b + n * pi
            #     s = da_temp + db_temp
            #     d = da_temp - db_temp
            #     if -pi < s < pi and 0 < d < pi/2:
            #         delta_a = da_temp
            #         delta_b = db_temp
            #         break

            if convention == "blatt":
                delta_a_bars[index] = rad_to_deg * delta_a
                delta_b_bars[index] = rad_to_deg * delta_b
                eps_bars[index] = eps * rad_to_deg

            # Bar phase shifts and mixing parameter
            # s2eps = 2 * r12 / sqrt(4 * r12**2 + (r11 - r22)**2)
            eps_bar = .5*np.arcsin(sin(2*eps) * sin(delta_a - delta_b))

            delta_a_bar = .5 * \
                (delta_a + delta_b + np.arcsin(tan(2*eps_bar)/tan(2*eps)))
            delta_b_bar = .5 * \
                (delta_a + delta_b - np.arcsin(tan(2*eps_bar)/tan(2*eps)))

            # These must be restricted too
            # while delta_a_bar - delta_b_bar <= -pi/2:
            #     delta_a += pi
            #     eps_bar *= -1
            # while delta_a - delta_b > pi/2:
            #     delta_b += pi
            #     eps_bar *= -1

            # S_stapp = array([
            #     [cos(2*eps_bar) * exp(2j*delta_a_bar), 1j*sin(2*eps_bar) * exp(1j*(delta_a_bar + delta_b_bar))],
            #     [1j*sin(2*eps_bar) * exp(1j*(delta_a_bar + delta_b_bar)), cos(2*eps_bar) * exp(2j*delta_b_bar)]
            # ])

            # S_blatt = array([
            #     [cos(eps)**2 * exp(2j * delta_a) + sin(eps)**2 * exp(2j*delta_b), .5*sin(2*eps) * (exp(2j*delta_a) - exp(2j*delta_b))],
            #     [.5*sin(2*eps) * (exp(2j*delta_a) - exp(2j*delta_b)), cos(eps)**2 * exp(2j * delta_b) + sin(eps)**2 * exp(2j*delta_a)]
            # ])

            # El = k_to_E(k_zero, "np")
            # if El > 1 and El < 30:
            #     print("========", El, k_zero, "========")
            #     print("check 1:", delta_a + delta_b == delta_a_bar + delta_b_bar, delta_a + delta_b, "=", delta_a_bar + delta_b_bar)
            #     print("check 2:", sin(delta_a_bar - delta_b_bar) == tan(2*eps_bar)/tan(2*eps), sin(delta_a_bar - delta_b_bar), '=', tan(2*eps_bar)/tan(2*eps))
            #     print("check 3:", sin(delta_a - delta_b) == sin(2*eps_bar)/sin(2*eps), sin(delta_a - delta_b), '=', sin(2*eps_bar)/sin(2*eps))
            #     print("blatt:", delta_a, delta_b, eps)
            #     print("stapp:", delta_a_bar, delta_b_bar, eps_bar)
            #     print("arctan:", tan(2*eps_bar)/tan(2*eps))
            #     print("sums:", delta_a + delta_b, -pi <= delta_a + delta_b <= pi, delta_a_bar + delta_b_bar, -pi <= delta_a_bar + delta_b_bar <= pi)
            #     print("diffs:", delta_a - delta_b, 0 <= delta_a - delta_b <= pi, delta_a_bar - delta_b_bar, -pi <= delta_a_bar - delta_b_bar <= pi)
            #     print("Sstapp\n", S_stapp)
            #     print("Sblatt\n", S_blatt)
            #     print("Sstapp = Sblatt:", np.allclose(S_stapp, S_blatt))
            #     print("Rs:", r11, r12, r21, r22, "Reps:", r_eps)
                # # print("?:", r_eps, delta_a, delta_b, -2*r12*k_zero*cos(delta_a)*cos(delta_b), eps_bar, D0)
                # print("???", np.arcsin((r11-r22)*tan(2*eps_bar)/(2*r12)), np.arcsin(tan(2*eps_bar)/tan(2*eps)))
                # print("d1:", delta_a, delta_a_bar)
                # print("d2:", delta_b, delta_b_bar)
                # print("sums:", delta_a + delta_b, delta_a_bar + delta_b_bar)
                # print("eps:", eps, eps_bar)
                # print("sig:", sin(delta_a)**2 + sin(delta_b)**2,
                #       2*sin(eps_bar)**2 + cos(2*eps_bar)*(sin(delta_a_bar)**2 +
                #                                           sin(delta_b_bar)**2))
                # print("same?", sin(delta_a)**2 + sin(delta_b)**2 ==
                #       2*sin(eps_bar)**2 + cos(2*eps_bar)*(sin(delta_a_bar)**2 +
                #                                           sin(delta_b_bar)**2))
            #     print("changes:", a_low, a_high, b_low, b_high)

            #     print("maxs:", np.amax(R), np.amax(F))
            #     print(delta_a_bar, delta_b_bar)
            #     print("cond:", linalg.cond(F))
            #     print(np.allclose(np.dot(F, R), V))
                # print(np.dot(F, R) == V)
                # print(D)


            if convention == "stapp":
                eps_bars[index] = eps_bar * rad_to_deg
                delta_a_bars[index] = rad_to_deg * delta_a_bar
                delta_b_bars[index] = rad_to_deg * delta_b_bar
        else:
            # Landau Eq. (18.13)
            phase_shift = arctan(- k_zero*R[-1, -1])
            # if phase_shift >= pi:
            #     phase_shift -= pi
            # if phase_shift < 0:
            #     phase_shift += pi
            phase_shifts[index] = rad_to_deg * phase_shift

    if is_coupled:
        return delta_a_bars, delta_b_bars, eps_bars
    else:
        return phase_shifts
