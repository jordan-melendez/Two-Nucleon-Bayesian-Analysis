# cython: boundscheck=False, wraparound=False, nonecheck=False
from functools import lru_cache
import numpy as np
cimport numpy as np
from numpy import zeros, ones
import itertools
from cython_gsl cimport gsl_sf_legendre_sphPlm, gsl_sf_coupling_3j
from libc.math cimport exp, cos, M_PI, sqrt
from collections import defaultdict


cdef int k_delta (int i, int j): 
    if i==j:
        return 1
    else:
        return 0


def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b),
        a, b = b, a + b
    return a

def show_Ylm(int ell, int m, double theta):
    # cdef int ell = 1;
    # cdef int m = 1;
    # cdef double theta = 0.75*M_PI;
    cdef double Ylm = gsl_sf_legendre_sphPlm(ell, m, cos(theta))
    # double Ylm2 = gsl_sf_legendre_sphPlm(ell, m, cos(-theta));
    print("Ylm_%i^%i(%g) = % .18e\n", ell, m, cos(theta), Ylm)
    return 0


# Define the rule to go from energy to momentum
def E_to_k(double E_lab, str interaction):
    """Return k in fm^{-1}.

    Parameters
    ----------
    energy      = float
                  lab energy given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """
    cdef double hbarc = 197.33  # Mev-fm
    cdef double p_rel = E_to_p(E_lab, interaction)

    return p_rel/hbarc


def E_to_p(double E_lab, str interaction):
    """Return p in MeV.

    Parameters
    ----------
    energy      = float
                  lab energy given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """

    cdef double m_p = 938.27208  # MeV/c^2
    cdef double m_n = 939.56541  # MeV/c^2
    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p
    cdef double p_rel = sqrt(
        E_lab * m2**2 * (E_lab + 2 * m1) /
        ((m1 + m2)**2 + 2 * m2 * E_lab)
        )
    return p_rel


def p_to_E(double p_rel, str interaction):
    """Return E_lab in MeV.

    Parameters
    ----------
    p_rel       = float
                  relative momentum given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """
    cdef double m_p = 938.27208  # MeV/c^2
    cdef double m_n = 939.56541  # MeV/c^2
    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p
    cdef double E_lab = (2 * p_rel**2 - 2 * m1 * m2 +
        2 * sqrt((m1**2 + p_rel**2) * (m2**2 + p_rel**2))) / (2 * m2)
    return E_lab


def k_to_E(double k_rel, str interaction):
    """Return E_lab in MeV.

    Parameters
    ----------
    k_rel       = float
                  relative momentum given in fm^{-1}.
    interaction = str
                  {"pp", "nn", "np"}
    """
    cdef double hbarc = 197.33  # Mev-fm
    cdef double E_lab = p_to_E(hbarc * k_rel, interaction)
    return E_lab


# @lru_cache(maxsize=None)
cdef double my_clebsch_gordan(int two_j1, int two_j2, int two_j3, int two_m1, int two_m2, int two_m3):
    return (-1)**(int((two_j1 - two_j2 + two_m3)/2)) * sqrt(two_j3 + 1) * \
        gsl_sf_coupling_3j(two_j1, two_j2, two_j3, two_m1, two_m2, -two_m3)


@lru_cache(maxsize=None)
def my_spherical_harmonics(int ell, int m, double costheta):
    if m >= 0:
        return gsl_sf_legendre_sphPlm(ell, m, costheta)
    else:
        return (-1)**(-m) * gsl_sf_legendre_sphPlm(ell, -m, costheta)


# cdef inline double my_wigner_3j(int two_j1, int two_j2, int two_j3, int two_m1, int two_m2, int two_m3):
#     return gsl_sf_coupling_3j(two_j1, two_j2, two_j3, two_m1, two_m2, two_m3)


cdef int singlet_triplet_index(int s, int m):
    cdef int index = 0
    if s != 0:
        index = m + 2
    return index


cdef complex M_singlet_triplet_element(S_mat, str interaction, np.ndarray[long, ndim=1] J_list, double E, int Sp, int mp, int S, int m, double theta, double phi):
    """Return M_{sp mp s m}(theta, phi) in millibarns"""
    cdef int fm_sq_to_mb = 10
    cdef complex M_term = 0
    cdef int maxjm, max0j, jp2, J, L, Lp
    for J in J_list:
        maxjm = max(abs(m-mp), J-1)
        max0j = max(0, J-1)
        jp2 = J + 2
        for L in range(max0j, jp2):
            for Lp in range(maxjm, jp2):      
                M_term += 1j**(L - Lp) * (2*J+1) * sqrt(2*L+1) * \
                    my_spherical_harmonics(Lp, m-mp, cos(theta)) * \
                    gsl_sf_coupling_3j(2*Lp, 2*Sp, 2*J, 2*(m-mp), 2*mp, -2*m) * \
                    gsl_sf_coupling_3j(2*L, 2*S, 2*J, 0, 2*m, -2*m) * \
                    (S_mat[E, J, Lp, Sp, L, S] - k_delta(Lp, L)*k_delta(Sp, S))
    M_term *= sqrt(4*M_PI)/(2j * E_to_k(E, interaction))*(-1)**(S-Sp)
    M_term *= sqrt(fm_sq_to_mb)  # Convert units
    return M_term


# def M_uncoupled_element(S_mat, interaction, np.ndarray[long, ndim=1] J_list, E, mp1, mp2, m1, m2, theta, phi):
#     cdef complex element = 0
#     for s in range(2):
#         for sp in range(2):
#             element += (
#                 my_clebsch_gordan(1, 1, 2*sp, 2*mp1, 2*mp2, 2*(mp1 + mp2)) *
#                 my_clebsch_gordan(1, 1, 2*s, 2*m1, 2*m2, 2*(m1 + m2)) *
#                 M_singlet_triplet_element(
#                     S_mat, interaction, J_list, E, sp, mp1 + mp2, s, m1 + m2, theta, phi
#                     )
#                 )
#     return element


def make_Clebsch_matrix():
    c_mat = zeros((4, 4), dtype=float)
    two_m_list = [1, -1]
    # "product" saves me from 2 nested for loops!
    for (m1_index, two_m1), (m2_index, two_m2) in itertools.product(enumerate(two_m_list), repeat=2):
        row = 2*m1_index + m2_index
        for s in range(2):
            for m in range(-s, s+1):
                col = singlet_triplet_index(s, m)
                c_mat[row, col] = my_clebsch_gordan(1, 1, 2*s, two_m1, two_m2, 2*m)
    return np.matrix(c_mat)


clebsch_matrix = make_Clebsch_matrix()


def make_M_singlet_triplet_matrix(S_mat, str interaction, np.ndarray[long, ndim=1] J_list, double E, double theta, double phi):
    cdef np.ndarray[complex, ndim=2] m_mat = ones((4, 4), dtype=complex)
    cdef int s, sp, m, mp, row, col
    for s in range(2):
        for sp in range(2):
            for m in range(-s, s+1):
                for mp in range(-sp, sp+1):
                    row = singlet_triplet_index(sp, mp)
                    col = singlet_triplet_index(s, m)
                    m_mat[row, col] = M_singlet_triplet_element(
                        S_mat, interaction, J_list, E, sp, mp, s, m, theta, phi)
    return np.matrix(m_mat)


def make_M_uncoupled_matrix(M_st):
    return clebsch_matrix @ M_st @ clebsch_matrix.T
