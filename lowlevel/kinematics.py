# Contains functions relating the kinematic variables of the interaction

from numpy import sqrt, vectorize, arctan, sin, cos, array, pi

# Constants: proton/neutron masses and hbar
m_p = 938.27208  # MeV/c^2
m_n = 939.56541  # MeV/c^2
hbarc = 197.33  # Mev-fm


# Define the rule to go from energy to momentum
def E_to_p_element(E_lab, interaction):
    """Return p in MeV.

    Parameters
    ----------
    energy      = float
                  lab energy given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """

    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p
    p_rel = sqrt(
        E_lab * m2**2 * (E_lab + 2 * m1) /
        ((m1 + m2)**2 + 2 * m2 * E_lab)
        )
    return p_rel


E_to_p = vectorize(E_to_p_element, excluded=["interaction"])
# print(E_to_p(300, "np"))


def E_to_k(E_lab, interaction):
    """Return k in fm^{-1}.

    Parameters
    ----------
    energy      = float
                  lab energy given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """

    p_rel = E_to_p(E_lab, interaction)
    return p_rel/hbarc


def p_to_E(p_rel, interaction):
    """Return E_lab in MeV.

    Parameters
    ----------
    p_rel       = float
                  relative momentum given in MeV.
    interaction = str
                  {"pp", "nn", "np"}
    """
    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p
    E_lab = (2 * p_rel**2 - 2 * m1 * m2 +
             2 * sqrt((m1**2 + p_rel**2) * (m2**2 + p_rel**2))) / (2 * m2)
    return E_lab


def k_to_E(k_rel, interaction):
    """Return E_lab in MeV.

    Parameters
    ----------
    k_rel       = float
                  relative momentum given in fm^{-1}.
    interaction = str
                  {"pp", "nn", "np"}
    """
    E_lab = p_to_E(hbarc * k_rel, interaction)
    return E_lab


def theta_cm_to_theta_1(theta_cm, p_rel, interaction):
    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p

    E1 = sqrt(m1**2 + p_rel**2)
    E2 = sqrt(m2**2 + p_rel**2)
    gamma = 1 / sqrt(1 - p_rel**2 / E2**2)

    tan_theta1 = sin(theta_cm) / (gamma * cos(theta_cm) + E1/m2)
    theta1 = arctan(tan_theta1)
    return theta1


def theta_cm_to_theta_2(theta_cm, p_rel, interaction):
    """Here theta_2 defined to be > 0"""
    if interaction == "pp":
        m1, m2 = m_p, m_p
    if interaction == "nn":
        m1, m2 = m_n, m_n
    if interaction == "np":
        m1, m2 = m_n, m_p

    E1 = sqrt(m1**2 + p_rel**2)
    E2 = sqrt(m2**2 + p_rel**2)
    gamma = 1 / sqrt(1 - p_rel**2 / E2**2)

    tan_theta2 = -sin(theta_cm) / (gamma * cos(theta_cm) - E2/m2)
    theta2 = arctan(tan_theta2)
    return theta2


def alpha_rotation(theta_cm, p_rel, interaction):
    # Tame craziness near backward angles
    if theta_cm > 178 * pi / 180:
        theta_cm = 178 * pi / 180
    theta1 = theta_cm_to_theta_1(theta_cm, p_rel, interaction)
    alpha = theta_cm / 2 - theta1
    return alpha


def beta_rotation(theta_cm, p_rel, interaction):
    theta2 = theta_cm_to_theta_2(theta_cm, p_rel, interaction)
    beta = theta_cm / 2 + theta2
    return beta


def n_rotation(t):
    """Create an active rotation matrix about normal (y) axis."""
    return array([[cos(t), 0, sin(t)],
                  [0, 1, 0],
                  [-sin(t), 0, cos(t)]])


def vec_lookup(dir_str, theta_cm_radians, p_rel, interaction):
    alpha = alpha_rotation(theta_cm_radians, p_rel, interaction)
    beta = beta_rotation(theta_cm_radians, p_rel, interaction)
    theta_2 = theta_cm_to_theta_2(theta_cm_radians, p_rel, interaction)
    s_hat = array([1, 0, 0])  # In scattering plane
    n_hat = array([0, 1, 0])  # Normal to scattering plane
    k_hat = array([0, 0, 1])  # Direction of beam
    vec_dict = {
        # No measurement
        "0": 0,
        # Beam frame
        "s": s_hat,
        "n": n_hat,
        "k": k_hat,
        # For com frame
        "m": n_rotation(theta_cm_radians/2) @ s_hat,
        "ell": n_rotation(theta_cm_radians/2) @ k_hat,
        # Scattered frame
        "sp": n_rotation(theta_cm_radians/2 + alpha) @ s_hat,
        "kp": n_rotation(theta_cm_radians/2 + alpha) @ k_hat,
        # Recoil particle frame
        "spp": n_rotation(-theta_2 + 2*beta - pi) @ s_hat,
        "kpp": n_rotation(-theta_2 + 2*beta - pi) @ k_hat,
    }
    return vec_dict[dir_str]

# for th in [t/100 for t in range(300, 400, 1)]:
#     print(th, alpha_rotation(th, 100, "np"))

# print(beta_rotation(pi, 100, "np"))

# th = pi/3
# prl = 500
# print(th, alpha_rotation(th, prl, "np"), beta_rotation(th, prl, "np"))

