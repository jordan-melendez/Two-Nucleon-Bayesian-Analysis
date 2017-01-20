###############################################################################
# Author: Jordan Melendez (melendez.27@osu.edu)
# Affiliation: The Ohio State University
# Created: Aug-09-2016
# Revised:
###############################################################################
# Define a function to create create points and weights for
# Gaussian quadrature.
###############################################################################

from numpy import copy, ones, linspace, cos, tan


def make_points_weights(self, N, start=0, end='infinity', inf_weight=1):
    """Generates N Gaussian quadrature points and weights and sets as
    self.points and self.weights.

    Functions for Gaussian quadrature procedure from Mark Newman:
    http://www-personal.umich.edu/~mejn/computational-physics/gaussxw.py
    I should email him and ask about discrepancies.
    """

    # Initial approximation to roots of the Legendre polynomial
    # using Abramowitz and Stegun 22.16.6.
    a = linspace(3, 4 * N - 1, N) / (4 * N + 2)
    # Where is pi in the second term??
    x = cos(pi * a + 1 / (8 * N * N * tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = ones(N, float)
        p1 = copy(x)
        # Build p0 = P_{N-1} and p1 = P_N using recurrence relation
        # found in Abramowitz and Stegun 22.7.10
        for k in range(1, N):
            p0, p1 = p1, ((2*k+1)*x*p1-k*p0)/(k+1)
        # Recurrence relation for derivative of P_N (almost??)
        # N+1 rather than N? Maybe it's more accurate.
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        # Key formula for Newton's method
        dx = p1/dp
        x -= dx
        # Check for convergence
        delta = max(abs(dx))

    # Calculate the weights
    # Corrects for the N+1 in the definition of dp
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    # Reverse x list so that x[0] < x[1] < ... < x[N]
    # Weight list is symmetric so is left alone
    x = x[::-1]

    # scale points and weights to desired interval
    if end == 'infinity':
        x = [(1+k)/(1-k)*inf_weight for k in x]
        w = [2*inf_weight/(1-x[i])**2 * w[i] for i in range(len(x))]
        x += start
    else:
        x = 0.5*(end-start)*x+0.5*(end+start)
        w = 0.5*(end-start)*w

    return x, w
