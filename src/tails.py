"""Tail certificates for the output power gain G_L**2 of compensated cascades.

Three bounds of increasing strength and decreasing generality:

tier 0   P[G_L**2 > c] <= 1/c
         any meshes, any input, any inter-mode loss correlation; needs only
         g**2 E[t**2] = 1 for every mode.
tier 1   P[G_L**2 > c] <= min_theta c**(-theta) m(theta)**L,  theta >= 1
         with m(theta) = g**(2 theta) E[t**(2 theta)]; same generality as
         tier 0 but uses the full loss marginal.
tier 2   P[G_L**2 > c] <= min_q c**(-q) M_n(q)**L,  q = 1, 2, ...
         Haar meshes only; M_n(q) is the exact q-th moment of one layer.

The mode-diagonal cascade (U_k = identity, input in a single mode) saturates
every tier-1 moment, so its exact tail is a floor for any mesh-independent
certificate.
"""
from __future__ import annotations

import math
from decimal import Decimal, getcontext

import numpy as np

from .model import gain, transmission_moment
from .uncertainty import clopper_pearson


def diagonal_moment(delta: float, theta: float) -> float:
    """m(theta) = E[(g t)**(2 theta)], the largest single-layer moment of order
    theta >= 1 that any mesh can produce."""
    return gain(delta) ** (2.0 * theta) * transmission_moment(delta, 2.0 * theta)


def haar_moment(n: int, delta: float, q: int) -> float:
    """Exact E[G_1**(2q)] for one Haar layer and integer q >= 0.

    With Dirichlet(1, ..., 1) weights w, E[(sum_i w_i x_i)**q] equals
    q! (n-1)! / (n+q-1)! times the tau**q coefficient of (sum_k E[x**k] tau**k)**n.
    """
    mu = np.array([transmission_moment(delta, 2.0 * k) for k in range(q + 1)])
    poly = np.array([1.0])
    for _ in range(n):
        poly = np.convolve(poly, mu)[: q + 1]
    log_dirichlet = math.lgamma(q + 1) + math.lgamma(n) - math.lgamma(n + q)
    return gain(delta) ** (2 * q) * poly[q] * math.exp(log_dirichlet)


def chernoff_bound(c: float, depth: int, moment, orders) -> float:
    """min over the given orders of c**(-order) * moment(order)**depth, capped at 1."""
    exponents = [depth * math.log(moment(q)) - q * math.log(c) for q in orders]
    return min(1.0, math.exp(min(exponents)))


def tier_bounds(c: float, depth: int, delta: float, n: int | None = None) -> dict:
    """Tier 0, tier 1 and (when n is given) the Haar tier-2 bound at threshold c."""
    out = {"tier0": min(1.0, 1.0 / c),
           "tier1": chernoff_bound(c, depth, lambda q: diagonal_moment(delta, q),
                                   np.linspace(1.0, 40.0, 1561))}
    if n is not None:
        out["tier2"] = chernoff_bound(c, depth, lambda q: haar_moment(n, delta, q),
                                      range(1, 17))
    return out


def diagonal_tail(c: float, depth: int, delta: float) -> float:
    """Exact P[G_L**2 > c] for the mode-diagonal cascade.

    ln G_L**2 = 2 L ln g - 2 S with S the sum of L independent U[0, delta]
    losses, so the tail is an Irwin-Hall distribution function. The
    alternating sum is evaluated in 100-digit decimal arithmetic.
    """
    x = (depth * math.log(gain(delta)) - 0.5 * math.log(c)) / delta
    if x <= 0.0:
        return 0.0
    if x >= depth:
        return 1.0
    getcontext().prec = 100
    xd = Decimal(repr(float(x)))
    total = Decimal(0)
    for k in range(int(math.floor(x)) + 1):
        total += (-1) ** k * math.comb(depth, k) * (xd - k) ** depth
    return float(total / math.factorial(depth))


def tail_table(g2: np.ndarray, thresholds, conf: float = 0.99) -> list:
    """Empirical exceedance of each threshold with one-sided Clopper-Pearson limits."""
    size = len(g2)
    rows = []
    for c in thresholds:
        k = int(np.sum(g2 > c))
        lower, upper = clopper_pearson(k, size, conf)
        rows.append({"c": float(c), "count": k, "p_hat": k / size,
                     "cp_lower": lower, "cp_upper": upper})
    return rows
