"""Loss-compensated coherent layers, mesh ensembles and loss laws.

A layer acts on the complex amplitudes of n guided modes as W = g U D. The
diagonal D holds the per-mode transmissions t_i exp(i phi_i), U is the unitary
realized by the interferometer mesh, and the scalar gain g restores the mean
transmitted power, g**2 * E[t**2] = 1. Amplitudes are dimensionless ratios of
output to input mode amplitude; log-gains are in nepers.
"""
from __future__ import annotations

import hashlib

import numpy as np

MASTER_SEED = 20260716

MESHES = ("haar", "givens", "dft")
LOSS_LAWS = ("uniform", "binary", "beta")


def rng_for(tag: str) -> np.random.Generator:
    """Random stream fixed by the master seed and a text tag.

    Tags are hashed with SHA-256 rather than Python's hash(), which is salted
    per process and would break run-to-run reproducibility.
    """
    key = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, key]))


def transmission_moment(delta: float, p: float) -> float:
    """E[t**p] for t = exp(-a) with a uniform on [0, delta]."""
    x = p * delta
    if x == 0.0:
        return 1.0
    return float(-np.expm1(-x) / x)


def gain(delta: float) -> float:
    """Amplitude gain g that makes g**2 * E[t**2] = 1."""
    return float(1.0 / np.sqrt(transmission_moment(delta, 2.0)))


def haar_unitaries(size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Stack of `size` Haar-random n x n unitaries, shape (size, n, n).

    QR of a complex Ginibre matrix with the phases of R's diagonal moved into Q,
    which makes the distribution exactly Haar.
    """
    z = (rng.standard_normal((size, n, n))
         + 1j * rng.standard_normal((size, n, n))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=1, axis2=2)
    return q * (d / np.abs(d))[:, None, :]


def draw_transmissions(size: int, n: int, delta: float, rng: np.random.Generator,
                       law: str = "uniform", corr: float = 0.0) -> np.ndarray:
    """Per-mode amplitude transmissions t, shape (size, n).

    law='uniform' is the model law, t = exp(-a) with a ~ U[0, delta]. For
    corr > 0 all modes of a layer share one draw with probability corr, which
    sets the mode-to-mode loss correlation to corr while every marginal stays
    exactly the model law.

    law='binary' and law='beta' keep t**2 inside [exp(-2 delta), 1] and match
    the model value of E[t**2]: 'binary' puts t**2 on the two end points (a
    mode is either lossless or maximally lossy), 'beta' draws
    t**2 = exp(-2 delta) + (1 - exp(-2 delta)) b with b ~ Beta(alpha, 3).
    """
    if delta == 0.0:
        return np.ones((size, n))
    if law == "uniform":
        if corr == 0.0:
            return np.exp(-rng.uniform(0.0, delta, (size, n)))
        shared = rng.random(size) < corr
        a_own = rng.uniform(0.0, delta, (size, n))
        a_common = rng.uniform(0.0, delta, (size, 1))
        a = np.where(shared[:, None], np.repeat(a_common, n, axis=1), a_own)
        return np.exp(-a)
    floor = np.exp(-2.0 * delta)
    p = (transmission_moment(delta, 2.0) - floor) / (1.0 - floor)
    if law == "binary":
        t2 = np.where(rng.random((size, n)) < p, 1.0, floor)
    elif law == "beta":
        b = rng.beta(3.0 * p / (1.0 - p), 3.0, (size, n))
        t2 = floor + (1.0 - floor) * b
    else:
        raise ValueError(f"unknown loss law: {law}")
    return np.sqrt(t2)


def _apply_mesh(v: np.ndarray, mesh: str, rng: np.random.Generator) -> np.ndarray:
    """Apply an independent random mesh unitary to every row of v (size, n).

    For a Haar unitary U drawn independently of v, U v is uniform on the
    sphere of radius |v|, so the Haar case is sampled directly in O(n)
    operations per row instead of building the n x n matrix. The equivalence
    with explicit Haar matrices is checked in the notebook (benchmark V5).
    """
    size, n = v.shape
    if mesh == "haar":
        z = rng.standard_normal((size, n)) + 1j * rng.standard_normal((size, n))
        scale = np.linalg.norm(v, axis=1) / np.linalg.norm(z, axis=1)
        return z * scale[:, None]
    if mesh == "dft":
        p_in = np.exp(2j * np.pi * rng.random((size, n)))
        p_out = np.exp(2j * np.pi * rng.random((size, n)))
        return p_out * np.fft.fft(p_in * v, axis=1, norm="ortho")
    if mesh == "givens":
        # n alternating columns of SU(2) rotations on neighboring modes.
        v = v.copy()
        for column in range(n):
            first = column % 2
            a, b = v[:, first:n - 1:2], v[:, first + 1:n:2]
            theta, alpha, beta = 2.0 * np.pi * rng.random((3, size, a.shape[1]))
            p = np.cos(theta) * np.exp(1j * alpha)
            q = np.sin(theta) * np.exp(1j * beta)
            v[:, first:n - 1:2], v[:, first + 1:n:2] = (p * a - q * b,
                                                        q.conj() * a + p.conj() * b)
        return v
    raise ValueError(f"unknown mesh: {mesh}")


def propagate(x: np.ndarray, depth: int, delta: float, rng: np.random.Generator,
              mesh: str = "haar", law: str = "uniform", corr: float = 0.0,
              gain_error: float = 0.0, profile: bool = False):
    """Send a batch of inputs x (size, n) through independent random cascades.

    Every row meets its own layers W_k = g (1 + gain_error) U_k D_k. With
    profile=True the natural log of the field norm after each layer is also
    returned, shape (depth + 1, size); row 0 is the input.
    """
    v = np.array(x, dtype=complex)
    size, n = v.shape
    g = gain(delta) * (1.0 + gain_error)
    logs = [np.log(np.linalg.norm(v, axis=1))] if profile else None
    for _ in range(depth):
        t = draw_transmissions(size, n, delta, rng, law=law, corr=corr)
        v = t * np.exp(2j * np.pi * rng.random((size, n))) * v
        v = g * _apply_mesh(v, mesh, rng)
        if profile:
            logs.append(np.log(np.linalg.norm(v, axis=1)))
    if profile:
        return v, np.array(logs)
    return v


def random_inputs(size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit input vectors drawn uniformly from the complex sphere in C^n."""
    z = rng.standard_normal((size, n)) + 1j * rng.standard_normal((size, n))
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def power_gains(size: int, n: int, depth: int, delta: float,
                rng: np.random.Generator, **kwargs) -> np.ndarray:
    """Output power gain G**2 = |y|**2 / |x|**2 for `size` random cascades."""
    y = propagate(random_inputs(size, n, rng), depth, delta, rng, **kwargs)
    return np.linalg.norm(y, axis=1) ** 2


def sample_increments(size: int, n: int, delta: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Exact single-layer log-gain increments xi for Haar meshes.

    The propagating direction is uniform on the sphere, so its squared moduli
    are Dirichlet(1, ..., 1) and xi = ln g + (1/2) ln(sum_i t_i**2 w_i).
    """
    w = rng.dirichlet(np.ones(n), size)
    t2 = draw_transmissions(size, n, delta, rng) ** 2
    return np.log(gain(delta)) + 0.5 * np.log(np.sum(w * t2, axis=1))


def random_layers(depth: int, n: int, delta: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Explicit Haar-mesh layer matrices of one cascade, shape (depth, n, n)."""
    t = draw_transmissions(depth, n, delta, rng)
    d = t * np.exp(2j * np.pi * rng.random((depth, n)))
    return gain(delta) * haar_unitaries(depth, n, rng) * d[:, None, :]
