"""Non-normal diagnostics and worst-case perturbation certificates.

For a cascade P = W_L ... W_1 the prefix before layer k is B_k = W_{k-1} ... W_1
and the suffix after it is A_k = W_L ... W_{k+1}, so that a perturbation
W_k -> W_k + E_k changes the output map by A_k E_k B_k to first order.
Layers are indexed from 0 in the code; every norm is the spectral norm.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import minimize

from .model import MASTER_SEED, random_layers


# --------------------------------------------------------------------------
# single-matrix diagnostics
# --------------------------------------------------------------------------
def spectral_radius(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def peak_transient(a: np.ndarray, kmax: int = 512) -> float:
    """M = max over 0 <= k <= kmax of ||A**k||."""
    p = np.eye(a.shape[0], dtype=complex)
    peak = 1.0
    for _ in range(kmax):
        p = a @ p
        peak = max(peak, np.linalg.norm(p, 2))
    return float(peak)


def _kreiss_objective(a: np.ndarray, eps: float, theta: float) -> float:
    """(|z| - 1) ||(z - A)^(-1)|| at z = (1 + eps) exp(i theta), with eps = |z| - 1 passed
    directly so that it is not formed by cancellation."""
    z = (1.0 + eps) * np.exp(1j * theta)
    smin = np.linalg.svd(z * np.eye(a.shape[0]) - a, compute_uv=False)[-1]
    return eps / smin if smin > 0 else np.inf


def kreiss_constant(a: np.ndarray, n_r: int = 60, n_theta: int = 120,
                    r_max: float = 3.0, starts: int = 5, eps_min: float = 1e-8) -> dict:
    """Kreiss constant K(A) = sup over |z| > 1 of (|z| - 1) ||(z - A)^(-1)||.

    A polar grid with |z| - 1 spaced geometrically from 1e-4 to r_max - 1 is
    scanned first. Its angles are n_theta equally spaced values together with the
    arguments of the eigenvalues of A, along which the resolvent peaks sharply
    when an eigenvalue lies on or near the unit circle. Nelder-Mead then refines
    from the `starts` best grid points in the variables (ln(|z| - 1 - eps_min),
    theta). The floor eps_min keeps the smallest singular value of z - A above
    rounding error when an eigenvalue lies on the unit circle; the objective
    tends continuously to the eigenvalue condition number there, so the floor
    costs a relative error of order eps_min. The result is a lower bound on the
    supremum.
    """
    epss = np.geomspace(1e-4, r_max - 1.0, n_r)
    thetas = np.concatenate([np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False),
                             np.mod(np.angle(np.linalg.eigvals(a)), 2.0 * np.pi)])
    n_theta = len(thetas)
    grid = np.array([[_kreiss_objective(a, e, th) for th in thetas] for e in epss])
    order = np.argsort(grid, axis=None)[::-1][:starts]
    best = (float(grid.flat[order[0]]), epss[order[0] // n_theta], thetas[order[0] % n_theta])
    for idx in order:
        e0, th0 = epss[idx // n_theta], thetas[idx % n_theta]
        res = minimize(lambda p: -_kreiss_objective(a, eps_min + np.exp(p[0]), p[1]),
                       np.array([np.log(e0 - eps_min), th0]), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000})
        if -res.fun > best[0]:
            best = (float(-res.fun), eps_min + float(np.exp(res.x[0])), float(res.x[1]))
    return {"K": best[0], "r": 1.0 + best[1], "theta": best[2],
            "K_grid": float(grid.flat[order[0]])}


def eig_conditioning(a: np.ndarray):
    """Eigenvalues, eigenvalue condition numbers kappa_j and kappa(V).

    With unit right eigenvectors r_j the left eigenvectors are the rows of V^{-1},
    so kappa_j = ||l_j|| and the Petermann factor of mode j is kappa_j**2.
    """
    w, v = np.linalg.eig(a)
    vn = v / np.linalg.norm(v, axis=0, keepdims=True)
    kappa_j = np.linalg.norm(np.linalg.inv(vn), axis=1)
    return w, kappa_j, float(np.linalg.cond(vn, 2))


def noise_gap(a: np.ndarray) -> float:
    """chi = n ||A||_2**2 / ||A||_F**2, the worst-to-mean output-noise ratio."""
    s = np.linalg.svd(a, compute_uv=False)
    return float(a.shape[0] * s[0] ** 2 / np.sum(s ** 2))


# --------------------------------------------------------------------------
# cascades of explicit matrices
# --------------------------------------------------------------------------
def product(layers) -> np.ndarray:
    p = np.eye(layers[0].shape[0], dtype=complex)
    for w in layers:
        p = w @ p
    return p


def partial_products(layers):
    """Prefixes B_k and suffixes A_k for k = 0, ..., L - 1."""
    n, depth = layers[0].shape[0], len(layers)
    prefix = [np.eye(n, dtype=complex)]
    for w in layers[:-1]:
        prefix.append(w @ prefix[-1])
    suffix = [np.eye(n, dtype=complex)]
    for w in layers[:0:-1]:
        suffix.append(suffix[-1] @ w)
    return prefix, suffix[::-1][:depth]


def certificates(layers, eps=1.0) -> dict:
    """First-order worst-case predictors for per-layer budgets ||E_k|| <= eps_k.

    S_cert = sum_k ||A_k|| ||B_k|| eps_k       (layer-resolved certificate)
    S_Lip  = sum_k prod_{j != k} ||W_j|| eps_k  (product of layer norms)
    S_spec = sum_k rho(A_k) rho(B_k) eps_k      (spectral-radius predictor)
    """
    depth = len(layers)
    eps = np.broadcast_to(np.asarray(eps, dtype=float), (depth,))
    prefix, suffix = partial_products(layers)
    wn = np.array([np.linalg.norm(w, 2) for w in layers])
    s_cert = s_lip = s_spec = 0.0
    for k in range(depth):
        s_cert += np.linalg.norm(suffix[k], 2) * np.linalg.norm(prefix[k], 2) * eps[k]
        s_lip += np.prod(np.delete(wn, k)) * eps[k]
        s_spec += spectral_radius(suffix[k]) * spectral_radius(prefix[k]) * eps[k]
    return {"S_cert": float(s_cert), "S_Lip": float(s_lip), "S_spec": float(s_spec)}


def exact_bound(layers, eps) -> float:
    """sum_k ||A_k|| eps_k prod_{j<k} (||W_j|| + eps_j), valid at any budget."""
    depth = len(layers)
    eps = np.broadcast_to(np.asarray(eps, dtype=float), (depth,))
    _, suffix = partial_products(layers)
    wn = np.array([np.linalg.norm(w, 2) for w in layers])
    return float(sum(np.linalg.norm(suffix[k], 2) * eps[k] * np.prod(wn[:k] + eps[:k])
                     for k in range(depth)))


def perturbation_error(layers, perturbations) -> float:
    """||prod(W_k + E_k) - prod(W_k)|| for explicit perturbations E_k."""
    return float(np.linalg.norm(product([w + e for w, e in zip(layers, perturbations)])
                                - product(layers), 2))


def attainability(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    """||A E B|| / (eps ||A|| ||B||) for the rank-one E = eps x y^H that pairs the
    top right singular vector x of A with the top left singular vector y of B."""
    _, sa, vah = np.linalg.svd(a)
    ub, sb, _ = np.linalg.svd(b)
    e = eps * np.outer(vah[0].conj(), ub[:, 0].conj())
    return float(np.linalg.norm(a @ e @ b, 2) / (eps * sa[0] * sb[0]))


def joint_worst_case(layers, starts: int = 4, iters: int = 300,
                     rng: np.random.Generator | None = None) -> float:
    """Lower bound on the joint first-order worst case S_joint.

    With every layer perturbed at unit budget, the largest first-order output
    change is S_joint = max over unit x, y of sum_k ||A_k^H y|| ||B_k x||,
    reached by rank-one E_k aligned with (A_k^H y, B_k x). S_joint <= S_cert,
    with equality only when all layers share one worst-case input and output
    direction. The objective is convex in x for fixed y and vice versa, so
    normalized gradient steps increase it monotonically; any value reached is
    attained by an explicit perturbation and therefore bounds S_joint from below.
    """
    depth = len(layers)
    n = layers[0].shape[0]
    rng = rng or np.random.default_rng(0)
    u, s, vh = np.linalg.svd(product(layers))
    inits = [(vh[0].conj(), u[:, 0])]
    for _ in range(starts - 1):
        z = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
        inits.append((z[0] / np.linalg.norm(z[0]), z[1] / np.linalg.norm(z[1])))

    def forward(x):
        out = [x]
        for w in layers[:-1]:
            out.append(w @ out[-1])
        return out

    def backward(y):
        out = [None] * depth
        out[-1] = y
        for k in range(depth - 1, 0, -1):
            out[k - 1] = layers[k].conj().T @ out[k]
        return out

    best = 0.0
    for x, y in inits:
        previous = -1.0
        for _ in range(iters):
            fu, bv = forward(x), backward(y)
            bn = np.array([np.linalg.norm(v) for v in fu])
            an = np.array([np.linalg.norm(v) for v in bv])
            value = float(an @ bn)
            if value <= previous * (1.0 + 1e-12):
                break
            previous = value
            acc = (bn[0] / an[0]) * bv[0]
            for k in range(1, depth):
                acc = layers[k] @ acc + (bn[k] / an[k]) * bv[k]
            y = acc / np.linalg.norm(acc)
            an = np.array([np.linalg.norm(v) for v in backward(y)])
            acc = (an[-1] / bn[-1]) * fu[-1]
            for k in range(depth - 2, -1, -1):
                acc = layers[k].conj().T @ acc + (an[k] / bn[k]) * fu[k]
            x = acc / np.linalg.norm(acc)
        best = max(best, previous)
    return best


def homogeneous_ratio(w: np.ndarray, depth: int) -> float:
    """S_cert / S_spec for `depth` copies of one layer W (scale free).

    With A = W / rho(W) every partial product has spectral radius one, so
    S_spec = depth and S_cert = sum_k ||A^(depth-1-k)|| ||A^k||.
    """
    a = w / spectral_radius(w)
    powers = [np.eye(w.shape[0], dtype=complex)]
    for _ in range(depth - 1):
        powers.append(a @ powers[-1])
    norms = np.linalg.norm(np.array(powers), ord=2, axis=(1, 2))
    return float(norms[::-1] @ norms / depth)


# --------------------------------------------------------------------------
# ensembles of deep random cascades
# --------------------------------------------------------------------------
def ensemble_certificates(size: int, n: int, depth: int, delta: float,
                          tag: str) -> dict:
    """Certificate ratios for `size` independent Haar cascades of given depth.

    Partial products are renormalized at every step and only their log-norms
    and rho/||.|| are kept, so depths of several thousand layers do not
    overflow. Layer k of every sample is drawn from its own child stream and is
    regenerated for the backward (suffix) pass instead of being stored.

    Returns per-sample arrays:
      ratio_spec     S_cert / S_spec
      ratio_rho          S_cert / (L rho(P)), eigenvalue estimate from the full map
      log_lip_over_cert  ln(S_Lip / S_cert), kept in logs because the ratio
                         grows exponentially with depth
      log_scert          ln S_cert for unit budgets
    """
    key = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")
    seeds = np.random.SeedSequence([MASTER_SEED, key]).spawn(depth)

    def layer(k):
        return random_layers(size, n, delta, np.random.default_rng(seeds[k]))

    def norm_and_radius(m):
        s = np.linalg.norm(m, ord=2, axis=(1, 2))
        m = m / s[:, None, None]
        return m, np.log(s), np.abs(np.linalg.eigvals(m)).max(axis=1)

    eye = np.broadcast_to(np.eye(n, dtype=complex), (size, n, n))
    log_b = np.zeros((depth, size))
    r_b = np.ones((depth, size))
    log_w = np.zeros((depth, size))
    b, acc = eye.copy(), np.zeros(size)
    for k in range(depth):
        w = layer(k)
        log_w[k] = np.log(np.linalg.norm(w, ord=2, axis=(1, 2)))
        if k > 0:
            log_b[k], r_b[k] = acc, rho_b
        b, ls, rho_b = norm_and_radius(w @ b)
        acc = acc + ls
    log_p, r_p = acc, rho_b

    log_a = np.zeros((depth, size))
    r_a = np.ones((depth, size))
    a, acc = eye.copy(), np.zeros(size)
    for k in range(depth - 2, -1, -1):
        a, ls, rho_a = norm_and_radius(a @ layer(k + 1))
        acc = acc + ls
        log_a[k], r_a[k] = acc, rho_a

    log_n = log_a + log_b
    top = log_n.max(axis=0)
    wts = np.exp(log_n - top)
    log_scert = top + np.log(wts.sum(axis=0))
    ratio_spec = wts.sum(axis=0) / (wts * r_a * r_b).sum(axis=0)
    ratio_rho = np.exp(log_scert - np.log(depth) - log_p - np.log(r_p))
    wmax = (-log_w).max(axis=0)
    log_slip = log_w.sum(axis=0) + wmax + np.log(np.exp(-log_w - wmax).sum(axis=0))
    return {"ratio_spec": ratio_spec, "ratio_rho": ratio_rho,
            "log_lip_over_cert": log_slip - log_scert, "log_scert": log_scert}
