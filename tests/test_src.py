"""Regression tests for the reusable modules in src/.

Run from the repository root with `python -m pytest -q`.
"""
import hashlib
import math

import numpy as np
import pytest

from src import model, operators, tails, uncertainty
from src.model import MASTER_SEED


# ------------------------------------------------------------------ model
def test_gain_restores_mean_power():
    for delta in (0.0, 0.05, 0.2, 1.0):
        assert math.isclose(model.gain(delta) ** 2 * model.transmission_moment(delta, 2.0),
                            1.0, rel_tol=1e-14)


def test_rng_for_is_reproducible_and_tag_dependent():
    a = model.rng_for("x").standard_normal(4)
    assert np.array_equal(a, model.rng_for("x").standard_normal(4))
    assert not np.array_equal(a, model.rng_for("y").standard_normal(4))


def test_haar_unitaries_are_unitary():
    u = model.haar_unitaries(20, 6, np.random.default_rng(1))
    eye = np.eye(6)
    assert np.allclose(np.einsum("bji,bjk->bik", u.conj(), u), eye, atol=1e-12)


@pytest.mark.parametrize("mesh", model.MESHES)
def test_meshes_preserve_norm(mesh):
    rng = np.random.default_rng(2)
    v = model.random_inputs(50, 8, rng)
    out = model.propagate(v, 5, 0.0, rng, mesh=mesh)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("law", model.LOSS_LAWS)
def test_loss_laws_share_the_second_moment(law):
    t = model.draw_transmissions(400_000, 8, 0.2, np.random.default_rng(3), law=law)
    target = model.transmission_moment(0.2, 2.0)
    assert abs(np.mean(t ** 2) - target) < 3e-3 * target
    assert t.min() >= math.exp(-0.2) - 1e-12 and t.max() <= 1.0 + 1e-12


def test_correlated_losses_keep_marginals_and_hit_target_correlation():
    rng = np.random.default_rng(4)
    for corr in (0.25, 0.75):
        a = -np.log(model.draw_transmissions(300_000, 8, 0.3, rng, corr=corr))
        assert abs(a.mean() - 0.15) < 2e-3
        assert abs(np.corrcoef(a[:, 0], a[:, 5])[0, 1] - corr) < 0.02


def test_zero_correlation_reproduces_the_independent_stream():
    a = model.draw_transmissions(10, 4, 0.5, np.random.default_rng(5), corr=0.0)
    b = np.exp(-np.random.default_rng(5).uniform(0.0, 0.5, (10, 4)))
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ operators
def test_unitary_cascade_certificates_equal_depth():
    layers = model.random_layers(7, 5, 0.0, np.random.default_rng(6))
    c = operators.certificates(layers)
    assert math.isclose(c["S_cert"], 7.0, rel_tol=1e-10)
    assert math.isclose(c["S_Lip"], 7.0, rel_tol=1e-10)


def test_ensemble_certificates_match_explicit_matrices():
    size, n, depth, delta, tag = 5, 4, 6, 0.5, "regression"
    res = operators.ensemble_certificates(size, n, depth, delta, tag)
    key = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")
    seeds = np.random.SeedSequence([MASTER_SEED, key]).spawn(depth)
    stacks = [model.random_layers(size, n, delta, np.random.default_rng(s)) for s in seeds]
    for i in range(size):
        layers = [stacks[k][i] for k in range(depth)]
        c = operators.certificates(layers)
        rho = operators.spectral_radius(operators.product(layers))
        assert math.isclose(res["ratio_spec"][i], c["S_cert"] / c["S_spec"], rel_tol=1e-9)
        assert math.isclose(res["ratio_rho"][i], c["S_cert"] / (depth * rho), rel_tol=1e-9)
        assert math.isclose(math.exp(res["log_lip_over_cert"][i]), c["S_Lip"] / c["S_cert"], rel_tol=1e-9)


def test_exact_bound_holds_for_finite_perturbations():
    rng = np.random.default_rng(7)
    for _ in range(50):
        layers = model.random_layers(6, 4, 0.4, rng)
        eps = rng.uniform(0.01, 0.2)
        pert = []
        for _ in layers:
            e = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
            pert.append(eps * e / np.linalg.norm(e, 2))
        assert operators.perturbation_error(layers, pert) <= operators.exact_bound(layers, eps) + 1e-12


def test_rank_one_perturbation_attains_the_layer_bound():
    rng = np.random.default_rng(8)
    a = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    b = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    assert math.isclose(operators.attainability(a, b, 0.1), 1.0, rel_tol=1e-12)


def test_joint_worst_case_is_bracketed_by_the_certificate():
    rng = np.random.default_rng(9)
    one = model.random_layers(1, 6, 0.5, rng)
    assert math.isclose(operators.joint_worst_case(one), operators.certificates(one)["S_cert"],
                        rel_tol=1e-9)
    layers = model.random_layers(12, 6, 0.5, rng)
    t = operators.joint_worst_case(layers, rng=rng)
    assert 0.0 < t <= operators.certificates(layers)["S_cert"] * (1 + 1e-12)


def test_kreiss_constant_of_normal_and_jordan_matrices():
    normal = np.diag([0.9, 0.5j, -0.7]).astype(complex)
    assert abs(operators.kreiss_constant(normal)["K"] - 1.0) < 1e-3
    jordan = np.array([[0.9, 1.0], [0.0, 0.9]], dtype=complex)
    k = operators.kreiss_constant(jordan)["K"]
    m = operators.peak_transient(jordan)
    assert k <= m * (1 + 1e-9) and m <= math.e * 2 * k


def test_homogeneous_ratio_matches_certificates():
    rng = np.random.default_rng(10)
    w = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    c = operators.certificates([w] * 9)
    assert math.isclose(operators.homogeneous_ratio(w, 9), c["S_cert"] / c["S_spec"], rel_tol=1e-9)


# ------------------------------------------------------------------ tails
def test_haar_moments():
    assert math.isclose(tails.haar_moment(8, 0.3, 1), 1.0, rel_tol=1e-12)
    xi = model.sample_increments(1_000_000, 8, 0.3, np.random.default_rng(11))
    m2 = np.exp(4.0 * xi)
    assert abs(tails.haar_moment(8, 0.3, 2) - m2.mean()) < 5 * m2.std() / math.sqrt(len(m2))


def test_diagonal_tail_matches_monte_carlo():
    rng = np.random.default_rng(12)
    depth, delta, c = 20, 0.4, 2.0
    a = rng.uniform(0.0, delta, (400_000, depth))
    g2 = np.exp(2 * depth * math.log(model.gain(delta)) - 2 * a.sum(axis=1))
    p = np.mean(g2 > c)
    assert abs(tails.diagonal_tail(c, depth, delta) - p) < 5 * math.sqrt(p * (1 - p) / len(g2))


def test_tier_ordering():
    for c in (2.0, 5.0, 20.0):
        b = tails.tier_bounds(c, 64, 0.2, n=8)
        assert b["tier2"] <= b["tier1"] <= b["tier0"]
        assert tails.diagonal_tail(c, 64, 0.2) <= b["tier1"]


def test_clopper_pearson_limits_bracket_the_estimate():
    lo, up = uncertainty.clopper_pearson(20, 100)
    assert 0.0 < lo < 0.2 < up < 1.0
    assert uncertainty.clopper_pearson(0, 100)[0] == 0.0
