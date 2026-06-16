"""Tests for the ThetaSpec.episode_reset_components extension and the
``reset_episode_theta`` helper. Covers M3 lifetime annotation + M4
per-episode posterior reset semantics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.factors import ThetaSpec
from learningpomdp.inference.state_inference import (
    reset_episode_theta,
    sample_persistent_theta,
    sample_persistent_theta_factored,
)


# --- ThetaSpec validation ---------------------------------------------------


def test_default_episode_reset_components_is_empty():
    spec = ThetaSpec(component_sizes=(6,))
    assert spec.episode_reset_components == ()
    assert spec.persistent_components == (0,)


def test_episode_reset_components_persistent_complement():
    spec = ThetaSpec(
        component_sizes=(2, 2, 2, 2),
        episode_reset_components=(0, 1),
    )
    assert spec.persistent_components == (2, 3)


def test_episode_reset_components_rejects_out_of_range():
    with pytest.raises(ValueError):
        ThetaSpec(component_sizes=(2, 2), episode_reset_components=(2,))


def test_episode_reset_components_rejects_duplicates():
    with pytest.raises(ValueError):
        ThetaSpec(component_sizes=(2, 2, 2), episode_reset_components=(0, 0))


# --- reset_episode_theta ----------------------------------------------------


def test_reset_is_identity_when_no_episode_components():
    spec = ThetaSpec(component_sizes=(6,))
    q = jnp.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    out = reset_episode_theta(q, spec)
    assert jnp.allclose(out, q)


def test_reset_uniforms_a_single_episodic_component():
    spec = ThetaSpec(
        component_sizes=(2, 2),
        episode_reset_components=(0,),
    )
    # q(θ_0, θ_1): 4 entries, axis 0 = episodic, axis 1 = persistent
    q = jnp.array([[0.4, 0.1],
                   [0.3, 0.2]])
    out = reset_episode_theta(q, spec)
    # After reset, axis 0 should be uniform conditional on axis 1.
    # axis 1 marginal: [0.7, 0.3] preserved.
    # Conditional q(θ_0 | θ_1) should be uniform = 0.5.
    expected = jnp.array([[0.35, 0.15],
                          [0.35, 0.15]])
    assert jnp.allclose(out, expected, atol=1e-6)


def test_reset_preserves_persistent_marginal():
    spec = ThetaSpec(
        component_sizes=(2, 2, 2, 2),
        episode_reset_components=(0, 1),
    )
    rng = np.random.default_rng(0)
    q_flat = rng.dirichlet(np.ones(16))
    q = jnp.asarray(q_flat.reshape(2, 2, 2, 2))
    out = reset_episode_theta(q, spec)
    # Persistent marginal (axes 2, 3) should be unchanged.
    persistent_in = q.sum(axis=(0, 1))
    persistent_out = out.sum(axis=(0, 1))
    assert jnp.allclose(persistent_in, persistent_out, atol=1e-6)


def test_reset_makes_episodic_marginal_uniform():
    spec = ThetaSpec(
        component_sizes=(2, 2, 2, 2),
        episode_reset_components=(0, 1),
    )
    rng = np.random.default_rng(7)
    q = jnp.asarray(rng.dirichlet(np.ones(16)).reshape(2, 2, 2, 2))
    out = reset_episode_theta(q, spec)
    # Episodic marginal should be uniform = 1/4.
    episodic = out.sum(axis=(2, 3))
    assert jnp.allclose(episodic, jnp.full((2, 2), 0.25), atol=1e-6)


def test_reset_normalization_preserved():
    spec = ThetaSpec(
        component_sizes=(2, 2, 2, 2),
        episode_reset_components=(0, 1),
    )
    rng = np.random.default_rng(13)
    q = jnp.asarray(rng.dirichlet(np.ones(16)).reshape(2, 2, 2, 2))
    out = reset_episode_theta(q, spec)
    assert jnp.allclose(out.sum(), 1.0, atol=1e-6)


def test_reset_accepts_flat_input():
    """Sanity: q passed as flat (n_static,) is reshaped internally."""
    spec = ThetaSpec(
        component_sizes=(2, 2),
        episode_reset_components=(0,),
    )
    q_factored = jnp.array([[0.4, 0.1],
                            [0.3, 0.2]])
    q_flat = q_factored.reshape(-1)
    out = reset_episode_theta(q_flat, spec)
    expected = jnp.array([[0.35, 0.15],
                          [0.35, 0.15]])
    # Output has the spec's factored shape.
    assert out.shape == (2, 2)
    assert jnp.allclose(out, expected, atol=1e-6)


# --- sample_persistent_theta ------------------------------------------------


def test_sample_persistent_returns_factored_shape():
    spec = ThetaSpec(
        component_sizes=(2, 2, 2, 2),
        episode_reset_components=(0, 1),
    )
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.dirichlet(np.ones(16)).reshape(2, 2, 2, 2))
    q_cond, theta_persistent = sample_persistent_theta(q, spec, np.random.default_rng(1))
    assert q_cond.shape == (2, 2, 2, 2)
    assert len(theta_persistent) == 4


def test_sample_persistent_axes_are_one_hot():
    spec = ThetaSpec(
        component_sizes=(2, 2, 2, 2),
        episode_reset_components=(0, 1),
    )
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.dirichlet(np.ones(16)).reshape(2, 2, 2, 2))
    q_cond, theta_persistent = sample_persistent_theta(q, spec, np.random.default_rng(3))
    # Persistent marginal of q_cond is one-hot at theta_persistent[2], theta_persistent[3].
    persistent_marg = q_cond.sum(axis=(0, 1))  # shape (2, 2)
    p2, p3 = theta_persistent[2], theta_persistent[3]
    assert p2 is not None and p3 is not None
    expected = jnp.zeros((2, 2)).at[p2, p3].set(1.0)
    assert jnp.allclose(persistent_marg, expected, atol=1e-6)
    # Episodic positions are None.
    assert theta_persistent[0] is None
    assert theta_persistent[1] is None


def test_sample_persistent_episodic_is_true_conditional():
    """The episodic axes of q_cond must equal q(θ_ep | θ̃_persist), i.e. the
    slice of the joint at the sampled persistent value, renormalised — not
    the marginal q(θ_ep)."""
    spec = ThetaSpec(
        component_sizes=(2, 2),
        episode_reset_components=(0,),
    )
    # Construct a q where conditional and marginal differ for axis 0.
    # Joint:
    #   q(0,0)=0.4, q(0,1)=0.1, q(1,0)=0.05, q(1,1)=0.45
    # Episodic marginal q(θ_0):       [0.45, 0.55]
    # Conditional q(θ_0 | θ_1=0):     [0.4, 0.05] / 0.45 = [0.8889, 0.1111]
    # Conditional q(θ_0 | θ_1=1):     [0.1, 0.45] / 0.55 = [0.1818, 0.8182]
    q = jnp.array([[0.4, 0.1], [0.05, 0.45]])

    # Force the sampler to pick θ_1=0 by zeroing the other slice via a
    # custom rng.choice mock would be invasive; instead, run many samples
    # and verify both branches produce the right conditional.
    seen_p1_0 = False
    seen_p1_1 = False
    for seed in range(50):
        q_cond, theta_persistent = sample_persistent_theta(
            q, spec, np.random.default_rng(seed)
        )
        p1 = theta_persistent[1]
        # Episodic axis 0 should be the conditional q(θ_0 | θ_1=p1).
        episodic_marg = q_cond.sum(axis=1)  # marginalize the persistent one-hot
        if p1 == 0:
            seen_p1_0 = True
            expected = jnp.array([0.4, 0.05]) / 0.45
            assert jnp.allclose(episodic_marg, expected, atol=1e-5), (
                f"seed={seed}: got {episodic_marg}, expected {expected}"
            )
        else:
            seen_p1_1 = True
            expected = jnp.array([0.1, 0.45]) / 0.55
            assert jnp.allclose(episodic_marg, expected, atol=1e-5)
    assert seen_p1_0 and seen_p1_1, "sampler did not exercise both branches"


def test_sample_persistent_distribution_matches_marginal():
    """Monte-Carlo gate: the empirical sampling distribution over the
    persistent value should match the persistent marginal of q."""
    spec = ThetaSpec(
        component_sizes=(2, 2, 2),
        episode_reset_components=(0,),
    )
    rng = np.random.default_rng(11)
    q = jnp.asarray(rng.dirichlet(np.ones(8)).reshape(2, 2, 2))
    persistent_marg = np.asarray(q.sum(axis=0))  # shape (2, 2)
    persistent_marg = persistent_marg / persistent_marg.sum()

    counts = np.zeros((2, 2))
    n_samples = 5000
    sampler_rng = np.random.default_rng(42)
    for _ in range(n_samples):
        _, tp = sample_persistent_theta(q, spec, sampler_rng)
        counts[tp[1], tp[2]] += 1
    empirical = counts / n_samples
    assert np.allclose(empirical, persistent_marg, atol=0.02), (
        f"empirical={empirical}\nexpected={persistent_marg}"
    )


def test_sample_persistent_is_identity_when_no_episodic():
    """Door-code degenerate: episode_reset_components=() ⇒ helper returns
    q reshaped to component_sizes and an empty persistent tuple — but
    PSRL/BAMCP regression depends on this branch returning what callers
    used to compute via flat sampling. Here we just assert the shape and
    that no exception fires; the byte-identical flat-sample regression is
    asserted via the planner regression tests."""
    spec = ThetaSpec(component_sizes=(6,))  # door-code shape, all persistent
    q = jnp.asarray(np.random.default_rng(0).dirichlet(np.ones(6)))
    q_cond, theta_persistent = sample_persistent_theta(q, spec, np.random.default_rng(1))
    assert q_cond.shape == (6,)
    # Persistent axis is one-hot at the sampled index.
    assert jnp.isclose(q_cond.sum(), 1.0)
    assert jnp.isclose(q_cond.max(), 1.0)
    assert len(theta_persistent) == 1
    assert theta_persistent[0] is not None


def test_sample_persistent_factored_returns_factored_shape():
    spec = ThetaSpec(
        component_sizes=(2, 2, 2, 2),
        episode_reset_components=(0, 1),
    )
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.dirichlet(np.ones(16)).reshape(2, 2, 2, 2))
    q_cond, theta_persistent = sample_persistent_theta_factored(
        q, spec, np.random.default_rng(1)
    )
    assert q_cond.shape == (2, 2, 2, 2)
    assert len(theta_persistent) == 4
    # Persistent positions hold ints, episodic hold None.
    assert theta_persistent[0] is None
    assert theta_persistent[1] is None
    assert theta_persistent[2] in (0, 1)
    assert theta_persistent[3] in (0, 1)


def test_sample_persistent_factored_independent_components():
    """Under independent persistent components, factored sampling matches
    each 1-D marginal (and is distributionally identical to flat sampling
    on the joint persistent marginal — RockSample's regime)."""
    spec = ThetaSpec(
        component_sizes=(2, 2, 2),
        episode_reset_components=(0,),
    )
    # Construct q where persistent components (axes 1, 2) are independent.
    p1 = np.array([0.7, 0.3])
    p2 = np.array([0.4, 0.6])
    p_episodic = np.array([0.6, 0.4])
    q_np = (
        p_episodic[:, None, None] * p1[None, :, None] * p2[None, None, :]
    )
    q = jnp.asarray(q_np)

    counts = np.zeros((2, 2))
    n_samples = 5000
    sampler_rng = np.random.default_rng(42)
    for _ in range(n_samples):
        _, tp = sample_persistent_theta_factored(q, spec, sampler_rng)
        counts[tp[1], tp[2]] += 1
    empirical = counts / n_samples
    expected = p1[:, None] * p2[None, :]
    assert np.allclose(empirical, expected, atol=0.02), (
        f"empirical={empirical}\nexpected={expected}"
    )


def test_sample_persistent_factored_diverges_when_components_correlated():
    """When persistent components are correlated, factored sampling
    (independent draws from 1-D marginals) and flat sampling on the joint
    differ — this is the regime where Factored BA-POMCP would diverge from
    persistent-only BAMCP."""
    spec = ThetaSpec(
        component_sizes=(2, 2),
        episode_reset_components=(),
    )
    # Highly correlated joint: q ≈ 0.5 δ(0,0) + 0.5 δ(1,1).
    q = jnp.array([[0.5, 0.0], [0.0, 0.5]])

    # Joint sampling from q produces only (0,0) or (1,1).
    flat_pairs = set()
    rng_flat = np.random.default_rng(7)
    for _ in range(200):
        _, tp = sample_persistent_theta(q, spec, rng_flat)
        flat_pairs.add((tp[0], tp[1]))
    assert flat_pairs == {(0, 0), (1, 1)}

    # Factored sampling samples each axis from the marginal [0.5, 0.5]
    # independently — every pair has probability 0.25.
    factored_pairs: dict = {}
    rng_fact = np.random.default_rng(13)
    for _ in range(2000):
        _, tp = sample_persistent_theta_factored(q, spec, rng_fact)
        factored_pairs[(tp[0], tp[1])] = factored_pairs.get((tp[0], tp[1]), 0) + 1
    assert set(factored_pairs.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}
