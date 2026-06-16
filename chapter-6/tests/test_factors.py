"""Unit tests for ThetaSpec / FactorSpec and helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.factors import (
    FactorSpec,
    ThetaSpec,
    align_to_full_components,
    expand_per_component_to_factor,
    insert_component_axes,
    marginalize_to_component,
)


# =============================================================================
# ThetaSpec
# =============================================================================


def test_theta_spec_single_component():
    spec = ThetaSpec(component_sizes=(24,))
    assert spec.n_components == 1
    assert spec.n_static == 24


def test_theta_spec_multi_component():
    spec = ThetaSpec(component_sizes=(2, 3, 4))
    assert spec.n_components == 3
    assert spec.n_static == 24


def test_theta_spec_ravel_unravel_roundtrip():
    spec = ThetaSpec(component_sizes=(2, 3, 4))
    for i in range(spec.n_static):
        comps = spec.unravel(i)
        assert spec.ravel(comps) == i


def test_theta_spec_ravel_unravel_roundtrip_single():
    spec = ThetaSpec(component_sizes=(120,))
    for i in range(spec.n_static):
        assert spec.ravel(spec.unravel(i)) == i


def test_theta_spec_ravel_wrong_arity():
    spec = ThetaSpec(component_sizes=(2, 3))
    with pytest.raises(ValueError):
        spec.ravel((1,))


def test_theta_spec_invalid_sizes():
    with pytest.raises(ValueError):
        ThetaSpec(component_sizes=(0, 2))
    with pytest.raises(ValueError):
        ThetaSpec(component_sizes=(-1,))


# =============================================================================
# FactorSpec
# =============================================================================


def test_factor_spec_independent():
    f = FactorSpec(theta_components=())
    assert f.is_theta_independent
    assert f.n_components == 0
    spec = ThetaSpec(component_sizes=(24,))
    assert f.shape_for(spec) == ()


def test_factor_spec_single_component():
    f = FactorSpec(theta_components=(0,))
    spec = ThetaSpec(component_sizes=(24,))
    assert not f.is_theta_independent
    assert f.shape_for(spec) == (24,)


def test_factor_spec_multi_component():
    f = FactorSpec(theta_components=(0, 2))
    spec = ThetaSpec(component_sizes=(2, 3, 4))
    assert f.shape_for(spec) == (2, 4)


def test_factor_spec_axis_for_component():
    f = FactorSpec(theta_components=(0, 2))
    assert f.axis_for_component(0, base_axis=2) == 2
    assert f.axis_for_component(2, base_axis=2) == 3


def test_factor_spec_axis_for_component_missing():
    f = FactorSpec(theta_components=(0, 2))
    with pytest.raises(ValueError):
        f.axis_for_component(1, base_axis=0)


def test_factor_spec_other_axes_for_component():
    f = FactorSpec(theta_components=(0, 1, 2))
    assert f.other_axes_for_component(1, base_axis=2) == (2, 4)


def test_factor_spec_unique_components():
    with pytest.raises(ValueError):
        FactorSpec(theta_components=(0, 0))


# =============================================================================
# marginalize_to_component
# =============================================================================


def test_marginalize_single_component_is_identity():
    f = FactorSpec(theta_components=(0,))
    msg = jnp.log(jnp.array([0.1, 0.5, 0.4]))  # (3,)
    out = marginalize_to_component(msg, f, target_component=0, leading_axes=0)
    np.testing.assert_allclose(out, msg)


def test_marginalize_multi_component():
    # factor sees components (0, 1) with sizes (2, 3)
    f = FactorSpec(theta_components=(0, 1))
    rng = np.random.default_rng(0)
    p = rng.random((2, 3))
    p = p / p.sum()
    msg = jnp.log(jnp.asarray(p))

    # Marginalize to component 0 (sum out component 1, which is at axis 1)
    out0 = marginalize_to_component(msg, f, target_component=0, leading_axes=0)
    expected0 = np.log(p.sum(axis=1))
    np.testing.assert_allclose(np.asarray(out0), expected0, atol=1e-6)

    # Marginalize to component 1 (sum out component 0, which is at axis 0)
    out1 = marginalize_to_component(msg, f, target_component=1, leading_axes=0)
    expected1 = np.log(p.sum(axis=0))
    np.testing.assert_allclose(np.asarray(out1), expected1, atol=1e-6)


def test_marginalize_with_leading_axes():
    f = FactorSpec(theta_components=(0, 1))
    rng = np.random.default_rng(1)
    p = rng.random((5, 2, 3))  # leading axis size 5 (e.g. time)
    msg = jnp.log(jnp.asarray(p))

    out0 = marginalize_to_component(msg, f, target_component=0, leading_axes=1)
    assert out0.shape == (5, 2)
    expected0 = np.log(p.sum(axis=2))
    np.testing.assert_allclose(np.asarray(out0), expected0, atol=1e-6)


# =============================================================================
# expand_per_component_to_factor
# =============================================================================


def test_expand_single_component():
    f = FactorSpec(theta_components=(0,))
    cav_0 = jnp.log(jnp.array([0.2, 0.3, 0.5]))
    out = expand_per_component_to_factor({0: cav_0}, f, leading_axes=0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(cav_0))


def test_expand_multi_component_outer_product():
    f = FactorSpec(theta_components=(0, 1))
    log_a = jnp.log(jnp.array([0.4, 0.6]))      # comp 0 size 2
    log_b = jnp.log(jnp.array([0.1, 0.3, 0.6]))  # comp 1 size 3
    out = expand_per_component_to_factor({0: log_a, 1: log_b}, f, leading_axes=0)
    assert out.shape == (2, 3)
    # In log-space, sum equals log of outer-product
    expected = np.log(np.outer(np.asarray(jnp.exp(log_a)), np.asarray(jnp.exp(log_b))))
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-6)


def test_expand_with_leading_axes():
    f = FactorSpec(theta_components=(0, 1))
    log_a = jnp.zeros((4, 2))   # leading 4, comp 0 size 2
    log_b = jnp.zeros((4, 3))   # leading 4, comp 1 size 3
    out = expand_per_component_to_factor({0: log_a, 1: log_b}, f, leading_axes=1)
    assert out.shape == (4, 2, 3)


# =============================================================================
# insert_component_axes
# =============================================================================


def test_insert_single_component_is_identity():
    f = FactorSpec(theta_components=(0,))
    msg = jnp.zeros(5)
    out = insert_component_axes(msg, f, target_component=0, leading_axes=0)
    assert out.shape == (5,)


def test_insert_multi_component_creates_singleton():
    f = FactorSpec(theta_components=(0, 1, 2))
    spec = ThetaSpec(component_sizes=(2, 3, 4))  # noqa: F841 (sanity)
    msg_for_1 = jnp.zeros(3)  # component 1 has size 3
    out = insert_component_axes(msg_for_1, f, target_component=1, leading_axes=0)
    assert out.shape == (1, 3, 1)


def test_insert_multi_component_with_leading():
    f = FactorSpec(theta_components=(0, 2))
    msg = jnp.zeros((7, 4))   # leading 7, target component (size 4)
    out = insert_component_axes(msg, f, target_component=2, leading_axes=1)
    assert out.shape == (7, 1, 4)


# =============================================================================
# align_to_full_components
# =============================================================================


def test_align_single_component_identity():
    spec = ThetaSpec(component_sizes=(24,))
    f = FactorSpec(theta_components=(0,))
    arr = jnp.zeros((10, 24))  # leading=1
    out = align_to_full_components(arr, f, spec, leading_axes=1)
    assert out.shape == (10, 24)


def test_align_theta_independent():
    spec = ThetaSpec(component_sizes=(2, 3))
    f = FactorSpec(theta_components=())
    arr = jnp.zeros((5,))
    out = align_to_full_components(arr, f, spec, leading_axes=1)
    assert out.shape == (5, 1, 1)


def test_align_subset_components_sorted():
    spec = ThetaSpec(component_sizes=(2, 3, 4))
    f = FactorSpec(theta_components=(0, 2))
    arr = jnp.zeros((5, 2, 4))  # leading=1, then (2, 4)
    out = align_to_full_components(arr, f, spec, leading_axes=1)
    assert out.shape == (5, 2, 1, 4)


def test_align_subset_components_unsorted():
    """When factor declares components in non-sorted order, axes get transposed."""
    spec = ThetaSpec(component_sizes=(2, 3, 4))
    f = FactorSpec(theta_components=(2, 0))   # axis 0 is component 2, axis 1 is component 0
    arr = jnp.arange(5 * 4 * 2).reshape((5, 4, 2)).astype(jnp.float32)
    out = align_to_full_components(arr, f, spec, leading_axes=1)
    assert out.shape == (5, 2, 1, 4)
    # Pick a sample: out[i, c0, 0, c2] should equal arr[i, c2, c0]
    for i in range(5):
        for c0 in range(2):
            for c2 in range(4):
                assert float(out[i, c0, 0, c2]) == float(arr[i, c2, c0])
