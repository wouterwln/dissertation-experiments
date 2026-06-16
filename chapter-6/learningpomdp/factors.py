"""Factor-graph specs for the (possibly factored) θ space.

ThetaSpec describes the structure of θ — a tuple of independent components.
FactorSpec describes which θ-components a particular factor (T, a B-channel, …)
depends on, in the order those components appear as axes of the factor's tensor.

The single-component case `ThetaSpec(component_sizes=(K!,))` is the M1/M2
degenerate. Multi-component is M4 territory; the abstraction supports it
end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


@dataclass(frozen=True)
class ThetaSpec:
    """Specification of the θ space as a product of independent components.

    component_sizes = (K!,) is the M1/M2 single-component case (door-code).
    component_sizes = (2,)*K is the M4 binary-factored case.

    `episode_reset_components` is the subset of component indices whose
    posterior is re-uniformed at every episode reset within a block (M4
    rock-quality θ). The remaining components persist across the block
    (M4 sensor-calibration θ). Default empty preserves M1/M2 behavior.
    """

    component_sizes: tuple[int, ...]
    episode_reset_components: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(s, int) and s > 0 for s in self.component_sizes):
            raise ValueError(
                f"component_sizes must be positive ints, got {self.component_sizes}"
            )
        if len(set(self.episode_reset_components)) != len(self.episode_reset_components):
            raise ValueError(
                f"episode_reset_components must be unique, "
                f"got {self.episode_reset_components}"
            )
        n = len(self.component_sizes)
        for k in self.episode_reset_components:
            if not 0 <= k < n:
                raise ValueError(
                    f"episode_reset_components index {k} out of range "
                    f"[0, {n}) for component_sizes={self.component_sizes}"
                )

    @property
    def n_components(self) -> int:
        return len(self.component_sizes)

    @property
    def n_static(self) -> int:
        if not self.component_sizes:
            return 1
        return int(np.prod(self.component_sizes))

    @property
    def persistent_components(self) -> tuple[int, ...]:
        """Component indices whose posterior persists across episodes."""
        episodic = set(self.episode_reset_components)
        return tuple(k for k in range(self.n_components) if k not in episodic)

    def ravel(self, components: tuple[int, ...]) -> int:
        if len(components) != self.n_components:
            raise ValueError(
                f"expected {self.n_components} components, got {len(components)}"
            )
        return int(np.ravel_multi_index(components, self.component_sizes))

    def unravel(self, flat_idx: int) -> tuple[int, ...]:
        return tuple(int(i) for i in np.unravel_index(flat_idx, self.component_sizes))


@dataclass(frozen=True)
class FactorSpec:
    """Specification of a factor's dependence on θ components.

    theta_components is the tuple of component indices this factor sees,
    in declaration order matching the factor tensor's θ axes.

    () = θ-independent
    (0,) = single-component (door-code today, all θ-dependent factors)
    (k1, k2, …) = subset; the order is the axis order of the factor's tensor.
    """

    theta_components: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(set(self.theta_components)) != len(self.theta_components):
            raise ValueError(
                f"theta_components must be unique, got {self.theta_components}"
            )

    @property
    def n_components(self) -> int:
        return len(self.theta_components)

    @property
    def is_theta_independent(self) -> bool:
        return len(self.theta_components) == 0

    def shape_for(self, theta_spec: ThetaSpec) -> tuple[int, ...]:
        """The factor's θ-axis shape suffix."""
        return tuple(theta_spec.component_sizes[i] for i in self.theta_components)

    def axis_for_component(self, target: int, base_axis: int = 0) -> int:
        """Axis position of a target component within the factor's tensor.

        base_axis is the offset of the first θ axis (e.g. base_axis=2 for
        a (n_obs, n_states, *theta_axes) tensor).
        """
        for i, k in enumerate(self.theta_components):
            if k == target:
                return base_axis + i
        raise ValueError(
            f"factor with theta_components={self.theta_components} does not see "
            f"component {target}"
        )

    def other_axes_for_component(
        self, target: int, base_axis: int = 0
    ) -> tuple[int, ...]:
        """Axes of all θ components other than `target`, offset by base_axis."""
        target_idx = self.theta_components.index(target)
        return tuple(
            base_axis + i
            for i in range(self.n_components)
            if i != target_idx
        )


def marginalize_to_component(
    log_msg: jnp.ndarray,
    factor: FactorSpec,
    target_component: int,
    leading_axes: int = 0,
) -> jnp.ndarray:
    """Marginalize a per-factor log message down to one of its components.

    Args:
        log_msg: tensor of shape (*leading, *factor.shape_for(theta_spec)).
        factor: the FactorSpec the message lives on.
        target_component: a component the factor sees.
        leading_axes: number of leading non-θ axes.

    Returns:
        log marginal over the target component, shape
        (*leading, theta_spec.component_sizes[target_component]).
    """
    other = factor.other_axes_for_component(target_component, base_axis=leading_axes)
    if not other:
        return log_msg
    return logsumexp(log_msg, axis=other)


def expand_per_component_to_factor(
    per_component_terms: dict[int, jnp.ndarray],
    factor: FactorSpec,
    leading_axes: int = 0,
) -> jnp.ndarray:
    """Outer-product (in log-space: sum) of per-component terms onto a factor's θ axes.

    Used for cavity reconstruction under factored q(θ): cavity_f over the
    components f sees is the *sum* of per-component log cavities, each
    broadcast to f's θ-axis layout.

    Args:
        per_component_terms: dict component_idx -> tensor of shape
            (*leading, component_sizes[component_idx]). Must contain every
            component in factor.theta_components.
        factor: the FactorSpec.
        leading_axes: number of leading non-θ axes shared across terms.

    Returns:
        tensor of shape (*leading, *factor.shape_for(theta_spec)).
    """
    if factor.is_theta_independent:
        # No θ axes; arbitrary call site choice. Return zeros over leading axes
        # broadcasted to scalar over (no) θ axes — i.e. just zeros of leading shape.
        any_term = next(iter(per_component_terms.values()), None)
        if any_term is None:
            return jnp.zeros(())
        return jnp.zeros(any_term.shape[:leading_axes])

    n_factor_axes = factor.n_components
    result = None
    for i, k in enumerate(factor.theta_components):
        term_k = per_component_terms[k]
        size_k = term_k.shape[-1]
        new_shape = list(term_k.shape[:leading_axes])
        for j in range(n_factor_axes):
            new_shape.append(size_k if j == i else 1)
        term_broadcast = term_k.reshape(new_shape)
        result = term_broadcast if result is None else result + term_broadcast
    return result


def align_to_full_components(
    arr: jnp.ndarray,
    factor: FactorSpec,
    theta_spec: ThetaSpec,
    leading_axes: int = 0,
) -> jnp.ndarray:
    """Reshape a per-factor tensor to align with the full (*component_sizes) axes.

    Input: arr of shape (*leading, *factor.shape_for(theta_spec)).
    Output: arr' of shape (*leading, *all_components_shape) where:
      all_components_shape[k] = component_sizes[k] if k in factor.theta_components
                                else 1.
    Components the factor doesn't see become size-1 (for broadcast).
    Components the factor sees are placed at their global component-index axis.
    """
    factor_components = factor.theta_components
    full_shape = list(arr.shape[:leading_axes])
    for k in range(theta_spec.n_components):
        if k in factor_components:
            full_shape.append(theta_spec.component_sizes[k])
        else:
            full_shape.append(1)

    # If factor_components is already sorted, we can just reshape;
    # otherwise we need to transpose the θ axes into sorted order first.
    sorted_components = tuple(sorted(factor_components))
    if factor_components == sorted_components:
        return arr.reshape(full_shape)
    sort_perm = [factor_components.index(c) for c in sorted_components]
    full_perm = list(range(leading_axes)) + [leading_axes + p for p in sort_perm]
    return arr.transpose(full_perm).reshape(full_shape)


def insert_component_axes(
    log_msg: jnp.ndarray,
    factor: FactorSpec,
    target_component: int,
    leading_axes: int = 0,
) -> jnp.ndarray:
    """Reshape a per-component message to fit at the right axis of a factor.

    Given a tensor of shape (*leading, component_sizes[k]) and a factor that
    sees component k at some axis, insert size-1 axes for the factor's other
    components so the result broadcasts correctly into the factor's tensor.

    Used when adding per-component cavities into a factor's region belief.
    """
    target_axis = factor.axis_for_component(target_component, base_axis=leading_axes)
    n_factor_axes = factor.n_components
    new_shape = list(log_msg.shape[:leading_axes])
    for j in range(n_factor_axes):
        if leading_axes + j == target_axis:
            new_shape.append(log_msg.shape[-1])
        else:
            new_shape.append(1)
    return log_msg.reshape(new_shape)
