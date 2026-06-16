"""Observation modality abstractions for multi-channel POMDPs.

Each observation modality (channel) has:
- A name for identification
- A cardinality (number of observation types)
- Theta-dependence specification (whether/how it depends on static config)

The ModalityRegistry collects modalities and provides computed metadata
for observation parsing, BP message passing, and parameter learning.
"""

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

import jax.numpy as jnp


class ThetaDependence(Enum):
    """How a modality's emission distribution depends on theta.

    INDEPENDENT: p(y | x) - emission does not depend on theta at all.
                 These channels should NOT contribute to theta inference.

    DEPENDENT: p(y | x, theta) - emission depends on both state and theta.
               Full theta-dependence at all states.

    CONDITIONAL: p(y | x, theta) only at specific states, uniform elsewhere.
                 E.g., cue observation that only reveals theta at cue location.
    """
    INDEPENDENT = auto()
    DEPENDENT = auto()
    CONDITIONAL = auto()


@dataclass(frozen=True)
class ObservationModality:
    """Specification for a single observation channel.

    Attributes:
        name: Human-readable identifier (e.g., "position", "cue", "reward").
        cardinality: Number of observation categories for this modality.
        theta_dependence: How this modality depends on theta.
        informative_states: States where theta-dependence is active.
                           Required for CONDITIONAL, must be None otherwise.
    """
    name: str
    cardinality: int
    theta_dependence: ThetaDependence
    informative_states: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.cardinality < 1:
            raise ValueError(f"cardinality must be >= 1, got {self.cardinality}")

        if self.theta_dependence == ThetaDependence.CONDITIONAL:
            if self.informative_states is None:
                raise ValueError(
                    "CONDITIONAL theta_dependence requires informative_states"
                )
            if len(self.informative_states) == 0:
                raise ValueError("informative_states cannot be empty for CONDITIONAL")
        elif self.informative_states is not None:
            raise ValueError(
                "informative_states is only valid for CONDITIONAL theta_dependence"
            )

    @property
    def contributes_to_theta(self) -> bool:
        """Whether this modality should contribute to theta inference."""
        return self.theta_dependence != ThetaDependence.INDEPENDENT


@dataclass(frozen=True)
class ModalityRegistry:
    """Collection of observation modalities with computed metadata.

    Provides derived properties for:
    - Observation vector parsing (sizes, offsets)
    - BP message passing (theta-dependent mask)
    - B tensor construction (max cardinality for padding)

    Attributes:
        modalities: Ordered tuple of ObservationModality objects.
    """
    modalities: tuple[ObservationModality, ...]

    def __post_init__(self) -> None:
        if len(self.modalities) == 0:
            raise ValueError("ModalityRegistry requires at least one modality")

        # Check for duplicate names
        names = [m.name for m in self.modalities]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate modality names: {names}")

    @property
    def n_channels(self) -> int:
        """Number of observation channels."""
        return len(self.modalities)

    @cached_property
    def channel_sizes(self) -> list[int]:
        """Cardinality of each channel, for observation vector parsing."""
        return [m.cardinality for m in self.modalities]

    @cached_property
    def total_obs_dim(self) -> int:
        """Total dimension of concatenated observation vector."""
        return sum(self.channel_sizes)

    @cached_property
    def channel_offsets(self) -> list[int]:
        """Start index of each channel in the concatenated observation vector."""
        offsets = [0]
        for size in self.channel_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        return offsets

    @cached_property
    def max_cardinality(self) -> int:
        """Maximum cardinality across all channels (for B tensor padding)."""
        return max(self.channel_sizes)

    @cached_property
    def theta_dependent_mask(self) -> list[bool]:
        """Per-channel mask: True if channel contributes to theta inference."""
        return [m.contributes_to_theta for m in self.modalities]

    @cached_property
    def theta_dependent_mask_jax(self) -> jnp.ndarray:
        """JAX array version of theta_dependent_mask for BP computations."""
        return jnp.array(self.theta_dependent_mask)

    def get_channel_slice(self, channel_idx: int, obs: jnp.ndarray) -> jnp.ndarray:
        """Extract single channel's observation from concatenated vector.

        Args:
            channel_idx: Index of the channel to extract.
            obs: Concatenated observation vector of shape (total_obs_dim,).

        Returns:
            Channel observation of shape (channel_cardinality,).
        """
        offset = self.channel_offsets[channel_idx]
        size = self.modalities[channel_idx].cardinality
        return obs[offset:offset + size]

    def get_modality_by_name(self, name: str) -> ObservationModality:
        """Look up a modality by name.

        Raises:
            KeyError: If no modality with the given name exists.
        """
        for m in self.modalities:
            if m.name == name:
                return m
        raise KeyError(f"No modality named '{name}'")

    def get_channel_index(self, name: str) -> int:
        """Get the channel index for a modality by name.

        Raises:
            KeyError: If no modality with the given name exists.
        """
        for i, m in enumerate(self.modalities):
            if m.name == name:
                return i
        raise KeyError(f"No modality named '{name}'")
