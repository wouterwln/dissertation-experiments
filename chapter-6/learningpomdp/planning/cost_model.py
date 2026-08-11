"""Hardware-independent per-decision planning cost for AIF-MP.

The paper's timing table puts the message-passing planners on an A100 GPU
and the sample-based planners on EPYC CPU cores, then draws conclusions
from the comparison. Cross-hardware wall-clock does not license those
conclusions. This count does not depend on hardware at all: it is an
analytic function of the planner's own hyperparameters and the problem
dimensions, so the same number holds on any machine.

The count is a **nominal ceiling**, not a measured cost. It is evaluated
at the nominal planning horizon `runner.horizon`, but under
`runner.clip_horizon: true` the runner shortens the horizon it hands
`plan()` as the episode budget runs down, so the realized per-decision
cost is at or below the reported number. Report it as a ceiling, and
compare cells within the sweep rather than reading the absolute value as
a measurement.

Only *contractions* (logsumexp reductions over a factor's full index set)
are counted, weighted by the number of scalar elements each one reads.
Elementwise kernel builds of the same tensors are excluded: they are the
same order but implementation-dependent in a way the contraction count is
not.

There is deliberately no sibling counter for the tree-search planners.
Their nominal ceiling is just `n_simulations * depth` — under the
canonical `planner.{bamcp_n_simulations, bamcp_depth}` that is
10000 * 6 = 60_000 simulated transitions per decision — but it is a
ceiling twice over and a long way above the realized cost: `depth` is
clamped to the remaining horizon (`bamcp_planner`: `max_depth =
min(self.depth, max(horizon, 1))`), and both `_simulate` and `_rollout`
return as soon as a simulation reaches a terminal outcome, which most
do well before `depth`. Quote it as a ceiling in prose; it needs no code
until something generates the compute table. Note also that it is in
different units from the count below — simulated transitions versus
tensor elements read — so the two are not a like-for-like ratio.
"""

from __future__ import annotations

from collections.abc import Sequence

# Contractions over the (x_new, x_old, u) dynamics slab per iteration of
# the AIF-MP fixed-point loop (`aif_planner._aif_planning_impl.body_fn`),
# counted from the code:
#   1  forward_pass         logsumexp(terms, axis=(1, 2))
#   2  backward_pass        logsumexp(terms, axis=(0, 1))  -> msg to u
#   3  backward_pass        logsumexp(terms_bwd, axis=(0, 2))
#   4  raw_log_dyn_channels logsumexp(log_theta_marg, axis=2, keepdims)
#   5  log_pair             logsumexp(log_theta_marg, axis=2)
#   6  damp_log_channel     logsumexp over the dyn-channel cond axis
# The action-channel damping contracts the smaller (T, x, u) tensor and is
# subdominant, so it is not counted.
_DYN_CONTRACTIONS_PER_ITERATION = 6

# Contractions per observation channel per iteration of the obs-channel
# loop (`aif_planner._precompute_obs_channels`):
#   theta-independent (`body_indep`): the kernel renormalizer, plus
#     damp_log_channel's -> 2.
#   theta-dependent (`body_dep`): the kernel renormalizer, the theta
#     marginalization, plus damp_log_channel's on the full tensor -> 3.
#     The marginal-channel updates run on the theta-free (n_obs, n_states)
#     tensor and are subdominant.
_OBS_CONTRACTIONS_THETA_INDEP = 2
_OBS_CONTRACTIONS_THETA_DEP = 3


def aif_mp_contraction_count(
    n_iterations: int,
    horizon: int,
    n_states: int,
    n_actions: int,
    obs_channels: Sequence[tuple[int, int]],
) -> int:
    """Tensor elements read by contractions in one AIF-MP `plan()` call.

    A ceiling at the given `horizon`, exact only for decisions that
    actually plan that far ahead — see the module docstring.

    Args:
        n_iterations: fixed-point iterations (`planner.iterations`; both
            the dynamics loop and the obs-channel loop run this many).
        horizon: planning horizon T. Callers pass the *nominal*
            `runner.horizon`, which makes the result a per-decision
            ceiling: under `clip_horizon` the horizon plan() actually
            receives shrinks as the episode budget runs down.
        n_states: |x|.
        n_actions: |u|.
        obs_channels: one (n_obs, n_theta) pair per observation channel,
            with n_theta = 1 for theta-independent channels and otherwise
            the product of the component sizes that channel sees (its
            FactorSpec's theta axes).

    Note the theta dependence sits entirely in the obs-channel term. The
    dynamics loop has no theta axis: T is marginalized over the prior
    before the loop (`_aif_planning_impl`, "log_base"), which is what
    prior-locked theta buys. The naive iterations x horizon x |x|^2 x
    |theta| sketch therefore overstates the dynamics side and misses the
    |u| factor.
    """
    dyn = _DYN_CONTRACTIONS_PER_ITERATION * horizon * n_states**2 * n_actions
    obs = 0
    for n_obs, n_theta in obs_channels:
        per_channel = (
            _OBS_CONTRACTIONS_THETA_INDEP if n_theta == 1
            else _OBS_CONTRACTIONS_THETA_DEP
        )
        obs += per_channel * n_obs * n_states * n_theta
    return n_iterations * (dyn + obs)
