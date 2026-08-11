"""Numba-JIT'd inner loop for BAMCP / Factored-BA-POMCP rollouts.

Dense array tree: parallel arrays indexed by node_idx, plus a 3D dense
``children[parent, action, obs_idx] -> child_idx`` lookup initialised
to -1 sentinel. Avoids the typed-Dict friction Numba has with
composite keys.

Statistical equivalence (not bit-identical) with the NumPy path is
asserted by ``tests/test_bamcp_numba_equivalence.py`` — the RNG paths
diverge (Numba uses ``np.random``; the NumPy path uses
``np.random.Generator``) but Monte-Carlo means match within 3 SE on
small envs.

Inputs are heavily preprocessed in the calling planner:

- ``T_eff``: 3D ``(n_states_new, n_states_old, n_actions)``. The
  caller handles theta-dep vs theta-indep T (by slicing or
  broadcasting before the JIT call — Numba indexes a 3D buffer).
- ``B_pos``: position channel ``(n_views, n_states)``. Argmax-only
  (deterministic identity in flat envs, many-to-one in spatial envs).
- ``B_lik_all``: per-θ effective likelihood channels
  ``(n_static, n_likelihood, max_obs, n_states)``. Each
  ``B_lik_all[theta_flat, c, :card_c, state]`` is the channel's
  observation distribution at that θ. Pre-built by the caller from
  ``B_list[c]`` and the channel's ``FactorSpec``.
- ``lik_obs_card``: per-channel actual cardinality.
- ``reward_table``: ``(n_states, n_actions, n_static)``.
- ``theta_samples``: per-simulation flat θ̃ (already sampled in
  Python from the persistent-only conditional).
- ``root_states``: per-simulation root state sample.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _sample_categorical(p, n):
    """Sample int in [0, n) proportional to p[:n]. Caller has
    normalised p so it sums to ~1; we draw once from uniform and walk
    the cumsum. O(n) but n is tiny (n_states ~70 or n_obs ~9)."""
    u = np.random.random()
    cumsum = 0.0
    for i in range(n):
        cumsum += p[i]
        if u < cumsum:
            return i
    return n - 1  # numerical-overflow safety


@njit(cache=True, fastmath=False)
def _argmax_int(x):
    """Numba-friendly argmax over a 1-D float array."""
    best_i = 0
    best_v = x[0]
    for i in range(1, x.shape[0]):
        if x[i] > best_v:
            best_v = x[i]
            best_i = i
    return best_i


@njit(cache=True, fastmath=False)
def _argmax_random_tiebreak(x):
    """Argmax over a 1-D float array, uniform random among all maximisers.

    Used only by the eps-greedy rollout's Q-hat lookup. ``_argmax_int``'s
    lowest-index tie-break pins that rollout to action 0 whenever Q-hat is
    flat (see ``bamcp_planner``'s module docstring); this mirrors the
    random tie-break ``_ucb_select_action`` already uses. No RNG draw is
    taken unless there is an actual tie, so the unique-maximiser case
    leaves the stream alone.
    """
    best_v = x[0]
    n_best = 1
    for i in range(1, x.shape[0]):
        if x[i] > best_v:
            best_v = x[i]
            n_best = 1
        elif x[i] == best_v:
            n_best += 1
    if n_best == 1:
        return _argmax_int(x)
    pick = np.random.randint(0, n_best)
    seen = 0
    best_i = 0
    for i in range(x.shape[0]):
        if x[i] == best_v:
            if seen == pick:
                best_i = i
                break
            seen += 1
    return best_i


@njit(cache=True, fastmath=False)
def _step_numba(
    state, theta_flat, action,
    T_eff, B_pos, B_lik_all, lik_obs_card,
    reward_table, terminal_table, has_terminal,
    n_likelihood,
):
    """One rollout step. Returns (next_state, obs_idx, reward, terminal).

    obs_idx is mixed-radix: obs_idx = ((view_idx * card_0 + obs_0) *
    card_1 + obs_1) * ... — uniquely identifies the agent's
    observation tuple.
    """
    # 1. Sample successor.
    next_state_dist = T_eff[:, state, action]
    n_states = next_state_dist.shape[0]
    next_state = _sample_categorical(next_state_dist, n_states)

    # 2. Position observation — deterministic argmax of B_pos[:, next_state].
    view_idx = _argmax_int(B_pos[:, next_state])

    # 3. Per-channel likelihood observations.
    obs_idx = view_idx
    for c in range(n_likelihood):
        card = lik_obs_card[c]
        # Inline categorical sample to avoid array allocation; walks
        # B_lik_all[theta_flat, c, :card, next_state].
        u = np.random.random()
        cumsum = 0.0
        sample = card - 1
        for o in range(card):
            cumsum += B_lik_all[theta_flat, c, o, next_state]
            if u < cumsum:
                sample = o
                break
        obs_idx = obs_idx * card + sample

    # 4. Reward and terminal lookup.
    reward = reward_table[state, action, theta_flat]
    if has_terminal:
        terminal = terminal_table[state, action, theta_flat]
    else:
        terminal = False
    return next_state, obs_idx, reward, terminal


@njit(cache=True, fastmath=False)
def _ucb_select_action(
    visit_count, action_visits, action_values,
    node_idx, n_actions, ucb_c,
):
    """Uniform random over unvisited actions; else UCB1-best.

    Mirrors the NumPy BAMCP path's `self._rng.choice(unvisited)`
    tie-break — required for statistical equivalence with the NumPy
    reference. Two-pass count-then-pick avoids per-call array
    allocation inside the JIT'd hot loop.
    """
    n_unvisited = 0
    for a in range(n_actions):
        if action_visits[node_idx, a] == 0:
            n_unvisited += 1
    if n_unvisited > 0:
        pick = np.random.randint(0, n_unvisited)
        seen = 0
        for a in range(n_actions):
            if action_visits[node_idx, a] == 0:
                if seen == pick:
                    return a
                seen += 1
    parent_visits = visit_count[node_idx]
    log_pv = np.log(max(parent_visits, 1))
    best_a = 0
    best_score = -1e30
    for a in range(n_actions):
        n_a = action_visits[node_idx, a]
        mean_v = action_values[node_idx, a] / n_a
        score = mean_v + ucb_c * np.sqrt(log_pv / n_a)
        if score > best_score:
            best_score = score
            best_a = a
    return best_a


@njit(cache=True, fastmath=False)
def _rollout_numba(
    state, theta_flat, depth,
    T_eff, B_pos, B_lik_all, lik_obs_card,
    reward_table, terminal_table, has_terminal,
    gamma, n_actions, n_likelihood,
    q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
):
    """Leaf rollout to depth-`depth`.

    ``use_q_rollout=False`` is the uniform-random rollout: the
    ``and``-short-circuit means no extra RNG draw is taken and no Q-hat
    element is touched, so the stream stays bit-identical to the
    pre-rollout-policy behaviour. ``use_q_rollout=True`` is epsilon-greedy
    in ``q_rollout`` (ties broken uniformly at random), updated off-policy
    from the step's ``(reward, next_state)``. ``q_rollout`` is owned and
    pre-allocated by the calling planner (a (1, 1) dummy when unused, and
    optimistically initialised otherwise) so it survives across plan()
    calls within a theta_true block.
    """
    total = 0.0
    gamma_t = 1.0
    for _ in range(depth):
        if use_q_rollout and np.random.random() >= rollout_epsilon:
            action = _argmax_random_tiebreak(q_rollout[state])
        else:
            action = np.random.randint(0, n_actions)
        next_state, _, reward, terminal = _step_numba(
            state, theta_flat, action,
            T_eff, B_pos, B_lik_all, lik_obs_card,
            reward_table, terminal_table, has_terminal,
            n_likelihood,
        )
        total += gamma_t * reward
        if use_q_rollout:
            if terminal:
                target = reward
            else:
                target = reward + gamma * q_rollout[next_state].max()
            q_rollout[state, action] += rollout_q_lr * (
                target - q_rollout[state, action]
            )
        if terminal:
            return total
        state = next_state
        gamma_t *= gamma
    return total


@njit(cache=True, fastmath=False)
def _simulate_iter(
    root_node_idx, root_state, theta_flat, max_depth,
    visit_count, total_value, action_visits, action_values, children,
    next_node_idx_arr,
    T_eff, B_pos, B_lik_all, lik_obs_card,
    reward_table, terminal_table, has_terminal,
    ucb_c, gamma, n_actions, n_likelihood,
    q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
):
    """One simulation: descend tree via UCB, expand at first leaf,
    rollout from leaf, back up the path."""
    path_node = np.empty(max_depth, dtype=np.int32)
    path_action = np.empty(max_depth, dtype=np.int32)
    path_reward = np.empty(max_depth, dtype=np.float64)
    path_len = 0

    node_idx = root_node_idx
    state = root_state
    rollout_value = 0.0
    expanded_leaf = False

    for d in range(max_depth):
        action = _ucb_select_action(
            visit_count, action_visits, action_values,
            node_idx, n_actions, ucb_c,
        )
        next_state, obs_idx, reward, terminal = _step_numba(
            state, theta_flat, action,
            T_eff, B_pos, B_lik_all, lik_obs_card,
            reward_table, terminal_table, has_terminal,
            n_likelihood,
        )
        path_node[path_len] = node_idx
        path_action[path_len] = action
        path_reward[path_len] = reward
        path_len += 1

        if terminal:
            break

        child = children[node_idx, action, obs_idx]
        if child < 0:
            # Allocate new child + rollout from leaf.
            new_idx = next_node_idx_arr[0]
            next_node_idx_arr[0] = new_idx + 1
            children[node_idx, action, obs_idx] = new_idx
            rollout_value = _rollout_numba(
                next_state, theta_flat, max_depth - d - 1,
                T_eff, B_pos, B_lik_all, lik_obs_card,
                reward_table, terminal_table, has_terminal,
                gamma, n_actions, n_likelihood,
                q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
            )
            expanded_leaf = True
            break

        node_idx = child
        state = next_state

    # Back up path.
    if expanded_leaf:
        value = rollout_value
    else:
        value = 0.0

    for i in range(path_len - 1, -1, -1):
        value = path_reward[i] + gamma * value
        ni = path_node[i]
        ai = path_action[i]
        visit_count[ni] += 1
        total_value[ni] += value
        action_visits[ni, ai] += 1
        action_values[ni, ai] += value


@njit(cache=True, fastmath=False)
def _run_bamcp_numba(
    n_simulations, max_depth, n_actions,
    theta_samples, root_states,
    T_eff, B_pos, B_lik_all, lik_obs_card,
    reward_table, terminal_table, has_terminal,
    ucb_c, gamma, n_likelihood, obs_stride_total,
    seed,
    q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
):
    """Top-level driver. Runs n_simulations rollouts, returns
    ``(visits, totals)`` at the root: per-action visit counts and
    summed returns, both float64[n_actions]. Callers compute Q(a) as
    totals/max(visits, 1) when they need per-action means.

    ``q_rollout`` is caller-owned and mutated in place (see
    ``_rollout_numba``); it is deliberately NOT allocated here beside the
    tree buffers, because it must outlive a single plan() call."""
    np.random.seed(seed)

    max_nodes = n_simulations + 1
    visit_count = np.zeros(max_nodes, dtype=np.int32)
    total_value = np.zeros(max_nodes, dtype=np.float64)
    action_visits = np.zeros((max_nodes, n_actions), dtype=np.int32)
    action_values = np.zeros((max_nodes, n_actions), dtype=np.float64)
    children = np.full(
        (max_nodes, n_actions, obs_stride_total), -1, dtype=np.int32,
    )

    next_node_idx_arr = np.zeros(1, dtype=np.int32)
    next_node_idx_arr[0] = 1  # root is node 0
    root_node_idx = 0

    for sim_idx in range(n_simulations):
        theta_flat = theta_samples[sim_idx]
        root_state = root_states[sim_idx]
        _simulate_iter(
            root_node_idx, root_state, theta_flat, max_depth,
            visit_count, total_value, action_visits, action_values, children,
            next_node_idx_arr,
            T_eff, B_pos, B_lik_all, lik_obs_card,
            reward_table, terminal_table, has_terminal,
            ucb_c, gamma, n_actions, n_likelihood,
            q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
        )

    visits = np.zeros(n_actions, dtype=np.float64)
    totals = np.zeros(n_actions, dtype=np.float64)
    for a in range(n_actions):
        visits[a] = float(action_visits[0, a])
        totals[a] = action_values[0, a]
    return visits, totals
