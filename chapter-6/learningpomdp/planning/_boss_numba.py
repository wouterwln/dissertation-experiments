"""Numba-JIT'd inner loop for BOSS — Best-of-Sampled-Set merged-MDP MCTS.

Mirrors ``_bamcp_numba`` but the action space is augmented to
``(action, k_sample)`` pairs. Tree node action stats carry an extra
K-axis. Observation sampling reuses the same B_lik_all table indexed
by the per-k θ̃.

See ``_bamcp_numba`` module docstring for the data-layout rationale
(dense child lookup, packed obs_idx, padded B_lik_all, reward table).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _sample_categorical(p, n):
    u = np.random.random()
    cumsum = 0.0
    for i in range(n):
        cumsum += p[i]
        if u < cumsum:
            return i
    return n - 1


@njit(cache=True, fastmath=False)
def _argmax_int(x):
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

    Rollout-only, mirroring ``_bamcp_numba._argmax_random_tiebreak``:
    lowest-index tie-breaking pins the eps-greedy rollout to action 0 under
    a flat Q-hat. No RNG draw unless there is an actual tie.
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
def _step_numba_boss(
    state, theta_flat, action,
    T_eff, B_pos, B_lik_all, lik_obs_card,
    reward_table, terminal_table, has_terminal,
    n_likelihood,
):
    next_state_dist = T_eff[:, state, action]
    n_states = next_state_dist.shape[0]
    next_state = _sample_categorical(next_state_dist, n_states)

    view_idx = _argmax_int(B_pos[:, next_state])

    obs_idx = view_idx
    for c in range(n_likelihood):
        card = lik_obs_card[c]
        u = np.random.random()
        cumsum = 0.0
        sample = card - 1
        for o in range(card):
            cumsum += B_lik_all[theta_flat, c, o, next_state]
            if u < cumsum:
                sample = o
                break
        obs_idx = obs_idx * card + sample

    reward = reward_table[state, action, theta_flat]
    if has_terminal:
        terminal = terminal_table[state, action, theta_flat]
    else:
        terminal = False
    return next_state, obs_idx, reward, terminal


@njit(cache=True, fastmath=False)
def _ucb_select_ak(
    visit_count, action_visits, action_values,
    node_idx, n_actions, K_samples, ucb_c,
):
    """Uniform random over unvisited (a, k); else UCB1-best over the
    augmented (action, k_sample) action space.

    Mirrors NumPy BOSS's `self._rng.choice(unvisited)` tie-break.
    """
    n_unvisited = 0
    for a in range(n_actions):
        for k in range(K_samples):
            if action_visits[node_idx, a, k] == 0:
                n_unvisited += 1
    if n_unvisited > 0:
        pick = np.random.randint(0, n_unvisited)
        seen = 0
        for a in range(n_actions):
            for k in range(K_samples):
                if action_visits[node_idx, a, k] == 0:
                    if seen == pick:
                        return a, k
                    seen += 1
    parent_visits = visit_count[node_idx]
    log_pv = np.log(max(parent_visits, 1))
    best_a = 0
    best_k = 0
    best_score = -1e30
    for a in range(n_actions):
        for k in range(K_samples):
            n_ak = action_visits[node_idx, a, k]
            mean_v = action_values[node_idx, a, k] / n_ak
            score = mean_v + ucb_c * np.sqrt(log_pv / n_ak)
            if score > best_score:
                best_score = score
                best_a = a
                best_k = k
    return best_a, best_k


@njit(cache=True, fastmath=False)
def _rollout_numba_boss(
    state, depth,
    thetas, K_samples,
    T_eff, B_pos, B_lik_all, lik_obs_card,
    reward_table, terminal_table, has_terminal,
    gamma, n_actions, n_likelihood,
    q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
):
    """Leaf rollout: per step pick (action, k).

    ``use_q_rollout=False`` keeps the original uniform (action, k) draw
    bit-for-bit. ``use_q_rollout=True`` makes the *real action*
    epsilon-greedy in ``q_rollout`` (updated off-policy from the step,
    ties broken uniformly at random); ``k_idx`` stays uniform because that
    draw is BOSS's merged-action-space mechanism, not the rollout policy.
    """
    total = 0.0
    gamma_t = 1.0
    for _ in range(depth):
        if use_q_rollout and np.random.random() >= rollout_epsilon:
            action = _argmax_random_tiebreak(q_rollout[state])
        else:
            action = np.random.randint(0, n_actions)
        k_idx = np.random.randint(0, K_samples)
        theta_flat = thetas[k_idx]
        next_state, _, reward, terminal = _step_numba_boss(
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
def _simulate_iter_boss(
    root_node_idx, root_state, max_depth,
    thetas, K_samples,
    visit_count, total_value, action_visits, action_values, children,
    next_node_idx_arr,
    T_eff, B_pos, B_lik_all, lik_obs_card,
    reward_table, terminal_table, has_terminal,
    ucb_c, gamma, n_actions, n_likelihood,
    q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
):
    path_node = np.empty(max_depth, dtype=np.int32)
    path_action = np.empty(max_depth, dtype=np.int32)
    path_k = np.empty(max_depth, dtype=np.int32)
    path_reward = np.empty(max_depth, dtype=np.float64)
    path_len = 0

    node_idx = root_node_idx
    state = root_state
    rollout_value = 0.0
    expanded_leaf = False

    for d in range(max_depth):
        action, k_idx = _ucb_select_ak(
            visit_count, action_visits, action_values,
            node_idx, n_actions, K_samples, ucb_c,
        )
        theta_flat = thetas[k_idx]
        next_state, obs_idx, reward, terminal = _step_numba_boss(
            state, theta_flat, action,
            T_eff, B_pos, B_lik_all, lik_obs_card,
            reward_table, terminal_table, has_terminal,
            n_likelihood,
        )
        path_node[path_len] = node_idx
        path_action[path_len] = action
        path_k[path_len] = k_idx
        path_reward[path_len] = reward
        path_len += 1

        if terminal:
            break

        # Children are keyed by (action, obs_idx), NOT (action, k, obs_idx).
        # k_sample is internal to the planner — different k choices that
        # produce the same observation share children. This is BOSS's
        # explicit design (per the merged-MDP construction).
        child = children[node_idx, action, obs_idx]
        if child < 0:
            new_idx = next_node_idx_arr[0]
            next_node_idx_arr[0] = new_idx + 1
            children[node_idx, action, obs_idx] = new_idx
            rollout_value = _rollout_numba_boss(
                next_state, max_depth - d - 1,
                thetas, K_samples,
                T_eff, B_pos, B_lik_all, lik_obs_card,
                reward_table, terminal_table, has_terminal,
                gamma, n_actions, n_likelihood,
                q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
            )
            expanded_leaf = True
            break

        node_idx = child
        state = next_state

    if expanded_leaf:
        value = rollout_value
    else:
        value = 0.0

    for i in range(path_len - 1, -1, -1):
        value = path_reward[i] + gamma * value
        ni = path_node[i]
        ai = path_action[i]
        ki = path_k[i]
        visit_count[ni] += 1
        total_value[ni] += value
        action_visits[ni, ai, ki] += 1
        action_values[ni, ai, ki] += value


@njit(cache=True, fastmath=False)
def _run_boss_numba(
    n_simulations, max_depth, n_actions,
    thetas, K_samples, root_states,
    T_eff, B_pos, B_lik_all, lik_obs_card,
    reward_table, terminal_table, has_terminal,
    ucb_c, gamma, n_likelihood, obs_stride_total,
    seed,
    q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
):
    """Top-level driver. Returns root visit counts marginalised over k:
    visits[a] = sum_k action_visits[root, a, k].

    ``q_rollout`` is caller-owned and mutated in place so it survives
    across plan() calls within a theta_true block."""
    np.random.seed(seed)

    max_nodes = n_simulations + 1
    visit_count = np.zeros(max_nodes, dtype=np.int32)
    total_value = np.zeros(max_nodes, dtype=np.float64)
    action_visits = np.zeros((max_nodes, n_actions, K_samples), dtype=np.int32)
    action_values = np.zeros((max_nodes, n_actions, K_samples), dtype=np.float64)
    children = np.full(
        (max_nodes, n_actions, obs_stride_total), -1, dtype=np.int32,
    )

    next_node_idx_arr = np.zeros(1, dtype=np.int32)
    next_node_idx_arr[0] = 1
    root_node_idx = 0

    for sim_idx in range(n_simulations):
        root_state = root_states[sim_idx]
        _simulate_iter_boss(
            root_node_idx, root_state, max_depth,
            thetas, K_samples,
            visit_count, total_value, action_visits, action_values, children,
            next_node_idx_arr,
            T_eff, B_pos, B_lik_all, lik_obs_card,
            reward_table, terminal_table, has_terminal,
            ucb_c, gamma, n_actions, n_likelihood,
            q_rollout, use_q_rollout, rollout_epsilon, rollout_q_lr,
        )

    # Marginalise over k at the root.
    visits = np.zeros(n_actions, dtype=np.float64)
    for a in range(n_actions):
        for k in range(K_samples):
            visits[a] += float(action_visits[0, a, k])
    return visits
