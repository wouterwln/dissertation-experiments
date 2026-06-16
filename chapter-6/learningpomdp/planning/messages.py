"""Message passing primitives for discrete factor graphs.

Ported from MinigridMP-AIF-JAX/inference/messages.py (dense ops only).
All computation is in log-space.
"""

import jax.numpy as jnp

LOG_ZERO = -1e12
EPSILON = 1e-10


def safe_log(x):
    """Log that maps 0 -> LOG_ZERO instead of -inf."""
    return jnp.where(x > 0, jnp.log(jnp.maximum(x, 1e-30)), LOG_ZERO)


def safe_log_div(log_num, log_den):
    """log(num/den) where 0/0 and nonzero/0 -> LOG_ZERO."""
    valid = (log_num > LOG_ZERO / 2) & (log_den > LOG_ZERO / 2)
    return jnp.where(valid, log_num - log_den, LOG_ZERO)


