"""Planning module — inference-based action selection."""

from .aif_planner import AIFPlanner
from .belief_state_q import BeliefStateQLearner
from .goal import Goal, GoalType
from .ids_planner import IDSPlanner
from .oracle_planner import OraclePlanner
from .psrl_planner import PSRLPlanner
from .vbp_planner import VBPPlanner

# Planners that internally execute JAX compute (potentially on GPU).
# The rock_sample runner forces workers=1 for these to avoid GPU OOM
# when spawning subprocess workers — JAX would re-init the GPU per
# worker. CPU-only planners (BAMCP/BOSS/Random/IDS) are safe to
# parallelise. IDS wraps BAMCP for its Q oracle, so its JAX touch is
# limited to converting the final action distribution.
GPU_PLANNER_TYPES = (AIFPlanner, VBPPlanner, PSRLPlanner, OraclePlanner)

__all__ = [
    "AIFPlanner",
    "BeliefStateQLearner",
    "GPU_PLANNER_TYPES",
    "Goal",
    "GoalType",
    "IDSPlanner",
    "OraclePlanner",
    "PSRLPlanner",
    "VBPPlanner",
]
