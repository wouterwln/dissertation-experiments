"""Planning via variational optimization."""

from .factorized_optimizer import (
    plan_actions_factorized,
    select_action_factorized,
    FactorizedPlanningConfig,
    FactorizedPlanningResult,
)

from .temporal_optimizer import (
    plan_actions_temporal,
    select_action_temporal,
    TemporalPlanningConfig,
    TemporalPlanningResult,
)

from .temporal_optimizer_meanfield import (
    plan_actions_meanfield,
    select_action_meanfield,
    MeanfieldPlanningConfig,
    MeanfieldPlanningResult,
)

from .sophisticated_planner import (
    SophisticatedPlanningConfig,
    SophisticatedPlanningResult,
    SophisticatedPlanner,
    convert_tmaze_tensors_to_pymdp,
)

__all__ = [
    # Factorized VFE planner (T-Maze)
    "plan_actions_factorized",
    "select_action_factorized",
    "FactorizedPlanningConfig",
    "FactorizedPlanningResult",
    # Temporal VFE planner (Epistemic Maze)
    "plan_actions_temporal",
    "select_action_temporal",
    "TemporalPlanningConfig",
    "TemporalPlanningResult",
    # Mean-field temporal planner (Epistemic Maze q_perp ablation)
    "plan_actions_meanfield",
    "select_action_meanfield",
    "MeanfieldPlanningConfig",
    "MeanfieldPlanningResult",
    # Sophisticated (pymdp) planner
    "SophisticatedPlanningConfig",
    "SophisticatedPlanningResult",
    "SophisticatedPlanner",
    "convert_tmaze_tensors_to_pymdp",
]
