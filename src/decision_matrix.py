import numpy as np
from typing import Dict, Any, Tuple
from enum import Enum

class CognitivePattern(Enum):
    P1_GROUNDED = "P1_GROUNDED"               # Healthy progressive deduction
    P2_WATCHLIST = "P2_WATCHLIST"             # Fuzzy hesitation / slight curvature
    P3_DEADLOCK = "P3_DEADLOCK"               # Circular trap / periodic orbit
    P4_SLIP = "P4_SLIP"                       # Fact/calculation slip (smooth manifold)
    P5_DELUSION = "P5_DELUSION"               # Isolated confabulation / weak anchor
    P6_COLLAPSE = "P6_COLLAPSE"               # Semantic collapse / Catatonic lock
    P7_CREATIVE_LEAP = "P7_CREATIVE_LEAP"     # Abrupt trajectory jump / paradigm shift

class DecisionAction(Enum):
    PASS_THROUGH = "PASS_THROUGH"             # Complete silence / zero intervention
    MINIMAL_ANCHOR = "MINIMAL_ANCHOR"         # v2 prompt injection (attitude recovery)
    ABNORMAL_TERMINATE = "ABNORMAL_TERMINATE" # Hard kill with error message
    OBSERVE = "OBSERVE"                       # Buffer and monitor

def diagnose_cognitive_state(
    step_idx: int,
    density: float,
    max_life: float,
    shadow: Dict[str, Any],
    collapse_streak: int = 0,
    is_deadlock_confirmed: bool = False
) -> Tuple[CognitivePattern, DecisionAction, str]:
    """
    Kouta Decision Matrix (P1-P7 deterministic diagnosis).
    Takes topological metrics, intermediate forked telemetry, and collapse streak.
    """
    rg = shadow.get("rg", 0.0)
    vel = shadow.get("mean_velocity", 0.0)
    term_vel = shadow.get("terminal_velocity", 0.0)
    term_accel = shadow.get("terminal_accel", 0.0)
    lyap = shadow.get("lyapunov_max", 0.0)

    # Warm-up condition: need at least 4 steps in buffer before diagnosing collapse or deadlocks
    if step_idx < 4:
        return (
            CognitivePattern.P1_GROUNDED,
            DecisionAction.PASS_THROUGH,
            "[DTC Warm-up] Buffering initial trajectory"
        )

    # 1. P6: Semantic Collapse / Catatonic Lock / Heat Death
    # Detect stalled step (terminal velocity near zero)
    is_stalled = bool(term_vel < 0.08)
    if is_stalled:
        # Require hysteresis: at least 2 consecutive stalled steps (or 1 stalled step with heavy spatial collapse)
        if collapse_streak >= 1 or (term_vel < 0.05 and rg < 0.40):
            return (
                CognitivePattern.P6_COLLAPSE,
                DecisionAction.ABNORMAL_TERMINATE,
                "abnormal termination: abnormal termination of thought for DTC"
            )
        else:
            # First stall: do not kill immediately, place on high alert (Watchlist)
            return (
                CognitivePattern.P2_WATCHLIST,
                DecisionAction.OBSERVE,
                "[DTC Observe] Potential semantic stall detected (streak=1)"
            )

    # 2. P3: Deadlock Loop (True Topological 1-Cycle)
    # Trigger ONLY when the core coprocessor lookahead counter confirms deadlock (2 consecutive cycles)
    # AND velocity / space confirms trapped trajectory (or large cycle persistence)
    if is_deadlock_confirmed and (vel < 0.95 or lyap <= 0.05 or max_life >= 0.15):
        return (
            CognitivePattern.P3_DEADLOCK,
            DecisionAction.MINIMAL_ANCHOR,
            "[DTC Action] Inject minimal anchor for trajectory attitude recovery"
        )

    # 3. P7: Creative Leap / Exploratory Shift
    # Low cycle persistence, high step velocity, expanding Rg, positive Lyapunov
    if density < 0.08 and vel >= 1.15 and rg >= 0.75 and lyap >= 0.10:
        return (
            CognitivePattern.P7_CREATIVE_LEAP,
            DecisionAction.PASS_THROUGH,
            "[DTC Non-Intervention] Creative leap protected (high kinetic energy)"
        )

    # 4. P2: Watchlist / Fuzzy Hesitation
    # Intermediate density with slowing velocity, or hesitation Rg with low velocity
    if ((0.04 <= density < 0.10) and vel < 0.90) or shadow.get("is_hesitation", False):
        return (
            CognitivePattern.P2_WATCHLIST,
            DecisionAction.OBSERVE,
            "[DTC Observe] Foot-dragging / curvature on watchlist"
        )

    # 5. P1: Grounded (Healthy Deduction)
    # Default progressive flow
    return (
        CognitivePattern.P1_GROUNDED,
        DecisionAction.PASS_THROUGH,
        "[DTC Pass] Normal deductive flow"
    )
