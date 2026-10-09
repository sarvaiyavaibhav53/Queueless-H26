"""
PRAVAAH — Centralized Queue Stability Classification & Thresholds.

Defines centralized, configurable traffic intensity stability bands:
- STABLE:   rho < 0.85
- WATCH:    0.85 <= rho < 1.0
- UNSTABLE: rho >= 1.0

IMPORTANT SYNTHETIC PROJECT ASSUMPTIONS:
These thresholds are synthetic engineering heuristics created for simulation
benchmarking in PS-02. They are NOT empirical or statutory government standards.
"""

from enum import Enum
from typing import NamedTuple


class StabilityState(str, Enum):
    """Categorical stability state for a service queue."""
    STABLE = "STABLE"
    WATCH = "WATCH"
    UNSTABLE = "UNSTABLE"


class StabilityThresholds(NamedTuple):
    """
    Configurable boundary thresholds for traffic intensity rho = lambda / mu.
    """
    stable_limit: float = 0.85
    watch_limit: float = 1.0


DEFAULT_STABILITY_THRESHOLDS = StabilityThresholds(stable_limit=0.85, watch_limit=1.0)


def classify_stability(
    rho: float,
    thresholds: StabilityThresholds = DEFAULT_STABILITY_THRESHOLDS,
) -> StabilityState:
    """
    Classifies traffic intensity rho into STABLE, WATCH, or UNSTABLE.

    Boundary semantics:
    - rho < stable_limit (0.85) -> STABLE
    - stable_limit (0.85) <= rho < watch_limit (1.0) -> WATCH
    - rho >= watch_limit (1.0) -> UNSTABLE
    """
    if rho < thresholds.stable_limit:
        return StabilityState.STABLE
    elif rho < thresholds.watch_limit:
        return StabilityState.WATCH
    else:
        return StabilityState.UNSTABLE


def is_stable(
    rho: float,
    thresholds: StabilityThresholds = DEFAULT_STABILITY_THRESHOLDS,
) -> bool:
    """Returns True if the traffic intensity represents a STABLE queue (rho < 0.85)."""
    return classify_stability(rho, thresholds) == StabilityState.STABLE
