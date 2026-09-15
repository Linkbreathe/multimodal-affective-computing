"""Causal inference and shadow-replay utilities for the Relax experiment."""

from src.adaptive.condition_grid import CONDITION_GRID, adjacent_conditions, is_legal_transition
from src.adaptive.controller import AdaptiveController, ControllerConfig

__all__ = [
    "CONDITION_GRID",
    "AdaptiveController",
    "ControllerConfig",
    "adjacent_conditions",
    "is_legal_transition",
]
