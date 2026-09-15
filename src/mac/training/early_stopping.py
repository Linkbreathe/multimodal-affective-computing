"""Early stopping based on validation metric."""
from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )
        if improved:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
