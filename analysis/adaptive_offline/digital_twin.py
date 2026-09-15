"""II.2 Digital twin / simulator built from the Part I offline models.

EVIDENCE LEVEL: C (simulation is NOT effectiveness evidence).  The twin is only
as trustworthy as its offline assumptions, and I.1/I.6 show the relaxation
response surface barely beats the mean -- this limitation propagates through the
whole simulator and is documented in ``environment_spec.md``.

Environment ingredients (each traced to a Part I source):
  - reward base surface         <- I.1 group relaxation cell means (0-1 scale)
  - safety / high_discomfort     <- I.3 per-condition high_discomfort rate
  - observation noise            <- I.6 within-condition residual SD
  - habituation drift            <- I.5 presentation-position slope
  - individual heterogeneity eta <- I.2 grid-averaged random-effect SD (CI)

The single knob ``eta`` is the SD (in relaxation 0-1 units) of the per-
participant random tilt of the reward surface, defined so that its grid-averaged
random-effect SD equals ``eta`` -- i.e. it is directly comparable to the I.2
heterogeneity estimate and its credible interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from io_common import CONDITIONS, condition_grid, condition_sort_key, load_labels

# grid neighbourhood on the 3x3 (intensity_index, frequency_index) lattice
_GRID = condition_grid().set_index("condition")
_POS = {c: (int(_GRID.loc[c, "intensity_index"]), int(_GRID.loc[c, "frequency_index"])) for c in CONDITIONS}
_POS_INV = {v: k for k, v in _POS.items()}
# grid-averaged sum of squared centred coordinates -> tilt rescaling constant
_CENTRED = np.array([[(i - 1) ** 2 + (f - 1) ** 2 for (i, f) in [_POS[c]]][0] for c in CONDITIONS], dtype=float)
_TILT_SCALE = float(np.sqrt(np.mean(_CENTRED)))  # ~1.155


def adjacent_conditions(condition: str, include_self: bool = True) -> list[str]:
    i, f = _POS[condition]
    out = []
    for di, df in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        if di == 0 and df == 0 and not include_self:
            continue
        ni, nf = i + di, f + df
        if 0 <= ni <= 2 and 0 <= nf <= 2:
            out.append(_POS_INV[(ni, nf)])
    return out


@dataclass
class TwinCalibration:
    reward_surface: dict[str, float]          # I.1 group relaxation by condition
    high_discomfort_rate: dict[str, float]    # I.3 per-condition risk
    obs_noise_sd: float                        # I.6 within-condition residual SD
    habituation_rate: float                    # I.5 per-step relaxation decay (>=0)
    eta_estimate: float                        # I.2 grid-avg random-effect SD
    eta_ci_low: float
    eta_ci_high: float

    @property
    def best_condition(self) -> str:
        return max(self.reward_surface, key=self.reward_surface.get)


def calibrate_from_data(eta_estimate: float = 0.10, eta_ci=(0.06, 0.13)) -> TwinCalibration:
    labels = load_labels()
    reward = labels.groupby("condition")["relaxation"].mean().reindex(CONDITIONS).to_dict()
    risk = labels.groupby("condition")["high_discomfort"].mean().reindex(CONDITIONS).to_dict()
    resid = labels["relaxation"] - labels["condition"].map(labels.groupby("condition")["relaxation"].mean())
    obs_noise = float(resid.std(ddof=1))
    # I.5 within-participant relaxation drift vs presentation position
    lab = labels.copy()
    lab["pos_c"] = lab["presentation_position"] - lab.groupby("participant_id")["presentation_position"].transform("mean")
    lab["rel_c"] = lab["relaxation"] - lab.groupby("participant_id")["relaxation"].transform("mean")
    slope = float(np.polyfit(lab["pos_c"], lab["rel_c"], 1)[0])
    habituation = max(0.0, -slope)  # relaxation decays as dwell/position grows
    return TwinCalibration(
        reward_surface={k: float(v) for k, v in reward.items()},
        high_discomfort_rate={k: float(v) for k, v in risk.items()},
        obs_noise_sd=obs_noise,
        habituation_rate=habituation,
        eta_estimate=eta_estimate,
        eta_ci_low=eta_ci[0],
        eta_ci_high=eta_ci[1],
    )


@dataclass
class DigitalTwin:
    calib: TwinCalibration
    eta: float = 0.10
    tau_safe: float = 0.20                 # safety constraint on P(high_discomfort)
    habituation_recovery: bool = True       # changing stimulus resets dwell (dishabituation)
    obs_noise_scale: float = 1.0
    habituation_scale: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def new_participant(self) -> dict:
        """Draw an individual reward surface = group + eta-scaled random tilt."""
        a = self.rng.normal(0.0, 1.0)
        b = self.rng.normal(0.0, 1.0)
        # rescale so the grid-averaged SD of the tilt equals eta
        scale = self.eta / _TILT_SCALE if _TILT_SCALE > 0 else 0.0
        surface = {}
        for c in CONDITIONS:
            i, f = _POS[c]
            tilt = scale * (a * (i - 1) + b * (f - 1))
            surface[c] = float(np.clip(self.calib.reward_surface[c] + tilt, 0.0, 1.0))
        return {"surface": surface, "dwell": {c: 0 for c in CONDITIONS}}

    def individual_optimum(self, participant: dict) -> str:
        return max(participant["surface"], key=participant["surface"].get)

    def true_reward(self, participant: dict, condition: str, dwell: int) -> float:
        base = participant["surface"][condition]
        decay = self.calib.habituation_rate * self.habituation_scale * max(0, dwell)
        return float(np.clip(base - decay, 0.0, 1.0))

    def monotony(self, dwell: int) -> float:
        return float(np.clip(self.calib.habituation_rate * self.habituation_scale * dwell * 3.0, 0.0, 1.0))

    def observe_reward(self, true_reward: float) -> float:
        noise = self.rng.normal(0.0, self.calib.obs_noise_sd * self.obs_noise_scale)
        return float(true_reward + noise)

    def risk(self, condition: str) -> float:
        return self.calib.high_discomfort_rate[condition]

    def sample_high_discomfort(self, condition: str) -> int:
        return int(self.rng.random() < self.risk(condition))

    def safe_feasible(self, condition: str) -> list[str]:
        feasible = [c for c in adjacent_conditions(condition, include_self=True)
                    if self.risk(c) <= self.tau_safe]
        # the global safe fallback (lowest-risk reachable cell) is always retained
        if not feasible:
            feasible = [min(adjacent_conditions(condition, include_self=True), key=self.risk)]
        return feasible
