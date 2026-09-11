"""The preregistered 3 x 3 Relax intensity/frequency condition grid."""

from __future__ import annotations

from typing import Iterable


LEVEL_NAMES = ("Low", "Medium", "High")
INTENSITY_VALUES = (0.08, 0.16, 0.25)
FREQUENCY_VALUES = (0.12, 0.26, 0.41)
CONDITION_GRID = {
    f"C{intensity * 3 + frequency + 1}": (intensity, frequency)
    for intensity in range(3)
    for frequency in range(3)
}


def coordinates(condition: str) -> tuple[int, int]:
    try:
        return CONDITION_GRID[str(condition)]
    except KeyError as error:
        raise ValueError(f"Unknown Relax condition: {condition!r}") from error


def condition_at(intensity: int, frequency: int) -> str:
    if not (0 <= intensity <= 2 and 0 <= frequency <= 2):
        raise ValueError(f"Condition coordinates are outside the 3 x 3 grid: {(intensity, frequency)}")
    return f"C{intensity * 3 + frequency + 1}"


def adjacent_conditions(condition: str) -> tuple[str, ...]:
    intensity, frequency = coordinates(condition)
    candidates = []
    for delta_intensity, delta_frequency in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_intensity = intensity + delta_intensity
        next_frequency = frequency + delta_frequency
        if 0 <= next_intensity <= 2 and 0 <= next_frequency <= 2:
            candidates.append(condition_at(next_intensity, next_frequency))
    return tuple(sorted(candidates, key=lambda value: int(value[1:])))


def transition_axis(source: str, target: str) -> str | None:
    source_intensity, source_frequency = coordinates(source)
    target_intensity, target_frequency = coordinates(target)
    if source_intensity != target_intensity and source_frequency == target_frequency:
        return "intensity"
    if source_frequency != target_frequency and source_intensity == target_intensity:
        return "frequency"
    return None


def is_legal_transition(source: str, target: str) -> bool:
    source_coordinates = coordinates(source)
    target_coordinates = coordinates(target)
    return sum(abs(left - right) for left, right in zip(source_coordinates, target_coordinates, strict=True)) == 1


def action_name(source: str, target: str) -> str:
    if source == target:
        return "hold"
    if not is_legal_transition(source, target):
        raise ValueError(f"Illegal non-adjacent transition: {source} -> {target}")
    source_intensity, source_frequency = coordinates(source)
    target_intensity, target_frequency = coordinates(target)
    if source_intensity != target_intensity:
        direction = "increase" if target_intensity > source_intensity else "decrease"
        return f"intensity_{direction}"
    direction = "increase" if target_frequency > source_frequency else "decrease"
    return f"frequency_{direction}"


def load(condition: str) -> int:
    intensity, frequency = coordinates(condition)
    return intensity + frequency


def nearest_condition(source: str, candidates: Iterable[str]) -> str | None:
    source_intensity, source_frequency = coordinates(source)
    values = list(candidates)
    if not values:
        return None
    return min(
        values,
        key=lambda candidate: (
            abs(coordinates(candidate)[0] - source_intensity)
            + abs(coordinates(candidate)[1] - source_frequency),
            load(candidate),
            int(candidate[1:]),
        ),
    )


def one_step_toward(source: str, target: str) -> str:
    """Return one conservative adjacent step from source toward target.

    Intensity is reduced before frequency when both axes must be reduced. This
    never jumps a level and therefore keeps the global transition invariant.
    """

    if source == target:
        return source
    source_intensity, source_frequency = coordinates(source)
    target_intensity, target_frequency = coordinates(target)
    if source_intensity > target_intensity:
        return condition_at(source_intensity - 1, source_frequency)
    if source_frequency > target_frequency:
        return condition_at(source_intensity, source_frequency - 1)
    if source_frequency < target_frequency:
        return condition_at(source_intensity, source_frequency + 1)
    return condition_at(source_intensity + 1, source_frequency)


def level_record(condition: str) -> dict[str, object]:
    intensity, frequency = coordinates(condition)
    return {
        "intensity_index": intensity,
        "frequency_index": frequency,
        "intensity_level": LEVEL_NAMES[intensity],
        "frequency_level": LEVEL_NAMES[frequency],
        "intensity_value": INTENSITY_VALUES[intensity],
        "frequency_value": FREQUENCY_VALUES[frequency],
    }


__all__ = [
    "CONDITION_GRID",
    "FREQUENCY_VALUES",
    "INTENSITY_VALUES",
    "LEVEL_NAMES",
    "action_name",
    "adjacent_conditions",
    "condition_at",
    "coordinates",
    "is_legal_transition",
    "level_record",
    "load",
    "nearest_condition",
    "one_step_toward",
    "transition_axis",
]
