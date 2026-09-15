"""Research-only experiment entry points isolated from runtime training."""

from mac.fusion.minimal_fusion import benchmark_minimal_fusion
from mac.fusion.minimal_fusion_dcnn import benchmark_minimal_fusion_dcnn
from mac.fusion.minimal_fusion_dcnn_hp import analyze_minimal_fusion_dcnn_hp
from mac.experiments.dynamic_texture_five import (
    run_classical_dynamic_texture,
    run_paired_dcnn_dynamic_texture,
)

__all__ = [
    "analyze_minimal_fusion_dcnn_hp",
    "benchmark_minimal_fusion",
    "benchmark_minimal_fusion_dcnn",
    "run_classical_dynamic_texture",
    "run_paired_dcnn_dynamic_texture",
]
