"""Compatibility shim: ``real_time_ml.experiments.dynamic_texture_five`` is now ``mac.experiments.dynamic_texture_five``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.experiments.dynamic_texture_five as _target

sys.modules[__name__] = _target
