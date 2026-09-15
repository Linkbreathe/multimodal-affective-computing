"""Compatibility shim: ``real_time_ml.modeling.condition_models`` is now ``mac.models.condition_models``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.models.condition_models as _target

sys.modules[__name__] = _target
