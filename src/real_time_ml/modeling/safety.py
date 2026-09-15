"""Compatibility shim: ``real_time_ml.modeling.safety`` is now ``mac.evaluation.safety``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.evaluation.safety as _target

sys.modules[__name__] = _target
