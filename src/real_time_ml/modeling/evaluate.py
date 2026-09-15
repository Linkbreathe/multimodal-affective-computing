"""Compatibility shim: ``real_time_ml.modeling.evaluate`` is now ``mac.evaluation.evaluate``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.evaluation.evaluate as _target

sys.modules[__name__] = _target
