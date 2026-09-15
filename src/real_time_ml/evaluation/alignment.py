"""Compatibility shim: ``real_time_ml.evaluation.alignment`` is now ``mac.evaluation.alignment``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.evaluation.alignment as _target

sys.modules[__name__] = _target
