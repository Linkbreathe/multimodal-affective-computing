"""Compatibility shim: ``real_time_ml.features.physio`` is now ``mac.features.physio``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.features.physio as _target

sys.modules[__name__] = _target
