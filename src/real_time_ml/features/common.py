"""Compatibility shim: ``real_time_ml.features.common`` is now ``mac.features.common``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.features.common as _target

sys.modules[__name__] = _target
