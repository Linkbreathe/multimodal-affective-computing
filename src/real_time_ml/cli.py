"""Compatibility shim: ``real_time_ml.cli`` is now ``mac.cli``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.cli as _target

sys.modules[__name__] = _target
