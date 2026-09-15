"""Compatibility shim: ``src.models.head_motion_1dcnn`` is now ``mac.models.head_motion_1dcnn``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.models.head_motion_1dcnn as _target

sys.modules[__name__] = _target
