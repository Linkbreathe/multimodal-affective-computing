"""Compatibility shim: ``src.fusion.tmc`` is now ``mac.fusion.tmc``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.tmc as _target

sys.modules[__name__] = _target
