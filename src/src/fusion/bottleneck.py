"""Compatibility shim: ``src.fusion.bottleneck`` is now ``mac.fusion.bottleneck``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.bottleneck as _target

sys.modules[__name__] = _target
