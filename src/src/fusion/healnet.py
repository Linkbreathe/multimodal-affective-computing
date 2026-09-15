"""Compatibility shim: ``src.fusion.healnet`` is now ``mac.fusion.healnet``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.healnet as _target

sys.modules[__name__] = _target
