"""Compatibility shim: ``src.fusion.factory`` is now ``mac.fusion.factory``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.factory as _target

sys.modules[__name__] = _target
