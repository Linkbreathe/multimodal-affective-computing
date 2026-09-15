"""Compatibility shim: ``src.fusion.late`` is now ``mac.fusion.late``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.late as _target

sys.modules[__name__] = _target
