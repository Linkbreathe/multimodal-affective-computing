"""Compatibility shim: ``src.fusion.distill_late`` is now ``mac.fusion.distill_late``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.distill_late as _target

sys.modules[__name__] = _target
