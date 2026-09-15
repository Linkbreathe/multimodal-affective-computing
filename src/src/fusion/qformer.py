"""Compatibility shim: ``src.fusion.qformer`` is now ``mac.fusion.qformer``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.qformer as _target

sys.modules[__name__] = _target
