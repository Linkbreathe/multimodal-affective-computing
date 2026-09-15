"""Compatibility shim: ``src.fusion.frozen_compression`` is now ``mac.fusion.frozen_compression``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.frozen_compression as _target

sys.modules[__name__] = _target
