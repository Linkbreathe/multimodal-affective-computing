"""Compatibility shim: ``src.fusion.frozen_compression_v2`` is now ``mac.fusion.frozen_compression_v2``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.frozen_compression_v2 as _target

sys.modules[__name__] = _target
