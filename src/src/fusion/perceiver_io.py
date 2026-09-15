"""Compatibility shim: ``src.fusion.perceiver_io`` is now ``mac.fusion.perceiver_io``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.perceiver_io as _target

sys.modules[__name__] = _target
