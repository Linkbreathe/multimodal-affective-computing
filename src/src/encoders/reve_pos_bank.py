"""Compatibility shim: ``src.encoders.reve_pos_bank`` is now ``mac.encoders.reve_pos_bank``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.reve_pos_bank as _target

sys.modules[__name__] = _target
