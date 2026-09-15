"""Compatibility shim: ``src.encoders.video_mae`` is now ``mac.encoders.video_mae``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.video_mae as _target

sys.modules[__name__] = _target
