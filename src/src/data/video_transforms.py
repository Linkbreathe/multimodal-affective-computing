"""Compatibility shim: ``src.data.video_transforms`` is now ``mac.data.video_transforms``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.video_transforms as _target

sys.modules[__name__] = _target
