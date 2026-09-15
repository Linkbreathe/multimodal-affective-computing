"""Compatibility shim: ``src.fusion.multimodal_lego`` is now ``mac.fusion.multimodal_lego``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.multimodal_lego as _target

sys.modules[__name__] = _target
