"""Compatibility shim: ``src.trainer.fusion_trainer`` is now ``mac.training.fusion_trainer``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.training.fusion_trainer as _target

sys.modules[__name__] = _target
