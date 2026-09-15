"""Compatibility shim: ``src.trainer.early_stopping`` is now ``mac.training.early_stopping``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.training.early_stopping as _target

sys.modules[__name__] = _target
