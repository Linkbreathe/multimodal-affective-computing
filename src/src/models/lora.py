"""Compatibility shim: ``src.models.lora`` is now ``mac.models.lora``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.models.lora as _target

sys.modules[__name__] = _target
