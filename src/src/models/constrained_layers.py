"""Compatibility shim: ``src.models.constrained_layers`` is now ``mac.models.constrained_layers``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.models.constrained_layers as _target

sys.modules[__name__] = _target
