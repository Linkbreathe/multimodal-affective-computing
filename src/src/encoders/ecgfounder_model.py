"""Compatibility shim: ``src.encoders.ecgfounder_model`` is now ``mac.encoders.ecgfounder_model``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.ecgfounder_model as _target

sys.modules[__name__] = _target
