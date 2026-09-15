"""Compatibility shim: ``src.data.relax_foundation`` is now ``mac.data.relax_foundation``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.relax_foundation as _target

sys.modules[__name__] = _target
