"""Compatibility shim: ``src.data.relax_dataset`` is now ``mac.data.relax_dataset``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.relax_dataset as _target

sys.modules[__name__] = _target
