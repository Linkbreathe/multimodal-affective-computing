"""Compatibility shim: ``src.data.dataset`` is now ``mac.data.dataset``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.dataset as _target

sys.modules[__name__] = _target
