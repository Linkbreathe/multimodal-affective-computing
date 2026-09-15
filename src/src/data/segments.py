"""Compatibility shim: ``src.data.segments`` is now ``mac.data.segments``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.segments as _target

sys.modules[__name__] = _target
