"""Compatibility shim: ``src.data.collate`` is now ``mac.data.collate``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.collate as _target

sys.modules[__name__] = _target
