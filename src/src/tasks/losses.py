"""Compatibility shim: ``src.tasks.losses`` is now ``mac.tasks.losses``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.tasks.losses as _target

sys.modules[__name__] = _target
