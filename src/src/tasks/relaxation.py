"""Compatibility shim: ``src.tasks.relaxation`` is now ``mac.tasks.relaxation``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.tasks.relaxation as _target

sys.modules[__name__] = _target
