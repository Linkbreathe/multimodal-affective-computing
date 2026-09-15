"""Compatibility shim: ``src.tasks.heads`` is now ``mac.tasks.heads``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.tasks.heads as _target

sys.modules[__name__] = _target
