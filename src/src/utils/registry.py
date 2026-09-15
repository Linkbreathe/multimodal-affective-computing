"""Compatibility shim: ``src.utils.registry`` is now ``mac.reporting.registry``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.reporting.registry as _target

sys.modules[__name__] = _target
