"""Compatibility shim: ``src.utils.config`` is now ``mac.config.simple``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.config.simple as _target

sys.modules[__name__] = _target
