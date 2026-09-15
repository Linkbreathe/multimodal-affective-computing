"""Compatibility shim: ``src.data.egoemotion.manifest`` is now ``mac.data.egoemotion.manifest``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.egoemotion.manifest as _target

sys.modules[__name__] = _target
