"""Compatibility shim: ``src.encoders.papagei`` is now ``mac.encoders.papagei``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.papagei as _target

sys.modules[__name__] = _target
