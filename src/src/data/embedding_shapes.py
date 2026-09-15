"""Compatibility shim: ``src.data.embedding_shapes`` is now ``mac.data.embedding_shapes``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.embedding_shapes as _target

sys.modules[__name__] = _target
