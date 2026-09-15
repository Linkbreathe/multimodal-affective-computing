"""Compatibility shim: ``src.data.relax_attention_video`` is now ``mac.data.relax_attention_video``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.relax_attention_video as _target

sys.modules[__name__] = _target
