"""Compatibility shim: ``src.encoders.eegpt_model`` is now ``mac.encoders.eegpt_model``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.eegpt_model as _target

sys.modules[__name__] = _target
