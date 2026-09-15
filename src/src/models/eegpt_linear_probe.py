"""Compatibility shim: ``src.models.eegpt_linear_probe`` is now ``mac.models.eegpt_linear_probe``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.models.eegpt_linear_probe as _target

sys.modules[__name__] = _target
