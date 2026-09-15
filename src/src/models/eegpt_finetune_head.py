"""Compatibility shim: ``src.models.eegpt_finetune_head`` is now ``mac.models.eegpt_finetune_head``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.models.eegpt_finetune_head as _target

sys.modules[__name__] = _target
