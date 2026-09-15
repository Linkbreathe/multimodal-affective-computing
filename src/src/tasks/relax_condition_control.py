"""Compatibility shim: ``src.tasks.relax_condition_control`` is now ``mac.tasks.relax_condition_control``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.tasks.relax_condition_control as _target

sys.modules[__name__] = _target
