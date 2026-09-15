"""Compatibility shim: ``src.data.relax_physio_preprocessing`` is now ``mac.preprocessing.relax_physio``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.preprocessing.relax_physio as _target

sys.modules[__name__] = _target
