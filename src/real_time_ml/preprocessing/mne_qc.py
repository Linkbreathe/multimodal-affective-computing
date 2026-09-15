"""Compatibility shim: ``real_time_ml.preprocessing.mne_qc`` is now ``mac.preprocessing.mne_qc``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.preprocessing.mne_qc as _target

sys.modules[__name__] = _target
