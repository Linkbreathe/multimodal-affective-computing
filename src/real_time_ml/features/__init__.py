"""Compatibility shim: ``real_time_ml.features`` is now ``mac.features``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.features import *  # noqa: F401,F403
