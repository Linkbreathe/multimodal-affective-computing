"""Compatibility shim: ``real_time_ml.preprocessing`` is now ``mac.preprocessing``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.preprocessing import *  # noqa: F401,F403
