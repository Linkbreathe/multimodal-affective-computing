"""Compatibility shim: ``real_time_ml.evaluation`` is now ``mac.evaluation``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.evaluation import *  # noqa: F401,F403
