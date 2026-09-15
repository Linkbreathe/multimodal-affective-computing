"""Compatibility shim: ``real_time_ml.experiments`` is now ``mac.experiments``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.experiments import *  # noqa: F401,F403
