"""Compatibility shim: ``real_time_ml.adaptive_control`` is now ``mac.adaptive.control``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.adaptive.control import *  # noqa: F401,F403
