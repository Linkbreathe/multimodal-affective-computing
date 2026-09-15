"""Compatibility shim: ``real_time_ml.modeling`` is now ``mac.models``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.models import *  # noqa: F401,F403
