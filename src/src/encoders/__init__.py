"""Compatibility shim: ``src.encoders`` is now ``mac.encoders``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.encoders import *  # noqa: F401,F403
