"""Compatibility shim: ``src.adaptive`` is now ``mac.adaptive.offline``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.adaptive.offline import *  # noqa: F401,F403
