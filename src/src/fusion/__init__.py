"""Compatibility shim: ``src.fusion`` is now ``mac.fusion``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.fusion import *  # noqa: F401,F403
