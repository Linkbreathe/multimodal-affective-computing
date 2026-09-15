"""Runtime compatibility namespace.

The implementation remains in :mod:`real_time_ml.realtime` while integrations
migrate; this avoids changing the Unity Shadow protocol.
"""

from mac.realtime.engine import InferenceEngine
from mac.realtime.replay import replay
from mac.realtime.serve import serve

__all__ = ["InferenceEngine", "replay", "serve"]
