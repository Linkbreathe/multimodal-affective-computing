"""Runtime compatibility namespace.

The implementation remains in :mod:`real_time_ml.realtime` while integrations
migrate; this avoids changing the Unity Shadow protocol.
"""

from real_time_ml.realtime.engine import InferenceEngine
from real_time_ml.realtime.replay import replay
from real_time_ml.realtime.serve import serve

__all__ = ["InferenceEngine", "replay", "serve"]
