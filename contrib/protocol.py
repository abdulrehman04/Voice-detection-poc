"""
WebSocket protocol messages for POC server.

These are for the testing WebSocket server only.
Production integrations should use the core events API directly.
"""
import json
from dataclasses import dataclass, asdict
from enum import Enum


class MsgType(str, Enum):
    """WebSocket message types."""
    HELLO = "hello"
    ENROLL = "enroll"
    END_ENROLL = "end_enroll"
    STATS = "stats"
    ENROLL_COMPLETE = "enroll_complete"
    SEGMENT_COMPLETE = "segment_complete"
    ERROR = "error"


@dataclass
class StatsMsg:
    """Real-time statistics message."""
    vad_prob: float
    similarity: float
    state: str
    vad_only: bool = False
    # Adaptive silence timeout info
    timeout: float = 0.0         # Current adaptive timeout (seconds)
    timeout_min: float = 0.25    # Configured minimum
    timeout_max: float = 0.8     # Configured maximum
    
    def json(self) -> str:
        return json.dumps({"type": MsgType.STATS.value, **asdict(self)})


@dataclass
class EnrollCompleteMsg:
    """Enrollment completion message."""
    profile_id: str
    score: float
    
    def json(self) -> str:
        return json.dumps({"type": MsgType.ENROLL_COMPLETE.value, **asdict(self)})


@dataclass
class SegmentCompleteMsg:
    """Verified segment completion message."""
    timestamp: int          # Current timestamp (ms)
    duration: float         # Segment duration (seconds)
    samples: int            # Number of audio samples
    vad_started_at: int     # When VAD detected speech (ms)
    user_confirmed_at: int  # When user confirmed (ms), 0 if auto
    speech_ended_at: int    # When speech ended (ms)
    processing_delay_ms: int  # Delay from speech end to complete
    chunk_size_kb: float    # Audio chunk size in KB
    
    def json(self) -> str:
        return json.dumps({"type": MsgType.SEGMENT_COMPLETE.value, **asdict(self)})


@dataclass
class ErrorMsg:
    """Error message."""
    message: str
    
    def json(self) -> str:
        return json.dumps({"type": MsgType.ERROR.value, "message": self.message})


def parse_message(data: str) -> tuple[MsgType, dict]:
    """Parse incoming JSON message."""
    d = json.loads(data)
    return MsgType(d.get("type", "")), d
