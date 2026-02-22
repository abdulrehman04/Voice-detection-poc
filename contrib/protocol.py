"""WebSocket protocol messages for POC server."""
import json
from dataclasses import dataclass, asdict
from enum import Enum


class MsgType(str, Enum):
    HELLO = "hello"
    ENROLL = "enroll"
    END_ENROLL = "end_enroll"
    CLEAR_PROFILE = "clear_profile"
    STATS = "stats"
    ENROLL_COMPLETE = "enroll_complete"
    SEGMENT_COMPLETE = "segment_complete"
    PROFILE_INFO = "profile_info"
    PROFILE_UPDATE = "profile_update"
    ERROR = "error"


@dataclass
class StatsMsg:
    vad_prob: float
    similarity: float
    state: str
    vad_only: bool = False
    timeout: float = 0.0
    timeout_min: float = 0.25
    timeout_max: float = 0.8
    
    def json(self) -> str:
        return json.dumps({"type": MsgType.STATS.value, **asdict(self)})


@dataclass
class EnrollCompleteMsg:
    profile_id: str
    score: float
    
    def json(self) -> str:
        return json.dumps({"type": MsgType.ENROLL_COMPLETE.value, **asdict(self)})


@dataclass
class SegmentCompleteMsg:
    timestamp: int
    duration: float
    samples: int
    vad_started_at: int
    user_confirmed_at: int
    speech_ended_at: int
    processing_delay_ms: int
    chunk_size_kb: float
    best_session: int = 0
    total_sessions: int = 0
    similarity: float = 0.0
    threshold_used: float = 0.0

    def json(self) -> str:
        return json.dumps({"type": MsgType.SEGMENT_COMPLETE.value, **asdict(self)})


@dataclass
class ProfileInfoMsg:
    profile_id: str
    sessions: int
    session_thresholds: list
    overall_threshold: float
    adaptive_progress: float = 0.0  # 0-1, how close to auto-enrolling

    def json(self) -> str:
        return json.dumps({"type": MsgType.PROFILE_INFO.value, **asdict(self)})


@dataclass
class ProfileUpdateMsg:
    profile_id: str
    sessions: int
    session_thresholds: list
    overall_threshold: float
    reason: str = "adaptive"

    def json(self) -> str:
        return json.dumps({"type": MsgType.PROFILE_UPDATE.value, **asdict(self)})


@dataclass
class ErrorMsg:
    message: str
    
    def json(self) -> str:
        return json.dumps({"type": MsgType.ERROR.value, "message": self.message})


def parse_message(data: str) -> tuple[MsgType, dict]:
    d = json.loads(data)
    return MsgType(d.get("type", "")), d
