"""
Configuration for Speaker Gate SDK.

Uses plain dataclass for zero external dependencies in production.
All values have sensible defaults - just instantiate GateConfig().
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GateConfig:
    """Speaker Gate configuration. All timing values in seconds unless noted."""
    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 512
    
    # VAD (Silero)
    vad_threshold: float = 0.8
    vad_neg_threshold: float = 0.35       # Lower than entry for proper hysteresis
    vad_min_speech_ms: int = 300
    vad_min_silence_ms: int = 300
    silence_timeout: float = 0.8          # Fallback before adaptive kicks in

    # Adaptive silence detection
    adaptive_silence: bool = True
    silence_timeout_min: float = 0.3      # Fast speakers
    silence_timeout_max: float = 1.5      # Slow speakers
    silence_gap_multiplier: float = 1.5
    silence_gap_history: int = 6
    silence_gap_percentile: float = 0.9

    # Buffering
    pre_buffer_sec: float = 0.5
    post_buffer_sec: float = 0.15

    # Utterance handling
    short_utterance_sec: float = 1.6
    embed_window_sec: float = 1.6         # Resemblyzer requires 1.6s
    embed_interval_sec: float = 0.2
    embed_prefix_sec: float = 1.6
    embed_suffix_sec: float = 10.0        # Grace period for short followups
    pending_grace_sec: float = 0.3
    max_segment_sec: float = 30.0

    # Similarity thresholds
    default_threshold: float = 0.69
    sim_hysteresis: float = 0.08
    
    # Trailing resume
    trailing_resume_max: float = 0.8

    # Match counting
    enter_count: int = 1
    exit_count: int = 2


@dataclass(frozen=True)
class ProfileConfig:
    """Configuration for profile threshold calculation."""
    base_threshold: float = 0.58
    max_threshold: float = 0.72
    min_consistency: float = 0.60
    max_consistency: float = 1.00
    
    def calculate_threshold(self, consistency: float) -> float:
        """Calculate optimal threshold based on enrollment consistency."""
        c = max(self.min_consistency, min(self.max_consistency, consistency))
        ratio = (c - self.min_consistency) / (self.max_consistency - self.min_consistency)
        return self.base_threshold + ratio * (self.max_threshold - self.base_threshold)
