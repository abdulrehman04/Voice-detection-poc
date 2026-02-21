"""
Configuration for Speaker Gate SDK.

Uses plain dataclass for zero external dependencies in production.
All values have sensible defaults - just instantiate GateConfig().
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GateConfig:
    """
    Speaker Gate configuration.
    
    Frozen dataclass ensures immutability after creation.
    All timing values are in seconds unless otherwise noted.
    
    Attributes:
        sample_rate: Audio sample rate in Hz (default: 16000)
        chunk_size: Samples per processing chunk (default: 512)
        
        vad_threshold: VAD probability threshold (0-1). Higher = less sensitive to noise
        vad_neg_threshold: Exit threshold when speech ends (typically threshold - 0.15)
        vad_min_speech_ms: Minimum speech duration in ms (filters short noise bursts)
        vad_min_silence_ms: Minimum silence duration in ms to split speech segments
        silence_timeout: Seconds of silence to end utterance (fallback for adaptive)
        adaptive_silence: Enable adaptive silence detection based on speech patterns
        silence_timeout_min: Minimum silence timeout (fast speakers)
        silence_timeout_max: Maximum silence timeout (slow speakers) 
        silence_gap_multiplier: Multiplier for detected gaps (gap * multiplier = timeout)
        silence_gap_history: Number of recent gaps to track for adaptation
        silence_gap_percentile: Percentile of gaps to use (0.9 = 90th percentile)
        
        pre_buffer_sec: Audio to keep before speech detection
        post_buffer_sec: Audio to keep after speech ends
        
        short_utterance_sec: Auto-approve short utterances if recent USER
        embed_window_sec: Audio window for embedding extraction
        embed_interval_sec: Minimum time between embedding checks
        embed_prefix_sec: Retroactive audio buffer for confirmed speech
        embed_suffix_sec: Grace period for short followup utterances
        
        pending_grace_sec: Minimum speech before rejecting
        max_segment_sec: Maximum segment length before forcing split
        
        default_threshold: Default similarity threshold
        sim_hysteresis: Hysteresis for enter/exit thresholds
        
        enter_count: Consecutive matches to confirm user
        exit_count: Consecutive mismatches to reject
    """
    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 512
    
    # VAD settings - Silero VAD parameters
    vad_threshold: float = 0.8        # Speech detection threshold. Higher = less sensitive to noise
    vad_neg_threshold: float = 0.35   # Exit threshold (when to stop detecting speech). Lower than entry for proper hysteresis
    vad_min_speech_ms: int = 300      # Minimum speech duration in ms (filters short noise bursts)
    vad_min_silence_ms: int = 300     # Minimum silence duration in ms to consider speech ended
    silence_timeout: float = 0.8      # Fallback/initial timeout
    
    # Adaptive silence detection - adjusts to speaker's tempo
    adaptive_silence: bool = True  # Enable adaptive timeout
    silence_timeout_min: float = 0.3  # Fast speakers (young, fluent)
    silence_timeout_max: float = 1.5   # Slow speakers (elderly, hesitant)
    silence_gap_multiplier: float = 1.5  # timeout = max_gap * multiplier
    silence_gap_history: int = 6  # Track last N gaps
    silence_gap_percentile: float = 0.9  # Use 90th percentile gap
    
    # Buffering
    pre_buffer_sec: float = 0.5  # VAD pre-buffer (was 1.0)
    post_buffer_sec: float = 0.15  # Shorter trailing (was 0.3)
    
    # Utterance handling
    short_utterance_sec: float = 1.6  # Match embed_window for "yes" case
    embed_window_sec: float = 1.6  # Model requires 1.6s
    embed_interval_sec: float = 0.2  # Check faster (was 0.3)
    embed_prefix_sec: float = 1.6  # Historical buffer = embed_window
    embed_suffix_sec: float = 3.0  # Grace for short followups (was 2.0)
    pending_grace_sec: float = 0.3  # Faster rejection of short noise (was 0.5)
    max_segment_sec: float = 30.0
    
    # Similarity thresholds
    default_threshold: float = 0.69  # Slightly lower for faster match
    sim_hysteresis: float = 0.08  # Wider hysteresis to avoid flapping
    
    # Trailing resume
    trailing_resume_max: float = 0.8  # Max pause duration to resume instead of completing segment

    # Match counting
    enter_count: int = 1  # Single match to confirm (was 2) - prefix buffer compensates
    exit_count: int = 2  # Keep 2 for exit to avoid false rejections


@dataclass(frozen=True)
class ProfileConfig:
    """Configuration for profile threshold calculation."""
    base_threshold: float = 0.65
    max_threshold: float = 0.78
    min_consistency: float = 0.60
    max_consistency: float = 1.00
    
    def calculate_threshold(self, consistency: float) -> float:
        """Calculate optimal threshold based on enrollment consistency."""
        c = max(self.min_consistency, min(self.max_consistency, consistency))
        ratio = (c - self.min_consistency) / (self.max_consistency - self.min_consistency)
        return self.base_threshold + ratio * (self.max_threshold - self.base_threshold)
