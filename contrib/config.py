"""POC configuration with env var support via pydantic-settings."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

# Project root
_ROOT = Path(__file__).parent.parent.parent.parent


class ContribSettings(BaseSettings):
    """All settings overridable via SPEAKER_GATE_ env vars."""
    # Server
    host: str = "0.0.0.0"
    port: int = 8765
    debug: bool = False
    
    # Paths
    profiles_dir: Path = _ROOT / "profiles"
    debug_audio_dir: Path = _ROOT / "debug_audio"
    
    # Core settings (passed to GateConfig)
    sample_rate: int = 16000
    chunk_size: int = 512
    vad_threshold: float = 0.5
    vad_neg_threshold: float = 0.35
    vad_min_speech_ms: int = 300
    vad_min_silence_ms: int = 300
    silence_timeout: float = 1.2
    adaptive_silence: bool = True
    silence_timeout_min: float = 0.3
    silence_timeout_max: float = 1.5
    silence_gap_multiplier: float = 1.5
    silence_gap_history: int = 6
    silence_gap_percentile: float = 0.9
    pre_buffer_sec: float = 0.5
    post_buffer_sec: float = 0.15
    short_utterance_sec: float = 2.0
    embed_window_sec: float = 1.6
    embed_interval_sec: float = 0.2
    embed_prefix_sec: float = 1.6
    embed_suffix_sec: float = 10.0
    pending_grace_sec: float = 0.3
    max_segment_sec: float = 30.0
    default_threshold: float = 0.69
    sim_hysteresis: float = 0.07
    trailing_resume_max: float = 0.8
    enter_count: int = 1
    exit_count: int = 2
    
    # Stats (POC only)
    stats_enabled: bool = True
    stats_interval_ms: int = 96
    
    model_config = {"env_prefix": "SPEAKER_GATE_"}
    
    def to_gate_config(self):
        from ..core import GateConfig
        return GateConfig(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            vad_threshold=self.vad_threshold,
            vad_neg_threshold=self.vad_neg_threshold,
            vad_min_speech_ms=self.vad_min_speech_ms,
            vad_min_silence_ms=self.vad_min_silence_ms,
            silence_timeout=self.silence_timeout,
            adaptive_silence=self.adaptive_silence,
            silence_timeout_min=self.silence_timeout_min,
            silence_timeout_max=self.silence_timeout_max,
            silence_gap_multiplier=self.silence_gap_multiplier,
            silence_gap_history=self.silence_gap_history,
            silence_gap_percentile=self.silence_gap_percentile,
            pre_buffer_sec=self.pre_buffer_sec,
            post_buffer_sec=self.post_buffer_sec,
            short_utterance_sec=self.short_utterance_sec,
            embed_window_sec=self.embed_window_sec,
            embed_interval_sec=self.embed_interval_sec,
            embed_prefix_sec=self.embed_prefix_sec,
            embed_suffix_sec=self.embed_suffix_sec,
            pending_grace_sec=self.pending_grace_sec,
            max_segment_sec=self.max_segment_sec,
            default_threshold=self.default_threshold,
            sim_hysteresis=self.sim_hysteresis,
            trailing_resume_max=self.trailing_resume_max,
            enter_count=self.enter_count,
            exit_count=self.exit_count,
        )


_settings: Optional[ContribSettings] = None


@lru_cache
def get_settings() -> ContribSettings:
    return ContribSettings()


def update_settings(**kwargs) -> ContribSettings:
    global _settings
    get_settings.cache_clear()
    _settings = ContribSettings(**kwargs)
    return _settings
