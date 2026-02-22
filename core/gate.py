"""
Speaker Gate — real-time speaker verification state machine.

Feed audio chunks via gate.feed(), receive events for speech detection,
speaker verification, and completed segments.
"""
import collections
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable
import numpy as np

from .config import GateConfig
from .events import (
    Event, EventEmitter, EventHandler,
    SpeechEvent, StateChangeEvent, SimilarityEvent, SegmentEvent
)
from .audio import VADProcessor, EmbeddingProcessor, AdaptiveSilenceDetector
from .models import Models

log = logging.getLogger(__name__)


class GateState(Enum):
    """Speaker gate states."""
    UNKNOWN = auto()   # No active speech
    PENDING = auto()   # Speech detected, verifying speaker
    USER = auto()      # Confirmed enrolled user speaking
    TRAILING = auto()  # User finished, collecting trailing audio


@dataclass
class _Context:
    """Internal processing context."""
    state: GateState = GateState.UNKNOWN
    last_confirmed: Optional[str] = None
    segment: list = field(default_factory=list)
    post_buffer: list = field(default_factory=list)
    
    speech_start: Optional[float] = None
    last_embed_time: float = 0
    silence_dur: float = 0
    speech_dur: float = 0
    
    matches: int = 0
    mismatches: int = 0
    last_sim: float = 0.0
    best_speech_sim: float = 0.0  # Peak similarity during active speech
    last_best_session: int = 0  # 1-indexed best matching session (0=unknown)
    sim_history: list = field(default_factory=list)  # Last N similarity scores
    last_user_end: Optional[float] = None
    
    # Timing
    vad_started_at: float = 0.0
    user_confirmed_at: float = 0.0
    speech_ended_at: float = 0.0


@dataclass
class _SegmentResult:
    """Completed segment with timing."""
    data: np.ndarray
    vad_started_at: float
    user_confirmed_at: float
    speech_ended_at: float


class SpeakerGate:
    """Real-time speaker verification gate with VAD-only fallback."""
    
    def __init__(
        self,
        profile_embedding: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        config: Optional[GateConfig] = None,
        models: Optional[Models] = None,
    ):
        self._config = config or GateConfig()
        self._profile: Optional[np.ndarray] = None
        self._profile_refs: list[np.ndarray] = []  # Multi-session references
        self._profile_thresholds: list[float] = []  # Per-session thresholds
        self._verification_enabled = True
        if profile_embedding is not None:
            self._profile = np.asarray(profile_embedding, dtype=np.float32)
            self._profile_refs = [self._profile]
        self._models = models or Models.get()

        # Processors
        self._vad = VADProcessor(self._config, self._models)
        self._emb = EmbeddingProcessor(self._config, self._models)
        self._silence_detector = AdaptiveSilenceDetector(self._config)

        # Thresholds
        self._sim_enter = threshold or self._config.default_threshold
        self._sim_exit = self._sim_enter - self._config.sim_hysteresis
        self._profile_thresholds = [self._sim_enter]
        
        # Buffers
        pre_samples = int(self._config.pre_buffer_sec * self._config.sample_rate)
        pending_samples = int(self._config.embed_prefix_sec * self._config.sample_rate)
        embed_samples = int(self._config.embed_window_sec * self._config.sample_rate)
        
        self._pre_buffer = collections.deque(maxlen=pre_samples)
        self._pending_buffer = collections.deque(maxlen=pending_samples)
        self._embed_samples = embed_samples
        
        # Context and events
        self._ctx = _Context()
        self._events = EventEmitter()
    
    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def state(self) -> GateState:
        """Current gate state."""
        return self._ctx.state
    
    @property
    def vad_only(self) -> bool:
        return self._profile is None or not self._verification_enabled

    @property
    def verification_enabled(self) -> bool:
        return self._verification_enabled and self._profile is not None

    @verification_enabled.setter
    def verification_enabled(self, value: bool) -> None:
        self._verification_enabled = value
        if not value:
            log.debug("Speaker verification disabled - VAD-only mode")
        else:
            log.debug(f"Speaker verification enabled (profile={'set' if self._profile is not None else 'not set'})")
    
    def set_profile(
        self,
        embedding: Optional[np.ndarray],
        threshold: Optional[float] = None,
        session_embeddings: Optional[list[np.ndarray]] = None,
        session_thresholds: Optional[list[float]] = None,
    ) -> None:
        """Set or clear the speaker profile. Pass None to switch to VAD-only."""
        if embedding is None:
            self._profile = None
            self._profile_refs = []
            self._profile_thresholds = []
            log.debug("Profile cleared - VAD-only mode")
        else:
            self._profile = np.asarray(embedding, dtype=np.float32)
            if session_embeddings:
                self._profile_refs = [np.asarray(e, dtype=np.float32) for e in session_embeddings]
            else:
                self._profile_refs = [self._profile]
            if threshold is not None:
                self._sim_enter = threshold
                self._sim_exit = threshold - self._config.sim_hysteresis
            if session_thresholds and len(session_thresholds) == len(self._profile_refs):
                self._profile_thresholds = session_thresholds
            else:
                self._profile_thresholds = [self._sim_enter] * len(self._profile_refs)
            log.debug(f"Profile set, threshold={self._sim_enter:.3f}, refs={len(self._profile_refs)}, per_session={self._profile_thresholds}")

    @property
    def similarity(self) -> float:
        return self._ctx.last_sim

    @property
    def vad_probability(self) -> float:
        return self._vad.last_probability

    @property
    def last_best_session(self) -> int:
        return self._ctx.last_best_session

    @property
    def num_sessions(self) -> int:
        return len(self._profile_refs)

    @property
    def threshold(self) -> float:
        return self._sim_enter

    @property
    def config(self) -> GateConfig:
        return self._config

    @property
    def adaptive_silence_timeout(self) -> float:
        return self._silence_detector.get_timeout()

    @property
    def silence_detector_adapted(self) -> bool:
        return self._silence_detector.is_adapted
    
    def on(self, event: Event, handler: EventHandler) -> Callable[[], None]:
        """Subscribe to event. Returns unsubscribe function."""
        return self._events.on(event, handler)

    def once(self, event: Event, handler: EventHandler):
        self._events.once(event, handler)

    def off(self, event: Event, handler: Optional[EventHandler] = None):
        self._events.off(event, handler)
    
    def feed(self, audio: np.ndarray) -> None:
        """Feed audio data for processing. Splits into chunk_size pieces internally."""
        chunk_size = self._config.chunk_size
        offset = 0
        
        while offset + chunk_size <= len(audio):
            self._process_chunk(audio[offset:offset + chunk_size])
            offset += chunk_size
        
        # Handle remaining samples
        if offset < len(audio):
            self._process_chunk(audio[offset:])
    
    def reset(self):
        self._pre_buffer.clear()
        self._pending_buffer.clear()
        
        last_user_end = self._ctx.last_user_end
        last_confirmed = self._ctx.last_confirmed
        
        self._ctx = _Context()
        self._ctx.last_user_end = last_user_end
        self._ctx.last_confirmed = last_confirmed
        
        self._vad.reset()
        self._silence_detector.reset()
    
    # ─────────────────────────────────────────────────────────────────
    # Internal processing
    # ─────────────────────────────────────────────────────────────────
    
    def _process_chunk(self, chunk: np.ndarray):
        now = time.time()
        prev_state = self._ctx.state
        
        # VAD
        is_speech = self._vad(chunk)
        self._pre_buffer.extend(chunk)
        self._pending_buffer.extend(chunk)
        
        # Update timing
        chunk_dur = len(chunk) / self._config.sample_rate
        self._ctx.silence_dur = 0 if is_speech else self._ctx.silence_dur + chunk_dur
        if is_speech:
            self._ctx.speech_dur += chunk_dur
        
        # Update adaptive silence detector
        self._silence_detector.update(is_speech, chunk_dur)
        
        # State machine
        handler = self._STATE_HANDLERS[self._ctx.state]
        seg_result = handler(self, chunk, is_speech, now)
        
        # Emit state change event
        if self._ctx.state != prev_state:
            self._events.emit(Event.STATE_CHANGED, StateChangeEvent(
                timestamp=now,
                previous=prev_state.name,
                current=self._ctx.state.name,
            ))
            
            # Additional semantic events
            if self._ctx.state == GateState.PENDING and prev_state == GateState.UNKNOWN:
                self._events.emit(Event.SPEECH_STARTED, SpeechEvent(
                    timestamp=now,
                    vad_probability=self._vad.last_probability,
                ))
            elif self._ctx.state == GateState.USER and prev_state == GateState.PENDING:
                self._events.emit(Event.USER_STARTED, SpeechEvent(
                    timestamp=now,
                    vad_probability=self._vad.last_probability,
                ))
            elif prev_state == GateState.PENDING and self._ctx.state == GateState.UNKNOWN:
                if self._ctx.last_confirmed == "OTHER":
                    self._events.emit(Event.OTHER_DETECTED, SpeechEvent(
                        timestamp=now,
                        vad_probability=self._vad.last_probability,
                    ))
        
        # Emit segment complete
        if seg_result is not None:
            self._events.emit(Event.SEGMENT_COMPLETE, SegmentEvent(
                timestamp=now,
                audio=seg_result.data,
                duration_sec=len(seg_result.data) / self._config.sample_rate,
                samples=len(seg_result.data),
                vad_started_at=seg_result.vad_started_at,
                user_confirmed_at=seg_result.user_confirmed_at,
                speech_ended_at=seg_result.speech_ended_at,
            ))
            self._events.emit(Event.USER_ENDED, SpeechEvent(
                timestamp=now,
                vad_probability=self._vad.last_probability,
            ))
    
    def _on_unknown(self, chunk, is_speech, now) -> Optional[_SegmentResult]:
        if is_speech:
            self._ctx.state = GateState.PENDING
            self._ctx.speech_start = now
            self._ctx.vad_started_at = now
            self._ctx.segment = list(self._pre_buffer) + list(chunk)
            self._ctx.matches = self._ctx.mismatches = 0
        return None
    
    def _on_pending(self, chunk, is_speech, now) -> Optional[_SegmentResult]:
        cfg = self._config
        self._ctx.segment.extend(chunk)
        speech_dur = now - self._ctx.speech_start if self._ctx.speech_start else 0
        time_since_user = (now - self._ctx.last_user_end) if self._ctx.last_user_end else float('inf')
        
        if self.vad_only:
            self._ctx.state = GateState.USER
            self._ctx.last_confirmed = "USER"
            self._ctx.user_confirmed_at = now
            return None
        
        silence_timeout = self._silence_detector.get_timeout()
        if self._ctx.silence_dur > silence_timeout:
            # Short utterance auto-approval (e.g. "yes" right after confirmed speech)
            is_recent_user = self._ctx.last_confirmed == "USER" and time_since_user < cfg.embed_suffix_sec
            is_short_utterance = speech_dur < cfg.short_utterance_sec
            
            if is_short_utterance and is_recent_user:
                # Use best score from active speech (tail audio is diluted with silence)
                sim = self._ctx.best_speech_sim if self._ctx.best_speech_sim > 0 else self._check_similarity()
                if sim is not None and sim > 0 and sim < self._sim_exit:
                    log.debug(f"Short utterance rejected: sim={sim:.3f} < exit={self._sim_exit:.3f}")
                    if self._ctx.speech_dur >= cfg.pending_grace_sec:
                        self._ctx.state = GateState.UNKNOWN
                        self._ctx.segment = []
                        self._ctx.last_confirmed = "OTHER"
                    else:
                        self._ctx.state = GateState.UNKNOWN
                        self._ctx.segment = []
                    return None
                log.debug(f"Short utterance ({speech_dur:.2f}s) approved (recent user {time_since_user:.1f}s ago, sim={sim})")
                self._ctx.state = GateState.TRAILING
                self._ctx.last_confirmed = "USER"
                self._ctx.user_confirmed_at = now
                self._ctx.speech_ended_at = now
                return None
            
            # Use best score from active speech (tail audio is diluted with silence)
            sim = self._ctx.best_speech_sim if self._ctx.best_speech_sim > 0 else self._check_similarity()
            threshold = self._sim_enter if not is_recent_user else self._sim_exit
            if sim is not None and sim > threshold:
                log.debug(f"Final check passed: sim={sim:.3f} > {threshold:.3f}")
                self._include_prefix_buffer(cfg)
                self._ctx.state = GateState.TRAILING
                self._ctx.last_confirmed = "USER"
                self._ctx.user_confirmed_at = now
                self._ctx.speech_ended_at = now
                return None
            
            if self._ctx.speech_dur >= cfg.pending_grace_sec:
                log.debug(f"Rejected: speech_dur={self._ctx.speech_dur:.2f}s, sim={sim}")
                self._ctx.state = GateState.UNKNOWN
                self._ctx.segment = []
                self._ctx.last_confirmed = "OTHER"
            else:
                self._ctx.state = GateState.UNKNOWN
                self._ctx.segment = []
            return None
        
        # Skip similarity checks during silence — tail audio is diluted with silence
        # and produces garbage scores that poison counters and history
        if self._should_embed(now) and self._ctx.silence_dur < 0.3:
            sim = self._check_similarity()
            if sim is not None:
                self._update_counters(sim)

                # Emit similarity update
                self._events.emit(Event.SIMILARITY_UPDATE, SimilarityEvent(
                    timestamp=now,
                    similarity=sim,
                    threshold=self._sim_enter,
                    is_match=sim > self._sim_enter,
                ))

                if self._ctx.matches >= cfg.enter_count:
                    self._include_prefix_buffer(cfg)
                    self._ctx.state = GateState.USER
                    self._ctx.last_confirmed = "USER"
                    self._ctx.user_confirmed_at = now
                    return None

                # Accumulation fallback: 3+ checks averaging above threshold
                hist = self._ctx.sim_history
                if len(hist) >= 3 and sum(hist[-3:]) / 3 > self._sim_enter:
                    avg = sum(hist[-3:]) / 3
                    log.debug(f"Accumulation confirm: avg({len(hist[-3:])} checks)={avg:.3f} > {self._sim_enter:.3f}")
                    self._include_prefix_buffer(cfg)
                    self._ctx.state = GateState.USER
                    self._ctx.last_confirmed = "USER"
                    self._ctx.user_confirmed_at = now
                    return None

                if self._ctx.mismatches >= cfg.exit_count and self._ctx.speech_dur >= cfg.pending_grace_sec:
                    self._ctx.state = GateState.UNKNOWN
                    self._ctx.segment = []
                    self._ctx.last_confirmed = "OTHER"
        
        return None
    
    def _include_prefix_buffer(self, cfg: GateConfig):
        """Prepend buffered audio from before user was confirmed."""
        pending_list = list(self._pending_buffer)
        segment_len = len(self._ctx.segment)
        pending_len = len(pending_list)

        if pending_len > segment_len:
            prefix_samples = min(
                pending_len - segment_len,  # Available prefix
                int(cfg.embed_prefix_sec * cfg.sample_rate)  # Max prefix
            )
            if prefix_samples > 0:
                prefix = pending_list[:prefix_samples]
                log.debug(f"Including {prefix_samples/cfg.sample_rate:.2f}s prefix audio")
                self._ctx.segment = prefix + self._ctx.segment
    
    def _on_user(self, chunk, is_speech, now) -> Optional[_SegmentResult]:
        cfg = self._config
        self._ctx.segment.extend(chunk)
        seg_dur = len(self._ctx.segment) / cfg.sample_rate

        silence_timeout = self._silence_detector.get_timeout()

        if seg_dur > cfg.max_segment_sec or self._ctx.silence_dur > silence_timeout:
            self._ctx.state = GateState.TRAILING
            self._ctx.post_buffer = list(chunk)
            self._ctx.speech_ended_at = now
            return None
        
        if self.vad_only:
            return None
        
        # Skip checks during silence — prevents false speaker-change from degraded scores
        if self._should_embed(now) and self._ctx.silence_dur < 0.3:
            sim = self._check_similarity()
            if sim is not None:
                self._update_counters(sim)

                self._events.emit(Event.SIMILARITY_UPDATE, SimilarityEvent(
                    timestamp=now,
                    similarity=sim,
                    threshold=self._sim_exit,
                    is_match=sim > self._sim_exit,
                ))

                if self._ctx.mismatches >= cfg.exit_count:
                    log.debug(f"Speaker change detected: sim={sim:.3f} < exit={self._sim_exit:.3f}")
                    self._ctx.state = GateState.TRAILING
                    self._ctx.speech_ended_at = now
        
        return None
    
    def _on_trailing(self, chunk, is_speech, now) -> Optional[_SegmentResult]:
        cfg = self._config
        self._ctx.post_buffer.extend(chunk)
        self._ctx.segment.extend(chunk)
        
        # Resume on short pauses (breathing, inter-sentence gaps)
        silence_timeout = self._silence_detector.get_timeout()
        resume_limit = max(0.3, min(cfg.trailing_resume_max, silence_timeout * 0.6))
        if is_speech and self._ctx.silence_dur < resume_limit:
            self._ctx.state = GateState.USER
            self._ctx.post_buffer = []
            return None
        
        post_dur = len(self._ctx.post_buffer) / cfg.sample_rate
        silence_timeout = self._silence_detector.get_timeout()

        if post_dur >= cfg.post_buffer_sec or self._ctx.silence_dur > silence_timeout:
            seg_data = np.array(self._ctx.segment, dtype=np.float32)
            seg_result = _SegmentResult(
                data=seg_data,
                vad_started_at=self._ctx.vad_started_at,
                user_confirmed_at=self._ctx.user_confirmed_at,
                speech_ended_at=self._ctx.speech_ended_at,
            )
            
            self._ctx.last_user_end = now
            last_user_end = self._ctx.last_user_end
            last_sim = self._ctx.last_sim
            last_best_session = self._ctx.last_best_session
            self.reset()
            self._ctx.last_user_end = last_user_end
            self._ctx.last_confirmed = "USER"
            self._ctx.last_sim = last_sim
            self._ctx.last_best_session = last_best_session
            
            return seg_result
        
        return None
    
    _STATE_HANDLERS = {
        GateState.UNKNOWN: _on_unknown,
        GateState.PENDING: _on_pending,
        GateState.USER: _on_user,
        GateState.TRAILING: _on_trailing,
    }
    
    def _should_embed(self, now: float) -> bool:
        if self.vad_only:
            return False
        return (len(self._ctx.segment) >= self._embed_samples and 
                now - self._ctx.last_embed_time > self._config.embed_interval_sec)
    
    def _check_similarity(self) -> Optional[float]:
        if self._profile is None:
            return None
        audio = np.array(self._ctx.segment[-self._embed_samples:], dtype=np.float32)

        # Multi-session: compare against all refs, use best match's threshold
        if len(self._profile_refs) > 1:
            emb = self._emb.embed(audio)
            if emb is None:
                return None
            sims = [float(np.inner(emb, ref)) for ref in self._profile_refs]
            best_idx = max(range(len(sims)), key=lambda i: sims[i])
            sim = sims[best_idx]

            # Switch to best-matching session's threshold
            if self._profile_thresholds and best_idx < len(self._profile_thresholds):
                self._sim_enter = self._profile_thresholds[best_idx]
                self._sim_exit = self._sim_enter - self._config.sim_hysteresis
            self._ctx.last_best_session = best_idx + 1  # 1-indexed
            log.debug(
                f"Similarity check: best_session={best_idx+1}/{len(self._profile_refs)}, "
                f"sim={sim:.3f}, sims={[f'{s:.3f}' for s in sims]}, "
                f"threshold={self._sim_enter:.3f}"
            )
        else:
            sim = self._emb.similarity(audio, self._profile)
            self._ctx.last_best_session = 1
            if sim is not None:
                log.debug(f"Similarity check: sim={sim:.3f}, threshold={self._sim_enter:.3f}")

        if sim is not None:
            self._ctx.last_sim = sim
            self._ctx.last_embed_time = time.time()
            if sim > self._ctx.best_speech_sim:
                self._ctx.best_speech_sim = sim
        return sim
    
    def _update_counters(self, sim: float):
        # Rolling history (last 5) for accumulation fallback
        self._ctx.sim_history.append(sim)
        if len(self._ctx.sim_history) > 5:
            self._ctx.sim_history.pop(0)

        # Decay counters instead of hard reset for smoother transitions
        if sim > self._sim_enter:
            self._ctx.matches += 1
            self._ctx.mismatches = max(0, self._ctx.mismatches - 1)
        elif sim < self._sim_exit:
            self._ctx.mismatches += 1
            self._ctx.matches = max(0, self._ctx.matches - 1)
