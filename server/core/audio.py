"""
Audio processing utilities for Speaker Gate SDK.

Provides clean interfaces for VAD and embedding extraction.
"""
import logging
from collections import deque
from typing import Optional, List
import numpy as np

from .config import GateConfig
from .models import Models

log = logging.getLogger(__name__)


class AdaptiveSilenceDetector:
    """Adjusts silence timeout based on speaker's inter-word gap patterns."""
    
    def __init__(self, config: GateConfig):
        self._config = config
        
        # Gap tracking
        self._gap_history: deque = deque(maxlen=config.silence_gap_history)
        self._current_gap: float = 0.0
        self._was_speech: bool = False
        self._in_active_speech: bool = False
        self._speech_chunks: int = 0  # Count speech chunks to know we're in active speech
        
        # ~200ms of speech before we start tracking gaps
        self._min_speech_chunks = int(0.2 * config.sample_rate / config.chunk_size)
        
        self._cached_timeout: float = config.silence_timeout
        self._needs_recalc: bool = False
    
    def update(self, is_speech: bool, chunk_duration: float) -> None:
        if is_speech:
            self._speech_chunks += 1

            # End of gap — record if it looks like an inter-word pause
            if not self._was_speech and self._in_active_speech and self._current_gap > 0:
                min_gap = 0.05
                max_gap = self._config.silence_timeout_max * 0.8  # Below timeout
                
                if min_gap < self._current_gap < max_gap:
                    self._gap_history.append(self._current_gap)
                    self._needs_recalc = True
                    log.debug(f"Recorded inter-word gap: {self._current_gap*1000:.0f}ms")
                
                self._current_gap = 0.0
            
            if self._speech_chunks >= self._min_speech_chunks:
                self._in_active_speech = True
                
        else:
            if self._in_active_speech:
                self._current_gap += chunk_duration
        
        self._was_speech = is_speech
    
    def get_timeout(self) -> float:
        if not self._config.adaptive_silence:
            return self._config.silence_timeout
        
        if not self._needs_recalc and self._cached_timeout > 0:
            return self._cached_timeout
        
        if len(self._gap_history) < 2:
            return self._config.silence_timeout
        
        gaps = sorted(self._gap_history)
        idx = int(len(gaps) * self._config.silence_gap_percentile)
        idx = min(idx, len(gaps) - 1)
        percentile_gap = gaps[idx]
        
        adaptive_timeout = percentile_gap * self._config.silence_gap_multiplier
        timeout = max(
            self._config.silence_timeout_min,
            min(self._config.silence_timeout_max, adaptive_timeout)
        )
        
        self._cached_timeout = timeout
        self._needs_recalc = False
        
        log.debug(
            f"Adaptive timeout: {timeout*1000:.0f}ms "
            f"(gap_p{int(self._config.silence_gap_percentile*100)}={percentile_gap*1000:.0f}ms, "
            f"history={len(self._gap_history)})"
        )
        
        return timeout
    
    def reset(self) -> None:
        self._current_gap = 0.0
        self._was_speech = False
        self._in_active_speech = False
        self._speech_chunks = 0
        # Gap history persists across utterances
    
    def full_reset(self) -> None:
        self.reset()
        self._gap_history.clear()
        self._cached_timeout = self._config.silence_timeout
        self._needs_recalc = False
    
    @property
    def gap_count(self) -> int:
        return len(self._gap_history)

    @property
    def gaps(self) -> List[float]:
        return list(self._gap_history)

    @property
    def is_adapted(self) -> bool:
        return len(self._gap_history) >= 2


class VADProcessor:
    """VAD with hysteresis thresholds and temporal filtering."""

    def __init__(self, config: GateConfig, models: Optional[Models] = None):
        self._config = config
        self._models = models or Models.get()
        self.last_probability: float = 0.0
        self._is_speech: bool = False

        # Temporal filtering
        self._pending_speech: bool = False
        self._pending_silence: bool = False
        self._speech_duration_ms: float = 0.0
        self._silence_duration_ms: float = 0.0
        self._chunk_duration_ms: float = (config.chunk_size / config.sample_rate) * 1000
    
    def __call__(self, chunk: np.ndarray) -> bool:
        if self._models.vad is None:
            return True
        
        try:
            import torch
            tensor = torch.from_numpy(chunk.astype(np.float32))
            with torch.no_grad():
                self.last_probability = self._models.vad(
                    tensor.unsqueeze(0), 
                    self._config.sample_rate
                ).item()
            
            raw_is_speech = self._detect_with_hysteresis()
            return self._apply_temporal_filter(raw_is_speech)
            
        except Exception as e:
            log.debug(f"VAD error: {e}")
            return True
    
    def _detect_with_hysteresis(self) -> bool:
        if self._pending_speech or self._is_speech:
            return self.last_probability >= self._config.vad_neg_threshold
        else:
            return self.last_probability > self._config.vad_threshold
    
    def _apply_temporal_filter(self, raw_is_speech: bool) -> bool:
        if raw_is_speech:
            self._silence_duration_ms = 0.0
            self._pending_silence = False
            
            if not self._is_speech:
                self._speech_duration_ms += self._chunk_duration_ms
                self._pending_speech = True
                
                if self._speech_duration_ms >= self._config.vad_min_speech_ms:
                    self._is_speech = True
                    self._pending_speech = False
                    log.debug(f"VAD: Speech confirmed after {self._speech_duration_ms:.0f}ms")
        else:
            self._speech_duration_ms = 0.0
            self._pending_speech = False
            
            if self._is_speech:
                self._silence_duration_ms += self._chunk_duration_ms
                self._pending_silence = True
                
                if self._silence_duration_ms >= self._config.vad_min_silence_ms:
                    self._is_speech = False
                    self._pending_silence = False
                    log.debug(f"VAD: Speech ended after {self._silence_duration_ms:.0f}ms silence")
        
        return self._is_speech
    
    def reset(self):
        self._models.reset_vad()
        self.last_probability = 0.0
        self._is_speech = False
        self._pending_speech = False
        self._pending_silence = False
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0

    def speech_mask(self, audio: np.ndarray, window_samples: int, step_samples: int, min_speech_ratio: float = 0.5) -> List[bool]:
        """Per-window boolean mask: True if window has enough speech content."""
        if len(audio) < self._config.chunk_size:
            return []

        self.reset()

        chunk_size = self._config.chunk_size
        chunk_decisions: List[bool] = []
        for i in range(0, len(audio) - chunk_size + 1, chunk_size):
            chunk = audio[i:i + chunk_size]
            chunk_decisions.append(self(chunk))

        self.reset()

        if not chunk_decisions:
            return []

        chunks_per_window = window_samples // chunk_size
        chunks_per_step = max(1, step_samples // chunk_size)
        mask: List[bool] = []

        offset_chunk = 0
        while offset_chunk + chunks_per_window <= len(chunk_decisions):
            window_decisions = chunk_decisions[offset_chunk:offset_chunk + chunks_per_window]
            ratio = sum(window_decisions) / len(window_decisions) if window_decisions else 0.0
            mask.append(ratio >= min_speech_ratio)
            offset_chunk += chunks_per_step

        speech_windows = sum(mask)
        log.debug(f"Speech mask: {speech_windows}/{len(mask)} windows pass ({speech_windows/len(mask)*100:.0f}%)")

        return mask

    def speech_ratio(self, audio: np.ndarray) -> float:
        """Fraction of audio containing speech (0.0-1.0)."""
        if len(audio) < self._config.chunk_size:
            return 0.0
        
        self.reset()
        
        speech_chunks = 0
        total_chunks = 0
        
        for i in range(0, len(audio) - self._config.chunk_size + 1, self._config.chunk_size):
            chunk = audio[i:i + self._config.chunk_size]
            if self(chunk):
                speech_chunks += 1
            total_chunks += 1
        
        self.reset()
        
        if total_chunks == 0:
            return 0.0
        
        return speech_chunks / total_chunks


class EmbeddingProcessor:
    """Voice embedding extraction and comparison."""

    def __init__(self, config: GateConfig, models: Optional[Models] = None):
        self._config = config
        self._models = models or Models.get()
    
    def embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract 256-dim voice embedding, or None if audio too short."""
        if len(audio) < self._config.sample_rate:
            return None
        
        try:
            return self._models.encoder.embed_utterance(audio)
        except Exception as e:
            log.debug(f"Embedding error: {e}")
            return None
    
    def similarity(self, audio: np.ndarray, reference: np.ndarray) -> Optional[float]:
        """Cosine similarity between audio and reference embedding."""
        emb = self.embed(audio)
        if emb is not None:
            return float(np.inner(emb, reference))
        return None
    
    def consistency(self, audio: np.ndarray) -> float:
        """Split-half consistency score (0-1) for enrollment quality."""
        min_samples = self._config.sample_rate * 2
        if len(audio) < min_samples:
            return 0.0
        
        mid = len(audio) // 2
        e1 = self.embed(audio[:mid])
        e2 = self.embed(audio[mid:])
        
        if e1 is not None and e2 is not None:
            return float(np.inner(e1, e2))
        return 0.0

    def embed_chunked(
        self,
        audio: np.ndarray,
        chunk_sec: Optional[float] = None,
        step_sec: Optional[float] = None,
        vad: Optional[VADProcessor] = None,
        min_speech_ratio: float = 0.5,
    ) -> tuple[Optional[np.ndarray], float, List[float]]:
        """Extract embeddings in overlapping windows (matches streaming behavior).

        Returns (median_embedding, consistency_score, pairwise_similarities).
        When vad is provided, windows with low speech content are skipped.
        """
        chunk_samples = int((chunk_sec or self._config.embed_window_sec) * self._config.sample_rate)
        step_samples = int((step_sec or self._config.embed_interval_sec) * self._config.sample_rate)

        if len(audio) < chunk_samples:
            log.warning(f"Audio too short for chunked embedding: {len(audio)} < {chunk_samples}")
            return None, 0.0, []

        speech_mask: Optional[List[bool]] = None
        if vad is not None:
            speech_mask = vad.speech_mask(audio, chunk_samples, step_samples, min_speech_ratio)

        embeddings: List[np.ndarray] = []
        offset = 0
        window_idx = 0
        skipped = 0

        while offset + chunk_samples <= len(audio):
            if speech_mask is not None and window_idx < len(speech_mask) and not speech_mask[window_idx]:
                offset += step_samples
                window_idx += 1
                skipped += 1
                continue

            chunk = audio[offset:offset + chunk_samples]
            emb = self.embed(chunk)
            if emb is not None:
                embeddings.append(emb)
            offset += step_samples
            window_idx += 1

        if skipped > 0:
            log.debug(f"Skipped {skipped} low-speech windows")

        if len(embeddings) < 2:
            log.warning(f"Not enough valid chunks for enrollment: {len(embeddings)}")
            if len(embeddings) == 1:
                return embeddings[0], 0.0, []
            return None, 0.0, []
        
        log.debug(f"Extracted {len(embeddings)} embeddings from {len(audio)/self._config.sample_rate:.1f}s audio")

        similarities: List[float] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.inner(embeddings[i], embeddings[j]))
                similarities.append(sim)
        
        stacked = np.stack(embeddings, axis=0)
        median_embedding = np.median(stacked, axis=0).astype(np.float32)
        
        norm = np.linalg.norm(median_embedding)
        if norm > 0:
            median_embedding = median_embedding / norm
        
        consistency_score = float(np.median(similarities)) if similarities else 0.0

        log.debug(
            f"Chunked enrollment: {len(embeddings)} chunks, "
            f"consistency={consistency_score:.3f}, "
            f"sim_range=[{min(similarities):.3f}, {max(similarities):.3f}]"
        )
        
        return median_embedding, consistency_score, similarities
