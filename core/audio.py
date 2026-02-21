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
    """
    Adaptive silence detection based on speaker's speech patterns.
    
    Tracks inter-word gaps during active speech and dynamically adjusts
    the silence timeout threshold. This handles:
    - Fast speakers (young, fluent): Short gaps → short timeout → fast response
    - Slow speakers (elderly, hesitant): Long gaps → longer timeout → complete sentences
    
    Algorithm:
    1. During active speech, detect "micro-silences" (gaps between words)
    2. Track recent gap durations (typically 100-400ms)
    3. Use percentile of gaps * multiplier as adaptive timeout
    4. Clamp between min and max bounds
    
    Example:
        detector = AdaptiveSilenceDetector(config)
        
        # During speech processing
        detector.update(is_speech=True, chunk_duration=0.032)
        detector.update(is_speech=False, chunk_duration=0.032)  # gap starts
        detector.update(is_speech=True, chunk_duration=0.032)   # gap ends, recorded
        
        # Get current adaptive timeout
        timeout = detector.get_timeout()
    """
    
    def __init__(self, config: GateConfig):
        """
        Initialize adaptive silence detector.
        
        Args:
            config: Gate configuration with adaptive silence settings
        """
        self._config = config
        
        # Gap tracking
        self._gap_history: deque = deque(maxlen=config.silence_gap_history)
        self._current_gap: float = 0.0
        self._was_speech: bool = False
        self._in_active_speech: bool = False
        self._speech_chunks: int = 0  # Count speech chunks to know we're in active speech
        
        # Minimum speech chunks before we consider it "active speech"
        # ~200ms of speech before we start tracking gaps
        self._min_speech_chunks = int(0.2 * config.sample_rate / config.chunk_size)
        
        # Cache for current timeout
        self._cached_timeout: float = config.silence_timeout
        self._needs_recalc: bool = False
    
    def update(self, is_speech: bool, chunk_duration: float) -> None:
        """
        Update detector with new chunk.
        
        Call this for each audio chunk during speech processing.
        
        Args:
            is_speech: Whether current chunk contains speech (from VAD)
            chunk_duration: Duration of chunk in seconds
        """
        if is_speech:
            self._speech_chunks += 1
            
            # Check if we're transitioning from silence to speech (end of gap)
            if not self._was_speech and self._in_active_speech and self._current_gap > 0:
                # Only record gaps that look like inter-word pauses
                # Too short = VAD noise, too long = actual end of speech
                min_gap = 0.05  # 50ms minimum gap
                max_gap = self._config.silence_timeout_max * 0.8  # Below timeout
                
                if min_gap < self._current_gap < max_gap:
                    self._gap_history.append(self._current_gap)
                    self._needs_recalc = True
                    log.debug(f"Recorded inter-word gap: {self._current_gap*1000:.0f}ms")
                
                self._current_gap = 0.0
            
            # Mark as active speech after enough consecutive speech
            if self._speech_chunks >= self._min_speech_chunks:
                self._in_active_speech = True
                
        else:
            # Silence - accumulate gap duration if in active speech
            if self._in_active_speech:
                self._current_gap += chunk_duration
        
        self._was_speech = is_speech
    
    def get_timeout(self) -> float:
        """
        Get current adaptive silence timeout.
        
        Returns:
            Silence timeout in seconds, adapted to speaker's tempo
        """
        if not self._config.adaptive_silence:
            return self._config.silence_timeout
        
        if not self._needs_recalc and self._cached_timeout > 0:
            return self._cached_timeout
        
        if len(self._gap_history) < 2:
            # Not enough data yet, use default
            return self._config.silence_timeout
        
        # Calculate percentile of gaps
        gaps = sorted(self._gap_history)
        idx = int(len(gaps) * self._config.silence_gap_percentile)
        idx = min(idx, len(gaps) - 1)
        percentile_gap = gaps[idx]
        
        # Apply multiplier: end-of-speech should be longer than inter-word gaps
        adaptive_timeout = percentile_gap * self._config.silence_gap_multiplier
        
        # Clamp to configured bounds
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
        """Reset detector state for new utterance."""
        self._current_gap = 0.0
        self._was_speech = False
        self._in_active_speech = False
        self._speech_chunks = 0
        # Keep gap history - speaker tempo persists across utterances
    
    def full_reset(self) -> None:
        """Fully reset detector including gap history."""
        self.reset()
        self._gap_history.clear()
        self._cached_timeout = self._config.silence_timeout
        self._needs_recalc = False
    
    @property
    def gap_count(self) -> int:
        """Number of tracked gaps."""
        return len(self._gap_history)
    
    @property
    def gaps(self) -> List[float]:
        """Recent gap durations (for debugging)."""
        return list(self._gap_history)
    
    @property
    def is_adapted(self) -> bool:
        """Whether detector has enough data to adapt."""
        return len(self._gap_history) >= 2


class VADProcessor:
    """
    Voice Activity Detection processor.
    
    Wraps the VAD model with configuration and state management.
    If no VAD model is available, always returns True (speech detected).
    
    Uses hysteresis thresholds:
    - vad_threshold: Probability must exceed this to START detecting speech
    - vad_neg_threshold: Probability must drop below this to STOP detecting speech
    This prevents rapid on/off switching at the threshold boundary.
    
    Also applies temporal filtering:
    - vad_min_speech_ms: Requires this duration of speech before confirming
    - vad_min_silence_ms: Requires this duration of silence before ending speech
    This filters out short noise bursts and brief pauses.
    """
    
    def __init__(self, config: GateConfig, models: Optional[Models] = None):
        """
        Initialize VAD processor.
        
        Args:
            config: Gate configuration
            models: Models container (uses singleton if not provided)
        """
        self._config = config
        self._models = models or Models.get()
        self.last_probability: float = 0.0
        self._is_speech: bool = False  # Track current speech state for hysteresis
        
        # Temporal filtering state
        self._pending_speech: bool = False  # Detected speech but not confirmed yet
        self._pending_silence: bool = False  # Detected silence but not confirmed yet
        self._speech_duration_ms: float = 0.0  # Accumulated speech duration
        self._silence_duration_ms: float = 0.0  # Accumulated silence duration
        self._chunk_duration_ms: float = (config.chunk_size / config.sample_rate) * 1000
    
    def __call__(self, chunk: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.
        
        Uses hysteresis: once speech is detected, it stays detected until
        probability drops below vad_neg_threshold (not just below vad_threshold).
        
        Also applies temporal filtering to require minimum durations.
        
        Args:
            chunk: Audio samples (float32, mono)
            
        Returns:
            True if speech detected, False otherwise
        """
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
            
            # Raw detection with hysteresis thresholds
            raw_is_speech = self._detect_with_hysteresis()
            
            # Apply temporal filtering
            return self._apply_temporal_filter(raw_is_speech)
            
        except Exception as e:
            log.debug(f"VAD error: {e}")
            return True
    
    def _detect_with_hysteresis(self) -> bool:
        """Apply hysteresis thresholds to raw probability."""
        if self._pending_speech or self._is_speech:
            # Currently in or pending speech - use neg_threshold to exit
            return self.last_probability >= self._config.vad_neg_threshold
        else:
            # Currently not in speech - use threshold to enter
            return self.last_probability > self._config.vad_threshold
    
    def _apply_temporal_filter(self, raw_is_speech: bool) -> bool:
        """
        Apply temporal filtering to require minimum durations.
        
        - Speech must persist for vad_min_speech_ms before confirming
        - Silence must persist for vad_min_silence_ms before ending speech
        """
        if raw_is_speech:
            # Reset silence counter
            self._silence_duration_ms = 0.0
            self._pending_silence = False
            
            if not self._is_speech:
                # Accumulate speech duration until threshold
                self._speech_duration_ms += self._chunk_duration_ms
                self._pending_speech = True
                
                if self._speech_duration_ms >= self._config.vad_min_speech_ms:
                    # Confirmed speech
                    self._is_speech = True
                    self._pending_speech = False
                    log.debug(f"VAD: Speech confirmed after {self._speech_duration_ms:.0f}ms")
        else:
            # Reset speech counter
            self._speech_duration_ms = 0.0
            self._pending_speech = False
            
            if self._is_speech:
                # Accumulate silence duration until threshold
                self._silence_duration_ms += self._chunk_duration_ms
                self._pending_silence = True
                
                if self._silence_duration_ms >= self._config.vad_min_silence_ms:
                    # Confirmed silence - end speech
                    self._is_speech = False
                    self._pending_silence = False
                    log.debug(f"VAD: Speech ended after {self._silence_duration_ms:.0f}ms silence")
        
        return self._is_speech
    
    def reset(self):
        """Reset VAD internal states."""
        self._models.reset_vad()
        self.last_probability = 0.0
        self._is_speech = False
        self._pending_speech = False
        self._pending_silence = False
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0

    def speech_mask(self, audio: np.ndarray, window_samples: int, step_samples: int, min_speech_ratio: float = 0.5) -> List[bool]:
        """
        Pre-compute per-window speech mask over entire audio.

        For each window position, returns True if the speech ratio
        within that window meets the minimum threshold.

        Args:
            audio: Audio samples (float32, mono, 16kHz)
            window_samples: Size of each embedding window in samples
            step_samples: Step between windows in samples
            min_speech_ratio: Minimum fraction of speech in window to pass (0-1)

        Returns:
            List of booleans, one per window position
        """
        if len(audio) < self._config.chunk_size:
            return []

        self.reset()

        # First pass: compute per-chunk VAD decisions
        chunk_size = self._config.chunk_size
        chunk_decisions: List[bool] = []
        for i in range(0, len(audio) - chunk_size + 1, chunk_size):
            chunk = audio[i:i + chunk_size]
            chunk_decisions.append(self(chunk))

        self.reset()

        if not chunk_decisions:
            return []

        # Second pass: compute speech ratio per window
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
        """
        Calculate the ratio of audio that contains speech.
        
        Processes audio in chunks and returns what fraction detected as speech.
        Useful for enrollment validation to reject silent/noise-only audio.
        
        Args:
            audio: Audio samples (float32, mono, 16kHz)
            
        Returns:
            Speech ratio (0.0-1.0), where 1.0 means all speech
        """
        if len(audio) < self._config.chunk_size:
            return 0.0
        
        self.reset()  # Clear VAD state before processing
        
        speech_chunks = 0
        total_chunks = 0
        
        # Process in VAD-sized chunks
        for i in range(0, len(audio) - self._config.chunk_size + 1, self._config.chunk_size):
            chunk = audio[i:i + self._config.chunk_size]
            if self(chunk):  # Uses __call__ which runs VAD
                speech_chunks += 1
            total_chunks += 1
        
        self.reset()  # Clean up after processing
        
        if total_chunks == 0:
            return 0.0
        
        return speech_chunks / total_chunks


class EmbeddingProcessor:
    """
    Voice embedding extraction and comparison.
    
    Extracts speaker embeddings from audio and computes similarity scores.
    """
    
    def __init__(self, config: GateConfig, models: Optional[Models] = None):
        """
        Initialize embedding processor.
        
        Args:
            config: Gate configuration  
            models: Models container (uses singleton if not provided)
        """
        self._config = config
        self._models = models or Models.get()
    
    def embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract voice embedding from audio.
        
        Args:
            audio: Audio samples (float32, mono, 16kHz)
            
        Returns:
            256-dim embedding vector, or None if audio too short
        """
        if len(audio) < self._config.sample_rate:
            return None
        
        try:
            return self._models.encoder.embed_utterance(audio)
        except Exception as e:
            log.debug(f"Embedding error: {e}")
            return None
    
    def similarity(self, audio: np.ndarray, reference: np.ndarray) -> Optional[float]:
        """
        Compute cosine similarity between audio and reference embedding.
        
        Args:
            audio: Audio samples
            reference: Reference embedding (256-dim)
            
        Returns:
            Similarity score (0-1), or None if embedding failed
        """
        emb = self.embed(audio)
        if emb is not None:
            return float(np.inner(emb, reference))
        return None
    
    def consistency(self, audio: np.ndarray) -> float:
        """
        Compute internal consistency of audio (split-half correlation).
        
        Used during enrollment to assess voice quality.
        
        Args:
            audio: Audio samples (should be at least 2 seconds)
            
        Returns:
            Consistency score (0-1), 0 if audio too short
        """
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
        """
        Extract embeddings from audio in chunks, mimicking streaming behavior.

        This matches exactly how real-time speaker verification works:
        - Takes embed_window_sec (1.6s) windows
        - Advances by embed_interval_sec (0.2s) steps
        - Overlap = 1.6 - 0.2 = 1.4s (same as streaming)

        When a VADProcessor is provided, windows with insufficient speech
        content are skipped. This dramatically improves enrollment quality
        by filtering out silence/noise windows.

        Args:
            audio: Audio samples (float32, mono, 16kHz)
            chunk_sec: Chunk duration in seconds (default: embed_window_sec = 1.6s)
            step_sec: Step/advance between chunks (default: embed_interval_sec = 0.2s)
            vad: Optional VADProcessor for speech filtering during enrollment
            min_speech_ratio: Minimum speech ratio per window when VAD is provided (0-1)

        Returns:
            Tuple of (median_embedding, consistency_score, chunk_similarities):
            - median_embedding: Median of all chunk embeddings (256-dim), or None if no valid chunks
            - consistency_score: Median of pairwise similarities (realistic baseline)
            - chunk_similarities: All pairwise similarity scores for analysis
        """
        # Use config defaults matching streaming behavior
        chunk_samples = int((chunk_sec or self._config.embed_window_sec) * self._config.sample_rate)
        step_samples = int((step_sec or self._config.embed_interval_sec) * self._config.sample_rate)

        # Minimum audio length check
        if len(audio) < chunk_samples:
            log.warning(f"Audio too short for chunked embedding: {len(audio)} < {chunk_samples}")
            return None, 0.0, []

        # Pre-compute speech mask if VAD is provided
        speech_mask: Optional[List[bool]] = None
        if vad is not None:
            speech_mask = vad.speech_mask(audio, chunk_samples, step_samples, min_speech_ratio)

        # Extract embeddings for each chunk
        embeddings: List[np.ndarray] = []
        offset = 0
        window_idx = 0
        skipped = 0

        while offset + chunk_samples <= len(audio):
            # Skip windows with insufficient speech
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
            log.debug(f"VAD-filtered enrollment: skipped {skipped} low-speech windows")
        
        if len(embeddings) < 2:
            log.warning(f"Not enough valid chunks for enrollment: {len(embeddings)}")
            if len(embeddings) == 1:
                return embeddings[0], 0.0, []
            return None, 0.0, []
        
        log.debug(f"Extracted {len(embeddings)} chunk embeddings from {len(audio)/self._config.sample_rate:.1f}s audio")
        
        # Compute pairwise similarities between all chunk embeddings
        # This represents how well chunks match each other (intra-speaker consistency)
        similarities: List[float] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.inner(embeddings[i], embeddings[j]))
                similarities.append(sim)
        
        # Compute median embedding (element-wise median across all chunks)
        # This is more robust than mean - outlier chunks don't skew the result
        stacked = np.stack(embeddings, axis=0)  # Shape: (n_chunks, 256)
        median_embedding = np.median(stacked, axis=0).astype(np.float32)
        
        # Normalize the median embedding (cosine similarity assumes unit vectors)
        norm = np.linalg.norm(median_embedding)
        if norm > 0:
            median_embedding = median_embedding / norm
        
        # Consistency score = median of pairwise similarities
        # This represents realistic chunk-to-chunk matching in streaming
        consistency_score = float(np.median(similarities)) if similarities else 0.0
        
        log.debug(
            f"Chunked enrollment: {len(embeddings)} chunks, "
            f"consistency={consistency_score:.3f}, "
            f"sim_range=[{min(similarities):.3f}, {max(similarities):.3f}]"
        )
        
        return median_embedding, consistency_score, similarities
