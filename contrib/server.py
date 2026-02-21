"""
WebSocket server for POC testing.

This server is for development and testing with the browser client.
For production, use the core SDK directly in your application.

The server wraps the core SpeakerGate and exposes it over WebSocket
with JSON message protocol and binary audio streaming.
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import websockets

from ..core import (
    SpeakerGate, GateConfig, Event, Models,
    Profile, FileProfileStore, ProfileStore,
    EmbeddingProcessor, VADProcessor,
)
from ..core.profiles import calculate_threshold
from .config import ContribSettings, get_settings
from .debug import SessionRecorder, AsyncAudioWriter
from .protocol import (
    MsgType, parse_message,
    StatsMsg, EnrollCompleteMsg, SegmentCompleteMsg, ErrorMsg
)

log = logging.getLogger(__name__)

# Enrollment validation constants
MIN_ENROLLMENT_DURATION_SEC = 15.0
MAX_ENROLLMENT_DURATION_SEC = 60.0
MIN_CONSISTENCY_SCORE = 0.50
MIN_SPEECH_RATIO = 0.30  # At least 30% of audio must contain speech


@dataclass
class Session:
    """WebSocket session state."""
    id: str
    gate: Optional[SpeakerGate] = None
    enrolling: bool = False
    enroll_id: Optional[str] = None
    enroll_audio: list = field(default_factory=list)
    
    # Stats tracking (POC only)
    last_stats_time: float = 0.0
    last_state: Optional[str] = None
    
    # Debug recording
    recorder: Optional[SessionRecorder] = None
    
    def close(self):
        if self.recorder:
            self.recorder.close()


class WSServer:
    """
    WebSocket server for POC testing.
    
    Handles:
    - Profile enrollment via audio streaming
    - Real-time speaker verification
    - Stats broadcasting (optional)
    - Debug audio recording (optional)
    """
    
    def __init__(
        self,
        settings: Optional[ContribSettings] = None,
        profile_store: Optional[ProfileStore] = None,
        models: Optional[Models] = None,
    ):
        """
        Initialize server.
        
        Args:
            settings: Server configuration
            profile_store: Profile storage (default: FileProfileStore)
            models: ML models (default: shared singleton)
        """
        self._settings = settings or get_settings()
        self._models = models or Models.get()
        self._store = profile_store or FileProfileStore(self._settings.profiles_dir)
        
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._gate_config = self._settings.to_gate_config()
        self._emb = EmbeddingProcessor(self._gate_config, self._models)
        self._vad = VADProcessor(self._gate_config, self._models)
        self._debug_writer = AsyncAudioWriter() if self._settings.debug else None
    
    async def handler(self, websocket):
        """Handle WebSocket connection."""
        session = Session(
            id=str(id(websocket)),
            recorder=SessionRecorder(str(id(websocket))) if self._settings.debug else None
        )
        log.info(f"Connected: {session.id}")
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_json(websocket, session, message)
                else:
                    await self._handle_audio(websocket, session, 
                                            np.frombuffer(message, dtype=np.float32))
        except websockets.exceptions.ConnectionClosed:
            log.info(f"Disconnected: {session.id}")
        except Exception as e:
            log.error(f"Handler error: {e}", exc_info=True)
        finally:
            session.close()
    
    async def _handle_json(self, ws, session: Session, raw: str):
        """Handle JSON control message."""
        try:
            msg_type, data = parse_message(raw)
        except (ValueError, KeyError) as e:
            await ws.send(ErrorMsg(f"Invalid message: {e}").json())
            return
        
        if msg_type == MsgType.HELLO:
            await self._handle_hello(ws, session, data)
        elif msg_type == MsgType.ENROLL:
            await self._handle_enroll_start(ws, session, data)
        elif msg_type == MsgType.END_ENROLL:
            await self._handle_enroll_end(ws, session)
    
    async def _handle_hello(self, ws, session: Session, data: dict):
        """Handle HELLO message - load profile and create gate."""
        profile_id = data.get("profile_id")
        vad_only = data.get("vad_only", False)
        
        if not profile_id and not vad_only:
            await ws.send(ErrorMsg("Missing profile_id").json())
            return
        
        # VAD-only mode: create gate without profile
        if vad_only:
            session.gate = SpeakerGate(
                profile_embedding=None,
                config=self._settings.to_gate_config(),
                models=self._models,
            )
            log.info(f"VAD-only mode enabled (no speaker verification)")
            return
        
        # Normal mode: load profile
        profile = self._store.load(profile_id)
        if profile:
            session.gate = SpeakerGate(
                profile_embedding=profile.embedding_array,
                threshold=profile.threshold,
                config=self._settings.to_gate_config(),
                models=self._models,
            )
            log.info(f"Loaded profile {profile_id}, threshold={profile.threshold:.3f}")
        else:
            # Profile not found - fall back to VAD-only mode
            session.gate = SpeakerGate(
                profile_embedding=None,
                config=self._settings.to_gate_config(),
                models=self._models,
            )
            log.warning(f"Profile not found: {profile_id}, falling back to VAD-only mode")
    
    async def _handle_enroll_start(self, ws, session: Session, data: dict):
        """Handle ENROLL message - start enrollment."""
        session.enrolling = True
        session.enroll_id = data.get("profile_id")
        session.enroll_audio = []
        log.info(f"Enrollment started: {session.enroll_id}")
    
    async def _handle_enroll_end(self, ws, session: Session):
        """Handle END_ENROLL message - complete enrollment."""
        session.enrolling = False
        
        if not session.enroll_audio or not session.enroll_id:
            await ws.send(ErrorMsg("No enrollment data").json())
            return
        
        audio = np.concatenate(session.enroll_audio)
        
        # Run enrollment in thread pool
        score, error = await asyncio.get_running_loop().run_in_executor(
            self._executor, self._enroll_sync, session.enroll_id, audio
        )
        
        if error:
            await ws.send(ErrorMsg(error).json())
        else:
            await ws.send(EnrollCompleteMsg(session.enroll_id, score).json())
    
    def _enroll_sync(self, profile_id: str, audio: np.ndarray) -> tuple[float, str]:
        """
        Synchronous enrollment with validation (runs in thread pool).
        
        Uses embed_chunked for streaming-consistent embeddings and
        speech_ratio for VAD validation.
        
        Returns:
            Tuple of (consistency_score, error_message).
            If successful, error_message is empty string.
        """
        try:
            duration_sec = len(audio) / self._settings.sample_rate
            
            # Validate duration
            if duration_sec < MIN_ENROLLMENT_DURATION_SEC:
                return 0.0, f"Audio too short. Minimum {MIN_ENROLLMENT_DURATION_SEC}s required, got {duration_sec:.1f}s."
            if duration_sec > MAX_ENROLLMENT_DURATION_SEC:
                return 0.0, f"Audio too long. Maximum {MAX_ENROLLMENT_DURATION_SEC}s allowed, got {duration_sec:.1f}s."
            
            # Validate speech content using VAD
            speech_ratio = self._vad.speech_ratio(audio)
            if speech_ratio < MIN_SPEECH_RATIO:
                return 0.0, (
                    f"Insufficient speech detected ({speech_ratio:.0%}). "
                    f"Please speak clearly for the entire recording. "
                    f"Minimum {MIN_SPEECH_RATIO:.0%} speech required."
                )
            
            # Extract embedding using chunked approach (matches streaming behavior)
            # Uses 1.6s windows with 0.2s step - same as real-time verification
            # Pass VAD to filter out silence/noise windows for cleaner enrollment
            embedding, consistency, chunk_sims = self._emb.embed_chunked(audio, vad=self._vad)
            if embedding is None:
                return 0.0, "Failed to extract voice embedding from audio."
            
            # Validate consistency (based on chunk-to-chunk similarity)
            if consistency < MIN_CONSISTENCY_SCORE:
                return 0.0, (
                    f"Audio quality too low. Consistency: {consistency:.2f} "
                    f"(min: {MIN_CONSISTENCY_SCORE}). "
                    f"Please try again with clearer audio."
                )
            
            threshold = calculate_threshold(consistency)
            
            profile = Profile(
                profile_id=profile_id,
                embedding=embedding.tolist(),
                threshold=threshold,
                consistency_score=consistency,
                duration_sec=duration_sec,
            )
            self._store.save(profile)
            log.info(
                f"Enrolled {profile_id}: consistency={consistency:.3f}, "
                f"speech_ratio={speech_ratio:.0%}, duration={duration_sec:.1f}s"
            )
            return consistency, ""
            
        except Exception as e:
            log.error(f"Enrollment failed: {e}")
            return 0.0, f"Enrollment failed: {e}"
    
    async def _handle_audio(self, ws, session: Session, chunk: np.ndarray):
        """Handle binary audio data."""
        # Enrollment mode
        if session.enrolling:
            session.enroll_audio.append(chunk)
            return
        
        # Debug recording
        if session.recorder:
            session.recorder.write(chunk)
        
        # No gate = no verification
        if not session.gate:
            return
        
        # Process audio through gate
        await self._process_audio(ws, session, chunk)
    
    async def _process_audio(self, ws, session: Session, audio: np.ndarray):
        """Process audio through speaker gate."""
        if session.gate is None:
            return
        
        chunk_size = self._settings.chunk_size
        offset = 0
        
        while offset + chunk_size <= len(audio):
            chunk = audio[offset:offset + chunk_size]
            await self._process_chunk(ws, session, chunk)
            offset += chunk_size
    
    async def _process_chunk(self, ws, session: Session, chunk: np.ndarray):
        """Process single audio chunk."""
        gate = session.gate
        if gate is None:
            return
        
        prev_state = gate.state.name
        
        # We need to capture segment events
        segment_event = None
        
        def on_segment(event):
            nonlocal segment_event
            segment_event = event
        
        # Temporarily subscribe to segment events
        unsubscribe = gate.on(Event.SEGMENT_COMPLETE, on_segment)
        
        try:
            gate.feed(chunk)
        finally:
            unsubscribe()
        
        # Handle segment completion
        if segment_event is not None:
            ts = int(time.time() * 1000)
            duration = segment_event.duration_sec
            chunk_size_kb = segment_event.samples * 4 / 1024  # float32 = 4 bytes
            processing_delay = ts - int(segment_event.speech_ended_at * 1000) \
                if segment_event.speech_ended_at else 0
            
            await ws.send(SegmentCompleteMsg(
                timestamp=ts,
                duration=duration,
                samples=segment_event.samples,
                vad_started_at=int(segment_event.vad_started_at * 1000),
                user_confirmed_at=int(segment_event.user_confirmed_at * 1000),
                speech_ended_at=int(segment_event.speech_ended_at * 1000),
                processing_delay_ms=processing_delay,
                chunk_size_kb=round(chunk_size_kb, 2),
            ).json())
            
            # Send audio binary
            await ws.send(segment_event.audio.tobytes())
            
            # Debug write
            if self._debug_writer:
                self._debug_writer.write(
                    f"segment_{session.id}_{ts}.wav",
                    segment_event.audio
                )
        
        # Stats broadcast (POC only)
        if self._settings.stats_enabled:
            now_ms = time.time() * 1000
            current_state = gate.state.name
            
            should_send = (
                now_ms - session.last_stats_time >= self._settings.stats_interval_ms or
                current_state != session.last_state
            )
            
            if should_send:
                await ws.send(StatsMsg(
                    vad_prob=gate.vad_probability,
                    similarity=gate.similarity,
                    state=current_state,
                    vad_only=gate.vad_only,
                    timeout=gate.adaptive_silence_timeout,
                    timeout_min=gate.config.silence_timeout_min,
                    timeout_max=gate.config.silence_timeout_max,
                ).json())
                session.last_stats_time = now_ms
                session.last_state = current_state
    
    async def run(self):
        """Run the WebSocket server."""
        log.info(f"Starting WS server on {self._settings.host}:{self._settings.port}")
        
        async with websockets.serve(
            self.handler,
            self._settings.host,
            self._settings.port
        ):
            await asyncio.Future()  # Run forever
    
    def start(self):
        """Start server (blocking)."""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            log.info("Server stopped")
        finally:
            if self._debug_writer:
                self._debug_writer.close()


def create_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    debug: bool = False,
    profiles_dir: Optional[str] = None,
    **kwargs
) -> WSServer:
    """
    Factory function to create configured server.
    
    Args:
        host: Server host
        port: Server port  
        debug: Enable debug mode
        profiles_dir: Profile storage directory
        **kwargs: Additional settings
        
    Returns:
        Configured WSServer instance
    """
    from pathlib import Path
    from .config import ContribSettings
    
    settings_kwargs = {
        "host": host,
        "port": port,
        "debug": debug,
        **kwargs
    }
    
    if profiles_dir:
        settings_kwargs["profiles_dir"] = Path(profiles_dir)
    
    settings = ContribSettings(**settings_kwargs)
    return WSServer(settings=settings)
