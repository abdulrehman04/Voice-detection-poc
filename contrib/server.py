"""WebSocket server for POC testing with browser client."""
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
    StatsMsg, EnrollCompleteMsg, SegmentCompleteMsg, ErrorMsg,
    ProfileInfoMsg, ProfileUpdateMsg,
)

log = logging.getLogger(__name__)

# Enrollment validation
MIN_ENROLLMENT_DURATION_SEC = 15.0
MAX_ENROLLMENT_DURATION_SEC = 60.0
MIN_CONSISTENCY_SCORE = 0.50
MIN_SPEECH_RATIO = 0.30
MIN_SESSION_SIMILARITY = 0.55

# Adaptive enrollment
ADAPTIVE_MIN_AUDIO_SEC = 20.0
ADAPTIVE_MAX_SESSIONS = 6
ADAPTIVE_MIN_SEGMENT_SEC = 5.0
ADAPTIVE_NOVELTY_THRESHOLD = 0.80
ADAPTIVE_COHERENCE_THRESHOLD = 0.70


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

    # Adaptive enrollment
    profile_id: Optional[str] = None
    ws: Optional[object] = None
    verified_audio: list = field(default_factory=list)
    verified_audio_sec: float = 0.0
    verified_audio_embedding: Optional[np.ndarray] = None

    # Debug recording
    recorder: Optional[SessionRecorder] = None
    
    def close(self):
        if self.recorder:
            self.recorder.close()


class WSServer:
    """WebSocket server for POC testing."""

    def __init__(
        self,
        settings: Optional[ContribSettings] = None,
        profile_store: Optional[ProfileStore] = None,
        models: Optional[Models] = None,
    ):
        self._settings = settings or get_settings()
        self._models = models or Models.get()
        self._store = profile_store or FileProfileStore(self._settings.profiles_dir)
        
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._gate_config = self._settings.to_gate_config()
        self._emb = EmbeddingProcessor(self._gate_config, self._models)
        self._vad = VADProcessor(self._gate_config, self._models)
        self._debug_writer = AsyncAudioWriter() if self._settings.debug else None
    
    async def handler(self, websocket):
        session = Session(
            id=str(id(websocket)),
            ws=websocket,
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
        elif msg_type == MsgType.CLEAR_PROFILE:
            await self._handle_clear_profile(ws, session, data)
    
    async def _handle_hello(self, ws, session: Session, data: dict):
        profile_id = data.get("profile_id")
        vad_only = data.get("vad_only", False)
        
        if not profile_id and not vad_only:
            await ws.send(ErrorMsg("Missing profile_id").json())
            return
        
        if vad_only:
            session.gate = SpeakerGate(
                profile_embedding=None,
                config=self._settings.to_gate_config(),
                models=self._models,
            )
            log.info(f"VAD-only mode enabled (no speaker verification)")
            return
        
        profile = self._store.load(profile_id)
        if profile:
            session.profile_id = profile_id
            session.gate = SpeakerGate(
                profile_embedding=profile.embedding_array,
                threshold=profile.threshold,
                config=self._settings.to_gate_config(),
                models=self._models,
            )
            refs = profile.session_embedding_arrays
            if len(refs) > 1:
                session.gate.set_profile(
                    profile.embedding_array,
                    threshold=profile.threshold,
                    session_embeddings=refs,
                    session_thresholds=profile.session_thresholds,
                )
                log.info(
                    f"Loaded profile {profile_id}, threshold={profile.threshold:.3f}, "
                    f"sessions={len(refs)}, per_session={profile.session_thresholds}"
                )
            else:
                log.info(f"Loaded profile {profile_id}, threshold={profile.threshold:.3f}")
            num_sessions = len(profile.session_embeddings or [profile.embedding])
            await ws.send(ProfileInfoMsg(
                profile_id=profile_id,
                sessions=num_sessions,
                session_thresholds=profile.session_thresholds or [profile.threshold],
                overall_threshold=profile.threshold,
            ).json())
        else:
            session.gate = SpeakerGate(
                profile_embedding=None,
                config=self._settings.to_gate_config(),
                models=self._models,
            )
            log.warning(f"Profile not found: {profile_id}, falling back to VAD-only mode")
    
    async def _handle_clear_profile(self, ws, session: Session, data: dict):
        profile_id = data.get("profile_id")
        if not profile_id:
            await ws.send(ErrorMsg("Missing profile_id").json())
            return

        deleted = self._store.delete(profile_id)
        if deleted:
            log.info(f"Profile cleared: {profile_id}")
        else:
            log.warning(f"Profile not found for clearing: {profile_id}")

        if session.gate:
            session.gate.set_profile(None)
        session.profile_id = None
        session.verified_audio = []
        session.verified_audio_sec = 0.0

        await ws.send(ProfileInfoMsg(
            profile_id=profile_id,
            sessions=0,
            session_thresholds=[],
            overall_threshold=0.0,
        ).json())

    async def _handle_enroll_start(self, ws, session: Session, data: dict):
        session.enrolling = True
        session.enroll_id = data.get("profile_id")
        session.enroll_audio = []
        log.info(f"Enrollment started: {session.enroll_id}")
    
    async def _handle_enroll_end(self, ws, session: Session):
        session.enrolling = False
        
        if not session.enroll_audio or not session.enroll_id:
            await ws.send(ErrorMsg("No enrollment data").json())
            return
        
        audio = np.concatenate(session.enroll_audio)

        score, error = await asyncio.get_running_loop().run_in_executor(
            self._executor, self._enroll_sync, session.enroll_id, audio
        )
        
        if error:
            await ws.send(ErrorMsg(error).json())
        else:
            await ws.send(EnrollCompleteMsg(session.enroll_id, score).json())
    
    def _enroll_sync(self, profile_id: str, audio: np.ndarray) -> tuple[float, str]:
        """Synchronous enrollment with validation. Returns (consistency, error)."""
        try:
            duration_sec = len(audio) / self._settings.sample_rate
            
            if duration_sec < MIN_ENROLLMENT_DURATION_SEC:
                return 0.0, f"Audio too short. Minimum {MIN_ENROLLMENT_DURATION_SEC}s required, got {duration_sec:.1f}s."
            if duration_sec > MAX_ENROLLMENT_DURATION_SEC:
                return 0.0, f"Audio too long. Maximum {MAX_ENROLLMENT_DURATION_SEC}s allowed, got {duration_sec:.1f}s."
            
            speech_ratio = self._vad.speech_ratio(audio)
            if speech_ratio < MIN_SPEECH_RATIO:
                return 0.0, (
                    f"Insufficient speech detected ({speech_ratio:.0%}). "
                    f"Please speak clearly for the entire recording. "
                    f"Minimum {MIN_SPEECH_RATIO:.0%} speech required."
                )
            
            embedding, consistency, chunk_sims = self._emb.embed_chunked(audio, vad=self._vad)
            if embedding is None:
                return 0.0, "Failed to extract voice embedding from audio."
            
            if consistency < MIN_CONSISTENCY_SCORE:
                return 0.0, (
                    f"Audio quality too low. Consistency: {consistency:.2f} "
                    f"(min: {MIN_CONSISTENCY_SCORE}). "
                    f"Please try again with clearer audio."
                )
            
            threshold = calculate_threshold(consistency)

            existing = self._store.load(profile_id)
            session_embeddings: list[list[float]] = []
            session_thresholds: list[float] = []

            if existing:
                if existing.session_embeddings:
                    session_embeddings = list(existing.session_embeddings)
                    if existing.session_thresholds and len(existing.session_thresholds) == len(session_embeddings):
                        session_thresholds = list(existing.session_thresholds)
                    else:
                        session_thresholds = [existing.threshold] * len(session_embeddings)
                else:
                    session_embeddings = [existing.embedding]
                    session_thresholds = [existing.threshold]
                log.info(f"Merging with existing profile ({len(session_embeddings)} previous sessions)")

                # Cross-session validation: new voice must match existing profile
                max_cross_sim = max(
                    float(np.inner(embedding, np.array(e, dtype=np.float32)))
                    for e in session_embeddings
                )
                if max_cross_sim < MIN_SESSION_SIMILARITY:
                    return 0.0, (
                        f"Voice doesn't match existing profile. "
                        f"Best similarity: {max_cross_sim:.2f} (min: {MIN_SESSION_SIMILARITY}). "
                        f"Multi-session enrollment is for the same speaker only."
                    )
                log.info(f"Cross-session check passed: best_sim={max_cross_sim:.3f}")

            session_embeddings.append(embedding.tolist())
            session_thresholds.append(threshold)

            # Merge all sessions into a single primary embedding
            if len(session_embeddings) > 1:
                all_embs = np.array(session_embeddings, dtype=np.float32)
                merged = np.median(all_embs, axis=0).astype(np.float32)
                norm = np.linalg.norm(merged)
                if norm > 0:
                    merged = merged / norm
                primary_embedding = merged.tolist()

                threshold = min(session_thresholds)
                log.info(
                    f"Multi-session: {len(session_embeddings)} sessions, "
                    f"per-session thresholds={[f'{t:.3f}' for t in session_thresholds]}"
                )
            else:
                primary_embedding = embedding.tolist()

            profile = Profile(
                profile_id=profile_id,
                embedding=primary_embedding,
                threshold=threshold,
                consistency_score=consistency,
                duration_sec=duration_sec,
                session_embeddings=session_embeddings,
                session_thresholds=session_thresholds,
            )
            self._store.save(profile)
            log.info(
                f"Enrolled {profile_id}: consistency={consistency:.3f}, "
                f"speech_ratio={speech_ratio:.0%}, duration={duration_sec:.1f}s, "
                f"sessions={len(session_embeddings)}"
            )
            return consistency, ""
            
        except Exception as e:
            log.error(f"Enrollment failed: {e}")
            return 0.0, f"Enrollment failed: {e}"
    
    async def _handle_audio(self, ws, session: Session, chunk: np.ndarray):
        if session.enrolling:
            session.enroll_audio.append(chunk)
            return
        
        if session.recorder:
            session.recorder.write(chunk)
        
        if not session.gate:
            return
        
        await self._process_audio(ws, session, chunk)
    
    async def _process_audio(self, ws, session: Session, audio: np.ndarray):
        if session.gate is None:
            return
        
        chunk_size = self._settings.chunk_size
        offset = 0
        
        while offset + chunk_size <= len(audio):
            chunk = audio[offset:offset + chunk_size]
            await self._process_chunk(ws, session, chunk)
            offset += chunk_size
    
    async def _process_chunk(self, ws, session: Session, chunk: np.ndarray):
        gate = session.gate
        if gate is None:
            return
        
        prev_state = gate.state.name
        segment_event = None
        
        def on_segment(event):
            nonlocal segment_event
            segment_event = event
        
        unsubscribe = gate.on(Event.SEGMENT_COMPLETE, on_segment)
        
        try:
            gate.feed(chunk)
        finally:
            unsubscribe()
        
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
                best_session=gate.last_best_session,
                total_sessions=gate.num_sessions,
                similarity=round(gate.similarity, 4),
                threshold_used=round(gate.threshold, 4),
            ).json())
            
            await ws.send(segment_event.audio.tobytes())

            # Adaptive enrollment: accumulate verified audio
            if session.profile_id and not session.enrolling and duration >= ADAPTIVE_MIN_SEGMENT_SEC:
                # Coherence check: reset if voice pattern changed significantly
                seg_emb = self._emb.embed(segment_event.audio)
                if seg_emb is not None and session.verified_audio_embedding is not None:
                    coherence = float(np.inner(seg_emb, session.verified_audio_embedding))
                    if coherence < ADAPTIVE_COHERENCE_THRESHOLD:
                        log.info(
                            f"Adaptive: voice pattern changed (coherence={coherence:.3f} "
                            f"< {ADAPTIVE_COHERENCE_THRESHOLD}), resetting accumulator"
                        )
                        session.verified_audio = []
                        session.verified_audio_sec = 0.0
                        session.verified_audio_embedding = None

                # Update running embedding (weighted average favoring accumulated)
                if seg_emb is not None:
                    if session.verified_audio_embedding is None:
                        session.verified_audio_embedding = seg_emb
                    else:
                        combined = session.verified_audio_embedding * 0.7 + seg_emb * 0.3
                        norm = np.linalg.norm(combined)
                        if norm > 0:
                            session.verified_audio_embedding = combined / norm

                session.verified_audio.append(segment_event.audio)
                session.verified_audio_sec += duration
                log.debug(
                    f"Adaptive: accumulated {session.verified_audio_sec:.1f}s "
                    f"verified audio ({len(session.verified_audio)} segments)"
                )
                if session.verified_audio_sec >= ADAPTIVE_MIN_AUDIO_SEC:
                    await self._try_adaptive_enroll(session)

            if self._debug_writer:
                self._debug_writer.write(
                    f"segment_{session.id}_{ts}.wav",
                    segment_event.audio
                )
        
        if self._settings.stats_enabled:
            now_ms = time.time() * 1000
            current_state = gate.state.name
            
            should_send = (
                now_ms - session.last_stats_time >= self._settings.stats_interval_ms or
                current_state != session.last_state
            )
            
            if should_send:
                stats = StatsMsg(
                    vad_prob=gate.vad_probability,
                    similarity=gate.similarity,
                    state=current_state,
                    vad_only=gate.vad_only,
                    timeout=gate.adaptive_silence_timeout,
                    timeout_min=gate.config.silence_timeout_min,
                    timeout_max=gate.config.silence_timeout_max,
                )
                await ws.send(stats.json())
                session.last_stats_time = now_ms
                session.last_state = current_state
    
    async def _try_adaptive_enroll(self, session: Session):
        if not session.profile_id or not session.gate:
            return

        profile = self._store.load(session.profile_id)
        if not profile:
            return

        audio = np.concatenate(session.verified_audio)
        result = await asyncio.get_running_loop().run_in_executor(
            self._executor, self._adaptive_enroll_sync, session.profile_id, audio, profile
        )

        # Reset accumulator regardless of result
        session.verified_audio = []
        session.verified_audio_sec = 0.0
        session.verified_audio_embedding = None

        if result:
            updated = self._store.load(session.profile_id)
            if updated:
                refs = updated.session_embedding_arrays
                session.gate.set_profile(
                    updated.embedding_array,
                    threshold=updated.threshold,
                    session_embeddings=refs,
                    session_thresholds=updated.session_thresholds,
                )
                log.info(
                    f"Adaptive: gate updated with {len(refs)} sessions, "
                    f"per_session={[f'{t:.3f}' for t in updated.session_thresholds]}"
                )
                if session.ws:
                    try:
                        await session.ws.send(ProfileUpdateMsg(
                            profile_id=session.profile_id,
                            sessions=len(refs),
                            session_thresholds=updated.session_thresholds,
                            overall_threshold=updated.threshold,
                            reason="adaptive",
                        ).json())
                    except Exception:
                        pass

    def _adaptive_enroll_sync(
        self, profile_id: str, audio: np.ndarray, profile: Profile
    ) -> bool:
        try:
            duration_sec = len(audio) / self._gate_config.sample_rate
            log.info(f"Adaptive enrollment: processing {duration_sec:.1f}s of verified audio")

            embedding, consistency, speech_ratio = self._emb.embed_chunked(
                audio, vad=self._vad, min_speech_ratio=0.5
            )

            if embedding is None or consistency < MIN_CONSISTENCY_SCORE:
                log.info(f"Adaptive: skipped, low consistency={consistency:.3f}")
                return False

            # Cross-session validation
            session_embeddings = list(profile.session_embeddings or [])
            if session_embeddings:
                max_cross_sim = max(
                    float(np.inner(embedding, np.array(e, dtype=np.float32)))
                    for e in session_embeddings
                )
                if max_cross_sim < MIN_SESSION_SIMILARITY:
                    log.warning(
                        f"Adaptive: rejected, cross-session sim={max_cross_sim:.3f} "
                        f"< {MIN_SESSION_SIMILARITY}"
                    )
                    return False

                # Novelty check: skip if voice already well-covered by existing session
                if max_cross_sim > ADAPTIVE_NOVELTY_THRESHOLD:
                    log.info(
                        f"Adaptive: skipped, redundant (best_sim={max_cross_sim:.3f} "
                        f"> {ADAPTIVE_NOVELTY_THRESHOLD}). Voice already well-covered."
                    )
                    return False

                log.info(
                    f"Adaptive: novel voice pattern detected "
                    f"(best_sim={max_cross_sim:.3f}, range={MIN_SESSION_SIMILARITY}-{ADAPTIVE_NOVELTY_THRESHOLD})"
                )

            threshold = calculate_threshold(consistency)
            session_thresholds = list(profile.session_thresholds or [])

            # Retire the most similar existing session to make room
            if len(session_embeddings) >= ADAPTIVE_MAX_SESSIONS:
                cross_sims = [
                    float(np.inner(embedding, np.array(e, dtype=np.float32)))
                    for e in session_embeddings
                ]
                retire_idx = max(range(len(cross_sims)), key=lambda i: cross_sims[i])
                retired_sim = cross_sims[retire_idx]
                log.info(
                    f"Adaptive: retiring S{retire_idx+1} (sim={retired_sim:.3f}) "
                    f"to make room for new session"
                )
                session_embeddings.pop(retire_idx)
                session_thresholds.pop(retire_idx)

            session_embeddings.append(embedding.tolist())
            session_thresholds.append(threshold)

            all_embs = np.array(session_embeddings, dtype=np.float32)
            merged = np.median(all_embs, axis=0).astype(np.float32)
            norm = np.linalg.norm(merged)
            if norm > 0:
                merged = merged / norm

            updated = Profile(
                profile_id=profile_id,
                embedding=merged.tolist(),
                threshold=min(session_thresholds),
                consistency_score=consistency,
                duration_sec=duration_sec,
                session_embeddings=session_embeddings,
                session_thresholds=session_thresholds,
            )
            self._store.save(updated)
            log.info(
                f"Adaptive enrolled {profile_id}: consistency={consistency:.3f}, "
                f"sessions={len(session_embeddings)}, "
                f"per_session={[f'{t:.3f}' for t in session_thresholds]}"
            )
            return True

        except Exception as e:
            log.error(f"Adaptive enrollment failed: {e}", exc_info=True)
            return False

    async def run(self):
        log.info(f"Starting WS server on {self._settings.host}:{self._settings.port}")
        
        async with websockets.serve(
            self.handler,
            self._settings.host,
            self._settings.port
        ):
            await asyncio.Future()  # Run forever
    
    def start(self):
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
