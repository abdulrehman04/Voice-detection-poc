"""
Speaker Gate SDK - Core Module

Production-ready speaker verification with event-driven architecture.
Zero dependencies on testing/debugging utilities.

Quick Start:
    from poc.server.core import SpeakerGate, GateConfig, Event
    
    # Create gate with profile embedding
    gate = SpeakerGate(profile_embedding, config=GateConfig())
    
    # Subscribe to events
    gate.on(Event.USER_STARTED, lambda e: print("User speaking"))
    gate.on(Event.SEGMENT_COMPLETE, lambda e: process_audio(e.audio))
    
    # Feed audio chunks
    for chunk in audio_stream:
        gate.feed(chunk)
"""
from .config import GateConfig
from .events import Event, EventData, SegmentEvent, StateChangeEvent, SpeechEvent
from .gate import SpeakerGate, GateState
from .models import Models, ModelLoader
from .profiles import Profile, ProfileStore, FileProfileStore
from .audio import VADProcessor, EmbeddingProcessor, AdaptiveSilenceDetector
from .exceptions import SpeakerGateError, ProfileNotFoundError, ProfileInvalidError

__version__ = "3.0.0"

__all__ = [
    # Main API
    "SpeakerGate",
    "GateConfig",
    "GateState",
    # Events
    "Event",
    "EventData",
    "SegmentEvent",
    "StateChangeEvent",
    "SpeechEvent",
    # Models
    "Models",
    "ModelLoader",
    # Profiles
    "Profile",
    "ProfileStore",
    "FileProfileStore",
    # Audio processors
    "VADProcessor",
    "EmbeddingProcessor",
    "AdaptiveSilenceDetector",
    # Exceptions
    "SpeakerGateError",
    "ProfileNotFoundError",
    "ProfileInvalidError",
]
