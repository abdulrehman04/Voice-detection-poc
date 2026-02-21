"""
Speaker Gate Server - Production-Ready SDK with Event-Driven Architecture.

Structure:
    server/
    ├── core/           # Production SDK (no external deps except ML models)
    │   ├── config.py   # GateConfig dataclass
    │   ├── events.py   # Event system (Event, EventData, etc.)
    │   ├── models.py   # ML model loaders
    │   ├── audio.py    # VAD & embedding processors
    │   ├── profiles.py # Profile model & storage interface
    │   └── gate.py     # SpeakerGate state machine
    │
    └── contrib/        # POC/Testing utilities (optional)
        ├── config.py   # Pydantic settings with env vars
        ├── debug.py    # Audio recording utilities
        ├── protocol.py # WebSocket message types
        └── server.py   # WebSocket server for testing

Quick Start (Production):
    from poc.server.core import SpeakerGate, GateConfig, Event
    
    # Create gate with profile embedding
    gate = SpeakerGate(profile_embedding, threshold=0.72)
    
    # Subscribe to events
    gate.on(Event.SEGMENT_COMPLETE, lambda e: process_audio(e.audio))
    gate.on(Event.USER_STARTED, lambda e: show_indicator())
    
    # Feed audio from your source
    for chunk in audio_stream:
        gate.feed(chunk)

Quick Start (POC Testing):
    python -m poc.server serve --port 8765 --debug
"""
# Re-export core API for convenience
from .core import (
    # Main classes
    SpeakerGate,
    GateConfig,
    GateState,
    # Events
    Event,
    EventData,
    SegmentEvent,
    StateChangeEvent,
    SpeechEvent,
    # Models
    Models,
    ModelLoader,
    # Profiles
    Profile,
    ProfileStore,
    FileProfileStore,
    # Audio
    VADProcessor,
    EmbeddingProcessor,
    # Exceptions
    SpeakerGateError,
    ProfileNotFoundError,
    ProfileInvalidError,
)

__version__ = "3.0.0"

__all__ = [
    # Core
    "SpeakerGate",
    "GateConfig",
    "GateState",
    "Event",
    "EventData",
    "SegmentEvent",
    "StateChangeEvent",
    "SpeechEvent",
    "Models",
    "ModelLoader",
    "Profile",
    "ProfileStore",
    "FileProfileStore",
    "VADProcessor",
    "EmbeddingProcessor",
    "SpeakerGateError",
    "ProfileNotFoundError",
    "ProfileInvalidError",
]
