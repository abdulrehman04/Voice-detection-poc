"""
Event system for Speaker Gate SDK.

Provides a clean, typed event API for integration with any application.
Events are emitted during audio processing to notify about state changes,
speech detection, and completed segments.

Example:
    gate = SpeakerGate(embedding)
    
    @gate.on(Event.SEGMENT_COMPLETE)
    def handle_segment(event: SegmentEvent):
        audio = event.audio  # numpy array
        duration = event.duration_sec
        save_audio(audio)
    
    @gate.on(Event.STATE_CHANGED)
    def handle_state(event: StateChangeEvent):
        print(f"State: {event.previous} -> {event.current}")
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional
import numpy as np


class Event(Enum):
    """Event types emitted by SpeakerGate."""
    
    # Speech detection
    SPEECH_STARTED = auto()      # VAD detected speech start
    SPEECH_ENDED = auto()        # VAD detected speech end
    
    # Speaker verification
    USER_STARTED = auto()        # Confirmed as enrolled user
    USER_ENDED = auto()          # User speech segment ended
    OTHER_DETECTED = auto()      # Detected non-enrolled speaker
    
    # State machine
    STATE_CHANGED = auto()       # Any state transition
    
    # Segments
    SEGMENT_COMPLETE = auto()    # Verified user segment ready
    
    # Similarity updates (optional, for debugging/UI)
    SIMILARITY_UPDATE = auto()   # New similarity score computed


@dataclass
class EventData:
    """Base class for event data."""
    timestamp: float  # Unix timestamp in seconds
    
    
@dataclass
class SpeechEvent(EventData):
    """Event data for speech detection events."""
    vad_probability: float = 0.0


@dataclass
class StateChangeEvent(EventData):
    """Event data for state transitions."""
    previous: str = ""
    current: str = ""


@dataclass
class SimilarityEvent(EventData):
    """Event data for similarity updates."""
    similarity: float = 0.0
    threshold: float = 0.0
    is_match: bool = False


@dataclass
class SegmentEvent(EventData):
    """Event data for completed segments."""
    audio: Optional[np.ndarray] = None
    duration_sec: float = 0.0
    samples: int = 0
    # Timing breakdown
    vad_started_at: float = 0.0      # When VAD first detected speech
    user_confirmed_at: float = 0.0   # When user identity confirmed (0 if auto-approved)
    speech_ended_at: float = 0.0     # When speech ended


# Type alias for event handlers
EventHandler = Callable[[EventData], None]


class EventEmitter:
    """
    Simple event emitter for the Speaker Gate.
    
    Supports multiple handlers per event type, handler removal,
    and one-time handlers.
    """
    
    def __init__(self):
        self._handlers: dict[Event, list[EventHandler]] = {e: [] for e in Event}
        self._once_handlers: dict[Event, list[EventHandler]] = {e: [] for e in Event}
    
    def on(self, event: Event, handler: EventHandler) -> Callable[[], None]:
        """
        Register an event handler.
        
        Args:
            event: Event type to listen for
            handler: Callback function receiving EventData
            
        Returns:
            Unsubscribe function to remove the handler
        """
        self._handlers[event].append(handler)
        
        def unsubscribe():
            if handler in self._handlers[event]:
                self._handlers[event].remove(handler)
        
        return unsubscribe
    
    def once(self, event: Event, handler: EventHandler):
        """Register a one-time event handler (auto-removes after first call)."""
        self._once_handlers[event].append(handler)
    
    def off(self, event: Event, handler: Optional[EventHandler] = None):
        """
        Remove event handler(s).
        
        Args:
            event: Event type
            handler: Specific handler to remove, or None to remove all
        """
        if handler is None:
            self._handlers[event].clear()
            self._once_handlers[event].clear()
        else:
            if handler in self._handlers[event]:
                self._handlers[event].remove(handler)
            if handler in self._once_handlers[event]:
                self._once_handlers[event].remove(handler)
    
    def emit(self, event: Event, data: EventData):
        """
        Emit an event to all registered handlers.
        
        Args:
            event: Event type
            data: Event data to pass to handlers
        """
        # Regular handlers
        for handler in self._handlers[event]:
            try:
                handler(data)
            except Exception:
                pass  # Don't let handler errors break processing
        
        # One-time handlers
        once = self._once_handlers[event]
        self._once_handlers[event] = []
        for handler in once:
            try:
                handler(data)
            except Exception:
                pass
    
    def clear(self):
        """Remove all handlers."""
        for event in Event:
            self._handlers[event].clear()
            self._once_handlers[event].clear()
