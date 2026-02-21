# Speaker Gate SDK

Production-ready speaker verification with clean separation of concerns.

## Architecture

```
voice_server/
├── core/               # 🚀 PRODUCTION SDK - Copy this to your project
│   ├── __init__.py     # Public API exports
│   ├── config.py       # GateConfig (plain dataclass, no deps)
│   ├── events.py       # Event system (Event, EventData, handlers)
│   ├── models.py       # ML model loaders (VAD, VoiceEncoder)
│   ├── audio.py        # Audio processors (VAD, Embedding)
│   ├── profiles.py     # Profile model & storage interface
│   ├── gate.py         # SpeakerGate state machine
│   ├── exceptions.py   # Custom exceptions
│   └── requirements.txt
│
└── contrib/            # 🧪 POC/TESTING - Don't deploy to production
    ├── __init__.py
    ├── config.py       # Pydantic settings with env vars
    ├── debug.py        # Audio recording utilities
    ├── protocol.py     # WebSocket message types
    ├── server.py       # WebSocket server
    └── requirements.txt
```

## Quick Start

### Production Integration

Copy the `core/` folder to your project and use the event-driven API:

```python
from speaker_gate.core import SpeakerGate, GateConfig, Event, SegmentEvent

# Load profile embedding from your database
embedding = db.get_user_embedding(user_id)

# Create gate with optional custom config
config = GateConfig(
    sample_rate=16000,
    default_threshold=0.72,
)
gate = SpeakerGate(embedding, config=config)

# Subscribe to events
@gate.on(Event.SEGMENT_COMPLETE)
def on_segment(event: SegmentEvent):
```

### VAD-Only Mode

Run without speaker verification - outputs all detected speech segments:

```python
from speaker_gate.core import SpeakerGate, Event

# Option 1: Start in VAD-only mode (no profile)
gate = SpeakerGate()

# Option 2: Disable verification at runtime
gate = SpeakerGate(profile_embedding=embedding)
gate.verification_enabled = False  # Switch to VAD-only

# Option 3: Dynamic profile switching
gate = SpeakerGate()  # Start VAD-only
gate.set_profile(embedding, threshold=0.72)  # Enable verification
gate.set_profile(None)  # Back to VAD-only

# Check current mode
if gate.vad_only:
    print("Running in VAD-only mode")

# All events still fire normally
@gate.on(Event.SEGMENT_COMPLETE)
def on_segment(event):
    # In VAD-only mode, ALL speech segments are emitted
    process_audio(event.audio)
```

### Event Handlers

```python
# Subscribe to events
@gate.on(Event.SEGMENT_COMPLETE)
def on_segment(event: SegmentEvent):
    """Called when verified user speech segment is complete."""
    audio = event.audio  # numpy array
    duration = event.duration_sec
    
    # Send to your STT service
    transcription = stt_service.transcribe(audio)
    process_command(transcription)

@gate.on(Event.USER_STARTED)
def on_user_started(event):
    """Called when enrolled user starts speaking."""
    show_speaking_indicator()

@gate.on(Event.OTHER_DETECTED)
def on_other(event):
    """Called when non-enrolled speaker detected."""
    log.info("Ignoring non-user speech")

# Feed audio from your streaming source
async for chunk in audio_stream:
    gate.feed(chunk)  # Events fire automatically
```

### POC/Testing Server

Run the WebSocket server for browser-based testing:

```bash
# Start server
python -m poc.server serve --port 8765 --debug

# List profiles
python -m poc.server profiles

# Delete profile
python -m poc.server delete user123

# Show info
python -m poc.server info
```

## Events

| Event | Data Class | Description |
|-------|-----------|-------------|
| `SPEECH_STARTED` | `SpeechEvent` | VAD detected speech start |
| `SPEECH_ENDED` | `SpeechEvent` | VAD detected speech end |
| `USER_STARTED` | `SpeechEvent` | Enrolled user confirmed |
| `USER_ENDED` | `SpeechEvent` | User segment ended |
| `OTHER_DETECTED` | `SpeechEvent` | Non-enrolled speaker |
| `STATE_CHANGED` | `StateChangeEvent` | Any state transition |
| `SEGMENT_COMPLETE` | `SegmentEvent` | Verified audio ready |
| `SIMILARITY_UPDATE` | `SimilarityEvent` | New similarity computed |

### Event Data

```python
@dataclass
class SegmentEvent:
    timestamp: float        # Unix timestamp
    audio: np.ndarray       # Verified audio samples
    duration_sec: float     # Segment duration
    samples: int            # Sample count
    vad_started_at: float   # When VAD detected speech
    user_confirmed_at: float  # When user verified (0 if auto)
    speech_ended_at: float  # When speech ended
```

## Configuration

### Core Config (GateConfig)

```python
from server.core import GateConfig

config = GateConfig(
    # Audio
    sample_rate=16000,
    chunk_size=512,
    
    # VAD
    vad_threshold=0.5,
    silence_timeout=0.6,
    
    # Thresholds
    default_threshold=0.72,
    sim_hysteresis=0.07,
    
    # Matching
    enter_count=2,  # Matches to confirm
    exit_count=2,   # Mismatches to reject
)
```

### Contrib Settings (Environment Variables)

```bash
# All settings can be set via env vars with SPEAKER_GATE_ prefix
export SPEAKER_GATE_PORT=9000
export SPEAKER_GATE_DEBUG=true
export SPEAKER_GATE_DEFAULT_THRESHOLD=0.75
```

## Custom Model Integration

```python
from server.core import SpeakerGate, Models, ModelLoader

class MyModelLoader(ModelLoader):
    def load_vad(self):
        return my_custom_vad()
    
    def load_encoder(self):
        return my_custom_encoder()

models = Models.load(MyModelLoader())
gate = SpeakerGate(embedding, models=models)
```

## Custom Profile Storage

```python
from server.core import ProfileStore, Profile

class RedisProfileStore(ProfileStore):
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def save(self, profile: Profile):
        self.redis.set(f"profile:{profile.profile_id}", profile.to_json())
    
    def load(self, profile_id: str) -> Profile | None:
        data = self.redis.get(f"profile:{profile_id}")
        return Profile.from_json(data) if data else None
    
    def delete(self, profile_id: str) -> bool:
        return self.redis.delete(f"profile:{profile_id}") > 0
    
    def list_ids(self) -> list[str]:
        keys = self.redis.keys("profile:*")
        return [k.split(":")[1] for k in keys]

# Use with contrib server
from server.contrib import WSServer
server = WSServer(profile_store=RedisProfileStore(redis))
```

## Integration Examples

### FastAPI WebSocket

```python
from fastapi import FastAPI, WebSocket
from server.core import SpeakerGate, Event

app = FastAPI()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    
    # Load profile
    profile = await db.get_profile(user_id)
    gate = SpeakerGate(profile.embedding, threshold=profile.threshold)
    
    # Setup event handlers
    @gate.on(Event.SEGMENT_COMPLETE)
    async def on_segment(event):
        result = await stt.transcribe(event.audio)
        await websocket.send_json({"transcription": result})
    
    # Process audio
    while True:
        data = await websocket.receive_bytes()
        audio = np.frombuffer(data, dtype=np.float32)
        gate.feed(audio)
```

### Synchronous Processing

```python
from server.core import SpeakerGate, Event
import soundfile as sf

# Load audio file
audio, sr = sf.read("input.wav")

# Create gate
gate = SpeakerGate(profile_embedding)

# Collect segments
segments = []
gate.on(Event.SEGMENT_COMPLETE, lambda e: segments.append(e.audio))

# Process
gate.feed(audio)

# segments now contains verified user speech
for i, seg in enumerate(segments):
    sf.write(f"segment_{i}.wav", seg, sr)
```

## Migration from v2

1. **Replace imports**: `from poc.server_v2 import ...` → `from poc.server.core import ...`

2. **Use events instead of stats**:
   ```python
   gate.on(Event.SEGMENT_COMPLETE, handle_segment)
   ```

3. **Config is now a dataclass**:
   ```python
   config = GateConfig(default_threshold=0.75)
   ```

4. **Models are injectable**:
   ```python
   models = Models(vad=my_vad, encoder=my_encoder)
   gate = SpeakerGate(embedding, models=models)
   ```

## Dependencies

### Core (Production)
- numpy
- torch
- silero-vad
- resemblyzer

### Contrib (Testing)
- websockets
- soundfile
- pydantic-settings
- typer
