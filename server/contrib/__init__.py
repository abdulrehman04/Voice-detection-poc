"""
Speaker Gate Contrib - Testing, Debugging, and POC Utilities.

This package contains utilities for development, testing, and POC deployment.
NOT required for production use - the core package is standalone.

Includes:
- WebSocket server for browser-based testing
- Debug audio recording
- Statistics tracking
- CLI interface
"""
from .config import ContribSettings, get_settings
from .debug import AsyncAudioWriter, SessionRecorder
from .server import WSServer, create_server

__all__ = [
    "ContribSettings",
    "get_settings",
    "AsyncAudioWriter", 
    "SessionRecorder",
    "WSServer",
    "create_server",
]
