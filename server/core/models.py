"""
ML Models for Speaker Gate SDK.

Provides clean interfaces for VAD and voice embedding models.
Supports both singleton pattern and custom model injection.

Example (using defaults):
    models = Models.get()  # Lazy-loaded singleton
    
Example (custom injection):
    my_vad = load_my_vad()
    my_encoder = MyEncoder()
    models = Models(vad=my_vad, encoder=my_encoder)
    gate = SpeakerGate(embedding, models=models)
"""
import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional, Any

log = logging.getLogger(__name__)


# Use Any for model types to avoid strict protocol matching
# The actual models (Silero VAD, Resemblyzer) have compatible interfaces
VADModel = Any
VoiceEncoderModel = Any


class ModelLoader(ABC):
    """
    Abstract model loader for custom model implementations.
    
    Implement this to provide your own VAD or encoder models.
    """
    
    @abstractmethod
    def load_vad(self) -> Optional[VADModel]:
        """Load and return VAD model, or None to disable VAD."""
        ...
    
    @abstractmethod
    def load_encoder(self) -> VoiceEncoderModel:
        """Load and return voice encoder model."""
        ...


class DefaultModelLoader(ModelLoader):
    """Default loader using Silero VAD and Resemblyzer."""
    
    def load_vad(self) -> Optional[VADModel]:
        try:
            from silero_vad import load_silero_vad
            return load_silero_vad(onnx=True)
        except Exception as e:
            log.warning(f"Failed to load Silero VAD: {e}")
            return None
    
    def load_encoder(self) -> VoiceEncoderModel:
        from resemblyzer import VoiceEncoder
        return VoiceEncoder()


class Models:
    """
    Container for ML models with optional singleton pattern.
    
    Usage:
        # Singleton (lazy loaded)
        models = Models.get()
        
        # Custom models
        models = Models(vad=my_vad, encoder=my_encoder)
        
        # Custom loader
        models = Models.load(MyModelLoader())
    """
    
    _instance: Optional["Models"] = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        vad: Optional[VADModel] = None,
        encoder: Optional[VoiceEncoderModel] = None,
        loader: Optional[ModelLoader] = None,
    ):
        """
        Initialize models container.
        
        Args:
            vad: Pre-loaded VAD model (None = disabled)
            encoder: Pre-loaded voice encoder
            loader: Model loader for lazy loading
        """
        self._loader = loader or DefaultModelLoader()
        self._vad = vad
        self._encoder = encoder
        self._loaded = vad is not None or encoder is not None
    
    def _ensure_loaded(self):
        if not self._loaded:
            log.info("Loading ML models...")
            if self._vad is None:
                self._vad = self._loader.load_vad()
            if self._encoder is None:
                self._encoder = self._loader.load_encoder()
            self._loaded = True
    
    @property
    def vad(self) -> Optional[VADModel]:
        """VAD model (may be None if disabled)."""
        self._ensure_loaded()
        return self._vad
    
    @property
    def encoder(self) -> VoiceEncoderModel:
        """Voice encoder model."""
        self._ensure_loaded()
        return self._encoder
    
    def reset_vad(self):
        """Reset VAD states."""
        if self._vad is not None:
            self._vad.reset_states()
    
    @classmethod
    def get(cls, loader: Optional[ModelLoader] = None) -> "Models":
        """
        Get singleton instance (lazy loaded).
        
        Args:
            loader: Optional custom model loader
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(loader=loader)
            return cls._instance
    
    @classmethod
    def load(cls, loader: ModelLoader) -> "Models":
        """Create new Models instance with custom loader."""
        return cls(loader=loader)
    
    @classmethod
    def reset_singleton(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
