"""Custom exceptions for Speaker Gate SDK."""


class SpeakerGateError(Exception):
    """Base exception for all Speaker Gate errors."""


class ProfileNotFoundError(SpeakerGateError):
    """Profile does not exist in storage."""
    
    def __init__(self, profile_id: str):
        super().__init__(f"Profile not found: {profile_id}")
        self.profile_id = profile_id


class ProfileInvalidError(SpeakerGateError):
    """Profile data is corrupted or invalid."""
    
    def __init__(self, profile_id: str, reason: str):
        super().__init__(f"Invalid profile {profile_id}: {reason}")
        self.profile_id = profile_id
        self.reason = reason


class AudioProcessingError(SpeakerGateError):
    """Audio processing failed."""


class ModelLoadError(SpeakerGateError):
    """Failed to load ML model."""
