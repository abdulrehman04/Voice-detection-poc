"""
Speaker profile management for Speaker Gate SDK.

Provides Profile model and abstract storage interface.
Includes a simple file-based implementation for standalone use.
"""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import numpy as np

from .config import ProfileConfig
from .exceptions import ProfileNotFoundError, ProfileInvalidError

log = logging.getLogger(__name__)


@dataclass
class Profile:
    """
    Speaker profile with embedding and threshold.
    
    Attributes:
        profile_id: Unique identifier
        embedding: 256-dim voice embedding as list
        threshold: Similarity threshold for this speaker
        consistency_score: Enrollment audio consistency
        duration_sec: Enrollment audio duration
        created_at: Profile creation timestamp
    """
    profile_id: str
    embedding: List[float]
    threshold: float
    consistency_score: float = 0.0
    duration_sec: float = 0.0
    created_at: Optional[str] = None  # ISO format string
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        
        if len(self.embedding) != 256:
            raise ValueError(f"Expected 256-dim embedding, got {len(self.embedding)}")
    
    @property
    def embedding_array(self) -> np.ndarray:
        """Get embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        """Deserialize from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Profile":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ProfileStore(ABC):
    """
    Abstract profile storage interface.
    
    Implement this to integrate with your existing storage
    (database, Redis, cloud storage, etc.)
    """
    
    @abstractmethod
    def save(self, profile: Profile) -> None:
        """Save profile to storage."""
        ...
    
    @abstractmethod
    def load(self, profile_id: str) -> Optional[Profile]:
        """Load profile from storage, return None if not found."""
        ...
    
    @abstractmethod
    def delete(self, profile_id: str) -> bool:
        """Delete profile, return True if existed."""
        ...
    
    @abstractmethod
    def list_ids(self) -> List[str]:
        """List all profile IDs."""
        ...
    
    def get(self, profile_id: str) -> Profile:
        """
        Load profile or raise ProfileNotFoundError.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Profile instance
            
        Raises:
            ProfileNotFoundError: If profile doesn't exist
        """
        profile = self.load(profile_id)
        if profile is None:
            raise ProfileNotFoundError(profile_id)
        return profile
    
    def exists(self, profile_id: str) -> bool:
        """Check if profile exists."""
        return self.load(profile_id) is not None


class FileProfileStore(ProfileStore):
    """
    File-based profile storage (JSON files).
    
    Suitable for standalone use and testing.
    For production, implement ProfileStore with your database.
    """
    
    def __init__(self, profiles_dir: Path):
        """
        Initialize file store.
        
        Args:
            profiles_dir: Directory for profile JSON files
        """
        self._dir = Path(profiles_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
    
    def _path(self, profile_id: str) -> Path:
        return self._dir / f"{profile_id}.json"
    
    def save(self, profile: Profile) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(profile.profile_id)
        path.write_text(profile.to_json())
        log.info(f"Saved profile {profile.profile_id}: threshold={profile.threshold:.3f}")
    
    def load(self, profile_id: str) -> Optional[Profile]:
        path = self._path(profile_id)
        if not path.exists():
            return None
        
        try:
            return Profile.from_json(path.read_text())
        except json.JSONDecodeError as e:
            raise ProfileInvalidError(profile_id, f"invalid JSON: {e}")
        except (ValueError, KeyError, TypeError) as e:
            raise ProfileInvalidError(profile_id, str(e))
    
    def delete(self, profile_id: str) -> bool:
        path = self._path(profile_id)
        if path.exists():
            path.unlink()
            log.info(f"Deleted profile {profile_id}")
            return True
        return False
    
    def list_ids(self) -> List[str]:
        return [f.stem for f in self._dir.glob("*.json")]


def calculate_threshold(consistency: float, config: Optional[ProfileConfig] = None) -> float:
    """
    Calculate optimal similarity threshold from enrollment consistency.
    
    Higher consistency = higher threshold (more strict matching).
    
    Args:
        consistency: Enrollment audio consistency score (0-1)
        config: Profile configuration (uses defaults if not provided)
        
    Returns:
        Similarity threshold
    """
    cfg = config or ProfileConfig()
    return cfg.calculate_threshold(consistency)
