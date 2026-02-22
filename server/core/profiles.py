"""Speaker profile management — model, storage interface, and file-based store."""
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
    """Speaker profile with embedding and threshold."""
    profile_id: str
    embedding: List[float]
    threshold: float
    consistency_score: float = 0.0
    duration_sec: float = 0.0
    created_at: Optional[str] = None  # ISO format string
    session_embeddings: Optional[List[List[float]]] = None  # Per-session reference embeddings
    session_thresholds: Optional[List[float]] = None  # Per-session similarity thresholds

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

        if len(self.embedding) != 256:
            raise ValueError(f"Expected 256-dim embedding, got {len(self.embedding)}")

        if self.session_embeddings is not None:
            for i, emb in enumerate(self.session_embeddings):
                if len(emb) != 256:
                    raise ValueError(f"Session embedding {i}: expected 256-dim, got {len(emb)}")

    @property
    def embedding_array(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)

    @property
    def session_embedding_arrays(self) -> List[np.ndarray]:
        """All session embeddings as arrays. Falls back to primary embedding."""
        if self.session_embeddings:
            return [np.array(e, dtype=np.float32) for e in self.session_embeddings]
        return [self.embedding_array]
    
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        # Backward compat
        if "session_embeddings" not in data:
            data["session_embeddings"] = None
        if "session_thresholds" not in data:
            data["session_thresholds"] = None
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Profile":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ProfileStore(ABC):
    """Abstract profile storage. Implement for your backend (DB, Redis, etc.)."""
    
    @abstractmethod
    def save(self, profile: Profile) -> None: ...

    @abstractmethod
    def load(self, profile_id: str) -> Optional[Profile]: ...

    @abstractmethod
    def delete(self, profile_id: str) -> bool: ...

    @abstractmethod
    def list_ids(self) -> List[str]: ...

    def get(self, profile_id: str) -> Profile:
        """Load profile or raise ProfileNotFoundError."""
        profile = self.load(profile_id)
        if profile is None:
            raise ProfileNotFoundError(profile_id)
        return profile

    def exists(self, profile_id: str) -> bool:
        return self.load(profile_id) is not None


class FileProfileStore(ProfileStore):
    """File-based profile storage (JSON). For testing/standalone use."""

    def __init__(self, profiles_dir: Path):
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
    """Map enrollment consistency to similarity threshold. Higher consistency = stricter."""
    cfg = config or ProfileConfig()
    return cfg.calculate_threshold(consistency)
