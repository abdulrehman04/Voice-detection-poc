"""
Debug utilities for audio recording.

These utilities are for development/testing only.
NOT required for production use.
"""
import queue
import threading
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

from .config import get_settings


class AsyncAudioWriter:
    """
    Asynchronous audio file writer.
    
    Writes audio files in a background thread to avoid blocking
    the main processing pipeline.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, sample_rate: int = 16000):
        """
        Initialize writer.
        
        Args:
            output_dir: Directory for output files (default from settings)
            sample_rate: Audio sample rate
        """
        self._dir = output_dir or get_settings().debug_audio_dir
        self._sr = sample_rate
        self._queue: queue.Queue = queue.Queue()
        self._running = True
        
        self._dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def write(self, filename: str, data: np.ndarray):
        """
        Queue audio for async writing.
        
        Args:
            filename: Output filename (without path)
            data: Audio samples
        """
        if self._running:
            self._queue.put((filename, np.array(data, copy=True)))
    
    def _worker(self):
        while self._running or not self._queue.empty():
            try:
                filename, data = self._queue.get(timeout=0.5)
                sf.write(self._dir / filename, data, self._sr)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass  # Ignore write errors
    
    def close(self):
        """Stop writer and wait for pending writes."""
        self._running = False
        self._thread.join(timeout=5.0)


class SessionRecorder:
    """
    Stream audio to a single file per session.
    
    Useful for recording entire sessions for debugging.
    """
    
    def __init__(
        self, 
        session_id: str, 
        output_dir: Optional[Path] = None, 
        sample_rate: int = 16000
    ):
        """
        Initialize recorder.
        
        Args:
            session_id: Unique session identifier
            output_dir: Directory for output files
            sample_rate: Audio sample rate
        """
        directory = output_dir or get_settings().debug_audio_dir
        directory.mkdir(parents=True, exist_ok=True)
        
        self._file = sf.SoundFile(
            directory / f"session_{session_id}.wav",
            mode='w',
            samplerate=sample_rate,
            channels=1
        )
        self._queue: queue.Queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def write(self, chunk: np.ndarray):
        """Queue audio chunk for writing."""
        if self._running:
            self._queue.put(np.array(chunk, copy=True))
    
    def _worker(self):
        while self._running or not self._queue.empty():
            try:
                chunk = self._queue.get(timeout=0.5)
                self._file.write(chunk)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def close(self):
        """Stop recording and close file."""
        self._running = False
        self._thread.join(timeout=5.0)
        self._file.close()
