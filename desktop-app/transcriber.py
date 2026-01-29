"""
Whisper Transcription Engine
Core module for audio-to-text transcription using OpenAI's Whisper model.
"""

import whisper
import os
from datetime import datetime
from typing import Callable, Optional


class Transcriber:
    """Handles audio transcription using OpenAI's Whisper model."""
    
    # Available model sizes with descriptions
    MODELS = {
        "tiny": "Fastest, least accurate (~1GB RAM)",
        "base": "Fast, good accuracy (~1.5GB RAM)",
        "small": "Balanced speed/accuracy (~2.5GB RAM)",
        "medium": "High accuracy, slower (~5GB RAM)",
        "large": "Best accuracy, slowest (~10GB RAM)"
    }
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the transcriber with specified model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        
    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Load the Whisper model into memory.
        
        Args:
            progress_callback: Optional callback for progress updates
        """
        if progress_callback:
            progress_callback(f"Loading {self.model_name} model...")
        
        self.model = whisper.load_model(self.model_name)
        
        if progress_callback:
            progress_callback(f"Model '{self.model_name}' loaded successfully!")
    
    def transcribe(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        include_timestamps: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> dict:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save output (defaults to audio file's directory)
            include_timestamps: Whether to include timestamps in output
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing transcription results
        """
        if not self.model:
            self.load_model(progress_callback)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if progress_callback:
            progress_callback("Transcribing audio... This may take a while.")
        
        # Perform transcription
        result = self.model.transcribe(audio_path, verbose=False)
        
        # Generate output file path
        if output_dir is None:
            output_dir = os.path.dirname(audio_path)
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_transcription_{timestamp}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write transcription to file
        self._write_output(result, output_path, audio_path, include_timestamps)
        
        if progress_callback:
            progress_callback(f"Transcription saved to: {output_filename}")
        
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
            "output_path": output_path
        }
    
    def _write_output(
        self,
        result: dict,
        output_path: str,
        audio_path: str,
        include_timestamps: bool
    ) -> None:
        """Write transcription results to a text file."""
        with open(output_path, "w", encoding="utf-8") as f:
            # Header
            f.write("=" * 60 + "\n")
            f.write("AUDIO TRANSCRIPTION\n")
            f.write("=" * 60 + "\n\n")
            
            # Metadata
            f.write(f"Source File: {os.path.basename(audio_path)}\n")
            f.write(f"Detected Language: {result.get('language', 'unknown')}\n")
            f.write(f"Model Used: {self.model_name}\n")
            f.write(f"Transcribed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "-" * 60 + "\n\n")
            
            if include_timestamps and result.get("segments"):
                # Timestamped transcription
                f.write("TIMESTAMPED TRANSCRIPTION:\n\n")
                for segment in result["segments"]:
                    start = self._format_time(segment["start"])
                    end = self._format_time(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"[{start} --> {end}]\n{text}\n\n")
                
                f.write("-" * 60 + "\n\n")
            
            # Full text
            f.write("FULL TRANSCRIPTION:\n\n")
            f.write(result["text"].strip())
            f.write("\n\n" + "=" * 60 + "\n")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS.mmm format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    @classmethod
    def get_available_models(cls) -> dict:
        """Return available model names and descriptions."""
        return cls.MODELS.copy()


# Supported audio formats
SUPPORTED_FORMATS = [
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", 
    ".wma", ".aac", ".opus", ".webm", ".mp4"
]


def is_supported_format(file_path: str) -> bool:
    """Check if the file format is supported for transcription."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_FORMATS
