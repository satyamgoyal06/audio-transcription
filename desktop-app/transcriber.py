"""
Whisper Transcription Engine with Speaker Diarization
Core module for audio-to-text transcription using MLX-Whisper (GPU accelerated) 
and pyannote-audio for speaker identification.
"""

import os
import subprocess
import json
from datetime import datetime
from typing import Callable, Optional

# Try to use MLX-Whisper (GPU accelerated for Apple Silicon)
try:
    import mlx_whisper
    USE_MLX = True
except ImportError:
    import whisper
    USE_MLX = False


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception:
        return 0.0


class Transcriber:
    """Handles audio transcription with speaker diarization."""
    
    # Available model sizes with descriptions
    MODELS = {
        "tiny": "Fastest, least accurate (~1GB RAM)",
        "base": "Fast, good accuracy (~1.5GB RAM)",
        "small": "Balanced speed/accuracy (~2.5GB RAM)",
        "medium": "High accuracy, slower (~5GB RAM)",
        "large": "Best accuracy, slowest (~10GB RAM)"
    }
    
    # Speed multipliers for MLX (faster than CPU)
    MODEL_SPEED = {
        "tiny": 50.0,
        "base": 30.0,
        "small": 15.0,
        "medium": 5.0,
        "large": 2.0
    }
    
    def __init__(self, model_name: str = "base", hf_token: str = None):
        """
        Initialize the transcriber with specified model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            hf_token: HuggingFace token for speaker diarization (optional)
        """
        self.model_name = model_name
        self.model = None
        self.diarization_pipeline = None
        self.hf_token = hf_token
        self.use_mlx = USE_MLX
        
    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Load the Whisper model into memory."""
        if progress_callback:
            backend = "MLX (GPU)" if self.use_mlx else "CPU"
            progress_callback(f"Loading {self.model_name} model on {backend}...")
        
        # MLX whisper loads on-demand during transcription
        if progress_callback:
            progress_callback(f"Model '{self.model_name}' ready!")
    
    def load_diarization(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Load speaker diarization pipeline."""
        if not self.hf_token:
            return False
            
        try:
            if progress_callback:
                progress_callback("Loading speaker diarization model...")
            
            from pyannote.audio import Pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            if progress_callback:
                progress_callback("Speaker diarization ready!")
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"Diarization unavailable: {str(e)[:50]}...")
            return False
    
    def transcribe(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        include_timestamps: bool = False,
        enable_diarization: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> dict:
        """
        Transcribe an audio file to text with optional speaker identification.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save output (defaults to audio file's directory)
            include_timestamps: Whether to include timestamps in output
            enable_diarization: Whether to identify speakers
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing transcription results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load diarization if enabled and token provided
        diarization_result = None
        if enable_diarization and self.hf_token:
            if self.load_diarization(progress_callback):
                if progress_callback:
                    progress_callback("Identifying speakers...")
                try:
                    diarization_result = self.diarization_pipeline(audio_path)
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Speaker identification failed: {str(e)[:30]}")
        
        if progress_callback:
            backend = "MLX GPU" if self.use_mlx else "CPU"
            progress_callback(f"Transcribing on {backend}... This may take a while.")
        
        # Perform transcription with MLX or standard Whisper
        if self.use_mlx:
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=f"mlx-community/whisper-{self.model_name}-mlx"
            )
        else:
            if not self.model:
                import whisper
                self.model = whisper.load_model(self.model_name)
            result = self.model.transcribe(audio_path, verbose=False)
        
        # Merge diarization with transcription
        if diarization_result and result.get("segments"):
            result = self._merge_diarization(result, diarization_result)
        
        # Generate output file path
        if output_dir is None:
            output_dir = os.path.dirname(audio_path)
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_transcription_{timestamp}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write transcription to file
        has_speakers = diarization_result is not None
        self._write_output(result, output_path, audio_path, include_timestamps, has_speakers)
        
        if progress_callback:
            progress_callback(f"Transcription saved to: {output_filename}")
        
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
            "output_path": output_path,
            "has_speakers": has_speakers
        }
    
    def _merge_diarization(self, transcription: dict, diarization) -> dict:
        """Merge speaker diarization results with transcription segments."""
        segments = transcription.get("segments", [])
        
        for segment in segments:
            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_mid = (seg_start + seg_end) / 2
            
            # Find which speaker is talking at this segment's midpoint
            speaker = "Unknown"
            for turn, _, spk in diarization.itertracks(yield_label=True):
                if turn.start <= seg_mid <= turn.end:
                    speaker = spk
                    break
            
            segment["speaker"] = speaker
        
        return transcription
    
    def _write_output(
        self,
        result: dict,
        output_path: str,
        audio_path: str,
        include_timestamps: bool,
        has_speakers: bool
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
            f.write(f"Backend: {'MLX (GPU)' if self.use_mlx else 'CPU'}\n")
            f.write(f"Speaker Identification: {'Yes' if has_speakers else 'No'}\n")
            f.write(f"Transcribed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "-" * 60 + "\n\n")
            
            segments = result.get("segments", [])
            
            if has_speakers and segments:
                # Conversation-style output with speakers
                f.write("CONVERSATION:\n\n")
                current_speaker = None
                
                for segment in segments:
                    speaker = segment.get("speaker", "Unknown")
                    text = segment["text"].strip()
                    
                    if speaker != current_speaker:
                        # New speaker
                        speaker_label = self._format_speaker(speaker)
                        if include_timestamps:
                            time_str = self._format_time_short(segment["start"])
                            f.write(f"\n[{time_str}] {speaker_label}:\n")
                        else:
                            f.write(f"\n{speaker_label}:\n")
                        current_speaker = speaker
                    
                    f.write(f"  {text}\n")
                
            elif include_timestamps and segments:
                # Timestamped but no speakers
                f.write("TIMESTAMPED TRANSCRIPTION:\n\n")
                for segment in segments:
                    start = self._format_time(segment["start"])
                    end = self._format_time(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"[{start} --> {end}]\n{text}\n\n")
            
            f.write("\n" + "-" * 60 + "\n\n")
            
            # Full text
            f.write("FULL TRANSCRIPTION:\n\n")
            f.write(result["text"].strip())
            f.write("\n\n" + "=" * 60 + "\n")
    
    @staticmethod
    def _format_speaker(speaker_id: str) -> str:
        """Format speaker ID to a readable name."""
        if speaker_id.startswith("SPEAKER_"):
            num = speaker_id.replace("SPEAKER_", "")
            return f"Speaker {int(num) + 1}"
        return speaker_id
    
    @staticmethod
    def _format_time_short(seconds: float) -> str:
        """Format seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
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
