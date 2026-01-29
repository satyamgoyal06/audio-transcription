"""
Modern GUI for Audio Transcription
A clean, dark-themed interface with speaker diarization support.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import threading
import os
import time

from transcriber import Transcriber, is_supported_format, SUPPORTED_FORMATS, get_audio_duration


class TranscriptionApp:
    """Main GUI application for audio transcription."""
    
    def __init__(self):
        # Create main window with dark theme
        self.root = ttkb.Window(
            title="🎙️ Audio Transcriber",
            themename="darkly",
            size=(750, 750),
            resizable=(True, True)
        )
        self.root.minsize(650, 650)
        
        # Center window on screen
        self._center_window()
        
        # State variables
        self.current_file = None
        self.transcriber = None
        self.is_transcribing = False
        self.audio_duration = 0.0
        self.start_time = 0.0
        self.progress_update_id = None
        
        # Build UI
        self._create_ui()
        
        # Enable drag and drop
        self._setup_drag_drop()
    
    def _center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")
    
    def _create_ui(self):
        """Create all UI components."""
        # Main container with padding
        main_frame = ttkb.Frame(self.root, padding=20)
        main_frame.pack(fill=BOTH, expand=YES)
        
        # === Header ===
        header_frame = ttkb.Frame(main_frame)
        header_frame.pack(fill=X, pady=(0, 15))
        
        title_label = ttkb.Label(
            header_frame,
            text="🎙️ Audio Transcriber",
            font=("Helvetica", 24, "bold"),
            bootstyle="inverse-primary"
        )
        title_label.pack(side=LEFT)
        
        subtitle = ttkb.Label(
            header_frame,
            text="MLX GPU Accelerated • Speaker Diarization",
            font=("Helvetica", 10),
            bootstyle="secondary"
        )
        subtitle.pack(side=LEFT, padx=(15, 0), pady=(10, 0))
        
        # === Drop Zone ===
        self.drop_frame = ttkb.Labelframe(
            main_frame,
            text="Audio File",
            padding=15,
            bootstyle="primary"
        )
        self.drop_frame.pack(fill=X, pady=(0, 10))
        
        self.drop_zone = ttkb.Frame(
            self.drop_frame,
            bootstyle="dark",
            padding=20
        )
        self.drop_zone.pack(fill=X)
        
        self.drop_label = ttkb.Label(
            self.drop_zone,
            text="📂 Click to select an audio file",
            font=("Helvetica", 12),
            bootstyle="light",
            cursor="hand2"
        )
        self.drop_label.pack(pady=15)
        
        # File info label
        self.file_info = ttkb.Label(
            self.drop_frame,
            text="",
            font=("Helvetica", 9),
            bootstyle="info"
        )
        self.file_info.pack(pady=(5, 0))
        
        # Bind click to open file dialog
        self.drop_zone.bind("<Button-1>", lambda e: self._browse_file())
        self.drop_label.bind("<Button-1>", lambda e: self._browse_file())
        
        # === Settings ===
        settings_frame = ttkb.Labelframe(
            main_frame,
            text="Settings",
            padding=15,
            bootstyle="info"
        )
        settings_frame.pack(fill=X, pady=(0, 10))
        
        # Model selection
        model_row = ttkb.Frame(settings_frame)
        model_row.pack(fill=X, pady=3)
        
        ttkb.Label(
            model_row,
            text="Model:",
            font=("Helvetica", 11),
            width=18
        ).pack(side=LEFT)
        
        self.model_var = tk.StringVar(value="base")
        models = list(Transcriber.MODELS.keys())
        
        self.model_combo = ttkb.Combobox(
            model_row,
            textvariable=self.model_var,
            values=models,
            state="readonly",
            width=12,
            bootstyle="primary"
        )
        self.model_combo.pack(side=LEFT, padx=(0, 10))
        
        self.model_desc = ttkb.Label(
            model_row,
            text=Transcriber.MODELS["base"],
            font=("Helvetica", 9),
            bootstyle="secondary"
        )
        self.model_desc.pack(side=LEFT)
        
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        
        # Speaker diarization option
        diar_row = ttkb.Frame(settings_frame)
        diar_row.pack(fill=X, pady=3)
        
        ttkb.Label(
            diar_row,
            text="Speaker ID:",
            font=("Helvetica", 11),
            width=18
        ).pack(side=LEFT)
        
        self.diarization_var = tk.BooleanVar(value=True)
        ttkb.Checkbutton(
            diar_row,
            text="Identify speakers (requires HF token)",
            variable=self.diarization_var,
            bootstyle="success-round-toggle",
            command=self._toggle_diarization
        ).pack(side=LEFT)
        
        # HuggingFace token input
        token_row = ttkb.Frame(settings_frame)
        token_row.pack(fill=X, pady=3)
        
        ttkb.Label(
            token_row,
            text="HuggingFace Token:",
            font=("Helvetica", 11),
            width=18
        ).pack(side=LEFT)
        
        self.token_var = tk.StringVar()
        self.token_entry = ttkb.Entry(
            token_row,
            textvariable=self.token_var,
            show="•",
            width=40,
            bootstyle="secondary"
        )
        self.token_entry.pack(side=LEFT, padx=(0, 5))
        
        self.show_token_btn = ttkb.Button(
            token_row,
            text="👁",
            width=3,
            command=self._toggle_token_visibility,
            bootstyle="secondary-outline"
        )
        self.show_token_btn.pack(side=LEFT)
        
        # Token help text
        token_help = ttkb.Label(
            settings_frame,
            text="Get token at huggingface.co/settings/tokens • Accept pyannote/speaker-diarization-3.1 terms first",
            font=("Helvetica", 8),
            bootstyle="secondary"
        )
        token_help.pack(anchor=W, pady=(3, 0))
        
        # Timestamps option
        ts_row = ttkb.Frame(settings_frame)
        ts_row.pack(fill=X, pady=3)
        
        ttkb.Label(
            ts_row,
            text="Timestamps:",
            font=("Helvetica", 11),
            width=18
        ).pack(side=LEFT)
        
        self.timestamps_var = tk.BooleanVar(value=False)
        ttkb.Checkbutton(
            ts_row,
            text="Include timestamps in output",
            variable=self.timestamps_var,
            bootstyle="primary-round-toggle"
        ).pack(side=LEFT)
        
        # === Action Buttons ===
        button_frame = ttkb.Frame(main_frame)
        button_frame.pack(fill=X, pady=(0, 10))
        
        self.transcribe_btn = ttkb.Button(
            button_frame,
            text="🎯 Transcribe",
            command=self._start_transcription,
            bootstyle="success",
            width=18
        )
        self.transcribe_btn.pack(side=LEFT, padx=(0, 10))
        
        self.open_output_btn = ttkb.Button(
            button_frame,
            text="📁 Open Output Folder",
            command=self._open_output_folder,
            bootstyle="info-outline",
            width=18,
            state=DISABLED
        )
        self.open_output_btn.pack(side=LEFT)
        
        # === Progress ===
        progress_frame = ttkb.Labelframe(
            main_frame,
            text="Progress",
            padding=10,
            bootstyle="secondary"
        )
        progress_frame.pack(fill=X, pady=(0, 10))
        
        # Progress bar (determinate mode for real progress)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttkb.Progressbar(
            progress_frame,
            mode="determinate",
            variable=self.progress_var,
            maximum=100,
            bootstyle="success-striped"
        )
        self.progress_bar.pack(fill=X, pady=(0, 5))
        
        # Progress percentage and time
        progress_info_frame = ttkb.Frame(progress_frame)
        progress_info_frame.pack(fill=X, pady=(0, 3))
        
        self.progress_percent = ttkb.Label(
            progress_info_frame,
            text="0%",
            font=("Helvetica", 11, "bold"),
            bootstyle="success"
        )
        self.progress_percent.pack(side=LEFT)
        
        self.time_label = ttkb.Label(
            progress_info_frame,
            text="",
            font=("Helvetica", 10),
            bootstyle="secondary"
        )
        self.time_label.pack(side=RIGHT)
        
        self.status_label = ttkb.Label(
            progress_frame,
            text="Ready. Select an audio file to begin.",
            font=("Helvetica", 10),
            bootstyle="secondary"
        )
        self.status_label.pack()
        
        # === Output Preview ===
        output_frame = ttkb.Labelframe(
            main_frame,
            text="Transcription Preview",
            padding=10,
            bootstyle="success"
        )
        output_frame.pack(fill=BOTH, expand=YES)
        
        self.output_text = tk.Text(
            output_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1a1a2e",
            fg="#eaeaea",
            insertbackground="#ffffff",
            selectbackground="#4a4a6a",
            height=10
        )
        self.output_text.pack(fill=BOTH, expand=YES, side=LEFT)
        
        scrollbar = ttkb.Scrollbar(
            output_frame,
            orient=VERTICAL,
            command=self.output_text.yview,
            bootstyle="primary-round"
        )
        scrollbar.pack(fill=Y, side=RIGHT)
        self.output_text.config(yscrollcommand=scrollbar.set)
        
        # Footer
        footer = ttkb.Label(
            main_frame,
            text="Supported: " + ", ".join(SUPPORTED_FORMATS),
            font=("Helvetica", 8),
            bootstyle="secondary"
        )
        footer.pack(pady=(5, 0))
    
    def _setup_drag_drop(self):
        """Setup drag and drop functionality."""
        pass
    
    def _toggle_token_visibility(self):
        """Toggle password visibility for token entry."""
        current = self.token_entry.cget("show")
        if current == "•":
            self.token_entry.config(show="")
            self.show_token_btn.config(text="🔒")
        else:
            self.token_entry.config(show="•")
            self.show_token_btn.config(text="👁")
    
    def _toggle_diarization(self):
        """Enable/disable token entry based on diarization checkbox."""
        if self.diarization_var.get():
            self.token_entry.config(state=NORMAL)
        else:
            self.token_entry.config(state=DISABLED)
    
    def _browse_file(self):
        """Open file dialog to select audio file."""
        filetypes = [
            ("Audio Files", " ".join(f"*{ext}" for ext in SUPPORTED_FORMATS)),
            ("All Files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if file_path:
            self._set_file(file_path)
    
    def _set_file(self, file_path: str):
        """Set the current file for transcription."""
        if not is_supported_format(file_path):
            messagebox.showerror(
                "Unsupported Format",
                f"This file format is not supported.\n\nSupported formats:\n{', '.join(SUPPORTED_FORMATS)}"
            )
            return
        
        self.current_file = file_path
        filename = os.path.basename(file_path)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Get audio duration
        self.audio_duration = get_audio_duration(file_path)
        duration_str = self._format_duration(self.audio_duration) if self.audio_duration > 0 else "Unknown"
        
        self.drop_label.config(text=f"📄 {filename}")
        self.file_info.config(text=f"Size: {size_mb:.2f} MB | Duration: {duration_str}")
        self._update_status(f"File loaded: {filename}")
        
        # Show estimated time
        if self.audio_duration > 0:
            model = self.model_var.get()
            est_time = self._estimate_transcription_time(model)
            self.time_label.config(text=f"Estimated: ~{self._format_duration(est_time)}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds to human-readable duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def _estimate_transcription_time(self, model: str) -> float:
        """Estimate transcription time based on audio duration and model."""
        if self.audio_duration <= 0:
            return 0
        speed = Transcriber.MODEL_SPEED.get(model, 30.0)
        base_time = self.audio_duration / speed
        
        # Add time for diarization if enabled
        if self.diarization_var.get() and self.token_var.get():
            base_time += self.audio_duration * 0.3  # Diarization adds ~30% time
        
        return base_time + 5  # Add 5 seconds for model loading
    
    def _on_model_change(self, event=None):
        """Update model description when selection changes."""
        model = self.model_var.get()
        self.model_desc.config(text=Transcriber.MODELS.get(model, ""))
        
        # Update estimated time
        if self.audio_duration > 0:
            est_time = self._estimate_transcription_time(model)
            self.time_label.config(text=f"Estimated: ~{self._format_duration(est_time)}")
    
    def _start_transcription(self):
        """Start the transcription process."""
        if not self.current_file:
            messagebox.showwarning("No File", "Please select an audio file first.")
            return
        
        if self.is_transcribing:
            messagebox.showinfo("In Progress", "Transcription is already in progress.")
            return
        
        # Check for HF token if diarization is enabled
        hf_token = None
        if self.diarization_var.get():
            hf_token = self.token_var.get().strip()
            if not hf_token:
                result = messagebox.askyesno(
                    "No HuggingFace Token",
                    "Speaker identification requires a HuggingFace token.\n\n"
                    "Continue without speaker identification?"
                )
                if not result:
                    return
                hf_token = None
        
        # Disable UI during transcription
        self.is_transcribing = True
        self.transcribe_btn.config(state=DISABLED)
        self.model_combo.config(state=DISABLED)
        self.token_entry.config(state=DISABLED)
        
        # Reset progress
        self.progress_var.set(0)
        self.progress_percent.config(text="0%")
        self.start_time = time.time()
        
        # Clear previous output
        self.output_text.delete(1.0, tk.END)
        
        # Start progress updates
        self._update_progress_display()
        
        # Run transcription in thread
        thread = threading.Thread(
            target=self._transcribe_thread, 
            args=(hf_token,),
            daemon=True
        )
        thread.start()
    
    def _update_progress_display(self):
        """Update progress bar and time estimates during transcription."""
        if not self.is_transcribing:
            return
        
        elapsed = time.time() - self.start_time
        model = self.model_var.get()
        est_total = self._estimate_transcription_time(model)
        
        if est_total > 0:
            # Calculate progress percentage
            progress = min((elapsed / est_total) * 100, 99)  # Cap at 99% until done
            self.progress_var.set(progress)
            self.progress_percent.config(text=f"{int(progress)}%")
            
            # Calculate time remaining
            if progress > 5:
                time_remaining = max(0, est_total - elapsed)
                elapsed_str = self._format_duration(elapsed)
                remaining_str = self._format_duration(time_remaining)
                self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: ~{remaining_str}")
            else:
                self.time_label.config(text=f"Elapsed: {self._format_duration(elapsed)}")
        
        # Schedule next update
        self.progress_update_id = self.root.after(500, self._update_progress_display)
    
    def _transcribe_thread(self, hf_token: str = None):
        """Transcription logic running in background thread."""
        try:
            model_name = self.model_var.get()
            self.transcriber = Transcriber(model_name, hf_token=hf_token)
            
            result = self.transcriber.transcribe(
                audio_path=self.current_file,
                include_timestamps=self.timestamps_var.get(),
                enable_diarization=self.diarization_var.get(),
                progress_callback=self._update_status
            )
            
            # Update UI with results (in main thread)
            self.root.after(0, lambda r=result: self._on_transcription_complete(r))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self._on_transcription_error(msg))
    
    def _on_transcription_complete(self, result: dict):
        """Handle successful transcription."""
        # Stop progress updates
        if self.progress_update_id:
            self.root.after_cancel(self.progress_update_id)
        
        # Set progress to 100%
        self.progress_var.set(100)
        self.progress_percent.config(text="100%")
        
        elapsed = time.time() - self.start_time
        self.time_label.config(text=f"Completed in {self._format_duration(elapsed)}")
        
        self.is_transcribing = False
        self.transcribe_btn.config(state=NORMAL)
        self.model_combo.config(state="readonly")
        if self.diarization_var.get():
            self.token_entry.config(state=NORMAL)
        self.open_output_btn.config(state=NORMAL)
        
        # Display transcription
        self.output_text.insert(tk.END, result["text"])
        
        speaker_info = "with speakers" if result.get("has_speakers") else "without speakers"
        self._update_status(
            f"✅ Complete! Language: {result['language']} | {speaker_info}"
        )
        
        # Store output path for "Open Folder" button
        self.output_path = result["output_path"]
        
        messagebox.showinfo(
            "Transcription Complete",
            f"Successfully transcribed {speaker_info}!\n\n"
            f"Time taken: {self._format_duration(elapsed)}\n\n"
            f"Output saved to:\n{result['output_path']}"
        )
    
    def _on_transcription_error(self, error_msg: str):
        """Handle transcription error."""
        # Stop progress updates
        if self.progress_update_id:
            self.root.after_cancel(self.progress_update_id)
        
        self.progress_var.set(0)
        self.progress_percent.config(text="0%")
        self.time_label.config(text="")
        
        self.is_transcribing = False
        self.transcribe_btn.config(state=NORMAL)
        self.model_combo.config(state="readonly")
        if self.diarization_var.get():
            self.token_entry.config(state=NORMAL)
        
        self._update_status(f"❌ Error: {error_msg[:50]}...")
        messagebox.showerror("Transcription Error", f"An error occurred:\n\n{error_msg}")
    
    def _update_status(self, message: str):
        """Update status label (thread-safe)."""
        if threading.current_thread() is threading.main_thread():
            self.status_label.config(text=message)
        else:
            self.root.after(0, lambda: self.status_label.config(text=message))
    
    def _open_output_folder(self):
        """Open the folder containing the output file."""
        if hasattr(self, 'output_path') and self.output_path:
            folder = os.path.dirname(self.output_path)
            if os.path.exists(folder):
                import subprocess
                subprocess.run(["open", folder])
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Entry point for the application."""
    app = TranscriptionApp()
    app.run()


if __name__ == "__main__":
    main()
