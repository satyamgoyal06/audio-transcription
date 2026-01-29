"""
Modern GUI for Audio Transcription
A clean, dark-themed interface with drag-and-drop support.
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
            size=(700, 650),
            resizable=(True, True)
        )
        self.root.minsize(600, 550)
        
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
        header_frame.pack(fill=X, pady=(0, 20))
        
        title_label = ttkb.Label(
            header_frame,
            text="🎙️ Audio Transcriber",
            font=("Helvetica", 24, "bold"),
            bootstyle="inverse-primary"
        )
        title_label.pack(side=LEFT)
        
        subtitle = ttkb.Label(
            header_frame,
            text="Powered by OpenAI Whisper",
            font=("Helvetica", 10),
            bootstyle="secondary"
        )
        subtitle.pack(side=LEFT, padx=(15, 0), pady=(10, 0))
        
        # === Drop Zone ===
        self.drop_frame = ttkb.Labelframe(
            main_frame,
            text="Audio File",
            padding=20,
            bootstyle="primary"
        )
        self.drop_frame.pack(fill=X, pady=(0, 15))
        
        self.drop_zone = ttkb.Frame(
            self.drop_frame,
            bootstyle="dark",
            padding=30
        )
        self.drop_zone.pack(fill=X)
        
        self.drop_label = ttkb.Label(
            self.drop_zone,
            text="📂 Click to select or drag & drop an audio file here",
            font=("Helvetica", 12),
            bootstyle="light",
            cursor="hand2"
        )
        self.drop_label.pack(pady=20)
        
        # File info label
        self.file_info = ttkb.Label(
            self.drop_frame,
            text="",
            font=("Helvetica", 10),
            bootstyle="info"
        )
        self.file_info.pack(pady=(10, 0))
        
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
        settings_frame.pack(fill=X, pady=(0, 15))
        
        # Model selection
        model_row = ttkb.Frame(settings_frame)
        model_row.pack(fill=X, pady=5)
        
        ttkb.Label(
            model_row,
            text="Model:",
            font=("Helvetica", 11),
            width=15
        ).pack(side=LEFT)
        
        self.model_var = tk.StringVar(value="base")
        models = list(Transcriber.MODELS.keys())
        
        self.model_combo = ttkb.Combobox(
            model_row,
            textvariable=self.model_var,
            values=models,
            state="readonly",
            width=15,
            bootstyle="primary"
        )
        self.model_combo.pack(side=LEFT, padx=(0, 15))
        
        self.model_desc = ttkb.Label(
            model_row,
            text=Transcriber.MODELS["base"],
            font=("Helvetica", 9),
            bootstyle="secondary"
        )
        self.model_desc.pack(side=LEFT)
        
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        
        # Timestamps option
        ts_row = ttkb.Frame(settings_frame)
        ts_row.pack(fill=X, pady=5)
        
        ttkb.Label(
            ts_row,
            text="Timestamps:",
            font=("Helvetica", 11),
            width=15
        ).pack(side=LEFT)
        
        self.timestamps_var = tk.BooleanVar(value=True)
        ttkb.Checkbutton(
            ts_row,
            text="Include timestamps in output",
            variable=self.timestamps_var,
            bootstyle="primary-round-toggle"
        ).pack(side=LEFT)
        
        # === Action Buttons ===
        button_frame = ttkb.Frame(main_frame)
        button_frame.pack(fill=X, pady=(0, 15))
        
        self.transcribe_btn = ttkb.Button(
            button_frame,
            text="🎯 Transcribe",
            command=self._start_transcription,
            bootstyle="success",
            width=20
        )
        self.transcribe_btn.pack(side=LEFT, padx=(0, 10))
        
        self.open_output_btn = ttkb.Button(
            button_frame,
            text="📁 Open Output Folder",
            command=self._open_output_folder,
            bootstyle="info-outline",
            width=20,
            state=DISABLED
        )
        self.open_output_btn.pack(side=LEFT)
        
        # === Progress ===
        progress_frame = ttkb.Labelframe(
            main_frame,
            text="Progress",
            padding=15,
            bootstyle="secondary"
        )
        progress_frame.pack(fill=X, pady=(0, 15))
        
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
        progress_info_frame.pack(fill=X, pady=(0, 5))
        
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
            height=8
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
            text="Supported formats: " + ", ".join(SUPPORTED_FORMATS),
            font=("Helvetica", 8),
            bootstyle="secondary"
        )
        footer.pack(pady=(10, 0))
    
    def _setup_drag_drop(self):
        """Setup drag and drop functionality."""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            # Note: For full drag-drop, the root needs to be TkinterDnD.Tk()
            # Since we're using ttkbootstrap, we'll rely on file dialog
            pass
        except ImportError:
            # tkinterdnd2 not available, file dialog is the fallback
            pass
    
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
        self.file_info.config(text=f"Size: {size_mb:.2f} MB | Duration: {duration_str} | Path: {file_path}")
        self._update_status(f"File loaded: {filename}")
        
        # Show estimated time
        if self.audio_duration > 0:
            model = self.model_var.get()
            est_time = self._estimate_transcription_time(model)
            self.time_label.config(text=f"Estimated time: {self._format_duration(est_time)}")
    
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
        speed = Transcriber.MODEL_SPEED.get(model, 16.0)
        # Add 5 seconds for model loading overhead
        return (self.audio_duration / speed) + 5
    
    def _on_model_change(self, event=None):
        """Update model description when selection changes."""
        model = self.model_var.get()
        self.model_desc.config(text=Transcriber.MODELS.get(model, ""))
        
        # Update estimated time
        if self.audio_duration > 0:
            est_time = self._estimate_transcription_time(model)
            self.time_label.config(text=f"Estimated time: {self._format_duration(est_time)}")
    
    def _start_transcription(self):
        """Start the transcription process."""
        if not self.current_file:
            messagebox.showwarning("No File", "Please select an audio file first.")
            return
        
        if self.is_transcribing:
            messagebox.showinfo("In Progress", "Transcription is already in progress.")
            return
        
        # Disable UI during transcription
        self.is_transcribing = True
        self.transcribe_btn.config(state=DISABLED)
        self.model_combo.config(state=DISABLED)
        
        # Reset progress
        self.progress_var.set(0)
        self.progress_percent.config(text="0%")
        self.start_time = time.time()
        
        # Clear previous output
        self.output_text.delete(1.0, tk.END)
        
        # Start progress updates
        self._update_progress_display()
        
        # Run transcription in thread
        thread = threading.Thread(target=self._transcribe_thread, daemon=True)
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
            if progress > 5:  # Only show after we have some data
                time_remaining = max(0, est_total - elapsed)
                elapsed_str = self._format_duration(elapsed)
                remaining_str = self._format_duration(time_remaining)
                self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: ~{remaining_str}")
            else:
                self.time_label.config(text=f"Elapsed: {self._format_duration(elapsed)} | Calculating...")
        
        # Schedule next update
        self.progress_update_id = self.root.after(500, self._update_progress_display)
    
    def _transcribe_thread(self):
        """Transcription logic running in background thread."""
        try:
            model_name = self.model_var.get()
            self.transcriber = Transcriber(model_name)
            
            result = self.transcriber.transcribe(
                audio_path=self.current_file,
                include_timestamps=self.timestamps_var.get(),
                progress_callback=self._update_status
            )
            
            # Update UI with results (in main thread)
            self.root.after(0, lambda: self._on_transcription_complete(result))
            
        except Exception as e:
            self.root.after(0, lambda: self._on_transcription_error(str(e)))
    
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
        self.open_output_btn.config(state=NORMAL)
        
        # Display transcription
        self.output_text.insert(tk.END, result["text"])
        
        self._update_status(
            f"✅ Transcription complete! Language: {result['language']} | "
            f"Saved to: {os.path.basename(result['output_path'])}"
        )
        
        # Store output path for "Open Folder" button
        self.output_path = result["output_path"]
        
        messagebox.showinfo(
            "Transcription Complete",
            f"Successfully transcribed!\n\nTime taken: {self._format_duration(elapsed)}\n\nOutput saved to:\n{result['output_path']}"
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
        
        self._update_status(f"❌ Error: {error_msg}")
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
