# 🎙️ Audio Transcriber

A modern, locally-run audio transcription tool powered by **OpenAI's Whisper** model. Convert any audio file to text with a beautiful desktop GUI — no internet required after initial setup.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=flat-square&logo=openai)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey?style=flat-square)

---

## ✨ Features

- 🎨 **Modern Dark UI** — Clean, intuitive interface built with ttkbootstrap
- 🔒 **100% Local Processing** — Your audio never leaves your machine
- 🌍 **Multi-language Support** — Automatic language detection and transcription
- ⏱️ **Timestamped Output** — Get precise timestamps for each segment
- 📊 **Multiple Model Sizes** — Choose between speed and accuracy
- 📁 **Multiple Formats** — Supports MP3, WAV, M4A, FLAC, OGG, and more

---

## 🖥️ Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│  🎙️ Audio Transcriber                Powered by Whisper    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │     📂 Click to select or drag & drop audio file    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Model: [base ▼]  Fast, good accuracy (~1.5GB RAM)         │
│  ☑️ Include timestamps in output                            │
│                                                             │
│  [🎯 Transcribe]  [📁 Open Output Folder]                   │
│                                                             │
│  ═══════════════════════════════════════════════════════   │
│  Ready. Select an audio file to begin.                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/audio-transcriber.git
   cd audio-transcriber
   ```

2. **Install FFmpeg** (if not already installed)
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Windows:**
   ```bash
   # Using chocolatey
   choco install ffmpeg
   
   # Or download from https://ffmpeg.org/download.html
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt update && sudo apt install ffmpeg
   ```

3. **Set up virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   cd desktop-app
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

---

## 📖 How It Works

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Audio File    │────▶│  Whisper Model   │────▶│   Text Output   │
│  (MP3/WAV/...)  │     │  (Local Process) │     │  (.txt file)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
    User drops file      Model transcribes        Timestamped text
    into the GUI         audio to text            saved to disk
```

### The Whisper Model

[Whisper](https://github.com/openai/whisper) is OpenAI's open-source automatic speech recognition (ASR) system. It was trained on 680,000 hours of multilingual data and demonstrates robust performance across many languages.

**Key capabilities:**
- Speech recognition in 99 languages
- Translation to English
- Language identification
- Robust to accents, background noise, and technical language

### Model Sizes

| Model  | Parameters | Required VRAM | Relative Speed | Best For |
|--------|------------|---------------|----------------|----------|
| tiny   | 39 M       | ~1 GB         | ~32x           | Quick drafts, testing |
| base   | 74 M       | ~1.5 GB       | ~16x           | Everyday use |
| small  | 244 M      | ~2.5 GB       | ~6x            | Better accuracy |
| medium | 769 M      | ~5 GB         | ~2x            | High-quality transcription |
| large  | 1550 M     | ~10 GB        | 1x             | Maximum accuracy |

---

## 📂 Project Structure

```
audio-transcriber/
├── desktop-app/
│   ├── main.py           # Application entry point
│   ├── gui.py            # GUI implementation (ttkbootstrap)
│   ├── transcriber.py    # Core transcription engine
│   └── requirements.txt  # Python dependencies
│
├── README.md             # This file
└── LICENSE               # MIT License
```

---

## 🎯 Usage

### Basic Usage

1. Launch the application with `python main.py`
2. Click the drop zone or drag & drop an audio file
3. Select your preferred model (base is recommended for most uses)
4. Toggle timestamps if needed
5. Click **Transcribe**
6. Find your transcription saved alongside the original audio file

### Output Format

The generated text file includes:

```
============================================================
AUDIO TRANSCRIPTION
============================================================

Source File: interview.mp3
Detected Language: en
Model Used: base
Transcribed: 2024-01-29 21:55:00

------------------------------------------------------------

TIMESTAMPED TRANSCRIPTION:

[00:00:00.000 --> 00:00:05.500]
Welcome to the podcast, today we're discussing...

[00:00:05.500 --> 00:00:12.300]
Thank you for having me, I'm excited to be here.

------------------------------------------------------------

FULL TRANSCRIPTION:

Welcome to the podcast, today we're discussing...
Thank you for having me, I'm excited to be here.

============================================================
```

---

## 🔧 Supported Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| MP3    | `.mp3`    | Most common |
| WAV    | `.wav`    | Uncompressed |
| M4A    | `.m4a`    | Apple format |
| FLAC   | `.flac`   | Lossless |
| OGG    | `.ogg`    | Open format |
| WMA    | `.wma`    | Windows Media |
| AAC    | `.aac`    | Advanced Audio |
| Opus   | `.opus`   | High quality |
| WebM   | `.webm`   | Web format |
| MP4    | `.mp4`    | Video (audio track) |

---

## 🛠️ Technical Details

### Dependencies

| Package | Purpose |
|---------|---------|
| `openai-whisper` | Core transcription model |
| `torch` | PyTorch for ML inference |
| `ttkbootstrap` | Modern themed Tkinter widgets |
| `tkinterdnd2` | Drag-and-drop support (optional) |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8+ GB |
| Storage | 2 GB | 5+ GB (for larger models) |
| Python | 3.8 | 3.10+ |
| OS | macOS 10.14+ / Windows 10 / Ubuntu 18.04+ | Latest |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Ideas for Contribution

- [ ] Add CLI mode for batch processing
- [ ] Implement audio waveform visualization
- [ ] Add export to SRT/VTT subtitle formats
- [ ] Create Windows installer (.exe)
- [ ] Add speaker diarization (who said what)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — The incredible speech recognition model
- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) — Beautiful themed Tkinter widgets
- [FFmpeg](https://ffmpeg.org/) — Audio processing powerhouse

---

## 📬 Contact

**Satyam Goyal**

- GitHub: [@yourusername](https://github.com/yourusername)
- Portfolio: [yourwebsite.com](https://yourwebsite.com)

---

<p align="center">
  Made with ❤️ and 🎙️
</p>
