# 🎙️ Audio Transcriber

A modern, locally-run audio transcription tool powered by **OpenAI's Whisper** model with **MLX GPU acceleration** for Apple Silicon and **speaker diarization** to identify who's talking.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=flat-square&logo=openai)
![MLX](https://img.shields.io/badge/Apple-MLX-000000?style=flat-square&logo=apple)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-lightgrey?style=flat-square)

---

## ✨ Features

- 🚀 **MLX GPU Acceleration** — Runs on Apple Silicon GPU for blazing fast transcription
- 🗣️ **Speaker Diarization** — Identifies who's talking in conversations
- 🎨 **Modern Dark UI** — Clean, intuitive interface built with ttkbootstrap
- 🔒 **100% Local Processing** — Your audio never leaves your machine
- 🌍 **Multi-language Support** — Automatic language detection and transcription
- 📊 **Multiple Model Sizes** — Choose between speed and accuracy
- 📁 **Multiple Formats** — Supports MP3, WAV, M4A, FLAC, OGG, and more

---

## 🖥️ Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│  🎙️ Audio Transcriber    MLX GPU Accelerated • Speaker ID  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │          📂 Click to select an audio file           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Model:           [base ▼]  Fast, good accuracy             │
│  Speaker ID:      ☑️ Identify speakers (requires HF token)  │
│  HuggingFace Token: [••••••••••••••••] [👁]                 │
│  Timestamps:      ☐ Include timestamps in output            │
│                                                             │
│  [🎯 Transcribe]  [📁 Open Output Folder]                   │
│                                                             │
│  ████████████████████░░░░░░░░░░░░  65%                     │
│  Elapsed: 1m 23s | Remaining: ~45s                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- Python 3.8 or higher
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/satyamgoyal06/audio-transcription.git
   cd audio-transcription
   ```

2. **Install FFmpeg** (if not already installed)
   ```bash
   brew install ffmpeg
   ```

3. **Set up virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r desktop-app/requirements.txt
   ```

5. **Run the application**
   ```bash
   python desktop-app/main.py
   ```

---

## 🗣️ Speaker Diarization Setup

To enable speaker identification (who said what), you need a HuggingFace token:

1. **Create account** at [huggingface.co](https://huggingface.co)

2. **Accept model terms** at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

3. **Get your token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

4. **Paste token** in the app's "HuggingFace Token" field

### Output with Speaker Identification

```
============================================================
AUDIO TRANSCRIPTION
============================================================

Source File: interview.mp3
Detected Language: en
Model Used: base
Backend: MLX (GPU)
Speaker Identification: Yes

------------------------------------------------------------

CONVERSATION:

Speaker 1:
  Welcome to the podcast. Today we're discussing AI.

Speaker 2:
  Thanks for having me. I'm excited to be here.

Speaker 1:
  Let's start with the basics. What is machine learning?

Speaker 2:
  Machine learning is a subset of AI that enables systems
  to learn from data without being explicitly programmed.

------------------------------------------------------------

FULL TRANSCRIPTION:

Welcome to the podcast. Today we're discussing AI...

============================================================
```

---

## 📖 How It Works

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Audio File    │────▶│   MLX Whisper    │────▶│   Transcription │
│  (MP3/WAV/...)  │     │   (Apple GPU)    │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  Speaker Labels │◀────│   Pyannote       │◀─────────────┘
│  (Who said what)│     │   (Diarization)  │
└─────────────────┘     └──────────────────┘
```

### Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Transcription | MLX-Whisper | GPU-accelerated speech-to-text |
| Speaker ID | Pyannote Audio | Identify different speakers |
| GUI | ttkbootstrap | Modern themed interface |
| Audio Processing | FFmpeg | Format conversion & duration |

### Model Sizes

| Model  | Speed | Accuracy | RAM | Best For |
|--------|-------|----------|-----|----------|
| tiny   | ⚡⚡⚡⚡ | ★★☆☆ | ~1 GB | Quick drafts |
| base   | ⚡⚡⚡ | ★★★☆ | ~1.5 GB | **Daily use** |
| small  | ⚡⚡ | ★★★★ | ~2.5 GB | Better accuracy |
| medium | ⚡ | ★★★★☆ | ~5 GB | High quality |
| large  | 🐢 | ★★★★★ | ~10 GB | Maximum accuracy |

---

## 📂 Project Structure

```
audio-transcription/
├── desktop-app/
│   ├── main.py           # Application entry point
│   ├── gui.py            # GUI with speaker diarization support
│   ├── transcriber.py    # MLX transcription + diarization engine
│   └── requirements.txt  # Python dependencies
│
├── README.md             # This file
├── LICENSE               # MIT License
└── .gitignore
```

---

## 🔧 Supported Audio Formats

| Format | Extension |
|--------|-----------|
| MP3    | `.mp3`    |
| WAV    | `.wav`    |
| M4A    | `.m4a`    |
| FLAC   | `.flac`   |
| OGG    | `.ogg`    |
| WMA    | `.wma`    |
| AAC    | `.aac`    |
| Opus   | `.opus`   |
| WebM   | `.webm`   |
| MP4    | `.mp4`    |

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
- [ ] Export to SRT/VTT subtitle formats
- [ ] Real-time microphone transcription
- [ ] Multi-language translation
- [ ] Custom speaker naming

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition model
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization
- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) — Beautiful themed Tkinter

---

## 📬 Contact

**Satyam Goyal**

- GitHub: [@satyamgoyal06](https://github.com/satyamgoyal06)

---

<p align="center">
  Made with ❤️ using 🎙️ Whisper + 🍎 MLX
</p>
