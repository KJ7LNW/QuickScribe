# QuickScribe

Real-time AI-powered dictation application with multiple audio source options and intelligent text processing. Record audio with configurable triggers, transcribe using local models or cloud APIs, and automatically inject professionally formatted text into any application.

**⚠️ Privacy Notice:** All processing modes send data to remote AI models. Raw audio mode sends audio directly to the LLM. Transcription mode sends audio to transcription service, then sends text to LLM for formatting.

## Features

- **Audio Sources**
  - Raw microphone (default): Direct audio to LLM
  - Transcription: Audio → transcription model → text → LLM
- **LLM Providers**
  - Cloud: Groq, Google Gemini, OpenAI, Anthropic, OpenRouter
  - Local: HuggingFace text generation models, llama.cpp GGUF models (requires transcription mode)
  - None: Passthrough mode (injects raw transcription without LLM processing)
- **Transcription Models** (when using transcription audio source)
  - HuggingFace Wav2Vec2 (local, phoneme-based)
  - HuggingFace Whisper/Speech2Text (local, GPU-accelerated)
  - HuggingFace NeMo TDT (local, NVIDIA transducer models)
  - OpenAI Whisper (cloud API)
  - Groq Whisper (cloud API)
  - VOSK (local, offline)
- **Text Processing**
  - Real-time streaming with incremental updates
  - Grammar correction, punctuation, technical term formatting
  - Conversation context across recordings
- **Input Control**
  - Keyboard triggers (configurable key)
  - POSIX signals (SIGUSR1/SIGUSR2)
- **Output**
  - macOS: Native keyboard injection via Accessibility API
  - Linux: xdotool with configurable keystroke rate

## Requirements

### Dependencies
```bash
pip install -r requirements.txt
```

### System Dependencies
**Linux**: `sudo apt-get install xdotool`

### Permissions
- Microphone access (all modes)
- Accessibility/input monitoring (keyboard triggers and text injection)
- **macOS**: System Settings → Privacy & Security → Accessibility

## Installation

1. **Clone and install dependencies:**
   ```bash
   git clone <repository-url>
   cd QuickScribe
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   ```bash
   # Create .env file with your API keys
   echo "GROQ_API_KEY=your_groq_key_here" > .env
   echo "GOOGLE_API_KEY=your_google_key_here" >> .env
   ```

## Configuration Options

### Core Arguments

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | | None | Model in format `provider/model` (e.g., `groq/llama-3.3-70b-versatile`) |
| `--audio-source` | `-a` | `raw` | Audio source: `raw` (audio→LLM) or `transcribe`/`trans` (audio→transcription→LLM) |
| `--transcription-model` | `-T` | `huggingface/...` | Transcription model when using `-a transcribe` (format: `provider/model`) |

### Model Format

Both `--model` and `--transcription-model` use format: `provider/identifier`

**LLM providers**: `groq`, `gemini`, `openai`, `anthropic`, `openrouter`, `huggingface`, `llamacpp`, `gguf`, `none`
**Transcription providers**: `huggingface`, `openai`, `groq`, `vosk`

**Note**: HuggingFace transcription provider supports multiple architectures (CTC, Seq2Seq, NeMo TDT) with automatic detection.

**LlamaCpp/GGUF provider**: `gguf` is an alias for `llamacpp`

**None provider**: Bypasses LLM processing and injects raw transcription output directly. Useful for:
- Phoneme passthrough from Wav2Vec2
- Direct text injection from Whisper/VOSK without formatting
- Testing transcription models without API costs
- Low-latency dictation (transcription only, no LLM processing)

The none provider extracts the first `<tx>` tag from transcription output and injects it verbatim. Works with any `-T` transcription model.

**Routing syntax**: `provider/model@routing_provider` (provider-specific feature)
- Example: `openrouter/google/gemini-2.5-flash@vertex`
- The `@routing_provider` suffix is passed to the provider for custom routing
- Supported by providers with routing capabilities (e.g., OpenRouter)

### Other Options

| Option | Default | Description |
|--------|---------|-------------|
| `--trigger-key` | `alt_r` | Keyboard trigger key |
| `--no-trigger-key` | disabled | Use SIGUSR1/SIGUSR2 signals instead of keyboard |
| `--xdotool-hz` | None | Keystroke rate for xdotool (Linux) |
| `--enable-reasoning` | `low` | Reasoning level: `none`, `low`, `medium`, `high` |
| `--temperature` | `0.2` | LLM temperature (0.0-2.0) |
| `--debug` / `-D` | disabled | Debug output |

## Usage

### Interactive Mode
```bash
python dictate.py
```

### Specify Model
```bash
# Raw audio → LLM
python dictate.py --model groq/llama-3.3-70b-versatile

# Transcription → Cloud LLM
python dictate.py -a transcribe -T openai/whisper-1 --model anthropic/claude-3-5-sonnet-20241022

# Transcription → Direct injection (no LLM)
python dictate.py -a transcribe -T vosk/vosk-model-small-en-us-0.15 --model none/

# NeMo TDT → Direct injection (no LLM)
python dictate.py -a transcribe -T huggingface/nvidia/parakeet-tdt-0.6b-v3 --model none/

# Transcription → Local GGUF LLM
python dictate.py -a transcribe -T vosk/vosk-model-small-en-us-0.15 --model gguf/bartowski/Qwen2.5-0.5B-Instruct-GGUF@Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

# Transcription → Local HuggingFace LLM
python dictate.py -a transcribe -T openai/whisper-1 --model huggingface/Qwen/Qwen2.5-0.5B-Instruct
```

### Signal Control (background mode)
```bash
python dictate.py --model groq/llama-3.3-70b-versatile --no-trigger-key &
PID=$!
kill -USR1 $PID  # Start recording
kill -USR2 $PID  # Stop recording
```

## How It Works

1. Hold trigger key (or send SIGUSR1 signal)
2. Audio captured → processed (raw or transcribed) → sent to LLM
3. Text streamed back and injected into active application
4. Conversation context maintained across recordings

## Audio Sources Explained

### Raw Audio (`-a raw`, default)
Audio sent directly to LLM for transcription and formatting.

**When to use:** LLM supports audio input (Gemini, OpenAI, Anthropic)
**Note:** Groq LLMs do not support audio; use transcription mode

### Transcription Mode (`-a transcribe`)
Two-stage: audio → transcription model → text → LLM → formatted text

**When to use:**
- LLM lacks audio support (Groq)
- Lower cost (cheap transcription + expensive LLM)
- Local/offline transcription (VOSK, Wav2Vec2)
- Fully local processing (HuggingFace transcription + HuggingFace LLM)

**Example:** "their are too errors hear" → transcription → "their are too errors hear" → LLM → "There are two errors here."

LLM corrects: homophones, grammar, punctuation, technical terms

### LlamaCpp GGUF Models (Local)

Run quantized GGUF models locally via llama.cpp bindings.

**Requirement:** Must use transcription mode (`-a transcribe`) because GGUF text models cannot process audio directly.

**Format:** `llamacpp/repo/model@filename.gguf` or `gguf/repo/model@filename.gguf`

**Syntax:**
```bash
# Using llamacpp prefix
--model llamacpp/bartowski/Qwen2.5-0.5B-Instruct-GGUF@Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

# Using gguf alias
--model gguf/unsloth/Qwen3-4B-GGUF@Qwen3-4B-Q4_K_M.gguf
```

**Examples:**
```bash
# Small model with VOSK transcription
python dictate.py -a transcribe -T vosk/vosk-model-small-en-us-0.15 \
  --model gguf/bartowski/Qwen2.5-0.5B-Instruct-GGUF@Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

# Larger model with Whisper transcription
python dictate.py -a transcribe -T openai/whisper-1 \
  --model llamacpp/unsloth/Qwen3-4B-GGUF@Qwen3-4B-Q4_K_M.gguf
```

**Notes:**
- Models downloaded from HuggingFace Hub on first use
- Quantization formats: Q4_K_M, Q5_K_M, Q6_K, Q8_0
- CPU and GPU inference supported
- No API key required
- Install: `pip install llama-cpp-python`
- GPU support: `CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python`

### HuggingFace LLM Models (Local)

Run text generation models locally via HuggingFace Transformers library.

**Requirements:**
- Must use transcription mode (`-a transcribe`) because HuggingFace text models cannot process audio directly
- CUDA GPU required (rejects CPU-only systems to prevent excessive RAM usage)
- Instruction-tuned models only (models with chat templates)

**Format:** `huggingface/repo/model` or `huggingface/repo/model@bits`

**Quantization options:**
- `@16` - Float16 (default, ~1-4GB VRAM)
- `@8` - 8-bit quantization (~0.5-2GB VRAM, requires bitsandbytes)
- `@4` - 4-bit quantization (~0.3-1GB VRAM, requires bitsandbytes)

**Supported architectures:**
- Causal LM: Qwen, Llama, Mistral, Phi, GPT-2
- Seq2Seq: T5, BART, Flan-T5

**Examples:**
```bash
# Float16 (default)
python dictate.py -a transcribe -T vosk/vosk-model-small-en-us-0.15 \
  --model huggingface/Qwen/Qwen2.5-0.5B-Instruct

# 4-bit quantization (lowest VRAM)
python dictate.py -a transcribe -T openai/whisper-1 \
  --model huggingface/Qwen/Qwen2.5-0.5B-Instruct@4

# 8-bit quantization
python dictate.py -a transcribe -T vosk/vosk-model-small-en-us-0.15 \
  --model huggingface/meta-llama/Llama-3.2-1B-Instruct@8
```

**Notes:**
- Models downloaded from HuggingFace Hub on first use
- Runtime quantization via bitsandbytes: `pip install bitsandbytes`
- Real streaming via TextIteratorStreamer
- No API key required

**Comparison with LlamaCpp:**
- HuggingFace: CUDA GPU required, runtime quantization, full model format
- LlamaCpp: CPU or GPU, pre-quantized GGUF files, broader hardware support

### Transcription Models

#### OpenAI Whisper (Cloud API)

Cloud-based transcription with best quality and ease of use.

**Model:** `whisper-1`

**Features:**
- No local model download
- Requires API key
- Best transcription quality

```bash
python dictate.py -a transcribe -T openai/whisper-1 --model anthropic/claude-3-5-sonnet-20241022
```

#### HuggingFace Seq2Seq Models (Local)

Local Whisper and Speech2Text models with GPU acceleration.

**Supported architectures:**
- Whisper (all variants)
- Speech2Text

**Features:**
- Plain text output
- GPU acceleration when available
- No streaming support

**Whisper models:**

```bash
# Standard Whisper
-T huggingface/openai/whisper-tiny        # ~39 MB, fastest, quick testing
-T huggingface/openai/whisper-base        # ~74 MB, fast, low-resource
-T huggingface/openai/whisper-small       # ~244 MB, balanced
-T huggingface/openai/whisper-medium      # ~769 MB, high accuracy
-T huggingface/openai/whisper-large-v3    # ~1.5 GB, best accuracy
-T huggingface/openai/whisper-large-v3-turbo

# Distil-Whisper (6x faster)
-T huggingface/distil-whisper/distil-large-v2  # ~756 MB, production use
-T huggingface/distil-whisper/distil-medium.en
```

**Speech2Text models (requires: pip install sentencepiece):**

```bash
-T huggingface/facebook/s2t-small-librispeech-asr
-T huggingface/facebook/s2t-large-librispeech-asr
```

**Example:**

```bash
python dictate.py -a transcribe -T huggingface/openai/whisper-small --model groq/llama-3.3-70b-versatile
```

#### HuggingFace NeMo TDT Models (Local)

NVIDIA NeMo ASR models using Token-and-Duration Transducer architecture.

**Supported models:**
- Parakeet TDT (FastConformer + TDT decoder)
- High accuracy with efficient decoding
- Built-in language detection (25+ languages)
- Timestamp support (word, segment, character level)

**Features:**
- Plain text output with punctuation and capitalization
- GPU acceleration strongly recommended
- No streaming support
- Supports audio up to 24 minutes (full attention) or 3 hours (local attention)

**Requirements:**
```bash
pip install nemo_toolkit[asr]
```

**Available models:**

```bash
-T huggingface/nvidia/parakeet-tdt-0.6b-v3  # 600M params, state-of-the-art accuracy
-T huggingface/nvidia/parakeet-tdt-0.6b-v2  # 600M params, previous version
-T huggingface/nvidia/parakeet_realtime_eou_120m-v1  # 120M params, real-time with end-of-utterance detection
```

**Examples:**

```bash
# With LLM formatting
python dictate.py -a transcribe -T huggingface/nvidia/parakeet-tdt-0.6b-v3 --model groq/llama-3.3-70b-versatile

# Direct injection (no LLM)
python dictate.py -a transcribe -T huggingface/nvidia/parakeet-tdt-0.6b-v3 --model none/
```

**Notes:**
- Models downloaded from HuggingFace Hub on first use (~2.5GB)
- CUDA GPU strongly recommended (CPU inference very slow)
- First load includes tokenizer and model initialization (~30 seconds)
- Automatic architecture detection (no manual configuration needed)

#### VOSK Models (Local, Offline)

Lightweight offline speech recognition with streaming support.

**Features:**
- Offline processing
- Streaming support
- Language-specific models
- Fast setup

**Usage:**

```bash
# Model name (auto-downloads to ~/.cache/vosk/)
python dictate.py -a transcribe -T vosk/vosk-model-small-en-us-0.15 --model groq/llama-3.3-70b-versatile
python dictate.py -a transcribe -T vosk/vosk-model-en-us-0.22-lgraph --model groq/llama-3.3-70b-versatile

# Local paths (relative, absolute, or home directory)
python dictate.py -a transcribe -T vosk/~/models/vosk-model-en-us --model groq/llama-3.3-70b-versatile
python dictate.py -a transcribe -T vosk//usr/share/vosk/model --model groq/llama-3.3-70b-versatile
python dictate.py -a transcribe -T vosk/./models/vosk-model --model groq/llama-3.3-70b-versatile
```

**Model downloads:** https://alphacephei.com/vosk/models

**Model loading:** Accepts model name (downloads automatically with progress bar) or local path. Path formats: relative (`models/vosk`), absolute (`/usr/share/vosk/model`), home directory (`~/models/vosk`). Downloaded models cache to `~/.cache/vosk/`. Override cache location with `VOSK_MODEL_PATH` environment variable.

#### HuggingFace CTC Models (Local, Advanced)

Connectionist Temporal Classification models for specialized use cases.

**Supported architectures:**
- Wav2Vec2
- HuBERT
- Data2VecAudio
- UniSpeech
- UniSpeechSat
- SEW
- SEWD
- MCTCT

**Features:**
- Multi-speed processing (0.80x, 0.85x, 0.90x, 0.95x)
- Phoneme (IPA) or character output
- Streaming support

**Phoneme-based models (IPA output):**

```bash
-T huggingface/facebook/wav2vec2-lv-60-espeak-cv-ft
```

**Character-based models (text output):**

```bash
-T huggingface/facebook/wav2vec2-base-960h
-T huggingface/facebook/hubert-large-ls960-ft
-T huggingface/facebook/data2vec-audio-base-960h
```

**Example:**

```bash
python dictate.py -a transcribe -T huggingface/facebook/wav2vec2-lv-60-espeak-cv-ft --model groq/llama-3.3-70b-versatile
```

#### Model Comparison

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| whisper-tiny | ~39 MB | Fastest | Quick testing |
| whisper-base | ~74 MB | Fast | Low-resource environments |
| whisper-small | ~244 MB | Medium | Balanced accuracy/speed |
| whisper-medium | ~769 MB | Slow | High accuracy needed |
| whisper-large-v3 | ~1.5 GB | Slowest | Best accuracy |
| distil-whisper/distil-large-v2 | ~756 MB | 6x faster | Production (faster alternative to large) |
| nvidia/parakeet-tdt-0.6b-v3 | ~2.5 GB | Fast (GPU) | State-of-the-art accuracy, multilingual |

#### Architecture Detection (HuggingFace)

HuggingFace provider automatically detects model architecture:

1. Checks for NeMo `.nemo` checkpoint files
2. If NeMo found, loads via nemo_toolkit
3. Otherwise, attempts to load as CTC model
4. If CTC fails, attempts to load as Seq2Seq model
5. If all fail, returns error with supported architecture types

You do not need to specify the architecture type - just use `huggingface/<model-id>`.

**Supported architectures:**
- **NeMo TDT**: FastConformer-TDT transducer models (nvidia/parakeet-*)
- **CTC**: Wav2Vec2, HuBERT, Data2VecAudio (multi-speed, phoneme/text output)
- **Seq2Seq**: Whisper, Speech2Text (GPU-accelerated text output)

#### Testing Model Compatibility

To test if a HuggingFace model is supported:

```bash
python dictate.py --audio-source transcribe -T huggingface/<model-id>
```

Watch for log messages:
- "Successfully loaded NeMo TDT model" → NeMo transducer architecture
- "Successfully loaded as CTC model" → Multi-speed + phoneme/text output
- "Successfully loaded as Seq2Seq model" → GPU-accelerated text output
- "not compatible with CTC or Seq2Seq" → Model not supported (or missing nemo_toolkit)

#### Requirements

**HuggingFace Models:**
```bash
# Core dependencies
pip install torch transformers huggingface_hub pyrubberband

# Optional (for specific models)
pip install sentencepiece  # Required for Speech2Text models
pip install nemo_toolkit[asr]  # Required for NeMo TDT models (nvidia/parakeet-*)
```

**OpenAI Models:**
```bash
pip install litellm soundfile
```

**Vosk Models:**
```bash
pip install vosk
```

## Troubleshooting

- **No microphone**: Check system permissions
- **macOS text injection fails**: Grant accessibility permissions (System Settings → Privacy & Security)
- **Linux text injection fails**: Install xdotool
- **High latency**: Set `--enable-reasoning none`, lower `--temperature`
