"""HuggingFace model loading with automatic architecture detection."""

import os
import sys
import tarfile

sys.path.insert(0, 'lib')
from pr_log import pr_info, pr_err, pr_notice

try:
    import torch
    from transformers import AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor
except ImportError:
    torch = None
    AutoModelForCTC = None
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None

try:
    from huggingface_hub.utils import is_offline_mode
except ImportError:
    is_offline_mode = None

try:
    from huggingface_hub import HfApi
    hf_api = HfApi()
except ImportError:
    hf_api = None

try:
    import nemo.collections.asr as nemo_asr
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
except ImportError:
    nemo_asr = None
    SaveRestoreConnector = None

NEMO_EXTRACT_CACHE = os.path.expanduser("~/.cache/nemo_extracted")

from .processor_utils import load_processor_with_fallback


def _resolve_device(device: str) -> str:
    """Resolve device string to a concrete 'cuda' or 'cpu' value."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            pr_notice("CUDA not available, using CPU for transcription")
            return "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            pr_err("--transcription-device cuda requested but CUDA is not available")
            raise ValueError("CUDA requested but not available")
        return device
    elif device == "cpu":
        return device
    else:
        raise ValueError(f"Unknown device: {device}")


def _apply_precision(model, precision: str, device: str):
    """Apply numeric precision conversion to model after loading."""
    if precision == "auto":
        if device == "cuda":
            return model.half()
        else:
            return model
    elif precision == "fp16":
        return model.half()
    elif precision == "bf16":
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            pr_err("--transcription-precision bf16 requested but GPU does not support bfloat16")
            raise ValueError("bfloat16 not supported on this GPU")
        return model.bfloat16()
    elif precision == "int8":
        if device == "cuda":
            pr_err("--transcription-precision int8 is CPU-only; use --transcription-device cpu")
            raise ValueError("int8 dynamic quantization is CPU-only")

        # torch.ao.quantization replaces Linear layers with int8-quantized versions
        return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    elif precision == "fp32":
        return model
    else:
        raise ValueError(f"Unknown precision: {precision}")


def _compute_torch_dtype(precision: str, device: str):
    """Return torch dtype to pass to from_pretrained, or None for framework default (fp32)."""
    if precision == "auto":
        if device == "cuda":
            return torch.float16
        else:
            return None
    elif precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else:
        return None


def _move_to_device(model, device: str):
    """Move model to device, falling back to CPU on OutOfMemoryError."""
    try:
        return model.to(device)
    except torch.cuda.OutOfMemoryError:
        pr_notice(f"CUDA out of memory moving model to {device}, falling back to CPU")
        return model.to("cpu")


def load_huggingface_model(
    model_path: str,
    cache_dir=None,
    force_download=False,
    local_files_only=False,
    device="auto",
    precision="auto"
):
    """
    Load HuggingFace model and automatically detect architecture type.

    Attempts to load model with AutoModelForCTC first, then AutoModelForSpeechSeq2Seq.
    Returns the successfully loaded model along with its processor and architecture type.

    Args:
        model_path: HuggingFace model identifier or path
        cache_dir: Optional cache directory
        force_download: Force re-download of model files
        local_files_only: Use only local cached files
        device: Target device — 'auto', 'cuda', or 'cpu'
        precision: Numeric precision — 'auto', 'fp32', 'fp16', 'bf16', or 'int8'

    Returns:
        Tuple of (model, processor, architecture_type)
        where architecture_type is 'ctc', 'whisper', or 'speech2text'

    Raises:
        ValueError: If model is not compatible with either CTC or Seq2Seq
        ImportError: If required libraries are not installed
    """
    if torch is None or AutoModelForCTC is None:
        raise ImportError("PyTorch and transformers libraries not installed")

    offline_mode = local_files_only
    if not offline_mode and is_offline_mode and callable(is_offline_mode):
        offline_mode = is_offline_mode()

    pr_info(f"Loading model: {model_path}")

    resolved_device = _resolve_device(device)

    # Check local extraction cache before any network calls
    cache_key = model_path.replace("/", "--")
    extract_dir = os.path.join(NEMO_EXTRACT_CACHE, cache_key)
    marker_file = os.path.join(extract_dir, ".extracted")

    if os.path.isfile(marker_file) and nemo_asr is not None and SaveRestoreConnector is not None:
        pr_info(f"Using cached NeMo extraction: {extract_dir}")

        connector = SaveRestoreConnector()
        connector.model_extracted_dir = extract_dir

        # Force CPU loading so precision conversion happens before device placement
        connector.map_location = torch.device('cpu')

        pr_info("Loading NeMo ASR model")
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_path,
            save_restore_connector=connector,
            map_location=torch.device('cpu')
        )
        model.eval()
        model = _apply_precision(model, precision, resolved_device)
        model = _move_to_device(model, resolved_device)

        pr_info(f"Successfully loaded NeMo TDT model: {model_path}")
        return model, None, 'nemo_tdt'

    if hf_api is not None:
        try:
            repo_files = hf_api.list_repo_files(model_path)
            nemo_files = [f for f in repo_files if f.endswith('.nemo')]

            if nemo_files:
                pr_info(f"Detected NeMo model: {nemo_files[0]}")

                if nemo_asr is None:
                    raise ImportError(
                        f"Model {model_path} is a NeMo model but nemo_toolkit not installed. "
                        "Install with: pip install nemo_toolkit[asr]"
                    )

                # Get the cached .nemo file path without loading
                nemo_file = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_path, return_model_file=True
                )

                # Extract if not already done (marker_file computed above)
                if not os.path.isfile(marker_file):
                    pr_info(f"Extracting NeMo model to {extract_dir}")
                    os.makedirs(extract_dir, exist_ok=True)
                    with tarfile.open(nemo_file, "r") as tar:
                        tar.extractall(extract_dir)
                    with open(marker_file, "w") as f:
                        f.write(nemo_files[0])
                else:
                    pr_info(f"Using cached extraction: {extract_dir}")

                # Load using the pre-extracted directory
                connector = SaveRestoreConnector()
                connector.model_extracted_dir = extract_dir

                # Force CPU loading so precision conversion happens before device placement
                connector.map_location = torch.device('cpu')

                pr_info("Loading NeMo ASR model")
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_path,
                    save_restore_connector=connector,
                    map_location=torch.device('cpu')
                )
                model.eval()
                model = _apply_precision(model, precision, resolved_device)
                model = _move_to_device(model, resolved_device)

                pr_info(f"Successfully loaded NeMo TDT model: {model_path}")
                return model, None, 'nemo_tdt'

        except ImportError:
            raise
        except Exception as nemo_error:
            pr_info(f"Not a NeMo model or detection failed: {nemo_error}")

    try:
        pr_info("Attempting to load as CTC model")
        model = AutoModelForCTC.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=offline_mode,
            torch_dtype=_compute_torch_dtype(precision, resolved_device)
        )

        processor = load_processor_with_fallback(
            model_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=offline_mode
        )

        model.eval()
        model = _apply_precision(model, precision, resolved_device)
        model = _move_to_device(model, resolved_device)
        pr_info(f"Successfully loaded as CTC model: {model_path}")
        return model, processor, 'ctc'

    except Exception as ctc_error:
        pr_info(f"Not a CTC model, trying Seq2Seq: {ctc_error}")

    try:
        pr_info("Attempting to load as Seq2Seq model")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=offline_mode,
            torch_dtype=_compute_torch_dtype(precision, resolved_device)
        )

        model = _apply_precision(model, precision, resolved_device)
        model = _move_to_device(model, resolved_device)

        processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=offline_mode
        )

        model.eval()

        model_type = model.config.model_type if hasattr(model.config, 'model_type') else 'seq2seq'

        if model_type == 'whisper':
            pr_info(f"Successfully loaded as Whisper model: {model_path}")
            return model, processor, 'whisper'
        elif model_type == 'speech_to_text':
            pr_info(f"Successfully loaded as Speech2Text model: {model_path}")
            return model, processor, 'speech2text'
        else:
            pr_info(f"Successfully loaded as Seq2Seq model ({model_type}): {model_path}")
            return model, processor, 'speech2text'

    except Exception as seq2seq_error:
        error_msg = str(seq2seq_error)
        pr_err(f"Failed to load as Seq2Seq: {error_msg}")

        if "SentencePiece" in error_msg:
            raise ImportError(
                f"Model {model_path} requires SentencePiece library. "
                "Install with: pip install sentencepiece"
            ) from None
        elif "protobuf" in error_msg:
            raise ImportError(
                f"Model {model_path} requires protobuf library. "
                "Install with: pip install protobuf"
            ) from None
        else:
            raise ValueError(
                f"Model {model_path} not compatible with CTC or Seq2Seq architectures. "
                f"Error: {error_msg}"
            ) from None
