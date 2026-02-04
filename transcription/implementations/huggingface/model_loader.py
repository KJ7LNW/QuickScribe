"""HuggingFace model loading with automatic architecture detection."""

import os
import sys
import tarfile

sys.path.insert(0, 'lib')
from pr_log import pr_info, pr_err

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


def load_huggingface_model(model_path: str, cache_dir=None, force_download=False, local_files_only=False):
    """
    Load HuggingFace model and automatically detect architecture type.

    Attempts to load model with AutoModelForCTC first, then AutoModelForSpeechSeq2Seq.
    Returns the successfully loaded model along with its processor and architecture type.

    Args:
        model_path: HuggingFace model identifier or path
        cache_dir: Optional cache directory
        force_download: Force re-download of model files
        local_files_only: Use only local cached files

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

                # Create persistent extraction directory
                cache_key = model_path.replace("/", "--")
                extract_dir = os.path.join(NEMO_EXTRACT_CACHE, cache_key)
                marker_file = os.path.join(extract_dir, ".extracted")

                # Extract if not already done
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

                pr_info("Loading NeMo ASR model")
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_path, save_restore_connector=connector
                )
                model.eval()

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
            local_files_only=offline_mode
        )

        processor = load_processor_with_fallback(
            model_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=offline_mode
        )

        model.eval()
        pr_info(f"Successfully loaded as CTC model: {model_path}")
        return model, processor, 'ctc'

    except Exception as ctc_error:
        pr_info(f"Not a CTC model, trying Seq2Seq: {ctc_error}")

    try:
        pr_info("Attempting to load as Seq2Seq model")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=offline_mode
        )

        model = model.to(device, dtype=torch_dtype)

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
