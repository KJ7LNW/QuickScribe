"""
llama.cpp provider for local GGUF model inference.

Supports quantized models from HuggingFace Hub in GGUF format.
Requires --audio-source transcribe mode (cannot process audio directly).
Uses llama-cpp-python bindings for efficient CPU/GPU inference.
"""
from typing import Optional
from .base_provider import AbstractProvider
from lib.pr_log import pr_err, pr_info, pr_debug


class LlamaCppProvider(AbstractProvider):
    """
    llama.cpp provider for local GGUF model inference.

    Loads GGUF models from HuggingFace Hub and runs inference locally.
    Only works with transcription mode since models require text input.
    Supports both CPU and GPU inference via llama-cpp-python.
    """

    def __init__(self, config, audio_processor):
        super().__init__(config, audio_processor)

        self.model = None
        self.repo_id = None
        self.gguf_file = None

    def initialize(self) -> bool:
        """Load GGUF model from HuggingFace Hub via llama-cpp-python."""

        # Validation
        if not self.route:
            pr_err("GGUF filename required after @ delimiter")
            pr_err(f"Example: llamacpp/bartowski/Qwen2.5-0.5B-Instruct-GGUF@Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
            return False

        if not self.route.endswith('.gguf'):
            pr_err(f"Invalid GGUF filename: {self.route}")
            pr_err("Route must end with .gguf extension")
            return False

        # Repository extraction
        raw_path = self.model_without_route
        if raw_path.lower().startswith('llamacpp/'):
            self.repo_id = raw_path[len('llamacpp/'):]
        elif raw_path.lower().startswith('gguf/'):
            self.repo_id = raw_path[len('gguf/'):]
        else:
            pr_err(f"Unexpected provider prefix in model path: {raw_path}")
            pr_err("Expected 'llamacpp/' or 'gguf/' prefix")
            return False

        self.gguf_file = self.route

        pr_info(f"Loading GGUF model: {self.repo_id}/{self.gguf_file}")

        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download

            # Model download
            pr_info("Downloading model from HuggingFace Hub...")
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.gguf_file
            )
            pr_info(f"Model cached at: {model_path}")

            # Model initialization
            pr_info("Initializing llama.cpp model...")
            self.model = Llama(
                model_path=model_path,
                n_ctx=0,
                n_gpu_layers=0,
                verbose=False
            )

            pr_info(f"Loaded GGUF model: {self.repo_id}/{self.gguf_file}")
            self._initialized = True
            return True

        except ImportError as e:
            pr_err(f"llama-cpp-python not installed: {e}")
            pr_err("Install with: pip install llama-cpp-python")
            pr_err("For GPU support: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python")
            return False

        except Exception as e:
            pr_err(f"Failed to load GGUF model: {e}")
            pr_err(f"Repository: {self.repo_id}")
            pr_err(f"File: {self.gguf_file}")
            return False

    def _generate_response(self, instructions: str, context, audio_data, text_data):
        """
        Generate response using llama.cpp with streaming.

        Uses OpenAI-compatible chat completion API from llama-cpp-python.
        """

        # Build messages in OpenAI-compatible format
        user_content = (
            self._build_context_text(context) + "\n\n" +
            self._build_text_input_explanation(text_data)
        )
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content}
        ]

        # Display what we're sending
        self._display_user_content(messages)

        # Generate streaming response
        response = self.model.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=self.config.max_tokens or 512,
            temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
            top_p=self.config.top_p
        )

        return response

    def _display_user_content(self, messages) -> None:
        """Display messages being sent to model."""
        pr_debug("=" * 60)
        pr_debug("SENDING TO MODEL:")

        for msg in messages:
            role_label = msg['role'].upper()
            msg_content = msg['content']
            msg_preview = msg_content[:200] + "..." if len(msg_content) > 200 else msg_content
            pr_debug(f"{role_label}: {msg_preview}")

        pr_debug("-" * 60)

    def _extract_text(self, chunk) -> Optional[str]:
        """Extract text from streaming chunk."""
        if not chunk:
            return None

        choices = chunk.get('choices', [])
        if not choices:
            return None

        delta = choices[0].get('delta', {})
        content = delta.get('content')

        return content

    def _extract_reasoning(self, chunk) -> Optional[str]:
        """No reasoning in llama.cpp models."""
        return None

    def _extract_thinking(self, chunk) -> Optional[list]:
        """No thinking blocks in llama.cpp models."""
        return None

    def _extract_usage(self, chunk) -> Optional[dict]:
        """Extract usage statistics if available."""
        if not chunk:
            return None

        usage = chunk.get('usage')
        return usage
