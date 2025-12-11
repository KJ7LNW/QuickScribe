"""
HuggingFace provider for local text generation models.

Supports causal LM models (Qwen, Llama, Mistral, Phi) and seq2seq models (T5, BART).
Requires --audio-source transcribe mode (cannot process audio directly).
Uses chat templates for system/user message separation and TextIteratorStreamer for real streaming.
"""
from typing import Optional
from threading import Thread
from .base_provider import AbstractProvider
from lib.pr_log import pr_err, pr_info, pr_debug


class HuggingFaceProvider(AbstractProvider):
    """
    HuggingFace provider for local text generation.

    Loads models from HuggingFace Hub and runs inference locally.
    Only works with transcription mode since models require text input.
    """

    def __init__(self, config, audio_processor):
        super().__init__(config, audio_processor)

        self.model = None
        self.tokenizer = None
        self.architecture = None
        self.device = None
        self.torch_dtype = None

    def initialize(self) -> bool:
        """Load HuggingFace model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

            raw_path = self.model_without_route
            model_path = (
                raw_path[len('huggingface/'):]
                if raw_path.lower().startswith('huggingface/')
                else raw_path
            )

            pr_info(f"Loading HuggingFace model: {model_path}")

            if not torch.cuda.is_available():
                pr_err("HuggingFace provider requires CUDA GPU")
                pr_err("CPU inference would use excessive RAM (>20GB for small models)")
                pr_err("Alternatives:")
                pr_err("  - Use GPU-enabled system")
                pr_err("  - Use Ollama with GGUF models: ollama/qwen2.5:0.5b-instruct-q4_K_M")
                pr_err("  - Use cloud API providers: groq, anthropic, openai")
                return False

            self.device = "cuda"
            self.torch_dtype = torch.float16

            quantization_config = None
            device_map = None

            if self.route:
                try:
                    bits = int(self.route)

                    if bits == 16:
                        pr_info("Using float16 dtype")

                    elif bits in (4, 8):
                        from transformers import BitsAndBytesConfig

                        if bits == 8:
                            pr_info("Loading with 8-bit quantization")
                            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        else:
                            pr_info("Loading with 4-bit quantization")
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4"
                            )

                        device_map = "auto"

                    else:
                        pr_err(f"Unsupported quantization: @{bits} (supported: @4, @8, @16)")
                        return False

                except ValueError:
                    pr_err(f"Invalid route format: @{self.route} (expected numeric bits: @4, @8, @16)")
                    return False

                except ImportError as e:
                    pr_err(f"Quantization requires bitsandbytes: {e}")
                    pr_err("Install with: pip install bitsandbytes")
                    return False

            load_params = {
                'dtype': self.torch_dtype,
                'low_cpu_mem_usage': True
            }

            if quantization_config is not None:
                load_params['quantization_config'] = quantization_config
                load_params['device_map'] = device_map
            else:
                load_params['device_map'] = self.device

            try:
                pr_info("Attempting to load as causal LM")
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_params)
                self.architecture = 'causal'

            except Exception as causal_error:
                pr_info(f"Not a causal LM, trying seq2seq: {causal_error}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **load_params)
                self.architecture = 'seq2seq'

            if quantization_config is None:
                self.model = self.model.to(self.device)

            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Chat template required for system/user message separation
            if self.tokenizer.chat_template is None:
                pr_err("Model lacks chat template - requires instruction-tuned model")
                pr_err(f"Model '{model_path}' is a base model without chat formatting")
                return False

            pr_info(f"Loaded {self.architecture} model on {self.device}")
            self._initialized = True
            return True

        except ImportError as e:
            pr_err(f"HuggingFace dependencies not installed: {e}")
            pr_err("Install with: pip install torch transformers")
            return False

        except Exception as e:
            pr_err(f"Failed to load HuggingFace model: {e}")
            return False

    def _generate_response(self, instructions: str, context, audio_data, text_data):
        """
        Generate response using HuggingFace model with real streaming.

        Uses chat template for system/user message separation and
        TextIteratorStreamer for token-by-token streaming output.
        """
        from transformers import TextIteratorStreamer

        # Build messages in OpenAI-compatible format
        user_content = (
            self._build_context_text(context) + "\n\n" +
            self._build_text_input_explanation(text_data)
        )
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content}
        ]

        # Apply model-specific chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        encoded = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        token_count = input_ids.shape[1]

        self._display_user_content({
            'messages': messages,
            'prompt': prompt,
            'tokens': token_count
        })

        # Real streaming via TextIteratorStreamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'streamer': streamer,
            'max_new_tokens': self.config.max_tokens or 512,
            'temperature': self.config.temperature if self.config.temperature > 0 else 1.0,
            'do_sample': self.config.temperature > 0,
            'pad_token_id': self.tokenizer.pad_token_id
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        try:
            for chunk in streamer:
                yield chunk
        finally:
            thread.join()

    def _display_user_content(self, content) -> None:
        """Display messages and prompt being sent to model."""
        pr_debug("=" * 60)
        pr_debug("SENDING TO MODEL:")

        for msg in content['messages']:
            role_label = msg['role'].upper()
            msg_content = msg['content']
            msg_preview = msg_content[:200] + "..." if len(msg_content) > 200 else msg_content
            pr_debug(f"{role_label}: {msg_preview}")

        pr_debug(f"Prompt: {content['tokens']} tokens, {len(content['prompt'])} chars")
        pr_debug("-" * 60)

    def _extract_text(self, chunk) -> Optional[str]:
        """Extract text from chunk (already plain text)."""
        return chunk

    def _extract_reasoning(self, chunk) -> Optional[str]:
        """No reasoning in HuggingFace models."""
        return None

    def _extract_thinking(self, chunk) -> Optional[list]:
        """No thinking blocks in HuggingFace models."""
        return None

    def _extract_usage(self, chunk) -> Optional[dict]:
        """No usage statistics for local models."""
        return None
