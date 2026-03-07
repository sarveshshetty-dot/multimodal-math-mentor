from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class LocalLLM:
    """Singleton wrapper to load TinyLlama into memory once."""
    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalLLM, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Loading TinyLlama fully locally (CPU)... This may take a moment.")
        try:
            self._pipeline = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                # Use device=-1 for CPU (avoids device_map which needs accelerate)
                device=-1,
                # Use dtype instead of deprecated torch_dtype
                torch_dtype=None,
            )
            print("TinyLlama loaded successfully!")
        except Exception as e:
            print(f"Failed to load pipeline: {e}")
            # Fallback: manual load
            try:
                tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                model = AutoModelForCausalLM.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    torch_dtype=torch.float32,
                )
                self._pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,
                )
                print("TinyLlama loaded via fallback!")
            except Exception as e2:
                print(f"Critical: Fallback also failed: {e2}")
                self._pipeline = None

    def generate(self, prompt: str, max_new_tokens: int = 200, return_full_text: bool = False) -> str:
        """Generates text from the loaded local LLM based on the prompt."""
        if self._pipeline is None:
            return "Error: Local LLM model failed to initialize. Please restart the app."

        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_k=50,
                top_p=0.95,
                return_full_text=return_full_text,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
            )
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Error during generation: {str(e)}"
