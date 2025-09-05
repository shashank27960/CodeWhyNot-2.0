import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
import torch
import re
import streamlit as st

class CodeLlamaGenerator:
    """
    Generates code from prompts using either Ollama (local LLM runner) or HuggingFace Transformers.
    Ensures Python code is returned by prompt engineering and post-processing.
    Supports quantized model loading for HuggingFace backend (4-bit via bitsandbytes if available).
    """
    def __init__(self, backend='ollama', hf_model='Salesforce/codegen-350M-mono', quantized=False):
        self.backend = backend
        self.hf_model = hf_model
        self.hf_pipeline = None
        self.quantized = quantized
        if backend == 'huggingface':
            self._init_hf_pipeline()

    def _load_hf_model(self, model_name, quantized):
        """Load HuggingFace model, optionally quantized (4-bit with bitsandbytes)."""
        try:
            if quantized:
                # Try to load quantized model (requires bitsandbytes and supported model)
                from transformers.utils.quantization_config import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            print(f"[HuggingFace quantized load error]: {e}")
            return None, None

    def _init_hf_pipeline(self):
        try:
            if self.quantized:
                model, tokenizer = self._load_hf_model(self.hf_model, True)
            else:
                model, tokenizer = self._load_hf_model(self.hf_model, False)
            if model is not None and tokenizer is not None:
                self.hf_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
            else:
                self.hf_pipeline = None
        except Exception as e:
            self.hf_pipeline = None
            print(f"[HuggingFace init error]: {e}")

    def _extract_code(self, text):
        # Try to extract code from triple backticks
        code_blocks = re.findall(r'```(?:python)?\n([\s\S]+?)```', text)
        if code_blocks:
            return code_blocks[0].strip()
        # Fallback: extract indented code
        lines = text.split('\n')
        code_lines = [line for line in lines if line.startswith('    ') or line.startswith('\t')]
        if code_lines:
            return '\n'.join(code_lines)
        # Fallback: return the whole text if it looks like code
        if 'def ' in text or 'class ' in text:
            return text.strip()
        return None

    def _cached_generate_hf(self, prompt, max_tokens):
        if self.hf_pipeline is None:
            self._init_hf_pipeline()
        if self.hf_pipeline is None:
            return "# HuggingFace pipeline not available."
        try:
            outputs = self.hf_pipeline(prompt, max_length=max_tokens, num_return_sequences=1)
            return outputs[0]['generated_text']
        except Exception as e:
            return f"# HuggingFace error: {e}"

    def generate_code(self, prompt, max_tokens=128):
        # Prompt engineering: force Python code
        prompt_for_code = f"Write Python code for: {prompt}"
        if self.backend == 'ollama':
            # NOTE: For best performance, ensure your Ollama model is quantized (e.g., GGUF 4-bit). See Ollama docs for details.
            raw = self._generate_ollama(prompt_for_code, max_tokens)
        elif self.backend == 'huggingface':
            raw = self._cached_generate_hf(prompt_for_code, max_tokens)
        else:
            return f"# Unknown backend: {self.backend}"
        code = self._extract_code(raw)
        if code:
            return code
        else:
            return f"# No Python code found in LLM output.\n# Raw output:\n{raw}"

    def _generate_ollama(self, prompt, max_tokens):
        url = 'http://localhost:11434/api/generate'
        data = {
            "model": "codellama",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        }
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            return f"# Ollama error: {e}"