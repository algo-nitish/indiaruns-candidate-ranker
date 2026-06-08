# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# LLM Client — Abstracted interface for OpenAI / Gemini / Ollama
# =============================================================================

import json
import re
import time
from typing import Optional
from rich.console import Console

console = Console()


class LLMClient:
    """
    Unified LLM client supporting OpenAI, Google Gemini, and Ollama.
    Provides a single `generate()` method regardless of backend.
    """

    def __init__(
        self,
        provider: str = "gemini",
        api_key: str = "",
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: str = "http://localhost:11434",
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._client = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            if not self.model:
                self.model = "gpt-4o"
            console.print(f"[green]✓ OpenAI client initialized (model: {self.model})[/green]")
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model or "gemini-2.0-flash")
            if not self.model:
                self.model = "gemini-2.0-flash"
            console.print(f"[green]✓ Gemini client initialized (model: {self.model})[/green]")
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    def _init_ollama(self):
        """Initialize Ollama client (local models)."""
        try:
            import requests
            # Test connection
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            if not self.model:
                self.model = "llama3.1"
            console.print(f"[green]✓ Ollama client initialized (model: {self.model})[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Ollama connection failed: {e}[/yellow]")
            console.print("[yellow]  Make sure Ollama is running: ollama serve[/yellow]")

    def generate(self, prompt: str, retries: int = 3) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt
            retries: Number of retry attempts on failure

        Returns:
            The generated text response
        """
        for attempt in range(retries):
            try:
                if self.provider == "openai":
                    return self._generate_openai(prompt)
                elif self.provider == "gemini":
                    return self._generate_gemini(prompt)
                elif self.provider == "ollama":
                    return self._generate_ollama(prompt)
            except Exception as e:
                console.print(f"[yellow]⚠ LLM attempt {attempt + 1}/{retries} failed: {e}[/yellow]")
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    console.print(f"[dim]Retrying in {wait}s...[/dim]")
                    time.sleep(wait)
                else:
                    raise

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise AI assistant that always returns valid JSON when asked."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _generate_gemini(self, prompt: str) -> str:
        """Generate using Google Gemini API."""
        response = self._client.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        return response.text.strip()

    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama (local)."""
        import requests
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def generate_json(self, prompt: str, retries: int = 3) -> dict | list:
        """
        Generate a response and parse it as JSON.
        Handles markdown code fences and common LLM output quirks.
        """
        raw = self.generate(prompt, retries=retries)

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            console.print(f"[yellow]⚠ JSON parse failed, attempting repair...[/yellow]")
            # Try to find JSON in the response
            json_match = re.search(r'[\[{].*[\]}]', raw, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nRaw: {raw[:500]}")


def create_llm_client_from_config() -> LLMClient:
    """Create an LLM client using settings from config."""
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
    from config.settings import (
        LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
        GEMINI_API_KEY, GEMINI_MODEL,
        OLLAMA_BASE_URL, OLLAMA_MODEL,
        LLM_TEMPERATURE, LLM_MAX_TOKENS,
    )

    provider_config = {
        "openai": {"api_key": OPENAI_API_KEY, "model": OPENAI_MODEL},
        "gemini": {"api_key": GEMINI_API_KEY, "model": GEMINI_MODEL},
        "ollama": {"api_key": "", "model": OLLAMA_MODEL, "base_url": OLLAMA_BASE_URL},
    }

    config = provider_config.get(LLM_PROVIDER, {})
    return LLMClient(
        provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        **config,
    )
