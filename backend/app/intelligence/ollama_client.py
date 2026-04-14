"""
AI Sentinel Lite - Phase 3: Ollama LLM Client
-----------------------------------------------
Sends structured scene JSON to a local Gemma 2B model via Ollama
and returns a single natural language sentence summarizing the scene.

Requires Ollama running locally: https://ollama.com
Requires gemma:2b pulled: `ollama pull gemma:2b`
"""

import urllib.request
import json


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e2b"

PROMPT_TEMPLATE = """You are an AI surveillance assistant. Based on the following scene data, write exactly ONE concise sentence describing what is happening.

Scene data:
{scene_json}

Rules:
- One sentence only. No lists. No extra commentary.
- Be specific about numbers of people and their activity.
- Example: "Two people are present indoors, one moving quickly while the other remains stationary."

Your one-sentence summary:"""


class OllamaClient:
    def __init__(self, model=MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url

    def is_running(self):
        """Check if Ollama server is reachable."""
        try:
            req = urllib.request.urlopen("http://localhost:11434", timeout=2)
            return True
        except Exception:
            return False

    def summarize(self, scene: dict) -> str:
        """
        Send scene dict to Gemma and return a 1-sentence summary.
        Returns a fallback string if Ollama is unreachable.
        """
        if not self.is_running():
            return "Ollama not running. Start it with: ollama serve"

        scene_json = json.dumps(scene, indent=2)
        prompt = PROMPT_TEMPLATE.format(scene_json=scene_json)

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except Exception as e:
            return f"LLM error: {e}"
