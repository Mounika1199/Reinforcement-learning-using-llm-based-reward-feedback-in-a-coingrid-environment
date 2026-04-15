"""
Ollama LLM client.

Sends prompts to a locally running Ollama server and returns the
model's text response.  The default model is ``gemma3:12b``.

Usage
-----
    from coingrid_llm_rl.llm.client import query_ollama

    response = query_ollama("Evaluate this: ...")
"""

from __future__ import annotations

import httpx


def query_ollama(
    prompt: str,
    model: str = "gemma3:12b",
    base_url: str = "http://localhost:11434",
    top_p: float = 0.9,
    top_k: int = 50,
    num_predict: int = 1000,
) -> str:
    """
    Query a locally running Ollama server synchronously.

    Parameters
    ----------
    prompt : str
        The full prompt string to send to the model.
    model : str
        Ollama model tag (default ``"gemma3:12b"``).
    base_url : str
        Base URL of the Ollama server (default ``"http://localhost:11434"``).
    top_p : float
        Nucleus-sampling probability threshold (default 0.9).
    top_k : int
        Top-k sampling cutoff (default 50).
    num_predict : int
        Maximum tokens to generate (default 1000).

    Returns
    -------
    str
        The model's text response, or an empty string on failure.

    Raises
    ------
    httpx.HTTPStatusError
        If the Ollama server returns a non-2xx status code.
    httpx.ConnectError
        If the Ollama server is not reachable.
    """
    with httpx.Client(timeout=None) as client:
        response = client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": num_predict,
                },
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")
