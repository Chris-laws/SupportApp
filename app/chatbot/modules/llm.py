from __future__ import annotations

import json
import requests


def query_ollama(
    prompt: str,
    model: str = "llama3",
    language: str = "Deutsch",
    *,
    options: dict | None = None,
    max_retries: int = 1,\n    timeout: int = 300,\n) -> str:
    full_prompt = f"Beantworte die folgende Frage auf {language}.\n\n{prompt}"
    default_options = {
        "temperature": 0.05,
        "top_p": 0.75,
        "repeat_penalty": 1.05,
        "num_predict": 64,
        "num_ctx": 4096,
    }

    payload_options = default_options | (options or {})

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": payload_options,
    }

    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.Timeout:
            print(f"ERROR: Anfrage an Ollama: Timeout nach {timeout} Sekunden!")
        except requests.exceptions.RequestException as exc:
            print(f"ERROR: Fehler bei Anfrage an Ollama: {exc}")
        except json.JSONDecodeError as exc:
            print(f"ERROR: Konnte Ollama-Antwort nicht dekodieren: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: Unerwarteter Fehler: {exc}")
        if attempt + 1 < max_retries:
            print("Versuche erneut...")
    return "Fehler bei der Kommunikation mit dem Modell."


