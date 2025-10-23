from __future__ import annotations

import json

import requests


def _safe(text: str) -> str:
    return text.encode("cp1252", errors="replace").decode("cp1252")


def query_ollama(
    prompt: str,
    model: str = "llama3",
    language: str = "Deutsch",
    *,
    options: dict | None = None,
    max_retries: int = 1,
    timeout: int = 300,
    stream: bool = False,
) -> str:
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
        "stream": stream,
        "options": payload_options,
    }

    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
            response.raise_for_status()

            if stream:
                answer = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        answer += data.get("response", "")
                    except Exception as exc:  # noqa: BLE001
                        print(_safe(f"WARN: Fehler beim Verarbeiten einer Zeile: {exc}"))
                return answer

            data = response.json()
            return data.get("response", "")

        except requests.exceptions.Timeout:
            print(_safe(f"ERROR: Anfrage an Ollama: Timeout nach {timeout} Sekunden!"))
        except requests.exceptions.RequestException as exc:
            print(_safe(f"ERROR: Fehler bei Anfrage an Ollama: {exc}"))
        except json.JSONDecodeError as exc:
            print(_safe(f"ERROR: Konnte Ollama-Antwort nicht dekodieren: {exc}"))
        except Exception as exc:  # noqa: BLE001
            print(_safe(f"ERROR: Unerwarteter Fehler: {exc}"))

        if attempt + 1 < max_retries:
            print(_safe("Versuche erneut..."))

    return "Fehler bei der Kommunikation mit dem Modell."
