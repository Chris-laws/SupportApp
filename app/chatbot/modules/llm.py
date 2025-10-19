import requests
import json

def query_ollama(prompt, model="llama3", language="Deutsch"):
    print(f">> Schicke Prompt an Ollama mit Streaming...")
    try:
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        full_prompt = f"Beantworte die folgende Frage auf {language}.\n\n{prompt}"
        
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "top_p": 0.7
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=600, stream=True)
        response.raise_for_status()
        
        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))  # <-- RICHTIG: bytes → str → dict
                    text_piece = data.get("response", "")
                    answer += text_piece
                except Exception as e:
                    print(f"WARN: Fehler beim Verarbeiten einer Zeile: {e}")

        print(f"OK: Antwort fertig: {answer[:300]}...")
        return answer

    except requests.exceptions.Timeout:
        print("ERROR: Anfrage an Ollama: Timeout nach 60 Sekunden!")
        return "Timeout bei der Anfrage an das Modell."
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Fehler bei Anfrage an Ollama: {e}")
        return f"Fehler bei der Kommunikation mit dem Modell: {e}"
    except Exception as e:
        print(f"ERROR: Unerwarteter Fehler: {e}")
        return "Ein unerwarteter Fehler ist aufgetreten."
