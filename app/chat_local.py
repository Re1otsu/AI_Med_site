import requests

def local_ai_chat(prompt: str, model: str = "llama3") -> str:
    """
    Offline/local chat via Ollama.
    Requires: ollama serve  (and model pulled: ollama pull llama3 or phi3)
    """
    url = "http://localhost:11434/api/generate"
    try:
        with requests.post(url, json={"model": model, "prompt": prompt}, stream=True, timeout=120) as r:
            r.raise_for_status()
            chunks = []
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    import json as _json
                    data = _json.loads(line.decode("utf-8"))
                    if "response" in data:
                        chunks.append(data["response"])
                except Exception:
                    pass
        return "".join(chunks).strip() or "Я не получил ответа от локальной модели."
    except Exception as e:
        return f"⚠️ Не удалось подключиться к Ollama: {e}\nУбедись, что запущено:  ollama serve"
