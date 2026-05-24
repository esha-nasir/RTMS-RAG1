import os
import requests


YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "yandex").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()

DOC_MODEL_URI = os.getenv("YANDEX_DOC_EMBED_MODEL", "text-search-doc/latest")
QUERY_MODEL_URI = os.getenv("YANDEX_QUERY_EMBED_MODEL", "text-search-query/latest")

EMBED_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
DEBUG_EMBED = os.getenv("DEBUG_EMBED", "0").strip() == "1"
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "8000"))

def _auth_candidates():
    token = (YANDEX_IAM_TOKEN or YANDEX_API_KEY or "").strip()
    if not token:
        return []
    # Try both schemes to handle API-key vs IAM-token setups.
    return [f"Bearer {token}", f"Api-Key {token}"]


def _ollama_embedding(text: str) -> list[float]:
    payload = {"model": OLLAMA_EMBED_MODEL, "input": text or ""}
    response = requests.post(f"{OLLAMA_URL}/api/embed", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    embeddings = data.get("embeddings") or []
    if embeddings and isinstance(embeddings[0], list):
        return embeddings[0]

    embedding = data.get("embedding")
    if isinstance(embedding, list):
        return embedding

    raise RuntimeError(f"Unexpected Ollama embedding response: {data}")

def get_embedding(text: str, kind: str = "doc"):
    """
    kind: "doc" or "query"
    Uses different embedding models for better retrieval.
    If your Yandex account doesn't support text-search-query/latest,
    set YANDEX_QUERY_EMBED_MODEL to text-search-doc/latest.
    """
    text = (text or "").strip()
    if len(text) > EMBED_MAX_CHARS:
        text = text[:EMBED_MAX_CHARS]

    if EMBED_PROVIDER == "ollama":
        return _ollama_embedding(text)

    if not FOLDER_ID or not _auth_candidates():
        raise RuntimeError("Missing env vars: YANDEX_FOLDER_ID and one of YANDEX_IAM_TOKEN / YANDEX_API_KEY")

    model = DOC_MODEL_URI if kind == "doc" else QUERY_MODEL_URI
    data = {"modelUri": f"emb://{FOLDER_ID}/{model}", "text": text}

    last_error = None
    r = None
    for auth in _auth_candidates():
        headers = {"Authorization": auth, "Content-Type": "application/json"}
        try:
            r = requests.post(EMBED_URL, headers=headers, json=data, timeout=60)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            last_error = e
            # Retry with the next auth scheme only for unauthorized responses.
            if getattr(e.response, "status_code", None) != 401:
                # Fallback for query model errors: retry once with doc model.
                if kind == "query":
                    fallback_data = {"modelUri": f"emb://{FOLDER_ID}/{DOC_MODEL_URI}", "text": text}
                    try:
                        r = requests.post(EMBED_URL, headers=headers, json=fallback_data, timeout=60)
                        r.raise_for_status()
                        break
                    except requests.exceptions.RequestException:
                        print(f"Request failed: {e}")
                        raise
                print(f"Request failed: {e}")
                raise
            continue
    else:
        print(f"Request failed: {last_error}")
        raise last_error

    if DEBUG_EMBED:
        print("Response Status Code:", r.status_code)
        print("Response Body:", r.text)

    try:
        j = r.json()
    except ValueError as e:
        print(f"Error decoding JSON: {e}")
        raise

    if "embedding" in j:
        return j["embedding"]

    if "error" in j:
        raise Exception(f"Yandex API error: {j['error']}")

    raise Exception(f"Unexpected response format: {j}")
