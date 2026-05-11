import json
from pathlib import Path
import re
import importlib

import requests


BASE_DIR = Path(__file__).parent
LIVE_JSON_PATH = BASE_DIR / "FRONTEND_LIVE_OUTPUT.json"
LIVE_ENDPOINT = "http://127.0.0.1:5000/live"
_NLP = None


def preprocess_query_text(text: str) -> str:
    """Normalize frontend query text for downstream embedding/search usage."""
    value = (text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def remove_filler_text(text: str) -> str:
    """Remove filler words and weak discourse from user query text."""
    global _NLP
    if _NLP is None:
        try:
            spacy = importlib.import_module("spacy")
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            # Graceful fallback when spaCy or model is unavailable.
            return text

    doc = _NLP(text)
    clean_tokens = []

    for token in doc:
        # Remove interjections such as "um" and "uh".
        if token.pos_ == "INTJ":
            continue

        # Remove discourse markers such as "you know".
        if token.dep_ == "discourse":
            continue

        # Remove weak adverbs but keep negations.
        if token.pos_ == "ADV" and token.dep_ == "advmod":
            if token.text.lower() not in {"not", "never"}:
                continue

        clean_tokens.append(token.text)

    return " ".join(clean_tokens)


def read_latest_query() -> str:
    try:
        response = requests.get(LIVE_ENDPOINT, timeout=3)
        if response.ok:
            payload = response.json()
            latest_from_api = (payload.get("last_text_input") or "").strip()
            if latest_from_api:
                return latest_from_api
    except Exception:
        # Fallback to file-based value when API is unavailable.
        pass

    try:
        with LIVE_JSON_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)
            return (payload.get("last_text_input") or "").strip()
    except Exception:
        return ""


if __name__ == "__main__":
    latest_query = read_latest_query()
    concise_query = remove_filler_text(latest_query)
    print("Concise query:", preprocess_query_text(concise_query))