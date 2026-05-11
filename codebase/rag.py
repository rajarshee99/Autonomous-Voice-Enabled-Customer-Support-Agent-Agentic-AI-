import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastembed import TextEmbedding
from pinecone import Pinecone


BASE_DIR = Path(__file__).parent
QDRANT_READY_PATH = BASE_DIR / "qdrant_ready.jasonl"
QUADRENT_READY_PATH = BASE_DIR / "quadrent_ready.jasonl"
ANSWER_CHUNK_PATH = BASE_DIR / "answer_chunk.jasonl"

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY",
    "",
)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ragcustomersupport")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "voice_support_chunks")

EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K_RESULTS = int(os.getenv("RAG_TOP_K", "5"))
MAX_SNIPPET_LENGTH = int(os.getenv("RAG_SNIPPET_LENGTH", "420"))
PINECONE_INDEX_DIM = int(os.getenv("PINECONE_INDEX_DIM", "512"))


def _resolve_ready_file_path() -> Path:
    env_path = (os.getenv("RAG_READY_FILE_PATH") or "").strip()
    if env_path:
        return Path(env_path)

    if QDRANT_READY_PATH.exists():
        return QDRANT_READY_PATH
    if QUADRENT_READY_PATH.exists():
        return QUADRENT_READY_PATH
    return QDRANT_READY_PATH


def current_ready_file_signature() -> dict[str, Any]:
    ready_file_path = _resolve_ready_file_path()
    signature: dict[str, Any] = {"source_file": ready_file_path.name}
    if not ready_file_path.exists():
        signature["source_mtime_ns"] = 0
        signature["source_size"] = 0
        return signature

    stat = ready_file_path.stat()
    signature["source_mtime_ns"] = int(getattr(stat, "st_mtime_ns", 0))
    signature["source_size"] = int(stat.st_size)
    return signature


def _safe_json_loads(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    if not value:
        return {}
    return json.loads(value)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = _safe_json_loads(line)
            except Exception:
                continue
            if isinstance(record, dict):
                record["_source_line"] = line_number
                records.append(record)
    return records


def _extract_chunk_text(record: dict[str, Any]) -> str:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}

    for key in ("chunk_text", "text", "content", "chunk", "body", "raw_text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("chunk_text", "text", "content", "chunk", "body", "raw_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _extract_title(record: dict[str, Any]) -> str:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    return str(
        record.get("title")
        or record.get("page_title")
        or record.get("name")
        or payload.get("title")
        or payload.get("page_title")
        or payload.get("name")
        or ""
    ).strip()


def _extract_url(record: dict[str, Any]) -> str:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    return str(
        record.get("url")
        or record.get("source_url")
        or record.get("source")
        or payload.get("url")
        or payload.get("source_url")
        or payload.get("source")
        or ""
    ).strip()


def _extract_query(record: dict[str, Any]) -> str:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    return str(
        record.get("query")
        or record.get("source_question")
        or payload.get("query")
        or payload.get("source_question")
        or ""
    ).strip()


def _normalize_record(record: dict[str, Any], index: int, source_file_name: str) -> dict[str, Any]:
    title = _extract_title(record)
    url = _extract_url(record)
    query = _extract_query(record)
    chunk_text = _extract_chunk_text(record)

    canonical_text = f"{title}|{url}|{query}|{chunk_text}".strip()
    source_hash = hashlib.sha1(canonical_text.encode("utf-8", errors="ignore")).hexdigest()

    payload = dict(record)
    payload.update(
        {
            "id": str(record.get("id") or record.get("chunk_id") or source_hash),
            "title": title,
            "url": url,
            "query": query,
            "chunk_text": chunk_text,
            "source_hash": source_hash,
            "chunk_index": index,
            "source_file": source_file_name,
        }
    )
    return payload


def _build_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def _build_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL_NAME)


def _fit_index_dim(vector: list[float], target_dim: int = PINECONE_INDEX_DIM) -> list[float]:
    values = [float(v) for v in vector]
    if len(values) == target_dim:
        return values
    if len(values) > target_dim:
        return values[:target_dim]
    return values + [0.0] * (target_dim - len(values))


def _reset_namespace(index) -> None:
    try:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
    except Exception as exc:
        message = str(exc).lower()
        if "namespace not found" in message or "404" in message:
            print(f"[rag] Namespace '{PINECONE_NAMESPACE}' does not exist yet; continuing")
            return
        raise


def _upsert_chunks(index, embedding_model: TextEmbedding, chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return

    texts = [str(chunk.get("chunk_text") or "") for chunk in chunks]
    vectors = list(embedding_model.embed(texts))

    to_upsert: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors, strict=False):
        metadata = {
            "title": str(chunk.get("title") or ""),
            "url": str(chunk.get("url") or ""),
            "query": str(chunk.get("query") or ""),
            "chunk_text": str(chunk.get("chunk_text") or ""),
            "source_hash": str(chunk.get("source_hash") or ""),
            "chunk_index": int(chunk.get("chunk_index") or 0),
        }
        to_upsert.append(
            {
                "id": str(chunk.get("id") or chunk.get("source_hash")),
                "values": _fit_index_dim([float(v) for v in vector]),
                "metadata": metadata,
            }
        )

    batch_size = 100
    for start in range(0, len(to_upsert), batch_size):
        batch = to_upsert[start : start + batch_size]
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)


def _truncate_text(text: str, limit: int = MAX_SNIPPET_LENGTH) -> str:
    value = re.sub(r"\s+", " ", (text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "..."


def _extract_source_websites(results: list[dict[str, Any]]) -> list[str]:
    websites: list[str] = []
    for result in results:
        url = str(result.get("url") or "").strip()
        if not url:
            continue
        parsed = urlparse(url if "://" in url else f"https://{url}")
        hostname = (parsed.netloc or parsed.path).strip().lower()
        if hostname and hostname not in websites:
            websites.append(hostname)
    return websites


def _build_answer_chunk_text(query: str, results: list[dict[str, Any]]) -> str:
    lines = [f"Query: {query}", f"Matches: {len(results)}"]
    for result in results:
        lines.append("")
        lines.append(f"[{result['rank']}] {result.get('title') or 'Untitled'}")
        lines.append(f"score: {result.get('score', 0):.4f}")
        if result.get("url"):
            lines.append(f"url: {result['url']}")
        if result.get("source_query"):
            lines.append(f"source_query: {result['source_query']}")
        lines.append(f"chunk: {result.get('chunk_text') or ''}")
    return "\n".join(lines).strip()


def _write_answer_chunk(payload: dict[str, Any], output_path: Path = ANSWER_CHUNK_PATH) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


def semantic_search(query: str, top_k: int = TOP_K_RESULTS) -> dict[str, Any]:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("query must not be empty")

    ready_file_path = _resolve_ready_file_path()
    ready_signature = current_ready_file_signature()
    chunks = [
        _normalize_record(record, index, ready_file_path.name)
        for index, record in enumerate(_load_jsonl_records(ready_file_path), start=1)
    ]
    chunks = [chunk for chunk in chunks if str(chunk.get("chunk_text") or "").strip()]

    embedding_model = _build_embedding_model()
    index = _build_pinecone_index()

    print(f"[rag] Loaded {len(chunks)} chunks from {ready_file_path.name}")

    if not chunks:
        payload = {
            "query": cleaned_query,
            "collection_name": PINECONE_INDEX_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "source_file": str(ready_file_path.name),
            "source_mtime_ns": int(ready_signature.get("source_mtime_ns") or 0),
            "source_size": int(ready_signature.get("source_size") or 0),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "result_count": 0,
            "source_websites": [],
            "top_chunks": [],
            "answer_chunk_text": f"Query: {cleaned_query}\nMatches: 0\nNo chunks were available in {ready_file_path.name}.",
        }
        _write_answer_chunk(payload)
        return payload

    _reset_namespace(index)
    print(f"[rag] Cleared namespace '{PINECONE_NAMESPACE}'")

    _upsert_chunks(index, embedding_model, chunks)
    print(f"[rag] Upserted {len(chunks)} chunks into Pinecone")

    query_vector = next(embedding_model.embed([cleaned_query]))

    query_result = index.query(
        vector=_fit_index_dim([float(value) for value in query_vector]),
        top_k=top_k,
        include_values=False,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )

    matches = query_result.get("matches", []) if isinstance(query_result, dict) else getattr(query_result, "matches", [])

    normalized_results: list[dict[str, Any]] = []
    for rank, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
        normalized_results.append(
            {
                "rank": rank,
                "id": str(match.get("id") if isinstance(match, dict) else getattr(match, "id", "")),
                "score": float(match.get("score") if isinstance(match, dict) else getattr(match, "score", 0.0) or 0.0),
                "title": str(metadata.get("title") or "").strip(),
                "url": str(metadata.get("url") or "").strip(),
                "source_query": str(metadata.get("query") or "").strip(),
                "chunk_text": _truncate_text(str(metadata.get("chunk_text") or "")),
                "source_hash": str(metadata.get("source_hash") or "").strip(),
                "chunk_index": metadata.get("chunk_index"),
            }
        )

    source_websites = _extract_source_websites(normalized_results)

    payload = {
        "query": cleaned_query,
        "collection_name": PINECONE_INDEX_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "source_file": str(ready_file_path.name),
        "source_mtime_ns": int(ready_signature.get("source_mtime_ns") or 0),
        "source_size": int(ready_signature.get("source_size") or 0),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_count": len(normalized_results),
        "source_websites": source_websites,
        "top_chunks": normalized_results,
        "answer_chunk_text": _build_answer_chunk_text(cleaned_query, normalized_results),
    }
    _write_answer_chunk(payload)
    return payload


def load_answer_chunk(output_path: Path = ANSWER_CHUNK_PATH) -> dict[str, Any]:
    if not output_path.exists():
        return {}

    try:
        content = output_path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        first_line = content.splitlines()[0].strip()
        return json.loads(first_line)
    except Exception:
        return {}


def run_rag_pipeline(query: str | None = None) -> dict[str, Any]:
    try:
        from question_validation import read_latest_query
    except Exception:
        read_latest_query = None

    effective_query = (query or "").strip()
    if not effective_query and read_latest_query is not None:
        effective_query = read_latest_query().strip()

    if not effective_query:
        raise ValueError("No query provided for semantic search")

    return semantic_search(effective_query)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pinecone semantic search and write answer_chunk.jasonl")
    parser.add_argument("query", nargs="?", default="", help="The search query to run")
    args = parser.parse_args()

    result = run_rag_pipeline(args.query)
    print(json.dumps(result, ensure_ascii=True, indent=2))

