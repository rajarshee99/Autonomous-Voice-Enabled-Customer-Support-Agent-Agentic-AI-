import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "webcrawler_result.jasonl"
OUTPUT_FILE = BASE_DIR / "chunking.jasonl"
STATE_FILE = BASE_DIR / ".chunking_state.json"

NOISE_DOMAINS = {
    "google.com",
    "www.google.com",
    "support.google.com",
    "bing.com",
    "www.bing.com",
    "r.bing.com",
    "duckduckgo.com",
    "www.duckduckgo.com",
}

NOISE_EXTENSIONS = (
    ".css",
    ".js",
    ".json",
    ".xml",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".map",
)

BOILERPLATE_PATTERNS = [
    "skip to main content",
    "privacy",
    "terms of use",
    "all rights reserved",
    "cookie policy",
    "sign in",
    "subscribe to rss",
]

QUERY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "have",
    "into",
    "your",
    "you",
    "are",
    "what",
    "when",
    "where",
    "how",
    "why",
    "who",
    "which",
    "about",
    "after",
    "before",
    "during",
    "site",
    "filetype",
    "http",
    "https",
    "www",
    "com",
    "org",
    "net",
}


@dataclass
class ProcessState:
    offset: int = 0
    line_no: int = 0


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _normalize_block(value: str) -> str:
    text = re.sub(r"\r\n?", "\n", (value or ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_for_key(value: str) -> str:
    text = _normalize_spaces(value).lower()
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_markdown(value: str) -> str:
    text = value or ""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"`{3}.*?`{3}", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text


def _tokenize_words(value: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", value or "")}


def _extract_query_terms(query: str, source_question: str, website: str) -> set[str]:
    raw = f"{query or ''} {source_question or ''}".lower()
    raw = re.sub(r"site:[^\s]+", " ", raw)
    raw = re.sub(r"filetype:[^\s]+", " ", raw)
    if website:
        raw = raw.replace(website.lower(), " ")
    return {w for w in re.findall(r"[a-z0-9]{3,}", raw) if w not in QUERY_STOPWORDS and not w.isdigit()}


def _query_overlap_count(text: str, query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    return len(_tokenize_words(text).intersection(query_terms))


def _is_probable_content_url(url: str) -> bool:
    value = (url or "").strip()
    if not value.startswith("http"):
        return False
    try:
        parsed = urlparse(value)
        domain = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
    except Exception:
        return False
    if domain in NOISE_DOMAINS:
        return False
    if path.endswith(NOISE_EXTENSIONS):
        return False
    if "/search?" in value.lower():
        return False
    return True


def _score_row(cleaned_text: str, query_terms: set[str], probable_url: bool) -> dict[str, Any]:
    normalized = _normalize_spaces(cleaned_text).lower()
    text_len = len(normalized)
    unique_words = len(_tokenize_words(cleaned_text))
    overlap = _query_overlap_count(cleaned_text, query_terms)
    link_like = len(re.findall(r"http[s]?://|www\.", normalized))
    boiler_hits = sum(1 for pattern in BOILERPLATE_PATTERNS if pattern in normalized)

    # RELAXED THRESHOLDS: Allowing more meaningful data through
    base_pass = text_len >= 50 and unique_words >= 8 and boiler_hits <= 5
    min_overlap = 0 if len(query_terms) <= 2 else 1  # Reduced overlap requirement
    strict_pass = (
        base_pass
        and probable_url
        and (overlap >= min_overlap if query_terms else True)
        and unique_words >= 10  # Lowered from 18
        and link_like <= 15  # Raised from 8
        and boiler_hits <= 5  # Raised from 2
    )

    # IMPROVED SCORING: Less harsh penalties for relaxed filtering
    score = (
        (overlap * 3.0)  # Reduced weight
        + min(unique_words / 15.0, 4.0)  # Reduced weight
        + min(text_len / 1000.0, 3.0)  # Reduced weight
        - (link_like * 0.2)  # Reduced penalty
        - (boiler_hits * 1.0)  # Reduced penalty
        - (0.2 if not probable_url else 0.0)  # Reduced penalty
    )
    return {
        "base_pass": base_pass,
        "strict_pass": strict_pass,
        "overlap": overlap,
        "score": round(score, 4),
        "url_quality": "probable_content" if probable_url else "noisy_fallback",
    }


def _chunk_text(text: str, max_chars: int = 1100) -> list[str]:
    value = (text or "").strip()
    if not value:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", value) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(paragraph), max_chars):
                part = paragraph[start : start + max_chars].strip()
                if part:
                    chunks.append(part)
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph
    if current:
        chunks.append(current.strip())
    return chunks


def _safe_load_state(path: Path) -> ProcessState:
    if not path.exists():
        return ProcessState()
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return ProcessState()
        data = json.loads(content)
        return ProcessState(offset=int(data.get("offset", 0)), line_no=int(data.get("line_no", 0)))
    except Exception:
        return ProcessState()


def _save_state(path: Path, state: ProcessState) -> None:
    path.write_text(
        json.dumps({"offset": state.offset, "line_no": state.line_no}, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_jsonl_line(line: str) -> dict[str, Any] | None:
    try:
        item = json.loads(line)
        if isinstance(item, dict):
            return item
    except Exception:
        return None
    return None


def _build_output_rows(
    input_rows: list[dict[str, Any]],
    seen_chunk_keys: set[str],
    min_chars_per_site_flush: int,
    relaxed_fallback_rows: int,
    final_flush: bool,
    website_buffers: dict[str, str],
    website_meta_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    strict_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fallback_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in input_rows:
        if (row.get("status") or "").strip().lower() != "success":
            continue
        raw_text = row.get("chunk_text") or ""
        if not str(raw_text).strip():
            continue

        website = (row.get("website") or "unknown").strip() or "unknown"
        query = row.get("query", "")
        source_question = row.get("source_question", "")
        url = (row.get("url") or "").strip()
        probable_url = _is_probable_content_url(url)

        cleaned_text = _normalize_block(_clean_markdown(str(raw_text)))
        query_terms = _extract_query_terms(query, source_question, website)
        metrics = _score_row(cleaned_text, query_terms, probable_url=probable_url)
        if not metrics["base_pass"]:
            continue

        candidate = {
            "website": website,
            "query": query,
            "source_question": source_question,
            "url": url,
            "title": row.get("title", ""),
            "crawl_timestamp": row.get("crawl_timestamp", ""),
            "text": cleaned_text,
            "quality_score": metrics["score"],
            "query_term_overlap": metrics["overlap"],
            "url_quality": metrics["url_quality"],
        }
        if metrics["strict_pass"]:
            strict_rows[website].append(candidate)
        else:
            fallback_rows[website].append(candidate)

    websites = set(strict_rows.keys()) | set(fallback_rows.keys())
    for website in websites:
        selected = strict_rows.get(website, [])
        filter_mode = "strict"
        if not selected:
            filter_mode = "fallback"
            selected = sorted(
                fallback_rows.get(website, []),
                key=lambda r: r.get("quality_score", 0.0),
                reverse=True,
            )[: max(1, relaxed_fallback_rows)]
        if not selected:
            continue

        combined = "\n\n".join(item["text"] for item in selected if item.get("text"))
        if not combined:
            continue

        prior = website_buffers.get(website, "")
        website_buffers[website] = f"{prior}\n\n{combined}".strip() if prior else combined
        best = max(selected, key=lambda r: r.get("quality_score", 0.0))
        website_meta_cache[website] = {**best, "filter_mode": filter_mode}

    out_rows: list[dict[str, Any]] = []
    for website, text in list(website_buffers.items()):
        if len(text) < min_chars_per_site_flush and not final_flush:
            continue

        chunks = _chunk_text(text)
        website_buffers[website] = ""
        if not chunks:
            continue

        meta = website_meta_cache.get(website, {})
        kept_chunks: list[str] = []
        for chunk in chunks:
            key = _normalize_for_key(chunk)
            if not key or key in seen_chunk_keys:
                continue
            seen_chunk_keys.add(key)
            kept_chunks.append(chunk)

        for idx, chunk in enumerate(kept_chunks, start=1):
            out_rows.append(
                {
                    "website": website,
                    "chunk_index": idx,
                    "chunk_count": len(kept_chunks),
                    "chunk_text": chunk,
                    "source_question": meta.get("source_question", ""),
                    "query": meta.get("query", ""),
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "crawl_timestamp": meta.get("crawl_timestamp", ""),
                    "query_term_overlap": meta.get("query_term_overlap", 0),
                    "quality_score": meta.get("quality_score", 0.0),
                    "url_quality": meta.get("url_quality", "probable_content"),
                    "filter_mode": meta.get("filter_mode", "strict"),
                    "status": "ready_for_rag",
                }
            )

    return out_rows


def _detect_input_reset(input_path: Path, state: ProcessState) -> bool:
    if not input_path.exists():
        return False
    return input_path.stat().st_size < state.offset


def process_new_lines(
    input_path: Path,
    output_path: Path,
    state_path: Path,
    website_buffers: dict[str, str],
    website_meta_cache: dict[str, dict[str, Any]],
    seen_chunk_keys: set[str],
    min_chars_per_site_flush: int,
    relaxed_fallback_rows: int,
    final_flush: bool = False,
) -> int:
    state = _safe_load_state(state_path)

    if _detect_input_reset(input_path, state):
        state = ProcessState()
        website_buffers.clear()
        website_meta_cache.clear()
        seen_chunk_keys.clear()
        output_path.write_text("", encoding="utf-8")
        _save_state(state_path, state)

    if not input_path.exists():
        return 0

    rows: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8", errors="replace") as handle:
        handle.seek(state.offset)
        while True:
            line = handle.readline()
            if not line:
                break
            state.offset = handle.tell()
            state.line_no += 1
            item = _parse_jsonl_line(line.strip())
            if item is not None:
                rows.append(item)

    output_rows = _build_output_rows(
        input_rows=rows,
        seen_chunk_keys=seen_chunk_keys,
        min_chars_per_site_flush=min_chars_per_site_flush,
        relaxed_fallback_rows=relaxed_fallback_rows,
        final_flush=final_flush,
        website_buffers=website_buffers,
        website_meta_cache=website_meta_cache,
    )
    _append_jsonl(output_path, output_rows)
    _save_state(state_path, state)
    return len(output_rows)


def run_once(
    input_path: Path,
    output_path: Path,
    state_path: Path,
    min_chars_per_site_flush: int,
    relaxed_fallback_rows: int,
) -> None:
    # One-shot run should always rebuild output from scratch.
    output_path.write_text("", encoding="utf-8")
    _save_state(state_path, ProcessState())
    rows_written = process_new_lines(
        input_path=input_path,
        output_path=output_path,
        state_path=state_path,
        website_buffers={},
        website_meta_cache={},
        seen_chunk_keys=set(),
        min_chars_per_site_flush=min_chars_per_site_flush,
        relaxed_fallback_rows=relaxed_fallback_rows,
        final_flush=True,
    )
    print(f"[CHUNKING] rows_written={rows_written}")


def run_watch(
    input_path: Path,
    output_path: Path,
    state_path: Path,
    poll_interval_sec: float,
    min_chars_per_site_flush: int,
    relaxed_fallback_rows: int,
) -> None:
    website_buffers: dict[str, str] = {}
    website_meta_cache: dict[str, dict[str, Any]] = {}
    seen_chunk_keys: set[str] = set()
    print(f"[CHUNKING] watching {input_path.name} for new lines...")
    while True:
        count = process_new_lines(
            input_path=input_path,
            output_path=output_path,
            state_path=state_path,
            website_buffers=website_buffers,
            website_meta_cache=website_meta_cache,
            seen_chunk_keys=seen_chunk_keys,
            min_chars_per_site_flush=min_chars_per_site_flush,
            relaxed_fallback_rows=relaxed_fallback_rows,
            final_flush=False,
        )
        if count > 0:
            print(f"[CHUNKING] appended {count} chunk row(s)")
        time.sleep(poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust preprocessing + chunking from webcrawler_result.jasonl")
    parser.add_argument("--once", action="store_true", help="Run one-shot and exit (default is watch mode).")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Watch polling interval in seconds.")
    parser.add_argument("--min-chars-per-site-flush", type=int, default=500, help="Flush smaller batches for faster output")
    parser.add_argument("--relaxed-fallback-rows", type=int, default=200, help="Allow up to 200 fallback rows per website (was 4)")
    parser.add_argument("--input-file", default=str(INPUT_FILE))
    parser.add_argument("--output-file", default=str(OUTPUT_FILE))
    parser.add_argument("--state-file", default=str(STATE_FILE))
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    state_path = Path(args.state_file)

    if args.once:
        run_once(
            input_path=input_path,
            output_path=output_path,
            state_path=state_path,
            min_chars_per_site_flush=args.min_chars_per_site_flush,
            relaxed_fallback_rows=args.relaxed_fallback_rows,
        )
    else:
        run_watch(
            input_path=input_path,
            output_path=output_path,
            state_path=state_path,
            poll_interval_sec=args.poll_interval,
            min_chars_per_site_flush=args.min_chars_per_site_flush,
            relaxed_fallback_rows=args.relaxed_fallback_rows,
        )


if __name__ == "__main__":
    main()
