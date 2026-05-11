import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RESULT_FILE = "webcrawler_result.jasonl"


BOILERPLATE_PATTERNS = [
    "skip to main content",
    "sign in",
    "sign out",
    "privacy",
    "terms of use",
    "all rights reserved",
    "feedback",
    "theme",
    "cookie",
    "ask learn",
    "table of contents",
    "home",
    "questions",
    "tags",
    "settings",
]


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _normalize_for_duplicate_key(value: str) -> str:
    text = _normalize_text(value)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_useful_chunk(chunk_text: str) -> bool:
    text = (chunk_text or "").strip()
    if not text:
        return False

    normalized = _normalize_text(text)
    if len(normalized) < 120:
        return False

    boilerplate_hits = sum(1 for p in BOILERPLATE_PATTERNS if p in normalized)
    if boilerplate_hits >= 2:
        return False

    words = re.findall(r"[a-zA-Z]{3,}", text)
    unique_words = {w.lower() for w in words}
    if len(unique_words) < 18:
        return False

    markdown_link_count = len(re.findall(r"\[[^\]]+\]\([^)]+\)", text))
    if markdown_link_count >= 12:
        return False

    return True


def _safe_json_loads(line: str) -> dict[str, Any] | None:
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def load_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    parse_errors = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        data = _safe_json_loads(line)
        if data is None:
            parse_errors += 1
            continue
        rows.append(data)
    return rows, parse_errors


def build_report(rows: list[dict[str, Any]], parse_errors: int, top_n: int) -> dict[str, Any]:
    status_counter = Counter((r.get("status") or "unknown").strip() for r in rows)
    query_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    url_counter = Counter()
    useful_url_counter = Counter()
    duplicate_chunk_counter = Counter()

    useful_success_rows = 0
    total_success_rows = 0

    for row in rows:
        query = (row.get("query") or "").strip() or "(missing query)"
        query_to_rows[query].append(row)

        url = (row.get("url") or "").strip()
        if url:
            url_counter[url] += 1

        status = (row.get("status") or "").strip().lower()
        chunk_text = row.get("chunk_text") or ""
        is_success = status == "success"

        if is_success:
            total_success_rows += 1
            if _is_useful_chunk(chunk_text):
                useful_success_rows += 1
                if url:
                    useful_url_counter[url] += 1

        duplicate_key = _normalize_for_duplicate_key(chunk_text)
        if duplicate_key:
            duplicate_chunk_counter[duplicate_key] += 1

    query_reports = []
    resolved_queries = 0
    unresolved_queries = []

    for query, q_rows in query_to_rows.items():
        q_statuses = Counter((r.get("status") or "unknown").strip().lower() for r in q_rows)
        q_success_rows = q_statuses.get("success", 0)
        q_failed_rows = len(q_rows) - q_success_rows

        q_success_urls = {
            (r.get("url") or "").strip()
            for r in q_rows
            if (r.get("status") or "").strip().lower() == "success" and (r.get("url") or "").strip()
        }
        q_useful_success_rows = sum(
            1
            for r in q_rows
            if (r.get("status") or "").strip().lower() == "success" and _is_useful_chunk(r.get("chunk_text") or "")
        )

        is_resolved = len(q_success_urls) > 0
        if is_resolved:
            resolved_queries += 1
        else:
            unresolved_queries.append(query)

        query_reports.append(
            {
                "query": query,
                "total_rows": len(q_rows),
                "success_rows": q_success_rows,
                "non_success_rows": q_failed_rows,
                "unique_success_urls": len(q_success_urls),
                "useful_success_rows": q_useful_success_rows,
                "resolved": is_resolved,
            }
        )

    duplicate_entries = [
        {"count": count, "chunk_preview": key[:180]}
        for key, count in duplicate_chunk_counter.items()
        if count > 1
    ]
    duplicate_entries.sort(key=lambda x: x["count"], reverse=True)

    duplicate_chunk_rows = sum(c - 1 for c in duplicate_chunk_counter.values() if c > 1)
    repeated_urls = [{"url": u, "count": c} for u, c in url_counter.items() if c > 1]
    repeated_urls.sort(key=lambda x: x["count"], reverse=True)

    total_queries = len(query_to_rows)
    coverage = (resolved_queries / total_queries) if total_queries else 0.0
    useful_ratio = (useful_success_rows / total_success_rows) if total_success_rows else 0.0

    report = {
        "summary": {
            "total_rows": len(rows),
            "parse_errors": parse_errors,
            "status_counts": dict(status_counter),
            "total_queries": total_queries,
            "resolved_queries": resolved_queries,
            "query_coverage_ratio": round(coverage, 4),
            "unique_urls": len(url_counter),
            "unique_useful_urls": len(useful_url_counter),
            "success_rows": total_success_rows,
            "useful_success_rows": useful_success_rows,
            "useful_success_ratio": round(useful_ratio, 4),
            "duplicate_chunk_rows": duplicate_chunk_rows,
            "repeated_urls_count": len(repeated_urls),
        },
        "unresolved_queries": sorted(unresolved_queries),
        "top_repeated_urls": repeated_urls[:top_n],
        "top_duplicate_chunks": duplicate_entries[:top_n],
        "query_reports": sorted(query_reports, key=lambda x: (x["resolved"], x["useful_success_rows"], x["success_rows"])),
    }
    return report


def print_human_report(report: dict[str, Any], top_n: int) -> None:
    s = report["summary"]
    print("=== Crawl Quality Audit ===")
    print(f"total_rows: {s['total_rows']}")
    print(f"parse_errors: {s['parse_errors']}")
    print(f"status_counts: {s['status_counts']}")
    print(f"total_queries: {s['total_queries']}")
    print(f"resolved_queries: {s['resolved_queries']}")
    print(f"query_coverage_ratio: {s['query_coverage_ratio']}")
    print(f"unique_urls: {s['unique_urls']}")
    print(f"unique_useful_urls: {s['unique_useful_urls']}")
    print(f"success_rows: {s['success_rows']}")
    print(f"useful_success_rows: {s['useful_success_rows']}")
    print(f"useful_success_ratio: {s['useful_success_ratio']}")
    print(f"duplicate_chunk_rows: {s['duplicate_chunk_rows']}")
    print(f"repeated_urls_count: {s['repeated_urls_count']}")

    print("\n--- Unresolved Queries ---")
    if not report["unresolved_queries"]:
        print("none")
    else:
        for query in report["unresolved_queries"]:
            print(f"- {query}")

    print(f"\n--- Top {top_n} Repeated URLs ---")
    if not report["top_repeated_urls"]:
        print("none")
    else:
        for item in report["top_repeated_urls"]:
            print(f"- {item['count']}x {item['url']}")

    print(f"\n--- Top {top_n} Duplicate Chunks ---")
    if not report["top_duplicate_chunks"]:
        print("none")
    else:
        for item in report["top_duplicate_chunks"]:
            print(f"- {item['count']}x {item['chunk_preview']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit crawler JSONL quality without changing crawler workflow.")
    parser.add_argument(
        "--input",
        default=DEFAULT_RESULT_FILE,
        help=f"Path to crawler result JSONL (default: {DEFAULT_RESULT_FILE})",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to save full JSON report.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many repeated URLs/duplicate chunks to show (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    top_n = max(1, args.top)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    rows, parse_errors = load_rows(input_path)
    report = build_report(rows, parse_errors, top_n)
    print_human_report(report, top_n)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
