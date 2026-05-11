import json
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote_plus, parse_qs, urlparse, urlunparse, unquote
import re
import requests

BASE_DIR = Path(__file__).parent
CRAWLER_RESULT_FILE = BASE_DIR / "webcrawler_result.jasonl"
CHUNKING_RESULT_FILE = BASE_DIR / "chunking.jasonl"
CHUNKING_STATE_FILE = BASE_DIR / ".chunking_state.json"
STOP_CRAWL_SIGNAL_FILE = BASE_DIR / ".stop_crawl"
SEARCH_ENGINE_DOMAINS = {
	"google.com",
	"www.google.com",
	"bing.com",
	"www.bing.com",
	"duckduckgo.com",
	"www.duckduckgo.com",
}
NON_CONTENT_PATH_HINTS = (
	"/accounts",
	"/preferences",
	"/settings",
	"/policies",
	"/privacy",
	"/terms",
	"/search",
)
MAX_QUERIES_PER_CRAWL = int(os.getenv("CRAWLER_MAX_QUERIES_PER_CRAWL", "8"))
MAX_QUERIES_PER_WEBSITE = int(os.getenv("CRAWLER_MAX_QUERIES_PER_WEBSITE", "2"))
MAX_SEARCH_VARIANTS_PER_QUERY = int(os.getenv("CRAWLER_MAX_SEARCH_VARIANTS_PER_QUERY", "2"))
MAX_SEARCH_ENGINES_PER_VARIANT = int(os.getenv("CRAWLER_MAX_SEARCH_ENGINES_PER_VARIANT", "2"))
MAX_PAGE_URLS_PER_QUERY = int(os.getenv("CRAWLER_MAX_PAGE_URLS_PER_QUERY", "2"))
MAX_CONTENT_CHARS_PER_PAGE = int(os.getenv("CRAWLER_MAX_CONTENT_CHARS_PER_PAGE", "12000"))
MAX_CHUNKS_PER_PAGE = int(os.getenv("CRAWLER_MAX_CHUNKS_PER_PAGE", "8"))
MIN_CHUNK_CHARS = int(os.getenv("CRAWLER_MIN_CHUNK_CHARS", "60"))


def _bootstrap_local_venv() -> None:
	"""Re-run this script with the bundled venv when the active interpreter is missing crawl4ai."""
	if os.environ.get("WEBCRAWLER_BOOTSTRAPPED") == "1":
		return

	venv_python = BASE_DIR / "venv" / "Scripts" / "python.exe"
	if not venv_python.exists():
		return

	try:
		import crawl4ai  # noqa: F401
		return
	except ModuleNotFoundError:
		os.environ["WEBCRAWLER_BOOTSTRAPPED"] = "1"
		try:
			os.execv(str(venv_python), [str(venv_python), *sys.argv])
		except OSError as exc:
			print(f"[CRAWLER] Failed to switch to local venv interpreter: {exc}")
			print("[CRAWLER] Continuing with the current interpreter.")


_bootstrap_local_venv()

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from the_intellegence import run_intelligence_pipeline


def _extract_domain_from_url(url: str) -> str:
	"""Extract domain name from URL for identification."""
	try:
		parsed = urlparse(url)
		domain = parsed.netloc.replace("www.", "")
		return domain or url
	except Exception:
		return url


def _extract_website_from_query(query: str) -> str:
	"""Extract website name from search query using site: operator."""
	match = re.search(r"site:([^\s]+)", query)
	if match:
		return match.group(1)
	return "unknown"


def _is_known_website(website: str) -> bool:
	normalized = (website or "").strip().lower()
	return normalized not in {"", "unknown"} and "." in normalized


def _normalize_whitespace(value: str) -> str:
	return re.sub(r"\s+", " ", (value or "")).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
	seen: set[str] = set()
	out: list[str] = []
	for value in values:
		if value and value not in seen:
			seen.add(value)
			out.append(value)
	return out


def _build_search_queries(query: str, website: str, source_question: str = "") -> list[str]:
	"""Generate fallback search query variants without changing overall pipeline behavior."""
	base_query = _normalize_whitespace(query)
	candidates: list[str] = [base_query]

	# If upstream produced site:unknown, drop that invalid constraint.
	if "site:unknown" in base_query.lower():
		candidates.append(re.sub(r"site:unknown", "", base_query, flags=re.IGNORECASE).strip())

	# Remove quotes for broader recall if exact matching is too strict.
	if '"' in base_query:
		candidates.append(base_query.replace('"', " ").strip())

	# Use the source question as a broader fallback.
	question_fallback = _normalize_whitespace(source_question)
	if question_fallback:
		if _is_known_website(website):
			candidates.append(f"{question_fallback} site:{website}")
		candidates.append(question_fallback)

	# If query has no site filter but website is known, add one focused variant.
	if _is_known_website(website) and "site:" not in base_query.lower():
		candidates.append(f"{base_query} site:{website}")

	return _dedupe_preserve_order([_normalize_whitespace(item) for item in candidates if item])


def _optimize_queries_for_crawl(queries: list[str]) -> list[str]:
	"""Keep crawl breadth controlled without changing the overall pipeline steps."""
	seen_queries: set[str] = set()
	seen_per_website: dict[str, int] = {}
	optimized: list[str] = []

	for raw_query in queries:
		query = _normalize_whitespace(raw_query)
		if not query:
			continue
		query_key = query.lower()
		if query_key in seen_queries:
			continue

		website = _extract_website_from_query(query).lower()
		count_for_site = seen_per_website.get(website, 0)
		if count_for_site >= max(1, MAX_QUERIES_PER_WEBSITE):
			continue

		seen_queries.add(query_key)
		seen_per_website[website] = count_for_site + 1
		optimized.append(query)

		if len(optimized) >= max(1, MAX_QUERIES_PER_CRAWL):
			break

	return optimized


def _build_search_urls(search_query: str) -> list[str]:
	"""Build resilient search URLs across multiple providers."""
	encoded = quote_plus(search_query)
	return [
		f"https://www.google.com/search?q={encoded}",
		f"https://www.bing.com/search?q={encoded}",
		f"https://duckduckgo.com/html/?q={encoded}",
	]


def _build_search_url(query: str, website: str) -> str:
	"""Backward-compatible primary search URL used for reporting."""
	search_queries = _build_search_queries(query, website)
	primary_query = search_queries[0] if search_queries else _normalize_whitespace(query)
	return _build_search_urls(primary_query)[0]


def _looks_like_search_engine_url(url: str) -> bool:
	domain = _extract_domain_from_url(url).lower()
	return domain in SEARCH_ENGINE_DOMAINS


def _clean_redirect_url(url: str) -> str:
	"""Extract real destination from common search redirect URLs."""
	if not url:
		return ""

	try:
		parsed = urlparse(url)
		domain = (parsed.netloc or "").lower()
		query = parse_qs(parsed.query)

		if "google." in domain and parsed.path == "/url":
			target = query.get("q", [""])[0]
			if target:
				return target

		if "duckduckgo.com" in domain and parsed.path.startswith("/l/"):
			target = query.get("uddg", [""])[0]
			if target:
				return unquote(target)

		if "bing.com" in domain:
			target = query.get("r", [""])[0]
			if target.startswith("http"):
				return target

	except Exception:
		return url

	return url


def _normalize_candidate_url(url: str) -> str:
	"""Normalize URL for de-duplication and safer crawling."""
	cleaned = _clean_redirect_url((url or "").strip())
	if not cleaned.startswith("http"):
		return ""

	try:
		parsed = urlparse(cleaned)
		path = parsed.path.rstrip("/") or "/"
		normalized = urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))
		return normalized
	except Exception:
		return cleaned


def _looks_like_non_content_url(url: str) -> bool:
	try:
		path = urlparse(url).path.lower()
	except Exception:
		return False
	return any(token in path for token in NON_CONTENT_PATH_HINTS)


def _website_matches_url(url: str, website: str) -> bool:
	"""Return True when a crawled URL belongs to the requested website scope."""
	if not url:
		return False

	if not _is_known_website(website):
		return not _looks_like_search_engine_url(url)

	url_domain = _extract_domain_from_url(url).lower().replace("www.", "")
	target_domain = _extract_domain_from_url(website).lower().replace("www.", "")
	return url_domain == target_domain or url_domain.endswith(f".{target_domain}")


def _extract_urls_from_text(value: str) -> list[str]:
	"""Extract candidate URLs from raw markdown/html when structured links are missing."""
	if not value:
		return []

	text = value.replace("&amp;", "&")
	candidates: list[str] = []

	# Markdown links.
	for match in re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", text, flags=re.IGNORECASE):
		candidates.append(match)

	# Plain URLs.
	for match in re.findall(r"https?://[^\s\"'<>]+", text, flags=re.IGNORECASE):
		candidates.append(match)

	# Google-style redirect URLs.
	for redirect in re.findall(r"/url\?q=(https?://[^&\s]+)", text, flags=re.IGNORECASE):
		candidates.append(unquote(redirect))

	return _dedupe_preserve_order(candidates)


def _extract_candidate_page_urls(search_result: Any, website: str, limit: int = 3) -> list[str]:
	"""Pull real page URLs from a search result page with resilient extraction and filtering."""
	links = getattr(search_result, "links", None) or {}
	strict_candidates: list[str] = []
	relaxed_candidates: list[str] = []

	for group_name in ("external", "internal"):
		for link in links.get(group_name, []) if isinstance(links, dict) else []:
			if not isinstance(link, dict):
				continue
			href = _normalize_candidate_url(link.get("href") or "")
			if not href or _looks_like_search_engine_url(href):
				continue
			if _looks_like_non_content_url(href):
				continue

			if _website_matches_url(href, website):
				strict_candidates.append(href)
			relaxed_candidates.append(href)

	# Fallback: parse raw search result text when structured links are absent.
	raw_search_text = (getattr(search_result, "markdown", None) or "") + "\n" + (getattr(search_result, "cleaned_html", None) or "")
	for href in _extract_urls_from_text(raw_search_text):
		normalized = _normalize_candidate_url(href)
		if not normalized or _looks_like_search_engine_url(normalized):
			continue
		if _looks_like_non_content_url(normalized):
			continue
		if _website_matches_url(normalized, website):
			strict_candidates.append(normalized)
		relaxed_candidates.append(normalized)

	strict_candidates = _dedupe_preserve_order(strict_candidates)
	relaxed_candidates = _dedupe_preserve_order(relaxed_candidates)

	if strict_candidates:
		return strict_candidates[:limit]

	# Avoid unrelated crawl drift: for known websites, skip relaxed cross-domain links.
	if _is_known_website(website):
		return []

	# For unknown targets, fall back to best available non-search URLs.
	return relaxed_candidates[:limit]


def _chunk_text(text: str, max_chars: int = 1200) -> list[str]:
	"""Split crawled text into RAG-friendly chunks without breaking paragraphs when possible."""
	clean_text = (text or "").strip()
	if not clean_text:
		return []
	if len(clean_text) > max(1000, MAX_CONTENT_CHARS_PER_PAGE):
		clean_text = clean_text[: max(1000, MAX_CONTENT_CHARS_PER_PAGE)]

	paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", clean_text) if paragraph.strip()]
	chunks: list[str] = []
	current_chunk = ""

	for paragraph in paragraphs:
		if len(paragraph) > max_chars:
			if current_chunk:
				chunks.append(current_chunk.strip())
				current_chunk = ""

			start = 0
			while start < len(paragraph):
				chunks.append(paragraph[start : start + max_chars].strip())
				start += max_chars
			continue

		candidate = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
		if len(candidate) <= max_chars:
			current_chunk = candidate
		else:
			if current_chunk:
				chunks.append(current_chunk.strip())
			current_chunk = paragraph

	if current_chunk:
		chunks.append(current_chunk.strip())

	filtered = [chunk.strip() for chunk in chunks if chunk and len(chunk.strip()) >= max(1, MIN_CHUNK_CHARS)]
	if not filtered:
		filtered = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
	return filtered[: max(1, MAX_CHUNKS_PER_PAGE)]


def _build_crawl_configs() -> tuple[BrowserConfig, CrawlerRunConfig]:
	"""Build conservative crawler settings to reduce browser load and hangs."""
	browser_type = os.getenv("CRAWLER_BROWSER_TYPE", "chromium")
	page_timeout_ms = int(os.getenv("CRAWLER_PAGE_TIMEOUT_MS", "30000"))

	browser_config = BrowserConfig(
		browser_type=browser_type,
		headless=True,
		light_mode=True,
		memory_saving_mode=True,
		max_pages_before_recycle=50,
		extra_args=[
			"--disable-gpu",
			"--disable-gpu-compositing",
			"--disable-software-rasterizer",
		],
		verbose=False,
	)
	run_config = CrawlerRunConfig(
		cache_mode=CacheMode.BYPASS,
		page_timeout=page_timeout_ms,
		wait_until="domcontentloaded",
		wait_for_images=False,
		process_iframes=False,
		scan_full_page=False,
		verbose=False,
	)
	return browser_config, run_config


def _should_stop_crawl() -> bool:
	"""Check whether app signaled the crawler to stop."""
	return STOP_CRAWL_SIGNAL_FILE.exists()


def _reset_chunking_outputs() -> None:
	"""Clear chunking output/state when a fresh crawl starts."""
	for path in (CHUNKING_RESULT_FILE, CHUNKING_STATE_FILE):
		try:
			if path.exists():
				path.write_text("", encoding="utf-8")
		except Exception as exc:
			print(f"[CRAWLER] Could not reset {path.name}: {exc}")


async def crawl_and_save_results(queries: list[str], source_question: str = "") -> dict[str, Any]:
	"""
	Crawl websites based on queries and save results to JSONL file.
	
	Args:
		queries: List of crawler-ready queries with site operators
		source_question: The original user question for context
	
	Returns:
		Dictionary with crawl statistics and status
	"""
	# Clear the results file at the start of a new crawl
	CRAWLER_RESULT_FILE.write_text("", encoding="utf-8")
	_reset_chunking_outputs()
	print(f"[CRAWLER] Cleared previous crawl results")
	try:
		if STOP_CRAWL_SIGNAL_FILE.exists():
			STOP_CRAWL_SIGNAL_FILE.unlink()
	except Exception:
		pass
	
	# Notify app that crawl is starting
	try:
		requests.post("http://localhost:5000/crawl-status", json={"action": "start"}, timeout=2)
	except Exception as e:
		print(f"[CRAWLER] Could not notify app of crawl start: {e}")
	
	optimized_queries = _optimize_queries_for_crawl(queries or [])
	results_stats = {
		"timestamp": datetime.now().isoformat(),
		"source_question": source_question,
		"total_queries": len(optimized_queries),
		"original_queries": len(queries or []),
		"successful_crawls": 0,
		"failed_crawls": 0,
		"websites_crawled": set(),
		"results_file": str(CRAWLER_RESULT_FILE),
	}

	if not optimized_queries:
		print("[CRAWLER] No queries provided. Skipping crawl.")
		results_stats["error"] = "No queries provided"
		try:
			requests.post("http://localhost:5000/crawl-status", json={"action": "end"}, timeout=2)
		except Exception:
			pass
		return results_stats

	print(f"[CRAWLER] Starting crawl for {len(optimized_queries)} queries (from {len(queries or [])})...")
	
	browser_config, run_config = _build_crawl_configs()
	async with AsyncWebCrawler(config=browser_config) as crawler:
		with open(CRAWLER_RESULT_FILE, "w", encoding="utf-8") as f:

			# Process each query
			for idx, query in enumerate(optimized_queries, 1):
				if _should_stop_crawl():
					print("[CRAWLER] Stop signal received. Ending crawl early.")
					break

				print(f"[CRAWLER] Processing query {idx}/{len(optimized_queries)}: {query[:80]}...")
				
				website = _extract_website_from_query(query)
				results_stats["websites_crawled"].add(website)
				
				# Notify app of current website being crawled
				try:
					requests.post(
						"http://localhost:5000/crawl-status",
						json={"action": "update", "website": website},
						timeout=2
					)
				except Exception as e:
					print(f"[CRAWLER] Could not notify app of website: {e}")

				try:
					page_urls: list[str] = []
					if query.startswith("http"):
						page_urls = [query]
					else:
						search_attempt_errors: list[str] = []
						search_queries = _build_search_queries(query, website, source_question)[: max(1, MAX_SEARCH_VARIANTS_PER_QUERY)]

						for search_query in search_queries:
							if _should_stop_crawl():
								break

							for search_url in _build_search_urls(search_query)[: max(1, MAX_SEARCH_ENGINES_PER_VARIANT)]:
								if _should_stop_crawl():
									break

								try:
									search_result = await crawler.arun(url=search_url, config=run_config, bypass_cache=True)
									candidates = _extract_candidate_page_urls(search_result, website, limit=max(1, MAX_PAGE_URLS_PER_QUERY))
									page_urls.extend(candidates)
									page_urls = _dedupe_preserve_order(page_urls)
								except Exception as search_exc:
									search_attempt_errors.append(f"{search_url} -> {str(search_exc)[:140]}")

								if len(page_urls) >= max(1, MAX_PAGE_URLS_PER_QUERY):
									break

							if len(page_urls) >= max(1, MAX_PAGE_URLS_PER_QUERY):
								break

						# Final fallback: crawl the site homepage when we have a known site but no result links.
						if not page_urls and _is_known_website(website):
							page_urls = [f"https://{website}"]

					if not page_urls:
						record = {
							"query": query,
							"website": website,
							"url": query if query.startswith("http") else _build_search_url(query, website),
							"crawl_timestamp": datetime.now().isoformat(),
							"status": "failed",
							"error": "No candidate links found after search fallbacks",
							"source_question": source_question,
							"search_attempts": len(_build_search_queries(query, website, source_question)) if not query.startswith("http") else 0,
							"diagnostic": search_attempt_errors[:5] if not query.startswith("http") else [],
						}
						f.write(json.dumps(record, ensure_ascii=False) + "\n")
						f.flush()
						results_stats["failed_crawls"] += 1
						print(f"  [FAIL] No candidate links found for: {website}")
						continue

					for page_url in page_urls:
						if _should_stop_crawl():
							print("[CRAWLER] Stop signal received during page crawl. Ending early.")
							break

						page_result = await crawler.arun(url=page_url, config=run_config, bypass_cache=True)

						if page_result and page_result.success:
							content_text = page_result.markdown or page_result.cleaned_html or ""
							content_chunks = _chunk_text(content_text)
							if not content_chunks:
								content_chunks = [""]

							for chunk_index, chunk_text in enumerate(content_chunks, start=1):
								if _should_stop_crawl():
									print("[CRAWLER] Stop signal received during chunk write. Ending early.")
									break

								record = {
									"query": query,
									"website": website,
									"url": page_url,
									"title": page_result.metadata.get("title", ""),
									"description": page_result.metadata.get("description", ""),
									"content_source": "markdown" if page_result.markdown else "cleaned_html",
									"chunk_index": chunk_index,
									"chunk_count": len(content_chunks),
									"chunk_text": chunk_text,
									"crawl_timestamp": datetime.now().isoformat(),
									"status": "success",
									"source_question": source_question,
								}
								f.write(json.dumps(record, ensure_ascii=False) + "\n")
								f.flush()

							results_stats["successful_crawls"] += 1
							print(f"  [OK] Successfully crawled: {page_url} ({len(content_chunks)} chunk(s))")

						else:
							# Record failure
							record = {
								"query": query,
								"website": website,
								"url": page_url,
								"crawl_timestamp": datetime.now().isoformat(),
								"status": "failed",
								"error": page_result.error_message if page_result else "Unknown error",
								"source_question": source_question,
							}
							
							f.write(json.dumps(record, ensure_ascii=False) + "\n")
							f.flush()
	
							results_stats["failed_crawls"] += 1
							print(f"  [FAIL] Failed to crawl: {website} - {page_result.error_message if page_result else 'Unknown error'}")

				except asyncio.TimeoutError:
					record = {
						"query": query,
						"website": website,
						"crawl_timestamp": datetime.now().isoformat(),
						"status": "timeout",
						"error": "Request timeout",
						"source_question": source_question,
					}
					f.write(json.dumps(record, ensure_ascii=False) + "\n")
					f.flush()
					results_stats["failed_crawls"] += 1
					print(f"  [TIMEOUT] {website}")

				except Exception as e:
					record = {
						"query": query,
						"website": website,
						"crawl_timestamp": datetime.now().isoformat(),
						"status": "error",
						"error": str(e),
						"source_question": source_question,
					}
					f.write(json.dumps(record, ensure_ascii=False) + "\n")
					f.flush()
					results_stats["failed_crawls"] += 1
					print(f"  [ERROR] Error crawling {website}: {str(e)[:100]}")

				# Small delay to avoid rate limiting
				await asyncio.sleep(0.5)

	# Convert set to list for JSON serialization
	results_stats["websites_crawled"] = list(results_stats["websites_crawled"])

	print(f"\n[CRAWLER] Crawl complete!")
	print(f"  Successful: {results_stats['successful_crawls']}")
	print(f"  Failed: {results_stats['failed_crawls']}")
	print(f"  Websites: {', '.join(results_stats['websites_crawled'])}")
	print(f"  Saved to: {CRAWLER_RESULT_FILE}")
	
	# Notify app that crawl is complete
	try:
		requests.post("http://localhost:5000/crawl-status", json={"action": "end"}, timeout=2)
	except Exception as e:
		print(f"[CRAWLER] Could not notify app of crawl end: {e}")

	return results_stats


def run_crawl_pipeline(queries: list[str], source_question: str = "") -> dict[str, Any]:
	"""
	Run the crawling pipeline synchronously.
	
	Args:
		queries: List of crawler-ready queries
		source_question: The original user question for context
	
	Returns:
		Dictionary with crawl statistics
	"""
	return asyncio.run(crawl_and_save_results(queries, source_question))


def run_full_intelligence_and_crawl_pipeline() -> dict[str, Any]:
	"""
	Complete pipeline: Get graph/queries from intelligence module, then crawl.
	This is the main entry point that orchestrates the entire process.
	
	Returns:
		Dictionary with graph, queries, and crawl statistics
	"""
	try:
		print("[CRAWLER] Starting full intelligence and crawl pipeline...")
		
		# Get graph and queries from intelligence module
		graph, queries = run_intelligence_pipeline()
		
		print(f"[CRAWLER] Received graph and {len(queries)} queries from intelligence module")
		print(f"[CRAWLER] Graph:\n{graph}\n")
		
		# Extract source question from first query or use default
		source_question = queries[0] if queries else "Unknown question"
		
		# Run the crawl pipeline
		crawl_stats = asyncio.run(crawl_and_save_results(queries, source_question))
		
		result = {
			"graph": graph,
			"queries": queries,
			"crawl_stats": crawl_stats,
			"status": "completed",
		}
		
		print(f"[CRAWLER] Full pipeline completed successfully")
		return result
		
	except Exception as e:
		print(f"[CRAWLER] Pipeline error: {e}")
		return {
			"graph": "",
			"queries": [],
			"crawl_stats": {},
			"status": "failed",
			"error": str(e),
		}


def read_crawler_results() -> list[dict[str, Any]]:
	"""Read all results from the JSONL file."""
	results = []
	if CRAWLER_RESULT_FILE.exists():
		try:
			with open(CRAWLER_RESULT_FILE, "r", encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if line:
						try:
							results.append(json.loads(line))
						except json.JSONDecodeError:
							continue
		except Exception as e:
			print(f"[CRAWLER] Error reading results: {e}")
	return results


def get_results_by_website(website: str) -> list[dict[str, Any]]:
	"""Get all results for a specific website."""
	all_results = read_crawler_results()
	return [r for r in all_results if r.get("website") == website]


def get_successful_results() -> list[dict[str, Any]]:
	"""Get only successful crawl results."""
	all_results = read_crawler_results()
	return [r for r in all_results if r.get("status") == "success"]


if __name__ == "__main__":
	# Run the complete pipeline: intelligence + crawl
	result = run_full_intelligence_and_crawl_pipeline()
	print(f"\n[CRAWLER] Final Result:")
	print(json.dumps(result, indent=2))
