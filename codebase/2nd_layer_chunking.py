import json
import re
import time
from pathlib import Path


DEFAULT_INPUT = Path("chunking.jasonl")
DEFAULT_OUTPUT = Path("2nd_layer_chunking.jasonl")
DEFAULT_QDRANT_OUTPUT = Path("qdrant_ready.jasonl")


SIMPLE_STOPWORDS = {
	"the",
	"and",
	"is",
	"in",
	"to",
	"of",
	"a",
	"for",
	"on",
	"with",
	"that",
	"this",
	"it",
	"as",
	"are",
}


NOISE_URL_EXTENSIONS = (
	".css",
	".js",
	".svg",
	".png",
	".jpg",
	".jpeg",
	".gif",
	".ico",
	".webp",
	".woff",
	".woff2",
	".ttf",
)


NOISE_TEXT_PATTERNS = [
	r"#sw_as",
	r"display\s*:\s*flex",
	r"box-shadow",
	r"clip-path",
	r"background-image",
	r"font-size\s*:",
	r"margin\s*:",
	r"padding\s*:",
	r"rgba\(",
	r"%3c/svg",
	r"help center",
	r"privacy policy",
	r"terms of service",
	r"send feedback",
	r"google apps",
	r"sign in",
	r"main menu",
	r"was this helpful",
	r"submit feedback",
	r"sign up with google",
	r"already have an account",
	r"collectives",
	r"hot network questions",
	r"ask question",
	r"post your answer",
	r"new to reddit\? create your account",
	r"continue with email",
	r"continue with phone number",
	r"user agreement",
	r"was this information helpful\?",
	r"your feedback helps",
	r"skip to main content",
	r"turn off suggestions",
	r"search instead for",
	r"did you mean",
	r"report inappropriate content",
	r"mark as new",
	r"bookmark",
	r"subscribe to rss feed",
	r"subscribe to rss",
	r"mute",
	r"permalink",
	r"print",
	r"options",
	r"quick links",
	r"learn, share, save",
	r"discover and save your favorite ideas",
	r"new here\? get started",
	r"log in to community",
	r"resources and legal",
	r"site map",
	r"customer support home",
	r"close post",
	r"amd\.com feedback",
	r"social",
	r"linkedin",
	r"twitter",
	r"facebook",
	r"instagram",
	r"our partners",
	r"privacy center",
	r"security & trust center",
	r"trial software downloads",
	r"support support support",
	r"ask the microsoft community",
	r"microsoft tech community",
	r"microsoft 365 training",
	r"accessibility center",
	r"communities help you ask and answer questions",
	r"lens go beyond words",
	r"search with your camera",
	r"gstatic",
	r"marketing-cms",
]


NOISE_LINE_TOKENS = {
	"back",
	"close",
	"cancel",
	"next",
	"previous",
	"home",
	"sign in",
	"log in",
	"create account",
	"my account",
	"select a different account",
	"you have multiple accounts",
	"choose the account you want to sign in with",
	"all rights reserved",
	"copyright",
	"disclaimer",
	"cookie",
	"privacy policy",
	"terms of service",
	"terms & conditions",
	"terms and conditions",
	"newsletter",
	"subscribe",
	"share",
	"follow us",
	"follow",
	"support",
	"search",
	"feedback",
	"help",
	"related topics",
	"quick links",
	"resources and legal",
	"site map",
	"customer support home",
	"report community issue",
	"what affected your experience",
	"resolved my issue",
	"clear instructions",
	"easy to follow",
	"no jargon",
	"pictures helped",
	"didn't match my screen",
	"incorrect instructions",
	"too technical",
	"not enough information",
	"not enough pictures",
	"by pressing submit, your feedback will be used to improve microsoft products and services",
	"auto-suggest helps you quickly narrow down your search results",
}


NOISE_LINE_PHRASES = (
	"report community issue",
	"auto-suggest helps you quickly narrow down your search results",
	"what affected your experience",
	"any additional feedback",
	"by pressing submit, your feedback will be used to improve microsoft products and services",
	"privacy statement",
	"terms & conditions",
	"report illegal content",
	"cookie policy",
	"©2026 cisco systems, inc.",
)


def safe_json_lines(path: Path):
	lines = []
	with path.open("r", encoding="utf-8") as f:
		for raw in f:
			s = raw.strip()
			if not s:
				continue
			try:
				obj = json.loads(s)
			except Exception:
				# try to recover common single-quotes / trailing commas issues by ignoring line
				continue
			lines.append(obj)
	return lines


def text_metrics(text: str):
	if not text:
		text = ""
	# basic cleaning
	txt = text.replace("\n", " ")
	tokens = re.findall(r"\w+", txt.lower())
	words = [t for t in tokens if re.search(r"[a-zA-Z]", t)]
	word_count = len(words)
	unique_words = len(set(words))
	unique_ratio = (unique_words / word_count) if word_count else 0.0
	stopwords = sum(1 for w in words if w in SIMPLE_STOPWORDS)
	stopword_ratio = (stopwords / word_count) if word_count else 0.0
	# count punctuation as characters that are not word characters or whitespace
	punctuation_count = len(re.findall(r"[^\w\s]", text)) if text else 0
	punctuation_ratio = (punctuation_count / max(1, len(text)))
	avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0
	has_url = 1.0 if re.search(r"https?://|www\\.", text) else 0.0
	numeric_tokens = sum(1 for t in tokens if t.isdigit())
	numeric_ratio = (numeric_tokens / word_count) if word_count else 0.0

	return {
		"word_count": word_count,
		"unique_ratio": unique_ratio,
		"stopword_ratio": stopword_ratio,
		"punctuation_ratio": punctuation_ratio,
		"avg_word_len": avg_word_len,
		"has_url": has_url,
		"numeric_ratio": numeric_ratio,
	}


def soft_clean_text(text: str):
	if not text:
		return ""
	cleaned = text.replace("\r", "\n")
	cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
	lines = [ln.strip() for ln in cleaned.split("\n")]
	good_lines = []
	seen_lines = set()
	for ln in lines:
		if not ln:
			continue
		line_lower = ln.lower()
		if any(re.search(pat, line_lower) for pat in NOISE_TEXT_PATTERNS):
			continue
		if any(phrase in line_lower for phrase in NOISE_LINE_PHRASES):
			continue
		if line_lower in NOISE_LINE_TOKENS:
			continue
		normalized = re.sub(r"\s+", " ", line_lower).strip(" *;|:.-")
		if normalized in seen_lines:
			continue
		seen_lines.add(normalized)
		word_count = len(re.findall(r"\w+", ln))
		if word_count <= 3 and len(ln) <= 24 and not re.search(r"[.!?/:#]", ln):
			continue
		if "|" in ln and word_count <= 14:
			pipe_parts = [part.strip() for part in ln.split("|") if part.strip()]
			if pipe_parts and sum(1 for part in pipe_parts if len(part.split()) <= 3) >= max(2, len(pipe_parts) - 1):
				continue
		# drop mostly-symbol lines
		alpha = sum(ch.isalpha() for ch in ln)
		if alpha == 0 and len(ln) > 12:
			continue
		if len(ln) > 25 and alpha / max(1, len(ln)) < 0.25:
			continue
		good_lines.append(ln)
	return "\n".join(good_lines).strip()


def is_hard_noise(item, metrics):
	text = (item.get("chunk_text") or "")
	text_lower = text.lower()
	cleaned_text = soft_clean_text(text)
	cleaned_metrics = text_metrics(cleaned_text)
	url = (item.get("url") or "").lower()
	title = (item.get("title") or "")
	title_lower = title.lower()
	lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]

	if any(url.endswith(ext) for ext in NOISE_URL_EXTENSIONS):
		return True

	# Known crawler fallback pages that are consistently non-content
	if (
		"support.google.com/websearch/answer/86640" in url
		or "support.google.com/websearch/answer/181196" in url
		or "/intl/en/about/products" in url
	):
		return True

	pattern_hits = sum(1 for pat in NOISE_TEXT_PATTERNS if re.search(pat, text_lower))
	link_hits = len(re.findall(r"\[[^\]]*\]\([^)]+\)", text)) + len(re.findall(r"https?://|www\.", text_lower))
	ui_token_hits = sum(
		1
		for token in (
			"skip to main content",
			"turn off suggestions",
			"search instead for",
			"did you mean",
			"mark as new",
			"bookmark",
			"subscribe",
			"mute",
			"permalink",
			"print",
			"quick links",
			"log in to community",
			"resources and legal",
			"site map",
			"customer support home",
			"amd.com feedback",
		)
		if token in text_lower
	)
	noise_line_count = 0
	content_line_count = 0
	for line in lines:
		line_lower = line.lower()
		if any(re.search(pat, line_lower) for pat in NOISE_TEXT_PATTERNS):
			noise_line_count += 1
			continue
		if len(re.findall(r"[a-zA-Z]", line)) >= 20:
			content_line_count += 1
	percent_escape_hits = len(re.findall(r"%[0-9a-fA-F]{2}", text))
	if pattern_hits >= 2:
		if cleaned_metrics["word_count"] < 18:
			return True
		return False

	# Dense navigation / footer / search scaffolding should be removed even if it
	# coexists with a small amount of real text.
	if cleaned_metrics["word_count"] >= 18 and cleaned_metrics["unique_ratio"] >= 0.30:
		return False
	if (ui_token_hits >= 4 and metrics["word_count"] < 250) or (ui_token_hits >= 6) or (link_hits >= 18 and metrics["word_count"] < 120):
		if cleaned_metrics["word_count"] < 18:
			return True
		return False
	if noise_line_count >= 4 and noise_line_count >= content_line_count:
		if cleaned_metrics["word_count"] < 18:
			return True
		return False
	if percent_escape_hits >= 10 and ui_token_hits >= 1:
		if cleaned_metrics["word_count"] < 18:
			return True
		return False

	# very short chunks are usually low-value boilerplate/noise
	if cleaned_metrics["word_count"] < 6:
		return True

	# keep bullet lists and technical text; only drop extreme symbol-heavy content
	if metrics["punctuation_ratio"] > 0.40 and cleaned_metrics["word_count"] < 20:
		return True

	if metrics["numeric_ratio"] > 0.70 and cleaned_metrics["word_count"] < 20:
		return True

	# navigation / chrome content from help sites
	if (
		("help center" in text_lower and "privacy policy" in text_lower)
		or ("google search help" in text_lower and "send feedback" in text_lower)
		or ("main menu" in text_lower and "google apps" in text_lower)
		or ("privacy statement" in text_lower and "cookies, ads & emails" in text_lower)
		or ("report community issue" in text_lower and "privacy statement" in text_lower)
	):
		return True

	# titles indicating non-content pages
	if "sorry, this page can't be found" in title_lower:
		return True

	# tag-cloud like content (many bullet markers + hyphenated tags)
	raw_tokens = re.findall(r"[\w-]+", text_lower)
	if len(raw_tokens) >= 40 and text.count("*") >= 6:
		hyphenated = sum(1 for t in raw_tokens if "-" in t)
		if hyphenated / max(1, len(raw_tokens)) > 0.12:
			return True

	# explicit forum/tag-cloud junk patterns
	if (
		("hot network questions" in text_lower)
		or ("post your answer" in text_lower)
		or ("sign up with google" in text_lower and "already have an account" in text_lower)
	):
		return cleaned_metrics["word_count"] < 18

	return False


def normalize(values):
	if not values:
		return []
	mn = min(values)
	mx = max(values)
	if mx == mn:
		return [0.5 for _ in values]
	return [(v - mn) / (mx - mn) for v in values]


def score_chunks(items):
	metrics_list = [text_metrics(item.get("chunk_text", "")) for item in items]

	wc = [m["word_count"] for m in metrics_list]
	ur = [m["unique_ratio"] for m in metrics_list]
	sr = [m["stopword_ratio"] for m in metrics_list]
	pr = [m["punctuation_ratio"] for m in metrics_list]
	aw = [m["avg_word_len"] for m in metrics_list]
	nu = [m["numeric_ratio"] for m in metrics_list]
	hu = [m["has_url"] for m in metrics_list]

	nwc = normalize(wc)
	nur = normalize(ur)
	nsr = normalize(sr)
	npr = normalize(pr)
	naw = normalize(aw)
	nnu = normalize(nu)

	scores = []
	for i in range(len(items)):
		# Compose an adaptive score that rewards length, uniqueness, average word length,
		# and penalizes stopword-heavy or very punctuation/noisy chunks.
		s = 0.0
		s += 0.45 * nwc[i]
		s += 0.20 * nur[i]
		s += 0.15 * naw[i]
		s -= 0.25 * nsr[i]
		s -= 0.10 * npr[i]
		s -= 0.05 * nnu[i]
		# small boost if a URL exists (often indicates code / references, but we keep it modest)
		s += 0.05 * hu[i]
		# clamp
		scores.append(max(-1.0, min(1.0, s)))

	# normalize scores to 0..1
	scores_norm = normalize(scores)
	return scores_norm, metrics_list


def adaptive_filter(items):
	scores, metrics_list = score_chunks(items)
	for it, sc in zip(items, scores):
		it["second_layer_score"] = sc
		it["chunk_text_clean"] = soft_clean_text(it.get("chunk_text", ""))

	# Keep all non-hard-noise rows. The goal is recall-first filtering,
	# not aggressive ranking-based dropping.
	for i, it in enumerate(items):
		hard_noise = is_hard_noise(it, metrics_list[i])
		it["second_layer_hard_noise"] = hard_noise
		keep = not hard_noise
		it["second_layer_keep"] = bool(keep)

	return items


def _build_qdrant_record(item: dict) -> dict:
	clean_text = (item.get("chunk_text_clean") or item.get("chunk_text") or "").strip()
	payload = {
		"website": item.get("website") or "",
		"source_question": item.get("source_question") or "",
		"query": item.get("query") or "",
		"url": item.get("url") or "",
		"title": item.get("title") or "",
		"crawl_timestamp": item.get("crawl_timestamp") or "",
		"chunk_index": item.get("chunk_index"),
		"chunk_count": item.get("chunk_count"),
		"filter_mode": item.get("filter_mode") or "",
		"quality_score": item.get("quality_score"),
		"query_term_overlap": item.get("query_term_overlap"),
		"second_layer_score": item.get("second_layer_score"),
		"second_layer_hard_noise": item.get("second_layer_hard_noise", False),
		"second_layer_keep": item.get("second_layer_keep", True),
	}
	return {
		"id": f"{payload['website']}::{payload['url']}::{payload['chunk_index']}",
		"text": clean_text,
		"payload": payload,
	}


def write_qdrant_ready(items: list[dict], output_path: Path) -> int:
	ready_items = [_build_qdrant_record(it) for it in items if it.get("second_layer_keep")]
	with output_path.open("w", encoding="utf-8") as out:
		for record in ready_items:
			out.write(json.dumps(record, ensure_ascii=False) + "\n")
	return len(ready_items)


def main(
	input_path: Path = DEFAULT_INPUT,
	output_path: Path = DEFAULT_OUTPUT,
	qdrant_output_path: Path = DEFAULT_QDRANT_OUTPUT,
):
	input_path = Path(input_path)
	if not input_path.exists():
		print(f"Input file not found: {input_path} - clearing outputs")
		# ensure downstream outputs are cleared when input disappears
		output_path.write_text("", encoding="utf-8")
		qdrant_output_path.write_text("", encoding="utf-8")
		return

	items = safe_json_lines(input_path)
	if not items:
		print("No items parsed from input; clearing outputs.")
		# If the input was truncated/cleared (e.g. website closed), clear downstream files too
		output_path.write_text("", encoding="utf-8")
		qdrant_output_path.write_text("", encoding="utf-8")
		return

	items = adaptive_filter(items)
	rows_to_write = [it for it in items if it.get("second_layer_keep")]

	# write out cleaned jsonl
	with output_path.open("w", encoding="utf-8") as out:
		for it in rows_to_write:
			# replace raw chunk with cleaned text for downstream RAG
			it["chunk_text"] = it.get("chunk_text_clean", it.get("chunk_text", ""))
			out.write(json.dumps(it, ensure_ascii=False) + "\n")

	qdrant_rows = write_qdrant_ready(rows_to_write, qdrant_output_path)

	kept = sum(1 for it in items if it.get("second_layer_keep"))
	print(
		f"Wrote {len(rows_to_write)} items to {output_path} and {qdrant_rows} qdrant-ready rows to {qdrant_output_path} "
		f"(kept={kept}/{len(items)}, ratio={kept/len(items):.2f})"
	)


def _input_signature(input_path: Path) -> tuple[int, int]:
	if not input_path.exists():
		return (0, 0)
	stat = input_path.stat()
	return (int(stat.st_mtime_ns), int(stat.st_size))


def run_watch(
	input_path: Path = DEFAULT_INPUT,
	output_path: Path = DEFAULT_OUTPUT,
	qdrant_output_path: Path = DEFAULT_QDRANT_OUTPUT,
	poll_interval: float = 1.0,
) -> None:
	last_signature: tuple[int, int] | None = None
	print(f"Watching {input_path.name} for changes...")
	while True:
		signature = _input_signature(input_path)
		if signature != last_signature:
			last_signature = signature
			main(input_path=input_path, output_path=output_path, qdrant_output_path=qdrant_output_path)
		time.sleep(poll_interval)


if __name__ == "__main__":
	import argparse

	p = argparse.ArgumentParser(description="Adaptive 2nd-layer chunking filter")
	p.add_argument("--watch", action="store_true", help="Continuously rebuild output when chunking.jasonl changes.")
	p.add_argument("--poll-interval", type=float, default=1.0, help="Watch polling interval in seconds.")
	p.add_argument("--input", default=str(DEFAULT_INPUT))
	p.add_argument("--output", default=str(DEFAULT_OUTPUT))
	p.add_argument("--qdrant_output", default=str(DEFAULT_QDRANT_OUTPUT))
	args = p.parse_args()
	if args.watch:
		run_watch(
			input_path=Path(args.input),
			output_path=Path(args.output),
			qdrant_output_path=Path(args.qdrant_output),
			poll_interval=args.poll_interval,
		)
	else:
		main(
			input_path=Path(args.input),
			output_path=Path(args.output),
			qdrant_output_path=Path(args.qdrant_output),
		)

