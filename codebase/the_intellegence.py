import json
import os
import re
from typing import Any

from openai import OpenAI

import question_validation as qv
from rag import current_ready_file_signature, load_answer_chunk, run_rag_pipeline


os.environ["GROQ_API_KEY"] = ""

MODEL_NAME = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

client = OpenAI(
	api_key=os.getenv("GROQ_API_KEY"),
	base_url=GROQ_BASE_URL,
)

ASCII_GRAPH_OUTPUT = ""
CRAWLER_READY_QUERIES: list[str] = []
SOURCE_QUESTION = ""
SOURCE_CONTEXT = ""
MIN_QUERY_COUNT = 10
MAX_QUERY_COUNT = 14

CRAWL_SITE_HINTS: dict[str, list[str]] = {
	"stackoverflow.com": ["code", "bug", "error", "python", "java", "javascript", "api", "exception"],
	"superuser.com": ["windows", "laptop", "desktop", "driver", "boot", "keyboard", "computer"],
	"serverfault.com": ["server", "dns", "ssl", "nginx", "apache", "linux", "infrastructure"],
	"reddit.com": ["issue", "troubleshoot", "community", "experience", "fix"],
	"github.com": ["issue", "repository", "sdk", "library", "integration", "bug"],
	"learn.microsoft.com": ["windows", "azure", "microsoft", "office", "auth", "identity"],
	"support.microsoft.com": ["windows", "outlook", "office", "account", "login"],
	"support.google.com": ["gmail", "android", "chrome", "google", "account", "login"],
	"developer.apple.com": ["ios", "mac", "apple", "xcode", "iphone", "ipad"],
	"support.apple.com": ["iphone", "ios", "mac", "apple", "icloud"],
	"docs.python.org": ["python", "pip", "venv", "traceback", "module"],
	"pypi.org": ["python", "package", "pip", "dependency"],
	"nodejs.org": ["node", "npm", "javascript", "package"],
	"npmjs.com": ["npm", "node", "package", "dependency"],
	"developer.mozilla.org": ["javascript", "web", "browser", "html", "css"],
	"docs.oracle.com": ["java", "jdbc", "oracle", "jvm"],
	"docs.aws.amazon.com": ["aws", "cloud", "s3", "lambda", "iam"],
	"cloud.google.com": ["gcp", "cloud", "kubernetes", "gke"],
	"learn.microsoft.com/azure": ["azure", "cloud", "container", "deployment"],
	"kubernetes.io": ["kubernetes", "k8s", "pod", "cluster", "container"],
	"docs.docker.com": ["docker", "container", "image", "compose"],
	"askubuntu.com": ["ubuntu", "linux", "apt", "systemd"],
	"archlinux.org": ["linux", "package", "kernel", "driver"],
	"ubuntu.com": ["ubuntu", "linux", "server"],
	"wiki.debian.org": ["debian", "linux", "package"],
	"mozilla.org": ["firefox", "browser", "security"],
	"support.mozilla.org": ["firefox", "browser", "profile", "addon"],
	"intel.com": ["cpu", "intel", "driver", "chipset"],
	"nvidia.com": ["gpu", "nvidia", "driver", "display"],
	"amd.com": ["gpu", "cpu", "amd", "driver"],
	"hp.com/support": ["printer", "hp", "cartridge", "firmware"],
	"support.hp.com": ["printer", "hp", "laptop", "driver"],
	"dell.com/support": ["dell", "laptop", "bios", "driver"],
	"support.lenovo.com": ["lenovo", "laptop", "bios", "driver"],
	"support.asus.com": ["asus", "laptop", "router", "bios"],
	"netgear.com/support": ["router", "wifi", "network"],
	"tp-link.com/support": ["router", "wifi", "network"],
	"community.cisco.com": ["network", "router", "switch", "vpn"],
	"openai.com": ["openai", "llm", "api", "chatgpt"],
	"console.groq.com": ["groq", "api", "model", "inference"],
	"community.openai.com": ["openai", "api", "community"],
	"status.openai.com": ["openai", "outage", "status"],
	"status.groq.com": ["groq", "outage", "status"],
	"medium.com": ["guide", "tutorial", "walkthrough", "explanation"],
	"dev.to": ["tutorial", "debug", "web", "python", "node"],
}


ANSWER_SYNTHESIS_MODEL = MODEL_NAME


def _tokenize_for_relevance(text: str) -> set[str]:
	return set(re.findall(r"[a-z0-9_\-]+", (text or "").lower()))


def _select_relevant_sites(question: str, context_text: str, limit: int = MAX_QUERY_COUNT) -> list[str]:
	tokens = _tokenize_for_relevance(f"{question} {context_text}")
	scored: list[tuple[int, str]] = []

	for site, hints in CRAWL_SITE_HINTS.items():
		score = sum(3 for hint in hints if hint in tokens)
		if site in {"stackoverflow.com", "github.com", "learn.microsoft.com", "support.google.com", "reddit.com"}:
			score += 2
		scored.append((score, site))

	scored.sort(key=lambda item: item[0], reverse=True)
	selected = [site for score, site in scored if score > 0][:limit]

	if len(selected) < min(8, limit):
		for site in ["stackoverflow.com", "superuser.com", "github.com", "learn.microsoft.com", "support.google.com", "reddit.com", "medium.com", "docs.python.org"]:
			if site not in selected:
				selected.append(site)
			if len(selected) >= min(8, limit):
				break

	return selected[:limit]


def _safe_list(value: Any) -> list[str]:
	if not isinstance(value, list):
		return []
	result: list[str] = []
	for item in value:
		if isinstance(item, str):
			cleaned = item.strip()
			if cleaned and cleaned not in result:
				result.append(cleaned)
	return result


def _parse_json_payload(text: str) -> dict[str, Any]:
	value = (text or "").strip()
	if not value:
		return {}
	if value.startswith("```"):
		value = value.removeprefix("```json").removeprefix("```").strip()
		if value.endswith("```"):
			value = value[:-3].strip()
	try:
		return json.loads(value)
	except Exception:
		start = value.find("{")
		end = value.rfind("}")
		if start != -1 and end != -1 and end > start:
			try:
				return json.loads(value[start : end + 1])
			except Exception:
				return {}
		return {}


def _normalize_question_text(question: str) -> str:
	value = (question or "").strip()
	if not value:
		return ""
	if not value.endswith("?"):
		value = f"{value}?"
	return value[0].upper() + value[1:]


def _fallback_clarifying_questions(question: str, context_text: str) -> list[str]:
	scope_text = (question or context_text or "this issue").strip()
	return [
		f"What exact error message or symptom do you see with {scope_text}?",
		"Which device, app, account, or browser is affected, including version if available?",
		"When did this start, and what have you already tried so far?",
	]


def _ensure_question_count(questions: list[str], fallback: list[str], question_count: int) -> list[str]:
	normalized: list[str] = []
	for question in questions:
		normalized_question = _normalize_question_text(question)
		if normalized_question and normalized_question not in normalized:
			normalized.append(normalized_question)
		if len(normalized) >= question_count:
			return normalized

	for question in fallback:
		normalized_question = _normalize_question_text(question)
		if normalized_question and normalized_question not in normalized:
			normalized.append(normalized_question)
		if len(normalized) >= question_count:
			return normalized

	return normalized[:question_count]


def _normalize_answer_type(value: str) -> str:
	allowed = {"Fact", "List", "Explanation", "Comparison"}
	normalized = (value or "").strip().title()
	if normalized in allowed:
		return normalized
	return "Explanation"


def _format_graph(
	question: str,
	intent: str,
	entities: list[str],
	constraints: list[str],
	decomposed: list[str],
	queries: list[str],
	answer_type: str,
) -> str:
	entities = entities or ["None"]
	constraints = constraints or ["None"]
	decomposed = decomposed or ["None"]
	queries = queries or ["None"]

	lines: list[str] = []
	lines.append("========== QUESTION GRAPH ==========")
	lines.append(f"[Q] {question}")
	lines.append("")
	lines.append(f" ├── [I] {intent or 'None'}")
	lines.append("")
	lines.append(" ├── [E]")
	for entity in entities:
		lines.append(f" │     ├── {entity}")
	lines.append("")
	lines.append(" ├── [C]")
	for index, constraint in enumerate(constraints):
		branch = "└──" if index == len(constraints) - 1 else "├──"
		lines.append(f" │     {branch} {constraint}")
	lines.append("")
	lines.append(" ├── [D]")
	for sub_question in decomposed:
		lines.append(f" │     ├── {sub_question}")
	lines.append("")
	lines.append(" ├── [R]")
	for search_query in queries:
		lines.append(f" │     ├── {search_query}")
	lines.append("")
	lines.append(" └── [A]")
	lines.append(f"       └── expected answer type: {answer_type}")
	lines.append("===================================")
	return "\n".join(lines)


def _generate_reasoning_payload(question: str, context_text: str) -> dict[str, Any]:
	selected_sites = _select_relevant_sites(question, context_text)
	selected_sites_text = ", ".join([f"site:{site}" for site in selected_sites]) if selected_sites else "site:stackoverflow.com"

	system_prompt = (
		"You are a structured reasoning and query generation engine. "
		"Return strict JSON only with keys: "
		"intent, entities, constraints, decomposed_questions, queries, expected_answer_type. "
		f"Rules: queries must be {MIN_QUERY_COUNT} to {MAX_QUERY_COUNT} items, unique, concise, crawler-ready. "
		"You must think and choose websites relevant to this exact problem context. "
		f"Prefer these relevant site operators first: {selected_sites_text}. "
		"At least 8 queries must include distinct site operators when possible. "
		"Use filetype:pdf in one query when relevant. "
		"If any list is empty, return [\"None\"]. "
		"expected_answer_type must be exactly one of: Fact, List, Explanation, Comparison."
	)

	user_prompt = (
		"Generate reasoning JSON for this user question and context.\n"
		f"Primary user question:\n{question}\n\n"
		"Collected question/answer context from question_validation.py (DETAILED_INPUT_HISTORY):\n"
		f"{context_text or 'None'}\n\n"
		f"Relevant site pool to consider first:\n{json.dumps(selected_sites, ensure_ascii=True)}\n\n"
		"Important: queries should align with entities and decomposed_questions."
	)

	response = client.chat.completions.create(
		model=MODEL_NAME,
		temperature=0,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
	)

	content = response.choices[0].message.content if response.choices else "{}"
	return _parse_json_payload(content or "{}")


def generate_relevant_clarifying_questions(
	question: str,
	context_text: str = "",
	answer_chunk: dict[str, Any] | None = None,
	question_count: int = 3,
) -> list[str]:
	cleaned_question = (question or "").strip()
	if not cleaned_question:
		return _fallback_clarifying_questions(question, context_text)[: max(1, question_count)]

	chunk_payload = answer_chunk or load_answer_chunk()
	saved_query = str(chunk_payload.get("query") or "").strip() if isinstance(chunk_payload, dict) else ""
	if not chunk_payload or saved_query != cleaned_question:
		try:
			chunk_payload = run_rag_pipeline(cleaned_question)
		except Exception:
			chunk_payload = chunk_payload or {}

	top_chunks = list(chunk_payload.get("top_chunks") or [])[:5] if isinstance(chunk_payload, dict) else []
	evidence: list[dict[str, str]] = []
	for chunk in top_chunks:
		if not isinstance(chunk, dict):
			continue
		evidence.append(
			{
				"title": str(chunk.get("title") or "").strip(),
				"url": str(chunk.get("url") or "").strip(),
				"snippet": str(chunk.get("chunk_text") or "").strip()[:300],
			}
		)

	system_prompt = (
		"You are a customer-support clarification assistant. "
		f"Generate exactly {question_count} relevant, non-redundant follow-up questions. "
		"Use the user issue, prior context, and retrieved evidence snippets. "
		"Do not ask generic filler questions and do not repeat details already present. "
		"Return JSON only with schema: {\"questions\": [string, string, string]}."
	)

	user_prompt = json.dumps(
		{
			"question": cleaned_question,
			"context_text": context_text or "",
			"retrieved_evidence": evidence,
		},
		ensure_ascii=True,
		indent=2,
	)

	generated: list[str] = []
	try:
		response = client.chat.completions.create(
			model=MODEL_NAME,
			temperature=0,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
		)
		content = response.choices[0].message.content if response.choices else "{}"
		payload = _parse_json_payload(content or "{}")
		generated = _safe_list(payload.get("questions"))
	except Exception:
		generated = []

	fallback = _fallback_clarifying_questions(cleaned_question, context_text)
	return _ensure_question_count(generated, fallback, max(1, question_count))


def generate_graph_and_queries(question: str, context_text: str) -> tuple[str, list[str]]:
	payload = _generate_reasoning_payload(question, context_text)
	relevant_sites = _select_relevant_sites(question, context_text)
	try:
		run_rag_pipeline(question)
	except Exception as exc:
		print(f"[rag] Failed to generate answer chunk: {exc}")

	intent = str(payload.get("intent") or "None").strip() or "None"
	entities = _safe_list(payload.get("entities"))
	constraints = _safe_list(payload.get("constraints"))
	decomposed = _safe_list(payload.get("decomposed_questions"))
	queries = _safe_list(payload.get("queries"))

	# Guarantee broader but relevant crawling coverage.
	for site in relevant_sites:
		candidate = f"{question} troubleshooting steps site:{site}"
		if candidate not in queries:
			queries.append(candidate)
		if len(queries) >= MAX_QUERY_COUNT - 1:
			break

	pdf_candidate = f"{question} root cause analysis filetype:pdf"
	if pdf_candidate not in queries:
		queries.append(pdf_candidate)

	queries = queries[:MAX_QUERY_COUNT]

	if len(queries) < MIN_QUERY_COUNT:
		for site in relevant_sites:
			candidate = f"{question} fix steps site:{site}"
			if candidate not in queries:
				queries.append(candidate)
			if len(queries) >= MIN_QUERY_COUNT:
				break

	answer_type = _normalize_answer_type(str(payload.get("expected_answer_type") or "Explanation"))

	graph = _format_graph(
		question=question,
		intent=intent,
		entities=entities,
		constraints=constraints,
		decomposed=decomposed,
		queries=queries,
		answer_type=answer_type,
	)

	return graph, queries


def generate_answer_from_answer_chunk(question: str, context_text: str = "", answer_chunk: dict[str, Any] | None = None) -> dict[str, Any]:
	current_question = (question or "").strip()
	chunk_payload = answer_chunk or load_answer_chunk()
	should_refresh_chunk = not chunk_payload
	saved_query = str(chunk_payload.get("query") or "").strip() if chunk_payload else ""

	if chunk_payload and current_question:
		if saved_query != current_question:
			should_refresh_chunk = True

	if chunk_payload and not should_refresh_chunk:
		current_signature = current_ready_file_signature()
		saved_file = str(chunk_payload.get("source_file") or "").strip()
		saved_mtime = int(chunk_payload.get("source_mtime_ns") or 0)
		saved_size = int(chunk_payload.get("source_size") or 0)
		if (
			saved_file != str(current_signature.get("source_file") or "")
			or saved_mtime != int(current_signature.get("source_mtime_ns") or 0)
			or saved_size != int(current_signature.get("source_size") or 0)
		):
			should_refresh_chunk = True

	if should_refresh_chunk:
		effective_query = current_question or saved_query
		chunk_payload = run_rag_pipeline(effective_query)

	source_websites = list(chunk_payload.get("source_websites") or [])
	if not source_websites:
		source_websites = []
		for item in chunk_payload.get("top_chunks") or []:
			url = str((item or {}).get("url") or "").strip()
			if not url:
				continue
			match = re.match(r"^(?:https?://)?([^/]+)", url, flags=re.IGNORECASE)
			hostname = (match.group(1) if match else url).strip().lower()
			if hostname and hostname not in source_websites:
				source_websites.append(hostname)

	answer_context = {
		"question": question or "",
		"context_text": context_text or "",
		"answer_chunk": chunk_payload,
		"source_websites": source_websites,
	}

	system_prompt = (
		"You are a support resolution writer. "
		"Use the retrieved answer chunk and the user context to produce a concise, practical answer. "
		"Prefer short paragraphs and bullet-like steps only when useful. "
		"Do not mention internal file names, embeddings, or vector search. "
		"If the evidence is thin, say so plainly and suggest the next best step. "
		"Return plain text only."
	)

	user_prompt = (
		"Synthesize the final customer-facing answer from this retrieval payload and context.\n\n"
		f"{json.dumps(answer_context, ensure_ascii=True, indent=2)}"
	)

	response = client.chat.completions.create(
		model=ANSWER_SYNTHESIS_MODEL,
		temperature=0.2,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
	)

	content = response.choices[0].message.content if response.choices else ""
	return {
		"answer": (content or "").strip(),
		"source_websites": source_websites,
		"answer_chunk": chunk_payload,
	}


def _build_context_from_question_validation(validation_result: dict[str, Any]) -> str:
	history = getattr(qv, "DETAILED_INPUT_HISTORY", None)
	if isinstance(history, list) and history:
		return "\n\n---\n\n".join([entry for entry in history if isinstance(entry, str) and entry.strip()]).strip()

	detailed_input_text = (validation_result.get("detailed_input") or "").strip()
	if detailed_input_text:
		return detailed_input_text

	return (
		(validation_result.get("input_query") or "").strip()
		or (validation_result.get("concise_query") or "").strip()
	)


def run_intelligence_pipeline() -> tuple[str, list[str]]:
	validation_result = qv.validate_query_text(qv.read_latest_query())
	question = (
		(validation_result.get("input_query") or "").strip()
		or (validation_result.get("concise_query") or "").strip()
	)
	context_text = _build_context_from_question_validation(validation_result)

	if not question:
		raise ValueError("No input question found in question_validation.py source output.")

	graph, queries = generate_graph_and_queries(question, context_text)
	return graph, queries


if __name__ == "__main__":
	validation_result = qv.validate_query_text(qv.read_latest_query())
	SOURCE_QUESTION = (
		(validation_result.get("input_query") or "").strip()
		or (validation_result.get("concise_query") or "").strip()
	)
	SOURCE_CONTEXT = _build_context_from_question_validation(validation_result)
	if not SOURCE_QUESTION:
		raise ValueError("No input question found in question_validation.py source output.")

	ASCII_GRAPH_OUTPUT, CRAWLER_READY_QUERIES = generate_graph_and_queries(SOURCE_QUESTION, SOURCE_CONTEXT)
	print(ASCII_GRAPH_OUTPUT)
	print()
	print(json.dumps(CRAWLER_READY_QUERIES, ensure_ascii=True, indent=2))
