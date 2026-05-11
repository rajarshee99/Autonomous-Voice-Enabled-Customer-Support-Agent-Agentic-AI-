from pathlib import Path
from datetime import datetime
import json
import threading
import atexit
import os

from flask import Flask, jsonify, request, send_from_directory

from voice_to_text import transcribe_audio_file
from query_preprocessing import preprocess_query_text
from question_validation import validate_query_text, reset_validation_session
from webcrawler import run_crawl_pipeline
from rag import current_ready_file_signature, load_answer_chunk, run_rag_pipeline
from the_intellegence import (
	_build_context_from_question_validation,
	generate_answer_from_answer_chunk,
	generate_graph_and_queries,
	generate_relevant_clarifying_questions,
)
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
LIVE_STATUS_FILE = BASE_DIR / "FRONTEND_LIVE_OUTPUT.md"
LIVE_STATUS_JSON_FILE = BASE_DIR / "FRONTEND_LIVE_OUTPUT.json"
CRAWLER_RESULT_FILE = BASE_DIR / "webcrawler_result.jasonl"
STOP_CRAWL_SIGNAL_FILE = BASE_DIR / ".stop_crawl"
CLEAR_CRAWLER_RESULTS_ON_FLUSH = os.getenv("CLEAR_CRAWLER_RESULTS_ON_FLUSH", "1") == "1"
CLEAR_ALL_JSONL_ON_FLUSH = os.getenv("CLEAR_ALL_JSONL_ON_FLUSH", "0") == "1"

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
LAST_TEXT_INPUT = ""
LAST_VOICE_TEXT = ""
LAST_PREPROCESSED_TEXT = ""
LAST_VALIDATION_RESULT = {}
LAST_LLM_ANSWER = ""
LAST_SOURCE_WEBSITES: list[str] = []

# Crawl progress tracking
IS_CRAWLING = False
CURRENT_CRAWL_WEBSITE = None
CRAWL_START_TIME = None
CRAWL_WEBSITES_SEEN = set()
RAG_REFRESH_THREAD = None
RAG_REFRESH_STOP = threading.Event()


def _pick_query_for_rag_refresh() -> str:
	candidates = [
		(LAST_PREPROCESSED_TEXT or "").strip(),
		(LAST_TEXT_INPUT or "").strip(),
		(LAST_VALIDATION_RESULT.get("concise_query") or "").strip() if isinstance(LAST_VALIDATION_RESULT, dict) else "",
		(LAST_VALIDATION_RESULT.get("input_query") or "").strip() if isinstance(LAST_VALIDATION_RESULT, dict) else "",
	]

	try:
		from question_validation import read_latest_query
		candidates.append((read_latest_query() or "").strip())
	except Exception:
		pass

	try:
		chunk_payload = load_answer_chunk()
		candidates.append(str(chunk_payload.get("query") or "").strip())
	except Exception:
		pass

	for candidate in candidates:
		if candidate:
			return candidate
	return ""


def _refresh_answer_chunk_if_stale() -> None:
	signature = current_ready_file_signature()
	if int(signature.get("source_size") or 0) <= 0:
		return

	chunk_payload = load_answer_chunk()
	chunk_file = str(chunk_payload.get("source_file") or "").strip()
	chunk_mtime = int(chunk_payload.get("source_mtime_ns") or 0)
	chunk_size = int(chunk_payload.get("source_size") or 0)
	current_file = str(signature.get("source_file") or "").strip()
	current_mtime = int(signature.get("source_mtime_ns") or 0)
	current_size = int(signature.get("source_size") or 0)

	is_stale = (
		not chunk_payload
		or chunk_file != current_file
		or chunk_mtime != current_mtime
		or chunk_size != current_size
	)
	if not is_stale:
		return

	query_for_refresh = _pick_query_for_rag_refresh()
	if not query_for_refresh:
		return

	try:
		run_rag_pipeline(query_for_refresh)
		print(f"[rag-refresh] answer_chunk.jasonl refreshed from {current_file} (size={current_size})")
	except Exception as exc:
		print(f"[rag-refresh] refresh failed: {exc}")


def start_rag_refresh_watcher(poll_interval: float = 2.0) -> None:
	global RAG_REFRESH_THREAD
	if RAG_REFRESH_THREAD and RAG_REFRESH_THREAD.is_alive():
		return

	RAG_REFRESH_STOP.clear()

	def _watcher():
		last_signature: tuple[str, int, int] | None = None
		while not RAG_REFRESH_STOP.is_set():
			try:
				sig = current_ready_file_signature()
				current_signature = (
					str(sig.get("source_file") or ""),
					int(sig.get("source_mtime_ns") or 0),
					int(sig.get("source_size") or 0),
				)
				if current_signature != last_signature:
					last_signature = current_signature
					_refresh_answer_chunk_if_stale()
			except Exception as exc:
				print(f"[rag-refresh] watcher error: {exc}")
			time.sleep(poll_interval)

	RAG_REFRESH_THREAD = threading.Thread(target=_watcher, daemon=True)
	RAG_REFRESH_THREAD.start()


def stop_rag_refresh_watcher() -> None:
	RAG_REFRESH_STOP.set()


def load_live_state_from_json():
	"""Restore last known values from JSON so restarts do not clear live state."""
	global LAST_TEXT_INPUT, LAST_VOICE_TEXT, LAST_PREPROCESSED_TEXT, LAST_VALIDATION_RESULT, LAST_LLM_ANSWER, LAST_SOURCE_WEBSITES

	if not LIVE_STATUS_JSON_FILE.exists():
		return

	try:
		payload = json.loads(LIVE_STATUS_JSON_FILE.read_text(encoding="utf-8"))
		LAST_TEXT_INPUT = (payload.get("last_text_input") or "").strip()
		LAST_VOICE_TEXT = (payload.get("last_voice_text") or "").strip()
		LAST_PREPROCESSED_TEXT = (payload.get("last_preprocessed_text") or "").strip()
		LAST_VALIDATION_RESULT = payload.get("last_validation_result") or {}
		LAST_LLM_ANSWER = (payload.get("last_llm_answer") or "").strip()
		LAST_SOURCE_WEBSITES = payload.get("last_source_websites") or []
	except Exception:
		# If file is malformed, keep defaults and continue.
		pass


def _load_crawled_websites() -> list[str]:
	"""Return the unique list of websites that were crawled most recently."""
	websites: list[str] = []
	if not CRAWLER_RESULT_FILE.exists():
		return websites

	try:
		with CRAWLER_RESULT_FILE.open("r", encoding="utf-8") as file_handle:
			for line in file_handle:
				line = line.strip()
				if not line:
					continue
				try:
					record = json.loads(line)
				except json.JSONDecodeError:
					continue

				website = (record.get("website") or "").strip()
				if website and website not in websites:
					websites.append(website)
	except Exception:
		return websites

	return websites


def _flush_crawler_results_file() -> None:
	"""Optionally clear crawler result JSONL based on env toggle."""
	if not CLEAR_CRAWLER_RESULTS_ON_FLUSH:
		return
	try:
		CRAWLER_RESULT_FILE.write_text("", encoding="utf-8")
	except Exception as exc:
		print(f"[cleanup] Failed to flush crawler results file: {exc}")


def _clear_all_jsonl_files() -> None:
	"""Clear all .jasonl files in the application directory."""
	try:
		for p in BASE_DIR.glob("*.jasonl"):
			try:
				p.write_text("", encoding="utf-8")
			except Exception as exc:
				print(f"[cleanup] Failed to clear {p}: {exc}")
	except Exception as exc:
		print(f"[cleanup] Failed enumerating .jasonl files: {exc}")


def _signal_crawl_stop() -> None:
	"""Signal background crawler to stop writing as soon as possible."""
	try:
		STOP_CRAWL_SIGNAL_FILE.write_text("stop", encoding="utf-8")
	except Exception as exc:
		print(f"[cleanup] Failed to write stop signal file: {exc}")


def _clear_crawl_stop_signal() -> None:
	"""Clear stale stop signal before starting a new crawl."""
	try:
		if STOP_CRAWL_SIGNAL_FILE.exists():
			STOP_CRAWL_SIGNAL_FILE.unlink()
	except Exception as exc:
		print(f"[cleanup] Failed to clear stop signal file: {exc}")


def update_live_markdown_file():
	"""Write latest frontend text input and latest voice transcription to a markdown file."""
	updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	websites_crawled = _load_crawled_websites()
	live_payload = {
		"updated_at": updated_at,
		"query": "<frontend text input>",
		"audio": "<frontend voice blob>",
		"text_response": "<response from /query>",
		"voice_transcription": "<response from /voice via voice_to_text.py>",
		"error_detail": "<error message if any>",
		"last_text_input": LAST_TEXT_INPUT or "",
		"last_preprocessed_text": LAST_PREPROCESSED_TEXT or "",
		"last_voice_text": LAST_VOICE_TEXT or "",
		"last_validation_result": LAST_VALIDATION_RESULT or {},
		"last_llm_answer": LAST_LLM_ANSWER or "",
		"last_source_websites": LAST_SOURCE_WEBSITES or [],
		"websites_crawled": websites_crawled,
	}

	content = (
		"# Frontend Live Variables\n\n"
		f"updated_at: {updated_at}\n\n"
		"query: <frontend text input>\n"
		"audio: <frontend voice blob>\n"
		"text_response: <response from /query>\n"
		"voice_transcription: <response from /voice via voice_to_text.py>\n"
		"error_detail: <error message if any>\n\n"
		f"last_text_input: {LAST_TEXT_INPUT or '(empty)'}\n"
		f"last_preprocessed_text: {LAST_PREPROCESSED_TEXT or '(empty)'}\n"
		f"last_voice_text: {LAST_VOICE_TEXT or '(empty)'}\n"
		f"last_validation_result: {json.dumps(LAST_VALIDATION_RESULT, ensure_ascii=True) if LAST_VALIDATION_RESULT else '(empty)'}\n"
		f"last_llm_answer: {LAST_LLM_ANSWER or '(empty)'}\n"
		f"last_source_websites: {', '.join(LAST_SOURCE_WEBSITES) if LAST_SOURCE_WEBSITES else '(empty)'}\n"
		f"websites_crawled: {', '.join(websites_crawled) if websites_crawled else '(empty)'}\n"
	)
	try:
		LIVE_STATUS_FILE.write_text(content, encoding="utf-8")
		LIVE_STATUS_JSON_FILE.write_text(json.dumps(live_payload, ensure_ascii=True, indent=2), encoding="utf-8")
	except Exception:
		# Avoid failing API requests if markdown write fails.
		pass


def _start_crawl_pipeline(user_text: str, preprocessed_text: str, validation_result: dict) -> None:
	"""Launch the existing intelligence + crawler pipeline in the background."""
	global IS_CRAWLING, CURRENT_CRAWL_WEBSITE, CRAWL_START_TIME, CRAWL_WEBSITES_SEEN, LAST_LLM_ANSWER, LAST_SOURCE_WEBSITES

	if IS_CRAWLING:
		print("[crawl] Crawl already in progress; skipping new start.")
		return

	IS_CRAWLING = True
	CURRENT_CRAWL_WEBSITE = None
	CRAWL_START_TIME = datetime.now()
	CRAWL_WEBSITES_SEEN = set()
	_clear_crawl_stop_signal()
	update_live_markdown_file()

	def _run_crawl_then_answer() -> tuple[str, list[str]]:
		context_text = _build_context_from_question_validation(validation_result)
		graph, queries = generate_graph_and_queries(preprocessed_text, context_text)
		print(f"[crawl] Generated {len(queries)} queries from intelligence module")
		print(f"[crawl] Graph:\n{graph}")
		run_crawl_pipeline(queries, preprocessed_text)
		# Re-run retrieval after crawl writes new chunks so answer_chunk.jasonl and LLM answer stay fresh.
		_refresh_answer_chunk_if_stale()
		refreshed_answer_payload = generate_answer_from_answer_chunk(preprocessed_text, context_text)
		final_answer = (refreshed_answer_payload.get("answer") or "").strip()
		final_sources = list(refreshed_answer_payload.get("source_websites") or [])
		if not final_sources:
			final_sources = _load_crawled_websites()
		return final_answer, final_sources

	def _worker():
		global IS_CRAWLING, CURRENT_CRAWL_WEBSITE, CRAWL_START_TIME, CRAWL_WEBSITES_SEEN, LAST_LLM_ANSWER, LAST_SOURCE_WEBSITES
		try:
			LAST_LLM_ANSWER, LAST_SOURCE_WEBSITES = _run_crawl_then_answer()
			update_live_markdown_file()
		except Exception as exc:
			print(f"[crawl] Background pipeline error: {exc}")
			IS_CRAWLING = False
			CURRENT_CRAWL_WEBSITE = None
			CRAWL_START_TIME = None
			CRAWL_WEBSITES_SEEN = set()
			update_live_markdown_file()

	threading.Thread(target=_worker, daemon=True).start()


def _run_crawl_pipeline_sync(preprocessed_text: str, validation_result: dict) -> tuple[str, list[str]]:
	"""Run crawl + retrieval synchronously so final answer is returned only after crawl."""
	global IS_CRAWLING, CURRENT_CRAWL_WEBSITE, CRAWL_START_TIME, CRAWL_WEBSITES_SEEN

	if IS_CRAWLING:
		print("[crawl] Crawl already in progress; waiting for existing crawl to finish before final answer.")

	IS_CRAWLING = True
	CURRENT_CRAWL_WEBSITE = None
	CRAWL_START_TIME = datetime.now()
	CRAWL_WEBSITES_SEEN = set()
	_clear_crawl_stop_signal()
	update_live_markdown_file()

	try:
		context_text = _build_context_from_question_validation(validation_result)
		graph, queries = generate_graph_and_queries(preprocessed_text, context_text)
		print(f"[crawl-sync] Generated {len(queries)} queries from intelligence module")
		print(f"[crawl-sync] Graph:\n{graph}")
		run_crawl_pipeline(queries, preprocessed_text)
		# Ensure answer chunk reflects the crawled corpus before synthesizing answer.
		for _ in range(5):
			_refresh_answer_chunk_if_stale()
			time.sleep(0.4)
		answer_payload = generate_answer_from_answer_chunk(preprocessed_text, context_text)
		answer_text = (answer_payload.get("answer") or "").strip()
		source_websites = list(answer_payload.get("source_websites") or [])
		if not answer_text:
			chunk_payload = load_answer_chunk()
			answer_text = (
				"I finished crawling, but could not synthesize a full response. "
				"Based on retrieved sources, please share the exact stop/error code shown on the BSOD screen "
				"so I can provide a targeted fix."
			)
			source_websites = list(chunk_payload.get("source_websites") or source_websites)
		if not source_websites:
			source_websites = _load_crawled_websites()
		return answer_text, source_websites
	finally:
		IS_CRAWLING = False
		CURRENT_CRAWL_WEBSITE = None
		CRAWL_START_TIME = None
		update_live_markdown_file()


# Chunker process management: ensure the chunker watcher runs alongside the app
CHUNKER_PROCESS = None
CHUNKER_LOG = BASE_DIR / "chunker.log"
SECOND_LAYER_PROCESS = None
SECOND_LAYER_LOG = BASE_DIR / "second_layer_chunker.log"

def start_chunker_process() -> None:
	global CHUNKER_PROCESS
	if CHUNKER_PROCESS is not None:
		return
	try:
		cmd = [str(Path(sys.executable)), str(BASE_DIR / "text preprocessing+chunking.py")]
		# Start chunker in watch mode (default) and capture logs
		CHUNKER_LOG.write_text("", encoding="utf-8")
		log_handle = open(CHUNKER_LOG, "a", encoding="utf-8")
		CHUNKER_PROCESS = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle)
		print(f"[APP] Started chunker process (pid={CHUNKER_PROCESS.pid})")
	except Exception as exc:
		print(f"[APP] Failed to start chunker process: {exc}")


def stop_chunker_process() -> None:
	global CHUNKER_PROCESS
	try:
		if CHUNKER_PROCESS:
			CHUNKER_PROCESS.terminate()
			CHUNKER_PROCESS.wait(timeout=5)
			CHUNKER_PROCESS = None
			print("[APP] Chunker process stopped")
	except Exception as exc:
		print(f"[APP] Error stopping chunker process: {exc}")


def start_second_layer_process() -> None:
	global SECOND_LAYER_PROCESS
	if SECOND_LAYER_PROCESS is not None:
		return
	try:
		cmd = [
			str(Path(sys.executable)),
			str(BASE_DIR / "2nd_layer_chunking.py"),
			"--watch",
			"--input",
			str(BASE_DIR / "chunking.jasonl"),
			"--output",
			str(BASE_DIR / "2nd_layer_chunking.jasonl"),
			"--qdrant_output",
			str(BASE_DIR / "qdrant_ready.jasonl"),
		]
		SECOND_LAYER_LOG.write_text("", encoding="utf-8")
		log_handle = open(SECOND_LAYER_LOG, "a", encoding="utf-8")
		SECOND_LAYER_PROCESS = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle)
		print(f"[APP] Started second-layer chunker process (pid={SECOND_LAYER_PROCESS.pid})")
	except Exception as exc:
		print(f"[APP] Failed to start second-layer chunker process: {exc}")


def stop_second_layer_process() -> None:
	global SECOND_LAYER_PROCESS
	try:
		if SECOND_LAYER_PROCESS:
			SECOND_LAYER_PROCESS.terminate()
			SECOND_LAYER_PROCESS.wait(timeout=5)
			SECOND_LAYER_PROCESS = None
			print("[APP] Second-layer chunker process stopped")
	except Exception as exc:
		print(f"[APP] Error stopping second-layer chunker process: {exc}")



@app.get("/")
def home():
	return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/logo")
def logo():
	"""Serve the project logo with a stable URL for frontend refreshes."""
	for filename in ("logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "logo.svg"):
		logo_path = FRONTEND_DIR / filename
		if logo_path.exists():
			return send_from_directory(FRONTEND_DIR, filename)
	return ("Logo not found", 404)


@app.post("/query")
def query():
	global LAST_TEXT_INPUT, LAST_PREPROCESSED_TEXT, LAST_VALIDATION_RESULT, LAST_LLM_ANSWER, LAST_SOURCE_WEBSITES

	payload = request.get_json(silent=True) or {}
	user_text = (payload.get("query") or "").strip()

	if not user_text:
		return jsonify({"text": "Please type a message."}), 400

	LAST_TEXT_INPUT = user_text
	LAST_VALIDATION_RESULT = validate_query_text(user_text)
	saved_context = LAST_VALIDATION_RESULT.get("saved_user_data") or {}
	context_concise_query = str(saved_context.get("concise_query") or "").strip()
	context_input_query = str(saved_context.get("input_query") or "").strip()
	LAST_PREPROCESSED_TEXT = (
		context_concise_query
		or context_input_query
		or (LAST_VALIDATION_RESULT.get("concise_query") or preprocess_query_text(user_text))
	).strip()
	if LAST_VALIDATION_RESULT.get("needs_more_info"):
		context_text = _build_context_from_question_validation(LAST_VALIDATION_RESULT)
		clarifying_questions = generate_relevant_clarifying_questions(
			LAST_PREPROCESSED_TEXT or user_text,
			context_text,
			question_count=3,
		)
		if clarifying_questions:
			LAST_VALIDATION_RESULT["top_level_questions"] = clarifying_questions
			LAST_VALIDATION_RESULT["follow_up_question"] = clarifying_questions[0]
			LAST_VALIDATION_RESULT["display_message"] = "Please answer these 3 quick questions so I can resolve this accurately."

		preliminary_answer = ""
		preliminary_sources: list[str] = []
		try:
			answer_payload = generate_answer_from_answer_chunk(LAST_PREPROCESSED_TEXT or user_text, context_text)
			preliminary_answer = (answer_payload.get("answer") or "").strip()
			preliminary_sources = list(answer_payload.get("source_websites") or [])
		except Exception as exc:
			print(f"[query] preliminary answer generation failed: {exc}")

		questions_block = "\n".join(
			[f"{index}. {question}" for index, question in enumerate(clarifying_questions, start=1)]
		)
		if preliminary_answer:
			LAST_LLM_ANSWER = (
				"Please answer these 3 questions first:\n"
				f"{questions_block}\n\n"
				"Current best answer from Pinecone retrieval:\n"
				f"{preliminary_answer}"
			).strip()
			LAST_SOURCE_WEBSITES = preliminary_sources
		else:
			LAST_LLM_ANSWER = (
				"Please answer these 3 questions first:\n"
				f"{questions_block}"
			).strip()
			LAST_SOURCE_WEBSITES = []
	else:
		context_text = _build_context_from_question_validation(LAST_VALIDATION_RESULT)
		if LAST_VALIDATION_RESULT.get("is_complete"):
			LAST_LLM_ANSWER = "Crawling complete data now. Preparing final answer..."
			LAST_SOURCE_WEBSITES = []
			update_live_markdown_file()
			try:
				LAST_LLM_ANSWER, LAST_SOURCE_WEBSITES = _run_crawl_pipeline_sync(
					LAST_PREPROCESSED_TEXT,
					LAST_VALIDATION_RESULT,
				)
			except Exception as exc:
				print(f"[query] sync crawl finalization failed: {exc}")
				try:
					fallback_answer_payload = generate_answer_from_answer_chunk(LAST_PREPROCESSED_TEXT, context_text)
					LAST_LLM_ANSWER = (fallback_answer_payload.get("answer") or "").strip()
					LAST_SOURCE_WEBSITES = list(fallback_answer_payload.get("source_websites") or [])
				except Exception as fallback_exc:
					print(f"[query] fallback synthesis failed: {fallback_exc}")
					LAST_LLM_ANSWER = ""
					LAST_SOURCE_WEBSITES = []
				if not LAST_LLM_ANSWER:
					LAST_LLM_ANSWER = (
						"I completed crawling but could not prepare a full final answer. "
						"Please share the exact BSOD stop/error code and when it appears."
					)
				if not LAST_SOURCE_WEBSITES:
					chunk_payload = load_answer_chunk()
					LAST_SOURCE_WEBSITES = list(chunk_payload.get("source_websites") or _load_crawled_websites())
		else:
			answer_payload = generate_answer_from_answer_chunk(LAST_PREPROCESSED_TEXT, context_text)
			LAST_LLM_ANSWER = (answer_payload.get("answer") or "").strip()
			LAST_SOURCE_WEBSITES = list(answer_payload.get("source_websites") or [])
	print(f"[query] last_text_input={LAST_TEXT_INPUT}")
	print(f"[query] last_preprocessed_text={LAST_PREPROCESSED_TEXT}")
	print(f"[query] last_validation_result={json.dumps(LAST_VALIDATION_RESULT, ensure_ascii=True)}")
	print(f"[query] last_llm_answer={LAST_LLM_ANSWER}")
	print(f"[query] last_source_websites={LAST_SOURCE_WEBSITES}")
	update_live_markdown_file()

	payload = dict(LAST_VALIDATION_RESULT)
	payload["llm_answer"] = LAST_LLM_ANSWER
	payload["source_websites"] = LAST_SOURCE_WEBSITES
	return jsonify(payload)


@app.post("/voice")
def voice():
	global LAST_VOICE_TEXT

	if "audio" not in request.files:
		return jsonify({"detail": "No audio file found."}), 400

	audio_file = request.files["audio"]
	if audio_file.filename == "":
		return jsonify({"detail": "Audio file is empty."}), 400

	text = transcribe_audio_file(audio_file)
	LAST_VOICE_TEXT = text or ""
	update_live_markdown_file()
	return jsonify({"text": text})


@app.post("/crawl-status")
def update_crawl_status():
	"""Update crawl progress tracking (called by crawler during execution)."""
	global IS_CRAWLING, CURRENT_CRAWL_WEBSITE, CRAWL_START_TIME, CRAWL_WEBSITES_SEEN
	
	data = request.get_json() or {}
	action = data.get("action")  # 'start', 'update', 'end'
	
	if action == "start":
		IS_CRAWLING = True
		CRAWL_START_TIME = datetime.now()
		CRAWL_WEBSITES_SEEN = set()
		CURRENT_CRAWL_WEBSITE = None
		print(f"[STATUS] Crawl started")
	elif action == "update":
		CURRENT_CRAWL_WEBSITE = data.get("website")
		if CURRENT_CRAWL_WEBSITE:
			CRAWL_WEBSITES_SEEN.add(CURRENT_CRAWL_WEBSITE)
		print(f"[STATUS] Crawling: {CURRENT_CRAWL_WEBSITE}")
	elif action == "end":
		IS_CRAWLING = False
		CURRENT_CRAWL_WEBSITE = None
		CRAWL_START_TIME = None
		print(f"[STATUS] Crawl completed")
	
	return jsonify({"status": "updated"})


@app.get("/live")
def live():
	return jsonify(
		{
			"updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
			"last_text_input": LAST_TEXT_INPUT,
			"last_preprocessed_text": LAST_PREPROCESSED_TEXT,
			"last_voice_text": LAST_VOICE_TEXT,
			"last_validation_result": LAST_VALIDATION_RESULT,
			"last_llm_answer": LAST_LLM_ANSWER,
			"last_source_websites": LAST_SOURCE_WEBSITES,
			"websites_crawled": _load_crawled_websites() if not IS_CRAWLING else list(CRAWL_WEBSITES_SEEN),
			"is_crawling": IS_CRAWLING,
			"current_website": CURRENT_CRAWL_WEBSITE,
			"crawl_start_time": CRAWL_START_TIME.isoformat() if CRAWL_START_TIME else None,
		}
	)


@app.post("/flush-crawler-results")
def flush_crawler_results():
	"""Allow frontend unload hooks to stop crawl; optional file clear via env flag."""
	global IS_CRAWLING, CURRENT_CRAWL_WEBSITE, CRAWL_START_TIME, CRAWL_WEBSITES_SEEN
	global LAST_TEXT_INPUT, LAST_VOICE_TEXT, LAST_PREPROCESSED_TEXT, LAST_VALIDATION_RESULT, LAST_LLM_ANSWER, LAST_SOURCE_WEBSITES
	_signal_crawl_stop()
	IS_CRAWLING = False
	CURRENT_CRAWL_WEBSITE = None
	CRAWL_START_TIME = None
	CRAWL_WEBSITES_SEEN = set()
	LAST_TEXT_INPUT = ""
	LAST_VOICE_TEXT = ""
	LAST_PREPROCESSED_TEXT = ""
	LAST_VALIDATION_RESULT = {}
	LAST_LLM_ANSWER = ""
	LAST_SOURCE_WEBSITES = []
	try:
		reset_validation_session(clear_history=True)
	except Exception as exc:
		print(f"[cleanup] Failed to reset validation session: {exc}")
	update_live_markdown_file()
	# By default, preserve downstream files (qdrant_ready.jasonl, answer_chunk.jasonl, etc.)
	# so semantic search context is not lost on tab close/reload.
	_flush_crawler_results_file()
	cleared_all_jsonl = False
	if CLEAR_ALL_JSONL_ON_FLUSH:
		_clear_all_jsonl_files()
		cleared_all_jsonl = True
	return jsonify(
		{
			"status": "flushed",
			"cleared_results_file": CLEAR_CRAWLER_RESULTS_ON_FLUSH,
			"cleared_all_jsonl": cleared_all_jsonl,
		}
	)


if __name__ == "__main__":
	load_live_state_from_json()
	update_live_markdown_file()
	# Ensure crawler/chunker cleanup at exit
	atexit.register(_flush_crawler_results_file)
	atexit.register(stop_chunker_process)
	atexit.register(stop_second_layer_process)
	atexit.register(stop_rag_refresh_watcher)
	# Start chunker watcher so crawl results are processed automatically
	start_chunker_process()
	start_second_layer_process()
	start_rag_refresh_watcher()
	app.run(debug=True)
