import json
import os
import re
from pathlib import Path

import requests
from openai import OpenAI

from query_preprocessing import preprocess_query_text, remove_filler_text


os.environ["GROQ_API_KEY"] = ""

BASE_DIR = Path(__file__).parent
LIVE_JSON_PATH = BASE_DIR / "FRONTEND_LIVE_OUTPUT.json"
LIVE_ENDPOINT = "http://127.0.0.1:5000/live"
MODEL_NAME = "openai/gpt-oss-20b"

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

PATIENCE_MESSAGE = "keep patience. we are trying to resolve your issue"
STORAGE_MESSAGE = "All the details you provide about the problem will be saved as text for the next step."
MIN_CLARIFICATION_TURNS = int(os.getenv("MIN_CLARIFICATION_TURNS", "1"))
MAX_CLARIFICATION_TURNS = int(os.getenv("MAX_CLARIFICATION_TURNS", "2"))
LAST_VALIDATION_TEXT = ""
LAST_USER_QA_TEXT = ""
BUSY_SIGN_MESSAGE = "Please wait. We are analyzing your issue with the collected details."
DETAILED_INPUT = ""
DETAILED_INPUT_HISTORY: list[str] = []
USER_INPUT_HISTORY: list[str] = []
USER_ANSWER_HISTORY: list[str] = []
ASSISTANT_QUESTION_HISTORY: list[str] = []
USER_ANSWER_TEXT = ""
SESSION_INITIAL_QUERY = ""
SESSION_ACTIVE = False
globals()["detailed input"] = DETAILED_INPUT
globals()["user answers"] = USER_ANSWER_TEXT

_TOPIC_PATTERNS = (
    r"\baccount\b",
    r"\blogin\b",
    r"\bpassword\b",
    r"\bkeyboard\b",
    r"\bprinter\b",
    r"\bcomputer\b",
    r"\blaptop\b",
    r"\bdesktop\b",
    r"\bphone\b",
    r"\bmobile\b",
    r"\bapp\b",
    r"\bapplication\b",
    r"\bwebsite\b",
    r"\bemail\b",
    r"\bbilling\b",
    r"\bpayment\b",
    r"\border\b",
    r"\binternet\b",
    r"\bwifi\b",
    r"\bnetwork\b",
    r"\brouter\b",
    r"\bsubscription\b",
    r"\bdelivery\b",
    r"\bbsod\b",
    r"\berror\b",
)

_ACTION_PATTERNS = (
    r"\breset\b",
    r"\bchange\b",
    r"\bupdate\b",
    r"\binstall\b",
    r"\bdownload\b",
    r"\bconnect\b",
    r"\blog\s*in\b",
    r"\bsign\s*in\b",
    r"\btrack\b",
    r"\bcancel\b",
    r"\brefund\b",
    r"\breplace\b",
    r"\brestart\b",
    r"\bconfigure\b",
    r"\bset\s*up\b",
    r"\bfix\b",
    r"\bresolve\b",
)

_DETAIL_PATTERNS = (
    r"\berror\s*\d+\b",
    r"\berror\s*code\b",
    r"\bversion\b",
    r"\bwindows\b",
    r"\bmac\b",
    r"\bios\b",
    r"\bandroid\b",
    r"\bchrome\b",
    r"\bedge\b",
    r"\bfirefox\b",
    r"\bafter\b",
    r"\bsince\b",
    r"\bwhen\b",
    r"\bbecause\b",
    r"\bwhile\b",
    r"\btoday\b",
    r"\byesterday\b",
    r"\bcartridge\b",
    r"\bmodel\b",
    r"\bpassword\s+reset\b",
    r"\bscreen\b",
    r"\bblue\s+screen\b",
)


def _build_readable_terms(original_query: str) -> list[str]:
    text = (original_query or "").strip()
    if not text:
        return []

    normalized_text = re.sub(r"(?<=[A-Za-z0-9])[,/;]+(?=[A-Za-z0-9])", " ", text)
    normalized_text = re.sub(r"[\r\n]+", " ", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    terms: list[str] = []

    key_group_matches = re.findall(r"(?:\b[A-Za-z]\b[\s,.-]*){2,}[A-Za-z]\b", text)
    for key_group in key_group_matches:
        merged_key = re.sub(r"[^A-Za-z]", "", key_group).lower()
        if merged_key and merged_key not in terms:
            terms.append(merged_key)

    phrase_patterns = (
        r"\bnot working\b",
        r"\bwon't work\b",
        r"\bdoesn't work\b",
        r"\bnot responding\b",
        r"\bno response\b",
        r"\bblue screen\b",
        r"\berror code\b",
    )
    for pattern in phrase_patterns:
        for match in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
            phrase = match.group(0).lower()
            if phrase not in terms:
                terms.append(phrase)

    leftover_text = normalized_text
    for pattern in phrase_patterns:
        leftover_text = re.sub(pattern, " ", leftover_text, flags=re.IGNORECASE)
    leftover_text = re.sub(r"\b(?:is|am|are|was|were|be|been|being|the|a|an|and|or|but|with|for|to|of|on|in|at|from|by|it|this|that|nothing)\b", " ", leftover_text, flags=re.IGNORECASE)
    leftover_text = re.sub(r"\s+", " ", leftover_text).strip()

    if leftover_text:
        for chunk in re.split(r"\s+", leftover_text):
            clean_chunk = chunk.strip().lower()
            if not clean_chunk:
                continue
            if len(clean_chunk) == 1 and clean_chunk.isalpha():
                continue
            if clean_chunk not in terms:
                terms.append(clean_chunk)

    return terms


def _build_readable_keywords(original_query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9#_\-]+", original_query or "")
    keywords: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if len(lowered) == 1 and lowered.isalpha():
            continue
        if lowered not in keywords:
            keywords.append(lowered)
    return keywords


def _build_question_generation_context(original_query: str, concise_query: str, gate: dict, saved_user_data: dict) -> dict:
    return {
        "original_query": original_query or "",
        "concise_query": concise_query or "",
        "topic": (saved_user_data or {}).get("topic") or (gate or {}).get("topic") or "unknown",
        "missing_details": list((gate or {}).get("missing_details") or []),
        "readable_terms": list((saved_user_data or {}).get("readable_terms") or []),
        "token_terms": list((saved_user_data or {}).get("token_terms") or []),
        "code_terms": list((saved_user_data or {}).get("code_terms") or []),
        "gate": gate or {},
    }


def _normalize_question_text(question: str) -> str:
    value = (question or "").strip()
    if not value:
        return ""
    if not value.endswith("?"):
        value = f"{value}?"
    return value[0].upper() + value[1:]


def _parse_json_payload(text: str) -> dict:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value, flags=re.IGNORECASE)

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start_index = value.find("{")
        end_index = value.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return json.loads(value[start_index : end_index + 1])
        raise


def _generic_fallback_questions(topic: str) -> list[str]:
    topic_text = topic if topic and topic != "unknown" else "the issue"
    return [
        f"What exactly is happening with {topic_text}?",
        "Which device, app, service, or account is affected?",
        "When did the problem start, and what have you already tried?",
    ]


def _generate_top_level_questions(original_query: str, concise_query: str, gate: dict, saved_user_data: dict, previous_qa_pairs: list[dict] = None, round_number: int = 1) -> list[str]:
    context = _build_question_generation_context(original_query, concise_query, gate, saved_user_data)
    
    # Include previous Q&A pairs in context for better follow-up questions
    if previous_qa_pairs:
        context["previous_qa_pairs"] = previous_qa_pairs

    system_prompt = (
        "You are a support query clarification engine. "
        "Generate exactly 3 short, specific, non-redundant follow-up questions for the user's issue. "
        "Do not answer the issue, do not add explanations, and do not ask generic or irrelevant questions. "
        "Use the user's original query and extracted context. "
        "If prior Q/A exists, ask sharper next questions and do not repeat already answered points. "
        "Return JSON only with this schema: {\"questions\": [string, string, string]}."
    )

    user_prompt = json.dumps(context, ensure_ascii=True)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content if response.choices else ""
        payload = _parse_json_payload(content or "")
    except Exception:
        payload = None

    questions: list[str] = []
    if payload and isinstance(payload, dict):
        list_payload = payload.get("questions")
        if isinstance(list_payload, list):
            for question in list_payload:
                if not isinstance(question, str):
                    continue
                normalized_question = _normalize_question_text(question)
                if normalized_question and normalized_question not in questions:
                    questions.append(normalized_question)
        single_question = payload.get("question")
        if isinstance(single_question, str):
            normalized_question = _normalize_question_text(single_question)
            if normalized_question and normalized_question not in questions:
                questions.append(normalized_question)

    if len(questions) < 3:
        for fallback_question in _generic_fallback_questions((gate or {}).get("topic") or "unknown"):
            normalized_question = _normalize_question_text(fallback_question)
            if normalized_question and normalized_question not in questions:
                questions.append(normalized_question)
            if len(questions) >= 3:
                break

    return questions[:3]


def _generate_question_sets(
    original_query: str,
    concise_query: str,
    gate: dict,
    saved_user_data: dict,
    previous_qa_pairs: list[dict] | None = None,
) -> list[list[str]]:
    questions = _generate_top_level_questions(
        original_query,
        concise_query,
        gate,
        saved_user_data,
        previous_qa_pairs or [],
        round_number=1,
    )
    return [questions]


def _build_question_rounds_text(question_sets: list[list[str]]) -> str:
    lines: list[str] = []
    for index, question_set in enumerate(question_sets, start=1):
        lines.append(f"Round {index}:")
        for question_index, question in enumerate(question_set, start=1):
            lines.append(f"{question_index}. {question}")
        lines.append("")

    return "\n".join(lines).strip()


def _build_user_qa_text(original_query: str, question_sets: list[list[str]]) -> str:
    lines = [f"User question: {original_query or '(empty)'}", ""]
    lines.append("Assistant follow-up question sets:")
    lines.append(_build_question_rounds_text(question_sets) if question_sets else "(none)")
    return "\n".join(lines).strip()


def _build_detailed_input_text(original_query: str, concise_query: str, question_sets: list[list[str]], status_label: str) -> str:
    lines: list[str] = []
    lines.append(f"question: {original_query or '(empty)'}")
    lines.append(f"answer: {status_label}")
    lines.append(f"question: concise_query")
    lines.append(f"answer: {concise_query or '(empty)'}")

    for question_set in question_sets:
        for question in question_set:
            lines.append(f"question: {question}")
            lines.append("answer: pending")

    return "\n".join(lines).strip()


def _append_to_detailed_input_history(entry: str) -> str:
    DETAILED_INPUT_HISTORY.append(entry)
    return "\n\n---\n\n".join(DETAILED_INPUT_HISTORY)


def _append_to_user_answer_history(entry: str) -> str:
    USER_ANSWER_HISTORY.append(entry)
    return "\n\n---\n\n".join(USER_ANSWER_HISTORY)


def read_latest_query() -> str:
    try:
        response = requests.get(LIVE_ENDPOINT, timeout=3)
        if response.ok:
            payload = response.json()
            latest_from_api = (payload.get("last_text_input") or "").strip()
            if latest_from_api:
                return latest_from_api
    except Exception:
        pass

    try:
        with LIVE_JSON_PATH.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
            return (payload.get("last_text_input") or "").strip()
    except Exception:
        return ""


def _count_matches(text: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


def _extract_topic(original_query: str) -> str:
    lower_text = (original_query or "").lower()
    for pattern in _TOPIC_PATTERNS:
        match = re.search(pattern, lower_text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return "unknown"


def _evaluate_gate(original_query: str, concise_query: str) -> dict:
    query_text = (concise_query or original_query or "").strip()
    lower_text = query_text.lower()
    topic_hits = _count_matches(lower_text, _TOPIC_PATTERNS)
    action_hits = _count_matches(lower_text, _ACTION_PATTERNS)
    detail_hits = _count_matches(lower_text, _DETAIL_PATTERNS)
    word_count = len(query_text.split())

    enough_data = topic_hits > 0 and (action_hits > 0 or detail_hits > 0 or word_count >= 6)

    missing_details = []
    if topic_hits == 0:
        missing_details.append("topic")
    if action_hits == 0:
        missing_details.append("specific action")
    if detail_hits == 0:
        missing_details.append("problem details")

    if enough_data:
        reason = "The query has enough topic and issue detail to pass forward."
    else:
        reason = "The query needs more details to resolve the issue."

    return {
        "enough_data": enough_data,
        "reason": reason,
        "word_count": word_count,
        "topic_hits": topic_hits,
        "action_hits": action_hits,
        "detail_hits": detail_hits,
        "missing_details": missing_details,
        "topic": _extract_topic(original_query),
    }


def _build_top_level_questions(original_query: str, gate: dict) -> list[str]:
    topic = (gate or {}).get("topic") or "unknown"
    topic_text = topic if topic != "unknown" else "the issue"
    return [
        f"What exact error, warning, or symptom are you seeing with {topic_text}?",
        "Which device, app, account, or browser is affected, and what version are you on?",
        "When did this start, and what changed or what steps have you already tried?",
    ]


def _build_saved_user_data(original_query: str, concise_query: str, gate: dict) -> dict:
    readable_terms = _build_readable_terms(original_query)
    keywords = _build_readable_keywords(original_query)
    code_terms = re.findall(r"#[A-Za-z0-9_\-]+|\b\d{3,}\b", original_query or "")
    return {
        "input_query": original_query or "",
        "original_query": original_query or "",
        "concise_query": concise_query or "",
        "raw_terms": readable_terms,
        "token_terms": keywords,
        "readable_terms": readable_terms,
        "code_terms": code_terms,
        "topic": (gate or {}).get("topic") or "unknown",
        "missing_details": list((gate or {}).get("missing_details") or []),
        "user_answers": USER_ANSWER_TEXT,
        "answer_cycles": list(USER_ANSWER_HISTORY),
        "assistant_questions": list(ASSISTANT_QUESTION_HISTORY),
        "gate": gate or {},
    }


def _build_result(
    original_query: str,
    concise_query: str,
    gate: dict,
    collection_cycle_count: int = 0,
    llm_context_query: str = "",
    llm_context_concise_query: str = "",
) -> dict:
    global LAST_VALIDATION_TEXT
    global LAST_USER_QA_TEXT
    global DETAILED_INPUT
    global USER_ANSWER_TEXT
    context_query = (llm_context_query or original_query or "").strip()
    context_concise_query = (llm_context_concise_query or concise_query or "").strip()
    saved_user_data = _build_saved_user_data(context_query, context_concise_query, gate)
    if gate.get("enough_data"):
        LAST_USER_QA_TEXT = f"User question: {original_query or '(empty)'}\nAssistant follow-up question sets: (none)"
        current_entry = _build_detailed_input_text(original_query, concise_query, [], "complete")
        DETAILED_INPUT = _append_to_detailed_input_history(current_entry)
        globals()["detailed input"] = DETAILED_INPUT
        show_busy_sign = bool(gate.get("forced_completion"))
        LAST_VALIDATION_TEXT = (
            "Validation Summary:\n"
            f"Status: complete\n"
            f"Topic: {gate.get('topic') or 'support_request'}\n"
            f"Message: {gate.get('reason') or 'The query has enough information to pass forward.'}\n"
            f"Storage: {STORAGE_MESSAGE}"
        )
        result = {
            "input_query": original_query,
            "concise_query": concise_query,
            "status": "complete",
            "is_complete": True,
            "needs_more_info": False,
            "should_end_answer_taking": True,
            "conversation_state": "ready_for_final_answer",
            "topic": gate.get("topic") or "support_request",
            "missing_details": [],
            "follow_up_question": BUSY_SIGN_MESSAGE if show_busy_sign else "",
            "display_message": BUSY_SIGN_MESSAGE if show_busy_sign else STORAGE_MESSAGE,
            "message": gate.get("reason") or "The query has enough information to pass forward.",
            "validation_source": "gate",
            "downstream_action": "generate_final_answer",
            "gate": gate,
            "storage_message": STORAGE_MESSAGE,
            "saved_user_data": saved_user_data,
            "problem_context": saved_user_data,
            "top_level_questions": [],
            "top_level_questions_source": "gate",
            "keep_patience_message": PATIENCE_MESSAGE,
            "collection_cycle_count": collection_cycle_count,
            "show_busy_sign_in_chat": show_busy_sign,
            "busy_sign_message": BUSY_SIGN_MESSAGE if show_busy_sign else "",
            "saved_user_qa_text": LAST_USER_QA_TEXT,
            "detailed_input": DETAILED_INPUT,
            "user_answers": USER_ANSWER_TEXT,
            "saved_text": LAST_VALIDATION_TEXT,
        }
    else:
        previous_qa_pairs: list[dict[str, str]] = []
        for index, asked_question in enumerate(ASSISTANT_QUESTION_HISTORY):
            if not asked_question:
                continue
            answer_text = USER_ANSWER_HISTORY[index] if index < len(USER_ANSWER_HISTORY) else "(pending)"
            previous_qa_pairs.append(
                {
                    "question": asked_question,
                    "answer": answer_text,
                }
            )

        question_sets = _generate_question_sets(
            context_query,
            context_concise_query,
            gate,
            saved_user_data,
            previous_qa_pairs,
        )
        top_level_questions = question_sets[-1] if question_sets else []
        follow_up_question = top_level_questions[0] if top_level_questions else PATIENCE_MESSAGE
        show_busy_sign = False
        LAST_USER_QA_TEXT = _build_user_qa_text(original_query, question_sets)
        current_entry = _build_detailed_input_text(original_query, concise_query, question_sets, "received")
        DETAILED_INPUT = _append_to_detailed_input_history(current_entry)
        globals()["detailed input"] = DETAILED_INPUT
        LAST_VALIDATION_TEXT = _build_question_rounds_text(question_sets)
        result = {
            "input_query": original_query,
            "concise_query": concise_query,
            "status": "needs_more_info",
            "is_complete": False,
            "needs_more_info": True,
            "should_end_answer_taking": False,
            "conversation_state": "collecting_details",
            "topic": gate.get("topic") or "support_request",
            "missing_details": gate.get("missing_details", []),
            "follow_up_question": follow_up_question,
            "display_message": follow_up_question,
            "message": "The query needs more information before downstream processing.",
            "validation_source": "gate",
            "downstream_action": "request_more_details",
            "gate": gate,
            "storage_message": STORAGE_MESSAGE,
            "saved_user_data": saved_user_data,
            "problem_context": saved_user_data,
            "top_level_questions": top_level_questions,
            "top_level_question_sets": question_sets,
            "top_level_questions_source": "llama",
            "keep_patience_message": PATIENCE_MESSAGE,
            "collection_cycle_count": collection_cycle_count,
            "show_busy_sign_in_chat": show_busy_sign,
            "busy_sign_message": BUSY_SIGN_MESSAGE if show_busy_sign else "",
            "saved_user_qa_text": LAST_USER_QA_TEXT,
            "detailed_input": DETAILED_INPUT,
            "user_answers": USER_ANSWER_TEXT,
            "saved_text": LAST_VALIDATION_TEXT,
        }

    return result


def validate_query_text(query_text: str) -> dict:
    global USER_INPUT_HISTORY
    global USER_ANSWER_TEXT
    global SESSION_INITIAL_QUERY
    global SESSION_ACTIVE
    global ASSISTANT_QUESTION_HISTORY
    original_query = (query_text or "").strip()
    cleaned_query = remove_filler_text(original_query)
    concise_query = preprocess_query_text(cleaned_query)

    if original_query and not SESSION_ACTIVE:
        SESSION_ACTIVE = True
        SESSION_INITIAL_QUERY = original_query
        USER_INPUT_HISTORY = [original_query]
        USER_ANSWER_HISTORY.clear()
        ASSISTANT_QUESTION_HISTORY.clear()
        USER_ANSWER_TEXT = ""
        globals()["user answers"] = USER_ANSWER_TEXT
    elif original_query:
        USER_INPUT_HISTORY.append(original_query)
        USER_ANSWER_TEXT = _append_to_user_answer_history(original_query)
        globals()["user answers"] = USER_ANSWER_TEXT

    combined_query = " ".join(USER_INPUT_HISTORY).strip() or SESSION_INITIAL_QUERY or original_query
    combined_cleaned_query = remove_filler_text(combined_query)
    combined_concise_query = preprocess_query_text(combined_cleaned_query)

    gate = _evaluate_gate(combined_query, combined_concise_query)
    answer_cycle_count = len(USER_ANSWER_HISTORY)

    min_turns = max(0, MIN_CLARIFICATION_TURNS)
    max_turns = max(min_turns, MAX_CLARIFICATION_TURNS)
    if answer_cycle_count < min_turns:
        gate["enough_data"] = False
        gate["reason"] = f"Collecting details ({answer_cycle_count}/{min_turns} minimum clarification turns)."
    elif not gate.get("enough_data") and answer_cycle_count >= max_turns:
        gate["enough_data"] = True
        gate["forced_completion"] = True
        gate["reason"] = f"Collected sufficient clarification turns ({answer_cycle_count}); proceeding to final resolution."

    result = _build_result(
        original_query,
        concise_query,
        gate,
        answer_cycle_count,
        combined_query,
        combined_concise_query,
    )

    if not result.get("is_complete"):
        next_question = str(result.get("follow_up_question") or "").strip()
        if next_question and (not ASSISTANT_QUESTION_HISTORY or ASSISTANT_QUESTION_HISTORY[-1] != next_question):
            ASSISTANT_QUESTION_HISTORY.append(next_question)

    if result.get("is_complete"):
        USER_INPUT_HISTORY.clear()
        USER_ANSWER_HISTORY.clear()
        ASSISTANT_QUESTION_HISTORY.clear()
        SESSION_INITIAL_QUERY = ""
        SESSION_ACTIVE = False

    return result


def reset_validation_session(clear_history: bool = True) -> None:
    """Reset in-memory conversation/session state for a fresh user session."""
    global LAST_VALIDATION_TEXT
    global LAST_USER_QA_TEXT
    global DETAILED_INPUT
    global USER_ANSWER_TEXT
    global SESSION_INITIAL_QUERY
    global SESSION_ACTIVE

    USER_INPUT_HISTORY.clear()
    USER_ANSWER_HISTORY.clear()
    ASSISTANT_QUESTION_HISTORY.clear()

    if clear_history:
        DETAILED_INPUT_HISTORY.clear()

    LAST_VALIDATION_TEXT = ""
    LAST_USER_QA_TEXT = ""
    DETAILED_INPUT = ""
    USER_ANSWER_TEXT = ""
    SESSION_INITIAL_QUERY = ""
    SESSION_ACTIVE = False
    globals()["detailed input"] = DETAILED_INPUT
    globals()["user answers"] = USER_ANSWER_TEXT


if __name__ == "__main__":
    latest_query = read_latest_query()
    validate_query_text(latest_query)
