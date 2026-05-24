import json
import math
import os
import re
import uuid
from datetime import datetime, timezone

import requests
from google import genai
from google.genai import types

from .retrieve import retrieve

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
MODEL_URI = os.getenv("YANDEX_GPT_MODEL", "yandexgpt/latest")
REFLEXION_MEMORY_PATH = os.getenv("REFLEXION_MEMORY_PATH", "reflexion_temporal_success_memory.jsonl")
REFLEXION_MEMORY_STATS_PATH = os.getenv("REFLEXION_MEMORY_STATS_PATH", "reflexion_temporal_success_memory.stats.jsonl")
REFLEXION_BENCHMARK_NO_RETRIEVAL = os.getenv("REFLEXION_BENCHMARK_NO_RETRIEVAL", "0").strip() == "1"
REFLEXION_DISABLE_MEMORY = os.getenv("REFLEXION_DISABLE_MEMORY", "0").strip() == "1"
REFLEXION_SIMPLE_MEMORY = os.getenv("REFLEXION_SIMPLE_MEMORY", "0").strip() == "1"
REFLEXION_MEMORY_PRIOR_ALPHA = float(os.getenv("REFLEXION_MEMORY_PRIOR_ALPHA", "1.0"))
REFLEXION_MEMORY_PRIOR_BETA = float(os.getenv("REFLEXION_MEMORY_PRIOR_BETA", "1.0"))
REFLEXION_TEMPORAL_DECAY_DAYS = float(os.getenv("REFLEXION_TEMPORAL_DECAY_DAYS", "30.0"))
REFLEXION_HALLUCINATION_PENALTY = float(os.getenv("REFLEXION_HALLUCINATION_PENALTY", "0.6"))
REFLEXION_SKEPTIC_MEMORY_RISK = float(os.getenv("REFLEXION_SKEPTIC_MEMORY_RISK", "0.35"))
REFLEXION_UNSUPPORTED_CLAIM_MEMORY_RISK_STEP = float(os.getenv("REFLEXION_UNSUPPORTED_CLAIM_MEMORY_RISK_STEP", "0.05"))
REFLEXION_EXTERNAL_HALLUCINATION_MEMORY_RISK = float(os.getenv("REFLEXION_EXTERNAL_HALLUCINATION_MEMORY_RISK", "0.75"))
REFLEXION_CASE_MATCH_BONUS = float(os.getenv("REFLEXION_CASE_MATCH_BONUS", "0.15"))
REFLEXION_FILE_MATCH_BONUS = float(os.getenv("REFLEXION_FILE_MATCH_BONUS", "0.1"))
REFLEXION_MIN_MEMORY_SCORE = float(os.getenv("REFLEXION_MIN_MEMORY_SCORE", "0.05"))
REFLEXION_GROUNDED_MIN_MEMORY_SCORE = float(os.getenv("REFLEXION_GROUNDED_MIN_MEMORY_SCORE", "0.35"))
REFLEXION_GROUNDED_MAX_MEMORY_HALLUCINATION_RISK = float(os.getenv("REFLEXION_GROUNDED_MAX_MEMORY_HALLUCINATION_RISK", "0.15"))
REFLEXION_GROUNDED_TOP_N = max(1, int(os.getenv("REFLEXION_GROUNDED_TOP_N", "1")))
REFLEXION_GROUNDED_MIN_MEMORY_CONTENT_OVERLAP = max(0, int(os.getenv("REFLEXION_GROUNDED_MIN_MEMORY_CONTENT_OVERLAP", "1")))
REFLEXION_GROUNDED_MIN_EVIDENCE_OVERLAP = float(os.getenv("REFLEXION_GROUNDED_MIN_EVIDENCE_OVERLAP", "0.18"))
REFLEXION_GROUNDED_MIN_FOCUS_HITS = max(0, int(os.getenv("REFLEXION_GROUNDED_MIN_FOCUS_HITS", "2")))
REFLEXION_RAGTRUTH_ALLOW_MEMORY_SAVE = os.getenv("REFLEXION_RAGTRUTH_ALLOW_MEMORY_SAVE", "0").strip() == "1"
REFLEXION_QRECC_ALLOW_MEMORY_SAVE = os.getenv("REFLEXION_QRECC_ALLOW_MEMORY_SAVE", "0").strip() == "1"
REFLEXION_RAGTRUTH_FORCE_GUIDANCE_ONLY_MEMORY = os.getenv("REFLEXION_RAGTRUTH_FORCE_GUIDANCE_ONLY_MEMORY", "1").strip() == "1"
REFLEXION_GROUNDED_FORCE_GUIDANCE_ONLY_MEMORY = os.getenv("REFLEXION_GROUNDED_FORCE_GUIDANCE_ONLY_MEMORY", "1").strip() == "1"
REFLEXION_MEMORY_GUIDED_PROMPT_FIRST = os.getenv("REFLEXION_MEMORY_GUIDED_PROMPT_FIRST", "1").strip() == "1"
REFLEXION_MEMORY_INCLUDE_VERIFIED_SPAN_HINT = os.getenv("REFLEXION_MEMORY_INCLUDE_VERIFIED_SPAN_HINT", "1").strip() == "1"
REFLEXION_RAGTRUTH_FIXED_RETRY_TOP_K = os.getenv("REFLEXION_RAGTRUTH_FIXED_RETRY_TOP_K", "1").strip() == "1"
REFLEXION_FAST_MODE = os.getenv("REFLEXION_FAST_MODE", "0").strip() == "1"
REFLEXION_ENABLE_SELF_REFLECTION = os.getenv("REFLEXION_ENABLE_SELF_REFLECTION", "1").strip() == "1"
REFLEXION_ENABLE_MULTI_LEVEL_REFLECTION = os.getenv("REFLEXION_ENABLE_MULTI_LEVEL_REFLECTION", "1").strip() == "1"
REFLEXION_ENABLE_EPISODE_REFLECTION = os.getenv("REFLEXION_ENABLE_EPISODE_REFLECTION", "1").strip() == "1"
REFLEXION_ENABLE_SKEPTIC = os.getenv("REFLEXION_ENABLE_SKEPTIC", "1").strip() == "1"
REFLEXION_DISABLE_BENCHMARK_MEMORY = os.getenv("REFLEXION_DISABLE_BENCHMARK_MEMORY", "1").strip() == "1"
REFLEXION_SAVE_ABSTENTION_MEMORY = os.getenv("REFLEXION_SAVE_ABSTENTION_MEMORY", "0").strip() == "1"


def _safe_benchmark_name(name: str) -> str:
    value = (name or "default").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value or "default"


def _memory_paths_for_benchmark(
    benchmark_name: str,
    task_type: str = "qa_generation",
) -> tuple[str, str]:
    memory_dir = (os.getenv("REFLEXION_MEMORY_DIR", "") or "").strip() or "."
    bench = _safe_benchmark_name(benchmark_name)
    task = _safe_benchmark_name(task_type)
    memory_path = os.path.join(memory_dir, f"{bench}.{task}.memory.jsonl")
    stats_path = os.path.join(memory_dir, f"{bench}.{task}.memory.stats.jsonl")
    return memory_path, stats_path


def _benchmark_family(
    benchmark_name: str = "default",
    task_type: str = "qa_generation",
    file_name: str | None = None,
    case_no: str | None = None,
) -> str:
    joined = " ".join(
        [
            str(benchmark_name or ""),
            str(task_type or ""),
            str(file_name or ""),
            str(case_no or ""),
        ]
    ).strip().lower()
    if "ragtruth" in joined:
        return "ragtruth"
    if "qrecc" in joined:
        return "qrecc"
    if any(marker in joined for marker in ("local_legal", "legalbench", "legal_bench", "case no", ".pdf")):
        return "local_legal"
    return "generic"


def _benchmark_profile(
    benchmark_name: str = "default",
    task_type: str = "qa_generation",
    file_name: str | None = None,
    case_no: str | None = None,
) -> dict:
    family = _benchmark_family(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    profile = {
        "family": family,
        "role": "a grounded QA assistant",
        "requires_citations": False,
        "prefer_primary_evidence": False,
        "allow_multi_context": True,
        "use_memory": True,
        "save_memory": True,
        "abstention_success": REFLEXION_SAVE_ABSTENTION_MEMORY,
    }
    if family == "ragtruth":
        profile.update(
            {
                "role": "a strict evidence-grounded QA assistant",
                "requires_citations": False,
                "prefer_primary_evidence": False,
                "allow_multi_context": True,
                "use_memory": not REFLEXION_DISABLE_BENCHMARK_MEMORY,
                "save_memory": REFLEXION_RAGTRUTH_ALLOW_MEMORY_SAVE,
                "abstention_success": False,
            }
        )
    elif family == "qrecc":
        profile.update(
            {
                "role": "a conversational QA assistant",
                "requires_citations": False,
                "prefer_primary_evidence": False,
                "allow_multi_context": True,
                "use_memory": not REFLEXION_DISABLE_BENCHMARK_MEMORY,
                "save_memory": REFLEXION_QRECC_ALLOW_MEMORY_SAVE,
                "abstention_success": False,
            }
        )
    elif family == "local_legal":
        profile.update(
            {
                "role": "a legal QA assistant",
                "requires_citations": True,
                "prefer_primary_evidence": True,
                "allow_multi_context": False,
                "use_memory": True,
                "save_memory": True,
            }
        )
    if task_type == "hallucination_detection":
        profile.update(
            {
                "requires_citations": False,
                "prefer_primary_evidence": False,
                "allow_multi_context": True,
                "use_memory": False,
                "save_memory": False,
                "abstention_success": False,
            }
        )
    return profile


def _build_gemini_client() -> genai.Client:
    api_key = (GEMINI_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("Missing env var: GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def _auth_candidates():
    token = (YANDEX_IAM_TOKEN or YANDEX_API_KEY or "").strip()
    if not token:
        return []
    return [f"Bearer {token}", f"Api-Key {token}"]


def _build_citation(c: dict) -> str:
    petitioner = (c.get("pet") or "").strip()
    respondent = (c.get("res") or "").strip()
    case_no = (c.get("case_no") or "").strip()
    file_name = (c.get("file_name") or "Unknown Source").strip()

    if petitioner and respondent:
        citation = f"{petitioner} vs {respondent}"
        if case_no:
            citation += f" (Case No: {case_no})"
        return citation

    return file_name


def _call_yandex_completion(prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> str:
    if not FOLDER_ID or not _auth_candidates():
        raise RuntimeError("Missing env vars: YANDEX_FOLDER_ID and one of YANDEX_IAM_TOKEN / YANDEX_API_KEY")

    data = {
        "modelUri": f"gpt://{FOLDER_ID}/{MODEL_URI}",
        "completionOptions": {"temperature": temperature, "maxTokens": max_tokens},
        "messages": [{"role": "user", "text": prompt}],
    }
    last_error = None
    for auth in _auth_candidates():
        headers = {"Authorization": auth, "Content-Type": "application/json"}
        try:
            response = requests.post(COMPLETION_URL, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            payload = response.json()
            return payload["result"]["alternatives"][0]["message"]["text"]
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if getattr(exc.response, "status_code", None) != 401:
                raise
            continue
    raise last_error


def _call_gemini_completion(prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> str:
    client = _build_gemini_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
        config=config,
    )
    return (getattr(response, "text", "") or "").strip()


def _call_ollama_completion(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    json_mode: bool = False,
) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    if json_mode:
        payload["format"] = "json"
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", "")).strip()


def _call_completion(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    json_mode: bool = False,
) -> str:
    if LLM_PROVIDER == "ollama":
        return _call_ollama_completion(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    if LLM_PROVIDER == "gemini":
        return _call_gemini_completion(prompt, temperature=temperature, max_tokens=max_tokens)
    if LLM_PROVIDER == "yandex":
        return _call_yandex_completion(prompt, temperature=temperature, max_tokens=max_tokens)
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _active_model_name() -> str:
    if LLM_PROVIDER == "ollama":
        return OLLAMA_MODEL
    if LLM_PROVIDER == "gemini":
        return GEMINI_MODEL
    if LLM_PROVIDER == "yandex":
        return MODEL_URI
    return LLM_PROVIDER


def _build_context_text(contexts: list[dict]) -> str:
    context_text = ""
    for c in contexts:
        citation = _build_citation(c)
        chunk_id = c.get("chunk_id")
        url = (c.get("url") or "").strip()
        link_part = f", Link: {url}" if url else ""

        text = (c.get("text") or "").strip()
        if not text:
            continue

        context_text += (
            f"[Source: {citation}{link_part}, Chunk: {chunk_id}]\n"
            f"{text}\n\n"
        )
    if not context_text.strip():
        return "[No relevant context found in vector DB]\n"
    return context_text


def _build_primary_context_text(contexts: list[dict]) -> str:
    if not contexts:
        return "[No relevant primary context found]\n"

    top_ctx = contexts[0]
    citation = _build_citation(top_ctx)
    chunk_id = top_ctx.get("chunk_id")
    url = (top_ctx.get("url") or "").strip()
    link_part = f", Link: {url}" if url else ""
    text = (top_ctx.get("text") or "").strip()
    if not text:
        return "[No relevant primary context found]\n"

    return (
        f"[Primary Source: {citation}{link_part}, Chunk: {chunk_id}]\n"
        f"{text}\n"
    )


def _is_abstention_answer(answer_text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(answer_text or "").strip().lower())
    if not normalized:
        return False
    prefixes = (
        "insufficient context",
        "there is no relevant statement",
        "the provided context does not state",
        "the context does not state",
        "not stated in the provided context",
        "not stated in the context",
        "cannot be determined from the provided context",
        "cannot be determined from the context",
        "no relevant statement in the provided context",
        "no relevant statement in the context",
    )
    return normalized.startswith(prefixes)


def _should_preserve_abstention(answer_text: str, contexts: list[dict], strict_extractive: bool) -> bool:
    if not _is_abstention_answer(answer_text):
        return False
    if not contexts:
        return True
    if strict_extractive:
        return True
    return False


def _primary_evidence_support_score(question: str, primary_context_text: str) -> dict[str, float | int | bool]:
    citation, primary_body = _split_primary_context(primary_context_text)
    if not citation or not primary_body:
        return {
            "token_overlap": 0.0,
            "focus_hits": 0,
            "quoted_match": False,
            "supported": False,
        }
    question_tokens = set(_normalize_token_list(question))
    body_tokens = set(_normalize_token_list(primary_body))
    token_overlap = len(question_tokens & body_tokens) / max(len(question_tokens), 1) if question_tokens else 0.0
    focus_terms = _question_focus_terms(question)
    focus_hits = len(focus_terms & body_tokens) if focus_terms else 0
    quoted_fragments = _extract_quoted_fragments(question)
    quoted_match = any(fragment and fragment in primary_body.lower() for fragment in quoted_fragments)
    supported = bool(
        quoted_match
        or token_overlap >= REFLEXION_GROUNDED_MIN_EVIDENCE_OVERLAP
        or focus_hits >= REFLEXION_GROUNDED_MIN_FOCUS_HITS
    )
    return {
        "token_overlap": round(token_overlap, 4),
        "focus_hits": focus_hits,
        "quoted_match": quoted_match,
        "supported": supported,
    }


def _should_defer_to_safe_grounded_path(
    question: str,
    primary_context_text: str,
    memory_items: list[dict],
    task_type: str,
) -> bool:
    if task_type != "grounded_qa":
        return False
    if not memory_items:
        return False
    evidence = _primary_evidence_support_score(question, primary_context_text)
    return not bool(evidence.get("supported", False))


def _answer_prompt(
    question: str,
    context_text: str,
    memory_text: str = "",
    reflection_hint: str = "",
    primary_context_text: str = "",
    strict_extractive: bool = False,
    constrained_grounding: bool = False,
    benchmark_name: str = "default",
    task_type: str = "qa_generation",
    file_name: str | None = None,
    case_no: str | None = None,
) -> str:
    profile = _benchmark_profile(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    requires_citations = bool(profile.get("requires_citations", True))
    prefer_primary = bool(profile.get("prefer_primary_evidence", True))
    memory_section = (
        "\nRelevant Past Reflections:\n"
        "Use these only as process guidance. They are not evidence. "
        "Do not copy facts, numbers, names, spans, or conclusions from memory unless the current evidence below supports them.\n"
        "Use the guidance to choose a concise answer shape, avoid previously unsupported additions, and prefer the smallest span supported by the current evidence.\n"
        f"{memory_text}\n"
        if memory_text.strip()
        else ""
    )
    reflection_section = f"\nCurrent Iteration Guidance:\n{reflection_hint}\n" if reflection_hint.strip() else ""
    primary_section = f"\nPrimary Evidence:\n{primary_context_text}\n" if primary_context_text.strip() else ""
    citation_rules = (
        """
- Then add exactly one citation line for the primary evidence. Add a second citation only if truly necessary.
- Every claim must be supported by at least one citation.
""".strip()
        if requires_citations
        else """
- Do not add legal citation lines unless the benchmark explicitly asks for them.
- Every factual claim must be directly supported by the provided evidence.
""".strip()
    )
    evidence_priority_rules = (
        """
- Treat the first source block as the primary evidence.
- Prefer a direct quoted or near-quoted statement from the primary evidence.
- Do not combine multiple chunks unless the primary evidence is clearly insufficient.
""".strip()
        if prefer_primary
        else """
- Use all provided evidence blocks when needed.
- Do not over-prioritize the first source if another block more directly answers the question.
- Combine evidence blocks only when each factual claim is explicitly supported.
""".strip()
    )
    mode_rules = """
Rules:
{evidence_priority_rules}
- Answer as extractively as possible; do not give a broad legal summary.
- Start with exactly one short answer line that directly addresses the asked phrase.
- Prefer wording copied or closely paraphrased from the evidence.
- Do not use multiple bullets unless the question explicitly asks for multiple points.
{citation_rules}
- If the provided evidence is insufficient, explicitly say "Insufficient context."
""".format(evidence_priority_rules=evidence_priority_rules, citation_rules=citation_rules).strip()
    if strict_extractive:
        citation_line_rule = (
            "- After the answer line, add exactly one citation line for the primary evidence."
            if requires_citations
            else "- Do not add a citation line."
        )
        mode_rules = f"""
Rules:
- Use ONLY the primary evidence unless it is completely unusable.
- Return exactly one short sentence or clause copied or closely paraphrased from the primary evidence.
- Do NOT explain, summarize, infer, generalize, or list possibilities.
- Do NOT mention parties, teams, roles, implications, or examples unless those exact words are in the evidence.
- Do NOT use bullets or numbering.
- If the answer is not explicitly stated in the primary evidence, return exactly: Insufficient context.
{citation_line_rule}
""".strip()
    elif constrained_grounding:
        primary_rule = (
            "- Prefer the primary evidence and keep the answer tightly grounded to it."
            if prefer_primary
            else "- Use the strongest provided evidence block(s); do not assume the first block is best."
        )
        citation_line_rule = (
            "- End with exactly one citation line for the strongest supporting evidence."
            if requires_citations
            else "- Do not add a citation line."
        )
        mode_rules = f"""
Rules:
- Use ONLY facts explicitly stated in the provided evidence.
{primary_rule}
- Do NOT add background knowledge, likely implications, or bridging assumptions.
- If a name, number, date, quote, location, cause, or outcome is not explicit in the evidence, leave it out.
- Keep the answer concise and literal; prefer omission over speculation.
- Every factual sentence must be supported by the provided evidence.
{citation_line_rule}
- If the evidence is insufficient, explicitly say "Insufficient context."
""".strip()
    citations_format = (
        """
Citations format (exact):
[Source: PETITIONER vs RESPONDENT (Case No: ...), Link: URL, Chunk: n]
""".strip()
        if requires_citations
        else ""
    )
    return f"""
You are {profile.get("role", "a grounded QA assistant")}.

Use ONLY the context below to answer the question.
{memory_section}{reflection_section}
{mode_rules}

{citations_format}

{primary_section}
Supporting Context:
{context_text}

Question:
{question}
""".strip()


def _use_strict_extractive_mode(
    task_type: str = "qa_generation",
    file_name: str | None = None,
    case_no: str | None = None,
) -> bool:
    task_type = (task_type or "").strip().lower()
    if task_type == "extractive_qa":
        return True
    if task_type == "grounded_qa":
        return False
    file_name = (file_name or "").strip().lower()
    case_no = (case_no or "").strip().lower()
    benchmark_markers = {"privacy_qa", "maud", "cuad", "contractnli"}
    return file_name in benchmark_markers or case_no in benchmark_markers


def _use_constrained_grounding_mode(
    task_type: str = "qa_generation",
    file_name: str | None = None,
    case_no: str | None = None,
    benchmark_name: str = "default",
) -> bool:
    task_type = (task_type or "").strip().lower()
    family = _benchmark_family(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    if family == "ragtruth":
        return True
    file_name = (file_name or "").strip().lower()
    case_no = (case_no or "").strip().lower()
    if task_type in {"grounded_qa", "qa_generation"} and ("ragtruth" in file_name or "ragtruth" in case_no):
        return True
    return task_type == "grounded_qa"


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start_idx:])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}


def _compact_text(text: str, limit: int = 320) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _compact_context_for_critic(context_text: str, limit: int = 2600) -> str:
    clean = (context_text or "").strip()
    if len(clean) <= limit:
        return clean
    # Keep the front of the evidence because it preserves source markers and early rationale.
    return clean[:limit].rstrip() + "\n...[truncated for critic]..."


def _split_primary_context(primary_context_text: str) -> tuple[str, str]:
    text = (primary_context_text or "").strip()
    if not text:
        return "", ""
    lines = text.splitlines()
    if not lines:
        return "", ""
    citation = lines[0].strip()
    body = "\n".join(lines[1:]).strip()
    return citation, body


def _normalize_token_list(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _question_focus_terms(question: str) -> set[str]:
    q = (question or "").lower()
    focus = set()
    if any(phrase in q for phrase in ["who can see", "who sees", "visible to", "who can view"]):
        focus.update({"visible", "public", "users", "buyers", "sellers", "share", "shared", "disclose", "disclosed", "contact", "profile", "published", "publish", "communications"})
    if any(phrase in q for phrase in ["what information", "what data", "store about me", "collect about me"]):
        focus.update({"collect", "collected", "gather", "gathered", "record", "recorded", "information", "data", "usage", "communications", "device", "browser", "ip", "geo", "location"})
    if any(phrase in q for phrase in ["protect", "secure", "hackers", "security"]):
        focus.update({"security", "secure", "protect", "unauthorized", "access", "confidentiality", "integrity"})
    if any(phrase in q for phrase in ["sell", "third parties", "share with"]):
        focus.update({"sell", "rent", "third", "parties", "provide", "shared", "disclose"})
    if any(phrase in q for phrase in ["hire workers", "tasks i hire", "workers for"]):
        focus.update({"workers", "buyers", "sellers", "users", "publish", "published", "profile", "gig", "communications"})
    if any(phrase in q for phrase in ["type of consideration", "merger consideration", "consideration"]):
        focus.update({"consideration", "cash", "stock", "merger", "share", "shares", "common", "per share"})
    if any(phrase in q for phrase in ["definition of", "what does", "what is the definition"]):
        focus.update({"means", "definition", "defined", "shall", "mean"})
    if any(phrase in q for phrase in ["compliance with covenants", "compliance with covenant"]):
        focus.update({"covenants", "covenant", "comply", "compliance", "performed", "perform", "breach"})
    if any(phrase in q for phrase in ["fiduciary exception", "no-shop clause", "no shop"]):
        focus.update({"fiduciary", "exception", "no-shop", "no", "shop", "superior", "proposal", "board"})
    return focus


def _extract_quoted_term(question: str) -> str:
    fragments = _extract_quoted_fragments(question)
    return fragments[0].strip().lower() if fragments else ""


def _question_type(question: str) -> str:
    q = (question or "").lower()
    if "definition of" in q or " what does " in f" {q} ":
        return "definition"
    if "type of consideration" in q or "merger consideration" in q:
        return "consideration"
    if "compliance with covenants" in q:
        return "covenants"
    if "fiduciary exception" in q or "no-shop clause" in q or "no shop" in q:
        return "fiduciary_exception"
    return "generic"


def _sentence_windows(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if not sentences:
        return []
    windows = []
    for i, sent in enumerate(sentences):
        windows.append(sent)
        if i + 1 < len(sentences):
            windows.append(f"{sent} {sentences[i + 1]}")
    return windows


def _rank_extractive_candidates(question: str, body: str) -> list[tuple[float, str]]:
    query_tokens = set(_normalize_token_list(question))
    quoted_fragments = _extract_quoted_fragments(question)
    quoted_term = _extract_quoted_term(question)
    focus_terms = _question_focus_terms(question)
    qtype = _question_type(question)
    candidates = _sentence_windows(body)
    ranked: list[tuple[float, str]] = []
    for candidate in candidates:
        lowered = candidate.lower()
        candidate_tokens = set(_normalize_token_list(candidate))
        if not candidate_tokens:
            continue
        overlap = len(query_tokens & candidate_tokens)
        focus_overlap = len(focus_terms & candidate_tokens)
        score = overlap + (1.5 * focus_overlap)
        if any(fragment and fragment in lowered for fragment in quoted_fragments):
            score += 2.5
        if len(candidate.split()) <= 8:
            score -= 0.5
        if "insufficient context" in lowered:
            score -= 2.0
        if any(marker in lowered for marker in ["execution version", "exhibit", "dated as of", "google analytics cookies", "help pages", "privacy policy."]):
            score -= 2.5
        if "means" in query_tokens and " means " not in f" {lowered} ":
            score -= 0.75
        if any(phrase in lowered for phrase in ["for more information", "this can include", "references to", "dollars are to the currency"]):
            score -= 1.5
        if quoted_term and quoted_term in lowered:
            score += 3.0
        if qtype == "definition":
            if " means " in f" {lowered} " or " shall mean " in f" {lowered} ":
                score += 3.0
            else:
                score -= 1.0
        elif qtype == "consideration":
            if any(term in lowered for term in ["consideration", "cash", "per share", "shares of common stock", "merger consideration"]):
                score += 2.5
            else:
                score -= 1.0
        elif qtype == "covenants":
            if any(term in lowered for term in ["covenants", "comply", "compliance", "performed", "perform in all material respects"]):
                score += 2.0
            else:
                score -= 0.75
        elif qtype == "fiduciary_exception":
            if any(term in lowered for term in ["fiduciary", "superior proposal", "no-shop", "no shop", "board"]):
                score += 2.0
            else:
                score -= 0.75
        ranked.append((score, candidate.strip()))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _verify_best_extractive_candidate(question: str, candidates: list[str]) -> int:
    if not candidates:
        return -1
    if len(candidates) == 1:
        return 0

    rendered_candidates = "\n".join(
        f"{idx + 1}. {candidate}" for idx, candidate in enumerate(candidates)
    )
    prompt = f"""
You are selecting the single best extractive answer sentence for a grounded QA task.
Choose the candidate that most directly and completely answers the question.
Prefer a sentence that answers the question explicitly over one that is only topically related.
If none of the candidates directly answers the question, return 0.

Return ONLY JSON:
{{"best_index": 0}}

Question:
{question}

Candidates:
{rendered_candidates}
""".strip()

    raw = _call_completion(prompt, temperature=0.0, max_tokens=120, json_mode=(LLM_PROVIDER == "ollama"))
    parsed = _extract_json_object(raw)
    try:
        best_index = int(parsed.get("best_index", 0))
    except Exception:
        best_index = 0

    if 1 <= best_index <= len(candidates):
        return best_index - 1
    return -1


def _extractive_span_answer(question: str, primary_context_text: str) -> str:
    citation, body = _split_primary_context(primary_context_text)
    if not citation or not body:
        return "Insufficient context."

    ranked_candidates = _rank_extractive_candidates(question, body)
    if not ranked_candidates:
        return f"Insufficient context.\n\n{citation}"

    top_score = ranked_candidates[0][0]
    top_candidates = [candidate for _, candidate in ranked_candidates[:5]]
    verified_index = _verify_best_extractive_candidate(question, top_candidates)
    best_candidate = top_candidates[verified_index] if verified_index >= 0 else ranked_candidates[0][1]

    if not best_candidate or top_score < 1.5:
        return f"Insufficient context.\n\n{citation}"
    return f"{best_candidate}\n\n{citation}"


def _strip_citation_block(answer_text: str) -> str:
    text = str(answer_text or "").strip()
    if not text:
        return ""
    return re.sub(r"\n\s*\[Source:.*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def _extract_material_markers(text: str) -> set[str]:
    raw = str(text or "")
    markers: set[str] = set()
    patterns = [
        r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
        r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b\d[\d,]*(?:\.\d+)?\b",
        r"\bSection\s+\d+[A-Za-z()/-]*\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, raw):
            markers.add(match.group(0).strip().lower())
    return markers


def _context_body_text(context_text: str) -> str:
    lines = []
    for line in str(context_text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("[source:") or stripped.lower().startswith("[primary source:"):
            continue
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()


def _combined_evidence_body(primary_context_text: str, context_text: str = "", prefer_primary: bool = True) -> str:
    _, primary_body = _split_primary_context(primary_context_text)
    if prefer_primary:
        return primary_body
    parts = []
    if primary_body:
        parts.append(primary_body)
    context_body = _context_body_text(context_text)
    if context_body and context_body not in parts:
        parts.append(context_body)
    return "\n\n".join(parts).strip()


def _unsupported_detail_issues(
    answer_text: str,
    primary_context_text: str,
    context_text: str = "",
    prefer_primary: bool = True,
) -> list[str]:
    answer_body = _strip_citation_block(answer_text)
    evidence_body = _combined_evidence_body(
        primary_context_text,
        context_text=context_text,
        prefer_primary=prefer_primary,
    )
    if not answer_body or not evidence_body:
        return []
    answer_markers = _extract_material_markers(answer_body)
    evidence_markers = _extract_material_markers(evidence_body)
    unsupported = sorted(marker for marker in answer_markers if marker not in evidence_markers)
    issues = []
    if unsupported:
        preview = ", ".join(unsupported[:3])
        scope = "primary evidence" if prefer_primary else "provided evidence"
        issues.append(f"unsupported material details not found in {scope}: {preview}")
    return issues


def _sentence_like_parts(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if not clean:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", clean) if part.strip()]
    return parts or [clean]


def _sentence_supported_by_evidence(sentence: str, evidence_body: str) -> bool:
    sent = str(sentence or "").strip()
    body = str(evidence_body or "")
    if not sent or not body:
        return False
    sent_norm = " ".join(sent.split()).lower()
    body_norm = " ".join(body.split()).lower()
    if sent_norm in body_norm:
        return True
    sent_tokens = set(_normalize_token_list(sent_norm))
    body_tokens = set(_normalize_token_list(body_norm))
    if not sent_tokens:
        return False
    overlap = len(sent_tokens & body_tokens) / max(len(sent_tokens), 1)
    return overlap >= 0.85 and not any(
        marker for marker in _extract_material_markers(sent_norm)
        if marker not in _extract_material_markers(body_norm)
    )


def _sentence_supported_by_primary(sentence: str, primary_body: str) -> bool:
    return _sentence_supported_by_evidence(sentence, primary_body)


def _trim_answer_to_supported_core(
    answer_text: str,
    primary_context_text: str,
    context_text: str = "",
    prefer_primary: bool = True,
    requires_citations: bool = True,
) -> str:
    text = str(answer_text or "").strip()
    if not text:
        return text
    citation, primary_body = _split_primary_context(primary_context_text)
    evidence_body = _combined_evidence_body(
        primary_context_text,
        context_text=context_text,
        prefer_primary=prefer_primary,
    )
    if not evidence_body:
        return text

    answer_body = _strip_citation_block(text)
    if not answer_body:
        return text

    supported_parts = [
        part for part in _sentence_like_parts(answer_body)
        if _sentence_supported_by_evidence(part, evidence_body)
    ]
    if supported_parts:
        trimmed = " ".join(supported_parts).strip()
        if trimmed:
            if requires_citations and citation:
                return f"{trimmed}\n\n{citation}"
            return trimmed

    span_fallback = _extractive_span_answer("", primary_context_text).strip() if prefer_primary else ""
    if span_fallback and not _is_abstention_answer(span_fallback):
        return span_fallback
    return text


def _should_prefer_span_first(question: str, task_type: str) -> bool:
    task_type = (task_type or "").strip().lower()
    if task_type != "grounded_qa":
        return False
    q = (question or "").lower()
    return bool(_extract_quoted_fragments(question)) or "what does the judgment say about" in q


def _candidate_span_answer(question: str, primary_context_text: str, task_type: str) -> str:
    if not _should_prefer_span_first(question, task_type):
        return ""
    candidate = _extractive_span_answer(question, primary_context_text).strip()
    if not candidate or _is_abstention_answer(candidate):
        return ""
    return candidate


def _should_force_span_grounded_answer(
    question: str,
    task_type: str,
    file_name: str | None = None,
    case_no: str | None = None,
) -> bool:
    task_type = (task_type or "").strip().lower()
    if task_type != "grounded_qa":
        return False
    q = (question or "").lower()
    file_name = (file_name or "").strip().lower()
    case_no = (case_no or "").strip().lower()
    legal_markers = (
        "judgment",
        "judgement",
        ".pdf",
        "case no",
        "supreme court",
        "high court",
    )
    if any(marker in q for marker in ("what does the judgment say", "what does the judgement say")):
        return True
    if _extract_quoted_fragments(question):
        return True
    return any(marker in file_name or marker in case_no for marker in legal_markers)


def _critic_prompt(question: str, answer_text: str, context_text: str) -> str:
    compact_context = _compact_context_for_critic(context_text)
    return f"""
You are the final RAG answer critic.
The adversarial skeptic has already tried to break the answer.
Your job is to decide whether the current answer is acceptable to return to the user.
Return VALID JSON ONLY. No prose. No markdown. No code fences.

Return ONLY JSON with keys:
- is_sufficient: boolean
- issues: array of short strings
- guidance: short actionable instruction for next attempt

Evaluation rules:
- Judge only from the provided context and the current answer.
- Prefer a short grounded answer over a broad or polished one.
- If the answer should be trimmed further, mark is_sufficient=false and say what to remove.
- If the answer is an appropriate abstention because evidence is insufficient, mark is_sufficient=true.

JSON example:
{{"is_sufficient": false, "issues": ["missing citations", "unsupported claim"], "guidance": "Quote the exact holding and cite the source."}}

Question:
{question}

Answer:
{answer_text}

Context:
{compact_context}
""".strip()


def _skeptic_prompt(
    question: str,
    answer_text: str,
    primary_context_text: str,
    context_text: str,
) -> str:
    compact_context = _compact_context_for_critic(context_text)
    return f"""
You are an adversarial skeptic for a RAG system.
Assume the answer may be wrong.
Try to falsify it using ONLY the provided evidence.
Do not rewrite the answer. Diagnose it.
Return VALID JSON ONLY. No prose. No markdown. No code fences.

Return ONLY JSON with keys:
- supported_claims: array of short strings
- unsupported_claims: array of short strings
- false_premise_risk: boolean
- counter_evidence: array of short strings
- verdict: one of "accept", "revise", "reject"
- guidance: short actionable instruction

Rules:
- Break the answer into atomic claims mentally and test each one against the evidence.
- If a detail is implied but not explicit, mark it unsupported.
- If the question appears to contain a false or loaded premise not supported by evidence, set false_premise_risk=true.
- Use "accept" only when the answer is tightly supported.
- Use "revise" when part of the answer is supported but extra claims should be removed.
- Use "reject" when the core answer is unsupported or misleading.

Question:
{question}

Answer:
{answer_text}

Primary Evidence:
{primary_context_text}

Supporting Context:
{compact_context}
""".strip()


def _reflect_prompt(question: str, answer_text: str, critique: dict) -> str:
    return f"""
You are writing a short self-reflection for a RAG agent.
Given a failed or weak attempt, summarize what went wrong and what to change next.
Return 2-4 concise bullet points.

Question:
{question}

Previous Answer:
{answer_text}

Critique:
{json.dumps(critique, ensure_ascii=True)}
""".strip()


def _retrieval_reflection_prompt(question: str, contexts: list[dict], critique: dict) -> str:
    context_preview = []
    for idx, ctx in enumerate(contexts[:3], 1):
        context_preview.append(
            {
                "rank": idx,
                "file_name": ctx.get("file_name", ""),
                "chunk_id": ctx.get("chunk_id", None),
                "score": ctx.get("score", None),
                "text": _compact_text(ctx.get("text", ""), limit=280),
            }
        )
    return f"""
You are writing a retrieval-level reflection for a RAG agent.
Focus only on whether retrieval quality helped or hurt the answer.
Return ONLY JSON with keys:
- summary: short string
- next_action: short string

Question:
{question}

Retrieved Context Summary:
{json.dumps(context_preview, ensure_ascii=True)}

Critique:
{json.dumps(critique, ensure_ascii=True)}
""".strip()


def _answer_reflection_prompt(question: str, answer_text: str, critique: dict) -> str:
    return f"""
You are writing an answer-level reflection for a RAG agent.
Focus only on grounding, completeness, citation use, and overclaiming.
Return ONLY JSON with keys:
- summary: short string
- next_action: short string

Question:
{question}

Answer:
{answer_text}

Critique:
{json.dumps(critique, ensure_ascii=True)}
""".strip()


def _episode_reflection_prompt(question: str, attempts: list[dict]) -> str:
    compact_attempts = []
    for item in attempts:
        compact_attempts.append(
            {
                "iter": item.get("iter"),
                "top_k_used": item.get("top_k_used"),
                "context_count": item.get("context_count"),
                "is_sufficient": bool(item.get("critique", {}).get("is_sufficient", False)),
                "issues": item.get("critique", {}).get("issues", []),
                "guidance": item.get("critique", {}).get("guidance", ""),
                "retrieval_reflection": item.get("multi_level_reflection", {}).get("retrieval_level", ""),
                "answer_reflection": item.get("multi_level_reflection", {}).get("answer_level", ""),
            }
        )
    return f"""
You are writing an episode-level reflection for a RAG agent.
Summarize the main recurring failure pattern across attempts and the best next strategy.
Return ONLY JSON with keys:
- summary: short string
- next_action: short string

Question:
{question}

Attempt Trace:
{json.dumps(compact_attempts, ensure_ascii=True)}
""".strip()


def _tokenize_for_overlap(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


_MEMORY_TOPIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "best", "by",
    "can", "could", "do", "does", "done", "for", "from", "give", "had", "has",
    "have", "how", "i", "in", "into", "is", "it", "its", "me", "more", "most",
    "of", "on", "or", "some", "such", "tell", "than", "that", "the", "their",
    "there", "these", "they", "this", "those", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "about",
    "describe", "explain", "information", "level", "levels", "type", "types",
}


def _memory_topic_token(token: str) -> str:
    token = (token or "").strip().lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _memory_content_tokens(text: str) -> set[str]:
    tokens = set()
    for raw in re.findall(r"[a-z0-9]+", (text or "").lower()):
        token = _memory_topic_token(raw)
        if len(token) < 4 or token in _MEMORY_TOPIC_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _memory_content_overlap(a_tokens: set[str], b_tokens: set[str]) -> int:
    if not a_tokens or not b_tokens:
        return 0
    overlap = len(a_tokens & b_tokens)
    related = 0
    for a_token in a_tokens - b_tokens:
        for b_token in b_tokens - a_tokens:
            if len(a_token) >= 5 and len(b_token) >= 5 and (
                a_token in b_token or b_token in a_token
            ):
                related += 1
                break
    return overlap + related


def _extract_quoted_fragments(text: str) -> list[str]:
    return [frag.strip().lower() for frag in re.findall(r'"([^"]+)"', text or "") if frag.strip()]


def _parse_yes_no_label(text: str) -> str:
    match = re.search(r"\b(yes|no)\b", (text or "").strip(), flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).capitalize()


def _hallucination_detection_prompt(
    question: str,
    candidate_answer: str,
    evidence: str,
    benchmark_family: str = "",
) -> str:
    family_note = f"\nBenchmark family: {benchmark_family}\n" if benchmark_family else ""
    return f"""
You are a strict hallucination detector for retrieval-augmented generation.
Use ONLY the provided evidence. Do not use outside knowledge.
Decide whether the candidate answer contains any unsupported, contradictory, or over-specific claim.
Do not punish harmless wording changes, compression, or omissions.
Mark hallucination only for claims present in the candidate answer.
Return VALID JSON ONLY. No prose. No markdown. No code fences.

Return exactly these keys:
- has_hallucination: boolean
- unsupported_spans: array of exact short spans copied from the candidate answer
- supported_spans: array of short spans or claims supported by the evidence
- reason: short string
- confidence: number from 0.0 to 1.0

Rules:
- If the candidate answer is fully supported by the evidence, has_hallucination=false and unsupported_spans=[].
- If a named entity, date, number, causal claim, quote, location, or outcome is not explicit in the evidence, mark that span unsupported.
- If the answer contradicts the evidence, mark the contradictory span unsupported.
- For summaries and QA, judge factual support, not whether the wording exactly matches a reference answer.
- For abstentions like "Insufficient context", mark hallucination=false unless the answer also adds unsupported facts.
{family_note}
Question:
{question}

Candidate Answer:
{candidate_answer}

Evidence:
{evidence}
""".strip()


def detect_hallucination(
    question: str,
    candidate_answer: str,
    evidence: str,
    max_attempts: int = 2,
    benchmark_family: str = "",
) -> dict:
    prompt = _hallucination_detection_prompt(
        question=question,
        candidate_answer=candidate_answer,
        evidence=evidence,
        benchmark_family=benchmark_family,
    )
    raw = ""
    parsed: dict = {}
    for attempt in range(max(1, int(max_attempts))):
        suffix = "" if attempt == 0 else "\n\nReturn valid JSON only, with no prose or markdown."
        raw = _call_completion(
            prompt + suffix,
            temperature=0.0,
            max_tokens=360,
            json_mode=True,
        )
        parsed = _extract_json_object(raw)
        if parsed:
            break

    if not parsed:
        yes_no_prompt = (
            "Use only the evidence. Does the candidate answer contain unsupported or contradictory information?\n"
            "Return exactly one token: Yes or No.\n\n"
            f"Question: {question}\n"
            f"Candidate Answer: {candidate_answer}\n"
            f"Evidence: {evidence}\n"
            "Output:"
        )
        raw = _call_completion(yes_no_prompt, temperature=0.0, max_tokens=5)
        label = _parse_yes_no_label(raw)
        return {
            "has_hallucination": label == "Yes",
            "unsupported_spans": [],
            "supported_spans": [],
            "reason": "Fallback Yes/No detector used because JSON detector did not parse.",
            "confidence": 0.5 if label else 0.0,
            "raw_output": raw,
            "detector_provider": LLM_PROVIDER,
            "detector_model": _active_model_name(),
            "benchmark_family": benchmark_family,
        }

    unsupported = parsed.get("unsupported_spans", [])
    supported = parsed.get("supported_spans", [])
    if not isinstance(unsupported, list):
        unsupported = [str(unsupported)]
    if not isinstance(supported, list):
        supported = [str(supported)]
    has_hallucination = parsed.get("has_hallucination")
    if not isinstance(has_hallucination, bool):
        has_hallucination = bool([span for span in unsupported if str(span).strip()])
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "has_hallucination": has_hallucination,
        "unsupported_spans": [str(span).strip() for span in unsupported if str(span).strip()],
        "supported_spans": [str(span).strip() for span in supported if str(span).strip()],
        "reason": str(parsed.get("reason", "")).strip(),
        "confidence": max(0.0, min(confidence, 1.0)),
        "raw_output": raw,
        "detector_provider": LLM_PROVIDER,
        "detector_model": _active_model_name(),
        "benchmark_family": benchmark_family,
    }


def _parse_detection_request(prompt: str) -> dict:
    raw = (prompt or "").strip()
    if not raw:
        return {}
    parsed = _extract_json_object(raw)
    if parsed:
        question = str(parsed.get("question", "") or parsed.get("query", "")).strip()
        candidate = str(
            parsed.get("candidate_answer", "")
            or parsed.get("candidate_response", "")
            or parsed.get("answer", "")
            or parsed.get("response", "")
        ).strip()
        evidence = str(
            parsed.get("evidence", "")
            or parsed.get("context", "")
            or parsed.get("fixed_context", "")
            or parsed.get("supporting_evidence", "")
        ).strip()
        if question and candidate and evidence:
            return {
                "question": question,
                "candidate_answer": candidate,
                "evidence": evidence,
                "benchmark_family": str(parsed.get("benchmark_family", "")).strip(),
            }

    patterns = {
        "question": r"Question:\s*(.*?)(?:\n\s*Candidate (?:Answer|Response|Summary):|\n\s*Answer:|\n\s*Evidence:|\Z)",
        "candidate_answer": r"Candidate (?:Answer|Response|Summary):\s*(.*?)(?:\n\s*Evidence:|\Z)",
        "evidence": r"Evidence:\s*(.*?)(?:\n\s*Output:|\Z)",
    }
    values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
        values[key] = match.group(1).strip() if match else ""
    if values["question"] and values["candidate_answer"] and values["evidence"]:
        return values
    return {}


def _detect_hallucination_label(prompt: str, max_attempts: int = 1) -> str:
    request = _parse_detection_request(prompt)
    if request:
        result = detect_hallucination(
            question=request["question"],
            candidate_answer=request["candidate_answer"],
            evidence=request["evidence"],
            max_attempts=max_attempts,
            benchmark_family=request.get("benchmark_family", ""),
        )
        return "Yes" if result.get("has_hallucination", False) else "No"

    attempts = max(1, int(max_attempts))
    current_prompt = (
        (prompt or "").strip()
        + "\n\nReturn exactly one token: Yes or No. Do not add explanation or citations."
    ).strip()
    last_output = ""
    for _ in range(attempts):
        last_output = _call_completion(current_prompt, temperature=0.0, max_tokens=5)
        if _parse_yes_no_label(last_output):
            return last_output
        current_prompt = (
            current_prompt
            + "\nYour previous output was invalid. Reply with only one token: Yes or No."
        )
    return last_output


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_datetime(raw: str) -> datetime | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_memory_defaults(item: dict) -> dict:
    row = dict(item or {})
    created_at = (
        row.get("created_at")
        or row.get("timestamp")
        or _utc_now().isoformat()
    )
    timestamp = row.get("timestamp") or created_at
    success_flag = bool(row.get("is_sufficient", False))
    success_count = int(row.get("success_count", 1 if success_flag else 0) or 0)
    failure_count = int(row.get("failure_count", 0 if success_flag else 1) or 0)
    alpha = _safe_float(row.get("alpha"), REFLEXION_MEMORY_PRIOR_ALPHA + success_count)
    beta = _safe_float(row.get("beta"), REFLEXION_MEMORY_PRIOR_BETA + failure_count)
    replay_count = int(row.get("replay_count", row.get("use_count", 0)) or 0)
    total_outcomes = max(success_count + failure_count, 1)
    row.setdefault("memory_id", str(uuid.uuid4()))
    row.setdefault("created_at", created_at)
    row.setdefault("timestamp", timestamp)
    row.setdefault("last_used_at", row.get("last_used_at", ""))
    row.setdefault("replay_count", replay_count)
    row.setdefault("use_count", replay_count)
    row.setdefault("success_count", success_count)
    row.setdefault("failure_count", failure_count)
    row.setdefault("alpha", alpha)
    row.setdefault("beta", beta)
    row.setdefault("origin_success", success_flag)
    row.setdefault("hallucination_risk", failure_count / total_outcomes)
    row.setdefault("task_type", "qa_generation")
    return row


def _load_memory_stats(path: str = REFLEXION_MEMORY_STATS_PATH) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    stats_by_id: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            memory_id = str(event.get("memory_id", "")).strip()
            if not memory_id:
                continue
            current = stats_by_id.setdefault(
                memory_id,
                {
                    "replay_count_delta": 0,
                    "success_count_delta": 0,
                    "failure_count_delta": 0,
                    "alpha_delta": 0.0,
                    "beta_delta": 0.0,
                    "last_used_at": "",
                    "last_outcome": "",
                },
            )
            current["replay_count_delta"] += int(event.get("replay_count_delta", 0) or 0)
            current["success_count_delta"] += int(event.get("success_count_delta", 0) or 0)
            current["failure_count_delta"] += int(event.get("failure_count_delta", 0) or 0)
            current["alpha_delta"] += _safe_float(event.get("alpha_delta"), 0.0)
            current["beta_delta"] += _safe_float(event.get("beta_delta"), 0.0)
            if str(event.get("last_used_at", "")).strip():
                current["last_used_at"] = str(event.get("last_used_at", "")).strip()
            if str(event.get("last_outcome", "")).strip():
                current["last_outcome"] = str(event.get("last_outcome", "")).strip()
    return stats_by_id


def _apply_memory_stats(row: dict, stats: dict | None) -> dict:
    current = _ensure_memory_defaults(row)
    if not stats:
        return current
    current["replay_count"] = int(current.get("replay_count", 0) or 0) + int(stats.get("replay_count_delta", 0) or 0)
    current["use_count"] = current["replay_count"]
    current["success_count"] = int(current.get("success_count", 0) or 0) + int(stats.get("success_count_delta", 0) or 0)
    current["failure_count"] = int(current.get("failure_count", 0) or 0) + int(stats.get("failure_count_delta", 0) or 0)
    current["alpha"] = _safe_float(current.get("alpha"), REFLEXION_MEMORY_PRIOR_ALPHA) + _safe_float(stats.get("alpha_delta"), 0.0)
    current["beta"] = _safe_float(current.get("beta"), REFLEXION_MEMORY_PRIOR_BETA) + _safe_float(stats.get("beta_delta"), 0.0)
    if str(stats.get("last_used_at", "")).strip():
        current["last_used_at"] = str(stats.get("last_used_at", "")).strip()
    if str(stats.get("last_outcome", "")).strip():
        current["last_outcome"] = str(stats.get("last_outcome", "")).strip()
    total = int(current.get("success_count", 0) or 0) + int(current.get("failure_count", 0) or 0)
    outcome_risk = int(current.get("failure_count", 0) or 0) / total if total > 0 else 0.0
    current["hallucination_risk"] = max(outcome_risk, _verification_memory_risk(current))
    return current


def _read_memory(path: str = REFLEXION_MEMORY_PATH, stats_path: str = REFLEXION_MEMORY_STATS_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    if REFLEXION_SIMPLE_MEMORY:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_ensure_memory_defaults(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return rows
    stats_by_id = _load_memory_stats(stats_path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                base_row = _ensure_memory_defaults(json.loads(line))
            except json.JSONDecodeError:
                continue
            memory_id = str(base_row.get("memory_id", "")).strip()
            rows.append(_apply_memory_stats(base_row, stats_by_id.get(memory_id)))
    return rows


def _append_memory(entry: dict, path: str = REFLEXION_MEMORY_PATH) -> None:
    if REFLEXION_DISABLE_MEMORY:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_ensure_memory_defaults(entry), ensure_ascii=True) + "\n")


def _append_memory_stat(entry: dict, path: str = REFLEXION_MEMORY_STATS_PATH) -> None:
    if REFLEXION_DISABLE_MEMORY or REFLEXION_SIMPLE_MEMORY:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _jaccard_similarity(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


def _memory_reliability(item: dict) -> float:
    alpha = max(_safe_float(item.get("alpha"), REFLEXION_MEMORY_PRIOR_ALPHA), 0.001)
    beta = max(_safe_float(item.get("beta"), REFLEXION_MEMORY_PRIOR_BETA), 0.001)
    return alpha / (alpha + beta)


def _verification_memory_risk(item: dict) -> float:
    risk = _safe_float(item.get("hallucination_risk"), 0.0)
    skeptic = item.get("skeptic", {}) or {}
    unsupported_count = int(skeptic.get("unsupported_claim_count", 0) or 0)
    if unsupported_count <= 0:
        unsupported_count = len(skeptic.get("unsupported_claims", []) or [])

    if bool(skeptic.get("has_hallucination", False)) or unsupported_count > 0:
        risk = max(
            risk,
            REFLEXION_SKEPTIC_MEMORY_RISK
            + REFLEXION_UNSUPPORTED_CLAIM_MEMORY_RISK_STEP * min(unsupported_count, 3),
        )

    skeptic_verdict = str(skeptic.get("verdict", "") or "").strip().lower()
    if skeptic_verdict in {"reject", "abstain"}:
        risk = max(risk, 0.8)
    elif skeptic_verdict == "revise":
        risk = max(risk, REFLEXION_SKEPTIC_MEMORY_RISK)

    issue_text = " ".join(str(issue) for issue in item.get("issues", []) or []).lower()
    if "unsupported" in issue_text or "skeptic" in issue_text:
        risk = max(risk, REFLEXION_SKEPTIC_MEMORY_RISK)

    external_eval = item.get("external_eval", {}) or {}
    if (
        bool(external_eval.get("generated_has_hallucination", False))
        or bool(external_eval.get("predicted_has_hallucination", False))
    ):
        risk = max(risk, REFLEXION_EXTERNAL_HALLUCINATION_MEMORY_RISK)

    return min(max(risk, 0.0), 1.0)


def _memory_hallucination_risk(item: dict) -> float:
    failure_count = int(item.get("failure_count", 0) or 0)
    success_count = int(item.get("success_count", 0) or 0)
    total = success_count + failure_count
    metadata_risk = _verification_memory_risk(item)
    if total <= 0:
        return metadata_risk
    return max(failure_count / total, metadata_risk)


def _memory_temporal_weight(item: dict, now: datetime) -> float:
    base_time = _parse_iso_datetime(item.get("last_used_at") or item.get("created_at") or item.get("timestamp"))
    if base_time is None:
        return 1.0
    age_days = max((now - base_time).total_seconds(), 0.0) / 86400.0
    decay_days = max(REFLEXION_TEMPORAL_DECAY_DAYS, 1.0)
    return math.exp(-age_days / decay_days)


def _score_memory(
    question: str,
    item: dict,
    case_no: str | None = None,
    file_name: str | None = None,
    now: datetime | None = None,
) -> tuple[float, dict]:
    now = now or _utc_now()
    q_tokens = _tokenize_for_overlap(question)
    m_tokens = _tokenize_for_overlap(item.get("question", ""))
    q_content_tokens = _memory_content_tokens(question)
    m_content_tokens = _memory_content_tokens(item.get("question", ""))
    content_overlap = _memory_content_overlap(q_content_tokens, m_content_tokens)
    overlap = len(q_tokens & m_tokens)
    similarity = _jaccard_similarity(q_tokens, m_tokens)
    if overlap <= 0 or similarity <= 0.0:
        return 0.0, {
            "similarity": similarity,
            "overlap": overlap,
            "content_overlap": content_overlap,
            "temporal_weight": 0.0,
            "reliability": 0.0,
            "risk_penalty": 0.0,
            "metadata_bonus": 0.0,
        }

    temporal_weight = _memory_temporal_weight(item, now)
    reliability = _memory_reliability(item)
    hallucination_risk = _memory_hallucination_risk(item)
    risk_penalty = max(0.05, 1.0 - (REFLEXION_HALLUCINATION_PENALTY * hallucination_risk))

    metadata_bonus = 0.0
    if case_no and case_no == (item.get("case_no") or "").strip():
        metadata_bonus += REFLEXION_CASE_MATCH_BONUS
    if file_name and file_name == (item.get("file_name") or "").strip():
        metadata_bonus += REFLEXION_FILE_MATCH_BONUS

    if REFLEXION_SIMPLE_MEMORY:
        score = similarity * (1.0 + metadata_bonus)
        return score, {
            "similarity": round(similarity, 4),
            "overlap": overlap,
            "content_overlap": content_overlap,
            "temporal_weight": 1.0,
            "reliability": 1.0,
            "risk_penalty": 1.0,
            "metadata_bonus": round(metadata_bonus, 4),
        }

    score = similarity * temporal_weight * reliability * risk_penalty * (1.0 + metadata_bonus)
    return score, {
        "similarity": round(similarity, 4),
        "overlap": overlap,
        "content_overlap": content_overlap,
        "temporal_weight": round(temporal_weight, 4),
        "reliability": round(reliability, 4),
        "risk_penalty": round(risk_penalty, 4),
        "metadata_bonus": round(metadata_bonus, 4),
    }


def _memory_for_query(
    question: str,
    top_n: int = 3,
    case_no: str | None = None,
    file_name: str | None = None,
    task_type: str = "qa_generation",
    benchmark_name: str = "default",
    memory_path: str | None = None,
    stats_path: str | None = None,
) -> list[dict]:
    if REFLEXION_DISABLE_MEMORY:
        return []
    now = _utc_now()
    grounded_mode = task_type == "grounded_qa"
    family = _benchmark_family(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    effective_top_n = REFLEXION_GROUNDED_TOP_N if grounded_mode else top_n
    candidates = []
    for item in _read_memory(
        path=memory_path or REFLEXION_MEMORY_PATH,
        stats_path=stats_path or REFLEXION_MEMORY_STATS_PATH,
    ):
        item_task_type = str(item.get("task_type", "qa_generation")).strip() or "qa_generation"
        if item_task_type != task_type:
            continue
        if grounded_mode:
            if not bool(item.get("origin_success", False)):
                continue
            if int(item.get("success_count", 0) or 0) <= 0:
                continue
            if _memory_hallucination_risk(item) > REFLEXION_GROUNDED_MAX_MEMORY_HALLUCINATION_RISK:
                continue
            item_case = str(item.get("case_no", "") or "").strip()
            item_file = str(item.get("file_name", "") or "").strip()
            same_case = bool(case_no and item_case and item_case == case_no)
            same_file = bool(file_name and item_file and item_file == file_name)
            allow_family_level_match = family == "ragtruth"
            if not (same_case or same_file or allow_family_level_match):
                continue
        reflection = (
            (item.get("multi_level_reflection", {}) or {}).get("episode_level")
            or item.get("final_reflection")
            or ""
        ).strip()
        if not reflection:
            continue
        score, score_details = _score_memory(question, item, case_no=case_no, file_name=file_name, now=now)
        min_score = REFLEXION_GROUNDED_MIN_MEMORY_SCORE if grounded_mode else REFLEXION_MIN_MEMORY_SCORE
        if grounded_mode and family == "ragtruth":
            min_score = min(min_score, 0.18)
        if (
            grounded_mode
            and REFLEXION_GROUNDED_MIN_MEMORY_CONTENT_OVERLAP > 0
            and int(score_details.get("content_overlap", 0) or 0) < REFLEXION_GROUNDED_MIN_MEMORY_CONTENT_OVERLAP
        ):
            continue
        if score >= min_score:
            enriched = dict(item)
            enriched["memory_score"] = round(score, 6)
            enriched["memory_score_details"] = score_details
            candidates.append((score, enriched))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in candidates[:effective_top_n]]


def _format_memory_snippets(memory_items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(memory_items, 1):
        reflection_text = (
            (item.get("multi_level_reflection", {}) or {}).get("episode_level")
            or item.get("final_reflection", "")
        )
        lines.append(f"{i}. Question: {item.get('question', '')}")
        if REFLEXION_SIMPLE_MEMORY:
            lines.append(f"   MemoryScore: {item.get('memory_score', 0.0):.4f}")
        else:
            details = item.get("memory_score_details", {}) or {}
            reliability = details.get("reliability", round(_memory_reliability(item), 4))
            temporal_weight = details.get("temporal_weight", round(_memory_temporal_weight(item, _utc_now()), 4))
            lines.append(
                "   MemoryScore: "
                f"{item.get('memory_score', 0.0):.4f} "
                f"(reliability={reliability}, temporal={temporal_weight}, "
                f"successes={item.get('success_count', 0)}, failures={item.get('failure_count', 0)})"
            )
        if str(item.get("exact_answer_span", "")).strip():
            lines.append(f"   ProvenSpan: {item.get('exact_answer_span', '')}")
        lines.append(f"   Reflection: {reflection_text}")
    return "\n".join(lines)


def _format_memory_guidance_snippets(memory_items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(memory_items, 1):
        details = item.get("memory_score_details", {}) or {}
        reliability = details.get("reliability", round(_memory_reliability(item), 4))
        temporal_weight = details.get("temporal_weight", round(_memory_temporal_weight(item, _utc_now()), 4))
        guidance_parts = [
            str(item.get("last_guidance", "") or "").strip(),
            str(((item.get("multi_level_reflection", {}) or {}).get("episode_level", "")) or "").strip(),
            str(item.get("final_reflection", "") or "").strip(),
        ]
        guidance = " ".join(part for part in guidance_parts if part).strip()
        if not guidance:
            guidance = "Use current evidence only; do not reuse prior answer facts unless they are directly supported here."
        guidance = re.sub(r"\s+", " ", guidance)
        previous_question = str(item.get("question", "") or "").strip()
        if previous_question:
            lines.append(f"{i}. Related previous question: {previous_question[:180]}")
        else:
            lines.append(f"{i}. Related previous memory")
        lines.append(f"   Guidance: {guidance[:260]}")
        proven_span = re.sub(r"\s+", " ", str(item.get("exact_answer_span", "") or "").strip())
        if REFLEXION_MEMORY_INCLUDE_VERIFIED_SPAN_HINT and proven_span:
            lines.append(
                "   Prior successful answer pattern, not evidence: "
                f"{proven_span[:220]}"
            )
        lines.append(
            "   MemoryScore: "
            f"{item.get('memory_score', 0.0):.4f} "
            f"(reliability={reliability}, temporal={temporal_weight})"
        )
    return "\n".join(lines)


def _should_use_guidance_only_memory(
    question: str,
    task_type: str,
    benchmark_name: str = "default",
    file_name: str | None = None,
    case_no: str | None = None,
) -> bool:
    family = _benchmark_family(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    if family == "ragtruth" and REFLEXION_RAGTRUTH_FORCE_GUIDANCE_ONLY_MEMORY:
        return True
    if task_type == "grounded_qa" and REFLEXION_GROUNDED_FORCE_GUIDANCE_ONLY_MEMORY:
        return True
    return _should_force_span_grounded_answer(
        question,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )


def _should_use_memory_guided_prompt_first(
    memory_items: list[dict],
    task_type: str,
    benchmark_name: str = "default",
    file_name: str | None = None,
    case_no: str | None = None,
    strict_extractive: bool = False,
) -> bool:
    if not REFLEXION_MEMORY_GUIDED_PROMPT_FIRST:
        return False
    if strict_extractive or not memory_items:
        return False
    if task_type != "grounded_qa":
        return False
    family = _benchmark_family(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    return family in {"qrecc", "ragtruth"}


def _retry_top_k(
    base_top_k: int,
    iter_index: int,
    benchmark_name: str = "default",
    task_type: str = "qa_generation",
    file_name: str | None = None,
    case_no: str | None = None,
) -> int:
    family = _benchmark_family(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    if family == "ragtruth" and REFLEXION_RAGTRUTH_FIXED_RETRY_TOP_K:
        return base_top_k
    return base_top_k + max(0, iter_index - 1)


def _update_memory_outcomes(
    memory_items: list[dict],
    is_success: bool,
    path: str = REFLEXION_MEMORY_STATS_PATH,
) -> None:
    if REFLEXION_DISABLE_MEMORY or REFLEXION_SIMPLE_MEMORY or not memory_items:
        return
    now_text = _utc_now().isoformat()
    for item in memory_items:
        memory_id = str(item.get("memory_id", "")).strip()
        if not memory_id:
            continue
        stat_event = {
            "timestamp": now_text,
            "memory_id": memory_id,
            "replay_count_delta": 1,
            "success_count_delta": 1 if is_success else 0,
            "failure_count_delta": 0 if is_success else 1,
            "alpha_delta": 1.0 if is_success else 0.0,
            "beta_delta": 0.0 if is_success else 1.0,
            "last_used_at": now_text,
            "last_outcome": "success" if is_success else "failure",
        }
        _append_memory_stat(stat_event, path=path)


def _quick_answer_checks(
    answer_text: str,
    strict_extractive: bool = False,
    task_type: str = "qa_generation",
    benchmark_name: str = "default",
    file_name: str | None = None,
    case_no: str | None = None,
) -> dict:
    profile = _benchmark_profile(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    requires_citations = bool(profile.get("requires_citations", True))
    has_source_marker = "[source:" in answer_text.lower()
    has_content = bool(str(answer_text or "").strip())
    if requires_citations:
        extractive_pass = has_source_marker and not _is_abstention_answer(answer_text)
        grounded_pass = has_source_marker and has_content
    else:
        extractive_pass = has_content and not _is_abstention_answer(answer_text)
        grounded_pass = has_content
    return {
        "has_source_marker": has_source_marker,
        "requires_citations": requires_citations,
        "has_content": has_content,
        "passes": grounded_pass if task_type == "grounded_qa" else (
            extractive_pass if strict_extractive else grounded_pass
        ),
    }


def _run_skeptic(
    question: str,
    answer_text: str,
    context_text: str,
    primary_context_text: str = "",
) -> dict:
    prompt = _skeptic_prompt(question, answer_text, primary_context_text, context_text)
    raw = _call_completion(prompt, temperature=0.0, max_tokens=500, json_mode=True)
    skeptic = _extract_json_object(raw)
    if not skeptic:
        retry_prompt = prompt + "\n\nReturn JSON only. No prose, no markdown, no code fences."
        raw = _call_completion(retry_prompt, temperature=0.0, max_tokens=260, json_mode=True)
        skeptic = _extract_json_object(raw)
    if not skeptic:
        skeptic = {
            "supported_claims": [],
            "unsupported_claims": ["skeptic parser fallback: unable to verify answer support"],
            "false_premise_risk": False,
            "counter_evidence": [],
            "verdict": "revise",
            "guidance": "Skeptic verification failed; trim to the supported core before accepting the answer.",
        }

    supported_claims = skeptic.get("supported_claims", [])
    unsupported_claims = skeptic.get("unsupported_claims", [])
    counter_evidence = skeptic.get("counter_evidence", [])
    if not isinstance(supported_claims, list):
        supported_claims = [str(supported_claims)]
    if not isinstance(unsupported_claims, list):
        unsupported_claims = [str(unsupported_claims)]
    if not isinstance(counter_evidence, list):
        counter_evidence = [str(counter_evidence)]

    verdict = str(skeptic.get("verdict", "accept")).strip().lower()
    if verdict not in {"accept", "revise", "reject"}:
        verdict = "accept"

    guidance = str(skeptic.get("guidance", "")).strip()
    if not guidance:
        if verdict == "reject":
            guidance = "Reject the answer and respond with an evidence-grounded abstention."
        elif verdict == "revise":
            guidance = "Remove unsupported claims and keep only the supported core."
        else:
            guidance = "Keep the answer tightly grounded to the primary evidence."

    normalized_supported = [str(item).strip() for item in supported_claims if str(item).strip()]
    normalized_unsupported = [str(item).strip() for item in unsupported_claims if str(item).strip()]
    normalized_counter = [str(item).strip() for item in counter_evidence if str(item).strip()]
    false_premise_risk = bool(skeptic.get("false_premise_risk", False))
    has_hallucination = bool(
        normalized_unsupported
        or normalized_counter
        or false_premise_risk
        or verdict in {"revise", "reject"}
    )

    return {
        "supported_claims": normalized_supported,
        "unsupported_claims": normalized_unsupported,
        "unsupported_claim_count": len(normalized_unsupported),
        "false_premise_risk": false_premise_risk,
        "counter_evidence": normalized_counter,
        "verdict": verdict,
        "guidance": guidance,
        "has_hallucination": has_hallucination,
        "raw_answer_before_guardrail": answer_text,
    }


def _abstention_with_primary_citation(primary_context_text: str, requires_citations: bool = True) -> str:
    citation, _ = _split_primary_context(primary_context_text)
    if not requires_citations:
        return "Insufficient context."
    if citation and "no relevant primary context found" in citation.lower():
        return "Insufficient context."
    if citation:
        return f"Insufficient context.\n\n{citation}"
    return "Insufficient context."


def _apply_skeptic_guardrail(
    question: str,
    answer_text: str,
    context_text: str,
    primary_context_text: str,
    benchmark_name: str = "default",
    task_type: str = "qa_generation",
    file_name: str | None = None,
    case_no: str | None = None,
) -> tuple[str, dict]:
    if not REFLEXION_ENABLE_SKEPTIC:
        return answer_text, {
            "verdict": "disabled",
            "supported_claims": [],
            "unsupported_claims": [],
            "unsupported_claim_count": 0,
            "false_premise_risk": False,
            "counter_evidence": [],
            "guidance": "",
            "has_hallucination": False,
            "guardrail_action": "disabled",
            "raw_answer_before_guardrail": answer_text,
            "final_answer_after_guardrail": answer_text,
        }

    profile = _benchmark_profile(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    prefer_primary = bool(profile.get("prefer_primary_evidence", True))
    requires_citations = bool(profile.get("requires_citations", True))
    skeptic = _run_skeptic(
        question,
        answer_text,
        context_text,
        primary_context_text=primary_context_text,
    )
    skeptic_verdict = skeptic.get("verdict", "accept")
    revised_answer = answer_text
    guardrail_action = "accept"
    if skeptic_verdict == "reject":
        revised_answer = _abstention_with_primary_citation(
            primary_context_text,
            requires_citations=requires_citations,
        )
        skeptic["verdict"] = "abstain"
        guardrail_action = "abstain"
    elif skeptic_verdict == "revise" or skeptic.get("unsupported_claims"):
        trimmed_answer = _trim_answer_to_supported_core(
            answer_text,
            primary_context_text,
            context_text=context_text,
            prefer_primary=prefer_primary,
            requires_citations=requires_citations,
        )
        if trimmed_answer.strip():
            revised_answer = trimmed_answer
        guardrail_action = "trim"
    skeptic["guardrail_action"] = guardrail_action
    skeptic["final_answer_after_guardrail"] = revised_answer
    return revised_answer, skeptic


def _is_benign_critic_issue(issue_text: str) -> bool:
    text = str(issue_text or "").strip().lower()
    if not text:
        return True
    benign_patterns = (
        "word count",
        "too short",
        "too long",
        "brevity",
        "concise",
        "style",
        "formatting",
    )
    return any(pattern in text for pattern in benign_patterns)


def _has_severe_grounding_issue(issues: list[str]) -> bool:
    severe_patterns = (
        "unsupported",
        "not supported",
        "missing citation",
        "missing citations",
        "uncited",
        "without citation",
        "not in context",
        "outside the context",
        "outside context",
        "contradict",
        "baseless",
        "hallucin",
        "made up",
        "invent",
        "speculat",
        "overclaim",
    )
    for issue in issues:
        text = str(issue or "").strip().lower()
        if not text or _is_benign_critic_issue(text):
            continue
        if any(pattern in text for pattern in severe_patterns):
            return True
    return False


def _is_grounding_sufficient(
    answer_text: str,
    critique: dict,
    strict_extractive: bool = False,
    task_type: str = "qa_generation",
    benchmark_name: str = "default",
    file_name: str | None = None,
    case_no: str | None = None,
) -> bool:
    if not critique.get("is_sufficient", False):
        return False
    profile = _benchmark_profile(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    if _is_abstention_answer(answer_text):
        return bool(profile.get("abstention_success", False))
    quick = _quick_answer_checks(
        answer_text,
        strict_extractive=strict_extractive,
        task_type=task_type,
        benchmark_name=benchmark_name,
        file_name=file_name,
        case_no=case_no,
    )
    if quick.get("requires_citations", True) and not quick.get("has_source_marker", False):
        return False
    if strict_extractive and not quick.get("passes", False):
        return False
    if _has_severe_grounding_issue(critique.get("issues", [])):
        return False
    if task_type == "grounded_qa" and not quick.get("passes", False):
        return False
    return True


def _is_qrecc_memory_success(
    answer_text: str,
    critique: dict,
    skeptic: dict | None = None,
    task_type: str = "qa_generation",
    benchmark_name: str = "default",
    file_name: str | None = None,
    case_no: str | None = None,
) -> bool:
    family = _benchmark_family(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    if family != "qrecc" or task_type != "grounded_qa":
        return False
    if _is_abstention_answer(answer_text):
        return False

    quick = _quick_answer_checks(
        answer_text,
        strict_extractive=False,
        task_type=task_type,
        benchmark_name=benchmark_name,
        file_name=file_name,
        case_no=case_no,
    )
    if not quick.get("passes", False):
        return False

    skeptic = skeptic or {}
    skeptic_verdict = str(skeptic.get("verdict", "")).strip().lower()
    if skeptic_verdict in {"reject", "abstain"}:
        return False

    filtered_issues = [
        str(issue or "").strip()
        for issue in (critique.get("issues", []) or [])
        if str(issue or "").strip().lower() != "skeptic found unsupported answer fragments"
    ]
    if _has_severe_grounding_issue(filtered_issues):
        return False
    return True


def _run_critic(
    question: str,
    answer_text: str,
    context_text: str,
    primary_context_text: str = "",
    strict_extractive: bool = False,
    task_type: str = "qa_generation",
    skeptic: dict | None = None,
    benchmark_name: str = "default",
    file_name: str | None = None,
    case_no: str | None = None,
) -> dict:
    profile = _benchmark_profile(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    prefer_primary = bool(profile.get("prefer_primary_evidence", True))
    quick = _quick_answer_checks(
        answer_text,
        strict_extractive=strict_extractive,
        task_type=task_type,
        benchmark_name=benchmark_name,
        file_name=file_name,
        case_no=case_no,
    )
    prompt = _critic_prompt(question, answer_text, context_text)
    raw = _call_completion(prompt, temperature=0.0, max_tokens=400, json_mode=True)
    critique = _extract_json_object(raw)
    if not critique:
        retry_prompt = prompt + "\n\nReturn JSON only. No prose, no markdown, no code fences."
        raw = _call_completion(retry_prompt, temperature=0.0, max_tokens=220, json_mode=True)
        critique = _extract_json_object(raw)
    if not critique and raw.strip():
        repair_prompt = (
            "Convert the following model output into VALID JSON ONLY with keys "
            "`is_sufficient`, `issues`, and `guidance`. "
            "If information is missing, infer conservatively.\n\n"
            f"Model output:\n{raw}"
        )
        repaired = _call_completion(repair_prompt, temperature=0.0, max_tokens=180, json_mode=True)
        critique = _extract_json_object(repaired)
    if not critique:
        critique = {
            "is_sufficient": quick["passes"],
            "issues": ["Critic parser fallback: invalid JSON response from judge model."],
            "guidance": "Strengthen citation-grounded claims and keep strict format.",
        }
    if "is_sufficient" not in critique:
        critique["is_sufficient"] = quick["passes"]
    if not isinstance(critique.get("issues"), list):
        critique["issues"] = [str(critique.get("issues", "Unknown issue"))]
    if "guidance" not in critique or not str(critique.get("guidance", "")).strip():
        critique["guidance"] = "Tighten grounded claims and include proper citations."
    critique["issues"] = [
        str(issue) for issue in critique.get("issues", [])
        if str(issue or "").strip()
    ]
    extra_issues = _unsupported_detail_issues(
        answer_text,
        primary_context_text,
        context_text=context_text,
        prefer_primary=prefer_primary,
    )
    if extra_issues:
        critique["issues"].extend(extra_issues)
        critique["is_sufficient"] = False
        critique["guidance"] = "Remove unsupported dates, numbers, and extra details; keep only the exact supported span."
    skeptic = skeptic or {}
    skeptic_verdict = str(skeptic.get("verdict", "")).strip().lower()
    skeptic_unsupported = skeptic.get("unsupported_claims", []) or []
    if skeptic_verdict == "abstain" and _is_abstention_answer(answer_text):
        critique["is_sufficient"] = bool(profile.get("abstention_success", False))
        if critique["is_sufficient"]:
            critique["issues"] = [
                issue for issue in critique["issues"]
                if "unsupported" not in issue.lower()
                and "insufficiently grounded" not in issue.lower()
            ]
            critique["guidance"] = "Abstention is appropriate because the available evidence does not support a grounded answer."
        else:
            critique["issues"].append("abstention is not counted as benchmark success")
            critique["guidance"] = "Avoid over-abstention unless the benchmark item is explicitly unanswerable."
    elif skeptic_verdict == "reject":
        critique["is_sufficient"] = False
        critique["issues"].append("skeptic rejected the core answer as unsupported")
        critique["guidance"] = str(skeptic.get("guidance", critique.get("guidance", ""))).strip()
    elif skeptic_verdict == "revise" and skeptic_unsupported:
        critique["is_sufficient"] = False
        critique["issues"].append("skeptic found unsupported answer fragments")
        critique["guidance"] = str(skeptic.get("guidance", critique.get("guidance", ""))).strip()
    if task_type == "grounded_qa":
        critique["issues"] = [
            issue for issue in critique["issues"]
            if "analysis" not in issue.lower()
            and "summary" not in issue.lower()
            and "explanation" not in issue.lower()
        ]
        if critique.get("is_sufficient", False) and not _is_grounding_sufficient(
            answer_text,
            critique,
            strict_extractive=strict_extractive,
            task_type=task_type,
            benchmark_name=benchmark_name,
            file_name=file_name,
            case_no=case_no,
        ):
            critique["is_sufficient"] = False
            if not _has_severe_grounding_issue(critique["issues"]):
                critique["issues"].append("answer may contain unsupported or insufficiently cited claims")
            critique["guidance"] = "Remove unsupported details and keep only claims explicitly backed by the evidence."
        elif quick["passes"] and not critique["issues"]:
            critique["guidance"] = "Grounded concise answer is acceptable for this benchmark."
    elif critique.get("is_sufficient", False) and not _is_grounding_sufficient(
        answer_text,
        critique,
        strict_extractive=strict_extractive,
        task_type=task_type,
        benchmark_name=benchmark_name,
        file_name=file_name,
        case_no=case_no,
    ):
        critique["is_sufficient"] = False
        if not _has_severe_grounding_issue(critique["issues"]):
            critique["issues"].append("answer may contain unsupported or insufficiently grounded claims")
        critique["guidance"] = "Revise the answer to include only explicit evidence-backed claims with citations."
    return critique


def _parse_reflection_json(raw: str, fallback_summary: str, fallback_action: str) -> dict:
    parsed = _extract_json_object(raw)
    summary = str(parsed.get("summary", "")).strip() if parsed else ""
    next_action = str(parsed.get("next_action", "")).strip() if parsed else ""
    if not summary:
        summary = fallback_summary
    if not next_action:
        next_action = fallback_action
    return {
        "summary": summary,
        "next_action": next_action,
    }


def _build_multi_level_reflection(question: str, answer_text: str, contexts: list[dict], critique: dict) -> dict:
    if critique.get("is_sufficient", False):
        return {
            "retrieval_level": "",
            "answer_level": "",
            "combined_guidance": critique.get("guidance", "").strip(),
        }

    retrieval_fallback_summary = (
        "No useful legal evidence was retrieved."
        if not contexts
        else f"Retrieved {len(contexts)} chunks, but the evidence set did not adequately support the answer."
    )
    retrieval_fallback_action = (
        "Increase retrieval breadth or use narrower metadata filters before answering."
        if not contexts
        else "Use more relevant chunks and avoid relying on weak or noisy evidence."
    )
    answer_fallback_summary = "The answer is not fully grounded in the retrieved legal context."
    answer_fallback_action = critique.get("guidance", "Tighten grounded claims and include proper citations.")

    retrieval_raw = _call_completion(
        _retrieval_reflection_prompt(question, contexts, critique),
        temperature=0.0,
        max_tokens=180,
        json_mode=True,
    )
    answer_raw = _call_completion(
        _answer_reflection_prompt(question, answer_text, critique),
        temperature=0.0,
        max_tokens=180,
        json_mode=True,
    )

    retrieval_level = _parse_reflection_json(
        retrieval_raw,
        fallback_summary=retrieval_fallback_summary,
        fallback_action=retrieval_fallback_action,
    )
    answer_level = _parse_reflection_json(
        answer_raw,
        fallback_summary=answer_fallback_summary,
        fallback_action=answer_fallback_action,
    )

    combined_guidance = " | ".join(
        part
        for part in [
            critique.get("guidance", "").strip(),
            retrieval_level.get("next_action", "").strip(),
            answer_level.get("next_action", "").strip(),
        ]
        if part
    )
    return {
        "retrieval_level": retrieval_level["summary"],
        "answer_level": answer_level["summary"],
        "combined_guidance": combined_guidance.strip(),
    }


def _build_episode_reflection(question: str, trace_attempts: list[dict]) -> dict:
    if not trace_attempts:
        return {"summary": "", "next_action": ""}
    if REFLEXION_FAST_MODE or not REFLEXION_ENABLE_EPISODE_REFLECTION:
        fallback_action = trace_attempts[-1].get("critique", {}).get(
            "guidance",
            "Retrieve stronger evidence and keep claims tightly grounded.",
        )
        return {
            "summary": "Episode reflection skipped in fast mode.",
            "next_action": fallback_action,
        }
    raw = _call_completion(
        _episode_reflection_prompt(question, trace_attempts),
        temperature=0.0,
        max_tokens=220,
        json_mode=True,
    )
    fallback_summary = "Earlier attempts exposed recurring grounding or retrieval weaknesses."
    fallback_action = trace_attempts[-1].get("critique", {}).get(
        "guidance",
        "Retrieve stronger evidence and keep claims tightly grounded.",
    )
    return _parse_reflection_json(raw, fallback_summary=fallback_summary, fallback_action=fallback_action)


def generate_answer(
    question: str,
    top_k: int = 5,
    case_no: str | None = None,
    file_name: str | None = None,
    source_file_path: str | None = None,
    task_type: str = "qa_generation",
    benchmark_name: str = "default",
):
    """
    Returns: (answer_text, contexts)
    Optional filters restrict retrieval to a case/file.
    """
    if task_type == "hallucination_detection":
        return _detect_hallucination_label(question, max_attempts=1), []

    profile = _benchmark_profile(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    prefer_primary = bool(profile.get("prefer_primary_evidence", True))
    requires_citations = bool(profile.get("requires_citations", True))
    contexts = [] if REFLEXION_BENCHMARK_NO_RETRIEVAL else retrieve(
        question,
        top_k=top_k,
        case_no=case_no,
        file_name=file_name,
        source_file_path=source_file_path,
    )
    memory_path, stats_path = _memory_paths_for_benchmark(
        benchmark_name=benchmark_name,
        task_type=task_type,
    )
    episodic = (
        _memory_for_query(
            question,
            top_n=3,
            case_no=case_no,
            file_name=file_name,
            task_type=task_type,
            benchmark_name=benchmark_name,
            memory_path=memory_path,
            stats_path=stats_path,
        )
        if profile.get("use_memory", True)
        else []
    )
    context_text = _build_context_text(contexts)
    primary_context_text = _build_primary_context_text(contexts)
    strict_extractive = _use_strict_extractive_mode(task_type=task_type, file_name=file_name, case_no=case_no)
    constrained_grounding = _use_constrained_grounding_mode(
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
        benchmark_name=benchmark_name,
    )
    if strict_extractive:
        answer_text = _extractive_span_answer(question, primary_context_text)
    else:
        span_first = _candidate_span_answer(question, primary_context_text, task_type=task_type)
        force_span_grounded = _should_force_span_grounded_answer(
            question,
            task_type=task_type,
            file_name=file_name,
            case_no=case_no,
        )
        if span_first:
            answer_text = span_first
        elif _should_defer_to_safe_grounded_path(question, primary_context_text, episodic, task_type):
            answer_text = _extractive_span_answer(question, primary_context_text)
        elif force_span_grounded:
            answer_text = _extractive_span_answer(question, primary_context_text)
        else:
            prompt = _answer_prompt(
                question,
                context_text,
                primary_context_text=primary_context_text,
                strict_extractive=False,
                constrained_grounding=constrained_grounding,
                benchmark_name=benchmark_name,
                task_type=task_type,
                file_name=file_name,
                case_no=case_no,
            )
            answer_text = _call_completion(
                prompt,
                temperature=0.0 if constrained_grounding else 0.2,
                max_tokens=1000,
            )
        if task_type == "grounded_qa":
            answer_text = _trim_answer_to_supported_core(
                answer_text,
                primary_context_text,
                context_text=context_text,
                prefer_primary=prefer_primary,
                requires_citations=requires_citations,
            )
    answer_text, _ = _apply_skeptic_guardrail(
        question,
        answer_text,
        context_text,
        primary_context_text,
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    return answer_text, contexts


def generate_answer_with_reflexion(
    question: str,
    top_k: int = 5,
    case_no: str | None = None,
    file_name: str | None = None,
    source_file_path: str | None = None,
    max_iters: int = 3,
    task_type: str = "qa_generation",
    benchmark_name: str = "default",
):
    """
    Reflexion loop:
    attempt -> critique -> multi-level reflection -> retry with guidance -> save episodic memory.
    Returns: (answer_text, contexts, trace)
    """
    if max_iters < 1:
        max_iters = 1

    if task_type == "hallucination_detection":
        answer_text = _detect_hallucination_label(question, max_attempts=max_iters)
        trace = {
            "mode": "hallucination_detection",
            "attempts": [{"iter": i, "answer": answer_text} for i in range(1, max_iters + 1)],
            "memory_hits": 0,
        }
        return answer_text, [], trace

    profile = _benchmark_profile(
        benchmark_name=benchmark_name,
        task_type=task_type,
        file_name=file_name,
        case_no=case_no,
    )
    prefer_primary = bool(profile.get("prefer_primary_evidence", True))
    requires_citations = bool(profile.get("requires_citations", True))
    memory_path, stats_path = _memory_paths_for_benchmark(
        benchmark_name=benchmark_name,
        task_type=task_type,
    )
    episodic = (
        _memory_for_query(
            question,
            top_n=3,
            case_no=case_no,
            file_name=file_name,
            task_type=task_type,
            benchmark_name=benchmark_name,
            memory_path=memory_path,
            stats_path=stats_path,
        )
        if profile.get("use_memory", True)
        else []
    )
    use_guidance_only_memory = _should_use_guidance_only_memory(
        question,
        task_type=task_type,
        benchmark_name=benchmark_name,
        file_name=file_name,
        case_no=case_no,
    )
    memory_text = (
        _format_memory_guidance_snippets(episodic)
        if use_guidance_only_memory
        else _format_memory_snippets(episodic)
    )

    trace_attempts = []
    final_answer = ""
    final_contexts = []
    final_reflection = ""
    reflection_hint = ""

    for i in range(1, max_iters + 1):
        current_top_k = _retry_top_k(
            top_k,
            i,
            benchmark_name=benchmark_name,
            task_type=task_type,
            file_name=file_name,
            case_no=case_no,
        )
        if REFLEXION_BENCHMARK_NO_RETRIEVAL:
            contexts = []
        else:
            contexts = retrieve(
                question,
                top_k=current_top_k,
                case_no=case_no,
                file_name=file_name,
                source_file_path=source_file_path,
            )
        context_text = _build_context_text(contexts)
        primary_context_text = _build_primary_context_text(contexts)
        strict_extractive = _use_strict_extractive_mode(task_type=task_type, file_name=file_name, case_no=case_no)
        constrained_grounding = _use_constrained_grounding_mode(
            task_type=task_type,
            file_name=file_name,
            case_no=case_no,
            benchmark_name=benchmark_name,
        )
        memory_guided_prompt_first = _should_use_memory_guided_prompt_first(
            episodic,
            task_type=task_type,
            benchmark_name=benchmark_name,
            file_name=file_name,
            case_no=case_no,
            strict_extractive=strict_extractive,
        )
        if strict_extractive:
            answer_text = _extractive_span_answer(question, primary_context_text)
        else:
            span_first = _candidate_span_answer(question, primary_context_text, task_type=task_type)
            force_span_grounded = _should_force_span_grounded_answer(
                question,
                task_type=task_type,
                file_name=file_name,
                case_no=case_no,
            )
            if memory_guided_prompt_first:
                prompt = _answer_prompt(
                    question,
                    context_text,
                    memory_text=memory_text,
                    reflection_hint=reflection_hint,
                    primary_context_text=primary_context_text,
                    strict_extractive=False,
                    constrained_grounding=constrained_grounding,
                    benchmark_name=benchmark_name,
                    task_type=task_type,
                    file_name=file_name,
                    case_no=case_no,
                )
                answer_text = _call_completion(
                    prompt,
                    temperature=0.0 if constrained_grounding else 0.2,
                    max_tokens=1000,
                )
            elif span_first:
                answer_text = span_first
            elif _should_defer_to_safe_grounded_path(question, primary_context_text, episodic, task_type):
                answer_text = _extractive_span_answer(question, primary_context_text)
            elif force_span_grounded:
                answer_text = _extractive_span_answer(question, primary_context_text)
            elif constrained_grounding and i > 1:
                answer_text = _extractive_span_answer(question, primary_context_text)
            else:
                prompt = _answer_prompt(
                    question,
                    context_text,
                    memory_text=memory_text,
                    reflection_hint=reflection_hint,
                    primary_context_text=primary_context_text,
                    strict_extractive=False,
                    constrained_grounding=constrained_grounding,
                    benchmark_name=benchmark_name,
                    task_type=task_type,
                    file_name=file_name,
                    case_no=case_no,
                )
                answer_text = _call_completion(
                    prompt,
                    temperature=0.0 if constrained_grounding else 0.2,
                    max_tokens=1000,
                )
            if task_type == "grounded_qa":
                answer_text = _trim_answer_to_supported_core(
                    answer_text,
                    primary_context_text,
                    context_text=context_text,
                    prefer_primary=prefer_primary,
                    requires_citations=requires_citations,
                )
        answer_text, skeptic = _apply_skeptic_guardrail(
            question,
            answer_text,
            context_text,
            primary_context_text,
            benchmark_name=benchmark_name,
            task_type=task_type,
            file_name=file_name,
            case_no=case_no,
        )
        critique = _run_critic(
            question,
            answer_text,
            context_text,
            primary_context_text=primary_context_text,
            strict_extractive=strict_extractive,
            task_type=task_type,
            skeptic=skeptic,
            benchmark_name=benchmark_name,
            file_name=file_name,
            case_no=case_no,
        )
        preserve_abstention = _should_preserve_abstention(answer_text, contexts, strict_extractive)
        if preserve_abstention:
            critique["is_sufficient"] = bool(profile.get("abstention_success", False))
            if critique["is_sufficient"]:
                critique["issues"] = []
                critique["guidance"] = "Preserve abstention because the retrieved evidence is insufficient for a grounded answer."
            else:
                critique["issues"] = ["abstention is not counted as benchmark success"]
                critique["guidance"] = "Avoid over-abstention unless the benchmark item is explicitly unanswerable."

        reflection = ""
        multi_level_reflection = {
            "retrieval_level": "",
            "answer_level": "",
            "combined_guidance": critique.get("guidance", "").strip(),
        }
        if not critique.get("is_sufficient", False):
            if not REFLEXION_FAST_MODE and REFLEXION_ENABLE_SELF_REFLECTION:
                reflection = _call_completion(
                    _reflect_prompt(question, answer_text, critique),
                    temperature=0.1,
                    max_tokens=220,
                )
            if not REFLEXION_FAST_MODE and REFLEXION_ENABLE_MULTI_LEVEL_REFLECTION:
                multi_level_reflection = _build_multi_level_reflection(question, answer_text, contexts, critique)
            reflection_hint = "\n".join(
                part
                for part in [
                    critique.get("guidance", "").strip(),
                    skeptic.get("guidance", "").strip(),
                    multi_level_reflection.get("retrieval_level", "").strip(),
                    multi_level_reflection.get("answer_level", "").strip(),
                    reflection.strip(),
                ]
                if part
            ).strip()

        trace_attempts.append(
            {
                "iter": i,
                "answer": answer_text,
                "skeptic": skeptic,
                "critique": critique,
                "reflection": reflection,
                "multi_level_reflection": multi_level_reflection,
                "top_k_used": current_top_k,
                "context_count": len(contexts),
                "memory_guided_prompt_first": bool(memory_guided_prompt_first),
            }
        )

        final_answer = answer_text
        final_contexts = contexts

        if critique.get("is_sufficient", False):
            break

    episode_reflection = _build_episode_reflection(question, trace_attempts)
    if episode_reflection.get("summary", "").strip():
        final_reflection = episode_reflection["summary"].strip()

    final_success = False
    if trace_attempts:
        final_success = _is_grounding_sufficient(
            final_answer,
            trace_attempts[-1]["critique"],
            strict_extractive=strict_extractive,
            task_type=task_type,
            benchmark_name=benchmark_name,
            file_name=file_name,
            case_no=case_no,
        )
        if not final_success:
            final_success = _is_qrecc_memory_success(
                final_answer,
                trace_attempts[-1]["critique"],
                skeptic=trace_attempts[-1].get("skeptic", {}),
                task_type=task_type,
                benchmark_name=benchmark_name,
                file_name=file_name,
                case_no=case_no,
            )
    latest_skeptic = trace_attempts[-1].get("skeptic", {}) if trace_attempts else {}
    latest_issues = trace_attempts[-1]["critique"].get("issues", []) if trace_attempts else []
    base_memory_risk = 0.0 if final_success else 1.0
    verification_memory_risk = _verification_memory_risk(
        {
            "hallucination_risk": base_memory_risk,
            "skeptic": latest_skeptic,
            "issues": latest_issues,
        }
    )

    memory_entry = {
        "memory_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "benchmark_name": benchmark_name,
        "task_type": task_type,
        "case_no": case_no or "",
        "file_name": file_name or "",
        "benchmark_profile": profile,
        "abstained": _is_abstention_answer(final_answer),
        "memory_saved": bool(profile.get("save_memory", True)),
        "replay_count": 0,
        "use_count": 0,
        "attempt_count": len(trace_attempts),
        "is_sufficient": final_success,
        "origin_success": final_success,
        "success_count": 1 if final_success else 0,
        "failure_count": 0 if final_success else 1,
        "alpha": REFLEXION_MEMORY_PRIOR_ALPHA + (
            1 if final_success else 0
        ),
        "beta": REFLEXION_MEMORY_PRIOR_BETA + (
            0 if final_success else 1
        ),
        "hallucination_risk": verification_memory_risk,
        "exact_answer_span": _strip_citation_block(final_answer),
        "source_chunk_id": final_contexts[0].get("chunk_id") if final_contexts else None,
        "final_reflection": final_reflection,
        "last_guidance": episode_reflection.get("next_action", "") or (
            trace_attempts[-1]["critique"].get("guidance", "") if trace_attempts else ""
        ),
        "issues": latest_issues,
        "skeptic": {
            "verdict": latest_skeptic.get("verdict", ""),
            "unsupported_claims": latest_skeptic.get("unsupported_claims", []),
            "unsupported_claim_count": latest_skeptic.get("unsupported_claim_count", 0),
            "false_premise_risk": latest_skeptic.get("false_premise_risk", False),
            "counter_evidence": latest_skeptic.get("counter_evidence", []),
            "guidance": latest_skeptic.get("guidance", ""),
            "has_hallucination": latest_skeptic.get("has_hallucination", False),
            "guardrail_action": latest_skeptic.get("guardrail_action", ""),
            "raw_answer_before_guardrail": latest_skeptic.get("raw_answer_before_guardrail", ""),
            "final_answer_after_guardrail": latest_skeptic.get("final_answer_after_guardrail", ""),
        },
        "multi_level_reflection": {
            "retrieval_level": trace_attempts[-1].get("multi_level_reflection", {}).get("retrieval_level", "") if trace_attempts else "",
            "answer_level": trace_attempts[-1].get("multi_level_reflection", {}).get("answer_level", "") if trace_attempts else "",
            "episode_level": episode_reflection.get("summary", ""),
        },
    }
    if profile.get("save_memory", True):
        _update_memory_outcomes(episodic, is_success=final_success, path=stats_path)
        _append_memory(memory_entry, path=memory_path)

    trace = {
        "benchmark_profile": profile,
        "memory_hits": len(episodic),
        "selected_memories": [
            {
                "memory_id": item.get("memory_id", ""),
                "score": item.get("memory_score", 0.0),
                "success_count": item.get("success_count", 0),
                "failure_count": item.get("failure_count", 0),
                "question": item.get("question", ""),
            }
            for item in episodic
        ],
        "attempts": trace_attempts,
        "episode_reflection": episode_reflection,
        "memory_entry": memory_entry,
    }
    return final_answer, final_contexts, trace
