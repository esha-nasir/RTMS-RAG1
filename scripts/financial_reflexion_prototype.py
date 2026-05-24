import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from google import genai
from google.genai import types


MOCK_FINANCIAL_DATA = [
    {
        "id": "Q3-2024-REPORT",
        "title": "TechCorp (TCHP) Q3 2024 Earnings Report Summary",
        "content": """
        TechCorp reported Q3 2024 revenue of $15.2 billion, up 12% year-over-year, exceeding consensus estimates of $14.8 billion.
        Net income was $3.1 billion. The primary driver of growth was the "Cloud Services" division, which saw a 25% increase
        in subscription revenue, reaching $7.5 billion. However, the legacy "Hardware Solutions" division declined by 5%.
        The company announced a major investment of $500 million into AI infrastructure for Q4.
        Management projects Q4 revenue to be between $15.5B and $16.0B. Operating expenses grew 8% due to hiring in the R&D sector.
        """,
        "tags": ["earnings", "TechCorp", "Q3", "2024", "revenue", "cloud", "AI", "TCHP"],
    },
    {
        "id": "ANALYST-NOTE-2025",
        "title": "Analyst Note: Future Outlook for BioPharma Inc. (BPHM)",
        "content": """
        BioPharma Inc. (BPHM) is projected to see significant revenue uplift starting in Q2 2025, driven by the anticipated Phase 3
        trial success of their oncology drug, 'OncoMend'. Current valuation suggests a target price of $150, assuming a 70% success
        rate for the trial. Key risks include regulatory approval delays and competition from GenRx's similar compound.
        The company has a stable cash reserve of $2.1 billion, enough to fund operations through 2026 without further debt.
        They are not currently considered a strong dividend payer.
        """,
        "tags": ["biotech", "BioPharma", "2025", "outlook", "OncoMend", "valuation", "risk", "BPHM"],
    },
    {
        "id": "MARKET-NEWS-NOV",
        "title": "Recent Market News: Interest Rate Hike Impact",
        "content": """
        The Federal Reserve announced a 25 basis point interest rate hike in November 2025. This action is expected to
        immediately impact lending and mortgage markets. Economists suggest that high-growth technology stocks, especially
        those reliant on venture capital funding (like smaller, pre-profit companies), may face pressure, while established,
        cash-rich financial institutions (e.g., MegaBank) are likely to benefit from wider net interest margins.
        Inflation expectations remain subdued at 2.5%.
        """,
        "tags": ["macro", "interest rate", "Fed", "inflation", "banking", "tech", "markets"],
    },
]


@dataclass
class ReflexionConfig:
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    memory_path: str = os.getenv(
        "FINANCIAL_REFLEXION_MEMORY_PATH",
        os.path.join(os.path.dirname(__file__), "financial_reflexion_memory.jsonl"),
    )
    max_iters: int = int(os.getenv("FINANCIAL_REFLEXION_MAX_ITERS", "3"))
    initial_top_k: int = int(os.getenv("FINANCIAL_REFLEXION_TOP_K", "1"))
    max_top_k: int = int(os.getenv("FINANCIAL_REFLEXION_MAX_TOP_K", "3"))
    use_google_search: bool = os.getenv("FINANCIAL_REFLEXION_USE_SEARCH", "1").strip() == "1"


def _build_client() -> genai.Client:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment before running this prototype.")
    return genai.Client(api_key=api_key)


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _score_document(query: str, doc: dict[str, Any]) -> int:
    query_lower = query.lower()
    query_words = _tokenize(query)
    score = 0

    for tag in doc["tags"]:
        tag_lower = tag.lower()
        if tag_lower in query_lower or query_lower in tag_lower:
            score += 5

    title_lower = doc["title"].lower()
    if title_lower in query_lower or query_lower in title_lower:
        score += 10

    content_lower = doc["content"].lower()
    for word in query_words:
        if len(word) > 2 and word in content_lower:
            score += 1

    boosted_terms = {
        "techcorp": "techcorp",
        "biopharma": "biopharma",
        "interest rate": "macro",
        "oncomend": "oncomend",
    }
    for needle, expected_tag in boosted_terms.items():
        if needle in query_lower and expected_tag in {tag.lower() for tag in doc["tags"]}:
            score += 8

    return score


def retrieve_relevant_contexts(query: str, data: list[dict[str, Any]], top_k: int = 1) -> list[dict[str, Any]]:
    scored = []
    for doc in data:
        score = _score_document(query, doc)
        enriched = dict(doc)
        enriched["retrieval_score"] = score
        scored.append(enriched)

    scored.sort(key=lambda item: item["retrieval_score"], reverse=True)
    selected = [doc for doc in scored[:top_k] if doc["retrieval_score"] > 0]
    if selected:
        return selected

    fallback_doc = next(doc for doc in data if doc["id"] == "MARKET-NEWS-NOV")
    fallback = dict(fallback_doc)
    fallback["retrieval_score"] = 0
    return [fallback]


def _format_context(context_documents: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, doc in enumerate(context_documents, start=1):
        blocks.append(
            f"[Document {idx}] {doc['title']} (id={doc['id']}, score={doc.get('retrieval_score', 0)})\n"
            f"{doc['content'].strip()}"
        )
    return "\n\n".join(blocks) if blocks else "[No relevant context found]"


def _read_memory(memory_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(memory_path):
        return []
    entries = []
    with open(memory_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _append_memory(memory_path: str, entry: dict[str, Any]) -> None:
    with open(memory_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _memory_for_query(query: str, memory_path: str, top_n: int = 3) -> list[dict[str, Any]]:
    q_tokens = _tokenize(query)
    scored = []
    for item in _read_memory(memory_path):
        reflection = (item.get("final_reflection") or "").strip()
        if not reflection:
            continue
        overlap = len(q_tokens.intersection(_tokenize(item.get("query", ""))))
        if overlap > 0:
            scored.append((overlap, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_n]]


def _format_memory(memory_items: list[dict[str, Any]]) -> str:
    if not memory_items:
        return ""
    lines = []
    for idx, item in enumerate(memory_items, start=1):
        lines.append(f"{idx}. Prior query: {item.get('query', '')}")
        lines.append(f"   Reflection: {item.get('final_reflection', '')}")
    return "\n".join(lines)


def _generation_prompt(
    user_query: str,
    context_text: str,
    memory_text: str = "",
    reflection_hint: str = "",
) -> str:
    memory_section = f"\nRelevant Past Reflections:\n{memory_text}\n" if memory_text.strip() else ""
    reflection_section = f"\nCurrent Iteration Guidance:\n{reflection_hint}\n" if reflection_hint.strip() else ""
    return f"""
You are a specialized financial research analyst.

Use ONLY the provided internal context and any grounded web search results enabled by the tool.
Do not invent company facts, dates, financial metrics, or macro events.
If the internal context is insufficient, say so explicitly.
Support each material claim with a source reference using either:
- [Internal: DOCUMENT_ID]
- [Web: source title]
{memory_section}{reflection_section}
Context:
{context_text}

User Query:
{user_query}
""".strip()


def _critic_prompt(user_query: str, answer_text: str, context_text: str) -> str:
    return f"""
You are a strict critic for a financial RAG agent.

Evaluate whether the answer:
1. Stays grounded in the provided internal context and grounded web search.
2. Actually answers the user's question.
3. Avoids unsupported claims.
4. Uses source references.

Return ONLY valid JSON with keys:
- is_sufficient: boolean
- issues: array of short strings
- guidance: short actionable instruction for the next attempt

User Query:
{user_query}

Answer:
{answer_text}

Internal Context:
{context_text}
""".strip()


def _reflection_prompt(user_query: str, answer_text: str, critique: dict[str, Any]) -> str:
    return f"""
You are writing a short self-reflection for a financial Reflexion agent.
Summarize what failed in the last attempt and what should change next.
Return 2-4 concise bullet points.

User Query:
{user_query}

Previous Answer:
{answer_text}

Critique JSON:
{json.dumps(critique, ensure_ascii=True)}
""".strip()


def _generate(
    client: genai.Client,
    config: ReflexionConfig,
    prompt: str,
    system_instruction: str,
    temperature: float,
) -> Any:
    tools = [types.Tool(google_search=types.GoogleSearch())] if config.use_google_search else None
    request_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        tools=tools,
    )
    return client.models.generate_content(
        model=config.model_name,
        contents=[prompt],
        config=request_config,
    )


def _extract_web_sources(response: Any) -> list[str]:
    sources = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return sources
    grounding = getattr(candidates[0], "grounding_metadata", None)
    if not grounding:
        return sources
    for chunk in getattr(grounding, "grounding_chunks", None) or []:
        web = getattr(chunk, "web", None)
        if web and getattr(web, "title", None) and getattr(web, "uri", None):
            sources.append(f"{web.title} ({web.uri})")
    return sources


def _quick_answer_checks(answer_text: str) -> dict[str, bool]:
    has_internal = "[internal:" in answer_text.lower()
    has_web = "[web:" in answer_text.lower()
    return {
        "has_reference": has_internal or has_web,
        "passes": has_internal or has_web,
    }


def _run_critic(
    client: genai.Client,
    config: ReflexionConfig,
    user_query: str,
    answer_text: str,
    context_text: str,
) -> dict[str, Any]:
    quick = _quick_answer_checks(answer_text)
    response = _generate(
        client=client,
        config=config,
        prompt=_critic_prompt(user_query, answer_text, context_text),
        system_instruction="Return only JSON. Do not add markdown fences.",
        temperature=0.0,
    )
    critique = _extract_json_object(getattr(response, "text", ""))
    if not critique:
        critique = {
            "is_sufficient": quick["passes"],
            "issues": ["Critic parser fallback: model did not return valid JSON."],
            "guidance": "Answer directly, remove unsupported claims, and add explicit source references.",
        }
    if "is_sufficient" not in critique:
        critique["is_sufficient"] = quick["passes"]
    if not isinstance(critique.get("issues"), list):
        critique["issues"] = [str(critique.get("issues", "Unknown issue"))]
    if not str(critique.get("guidance", "")).strip():
        critique["guidance"] = "Tighten grounding and cite each key claim."
    return critique


def run_financial_reflexion(
    user_query: str,
    data: list[dict[str, Any]] | None = None,
    config: ReflexionConfig | None = None,
) -> dict[str, Any]:
    data = data or MOCK_FINANCIAL_DATA
    config = config or ReflexionConfig()
    client = _build_client()

    episodic_memory = _memory_for_query(user_query, config.memory_path, top_n=3)
    memory_text = _format_memory(episodic_memory)

    trace_attempts = []
    final_answer = ""
    final_contexts = []
    final_reflection = ""
    reflection_hint = ""

    for attempt in range(1, max(1, config.max_iters) + 1):
        top_k = min(config.initial_top_k + attempt - 1, config.max_top_k)
        contexts = retrieve_relevant_contexts(user_query, data, top_k=top_k)
        context_text = _format_context(contexts)

        answer_response = _generate(
            client=client,
            config=config,
            prompt=_generation_prompt(
                user_query=user_query,
                context_text=context_text,
                memory_text=memory_text,
                reflection_hint=reflection_hint,
            ),
            system_instruction=(
                "You are a concise financial RAG assistant. "
                "Use only provided internal context and grounded search results."
            ),
            temperature=0.2,
        )
        answer_text = getattr(answer_response, "text", "").strip()
        critique = _run_critic(client, config, user_query, answer_text, context_text)

        reflection = ""
        if not critique.get("is_sufficient", False):
            reflection_response = _generate(
                client=client,
                config=config,
                prompt=_reflection_prompt(user_query, answer_text, critique),
                system_instruction="Return short bullet points only.",
                temperature=0.1,
            )
            reflection = getattr(reflection_response, "text", "").strip()
            reflection_hint = f"{critique.get('guidance', '')}\n{reflection}".strip()

        trace_attempts.append(
            {
                "attempt": attempt,
                "top_k": top_k,
                "contexts": [doc["id"] for doc in contexts],
                "answer": answer_text,
                "critique": critique,
                "reflection": reflection,
                "web_sources": _extract_web_sources(answer_response),
            }
        )

        final_answer = answer_text
        final_contexts = contexts
        if reflection:
            final_reflection = reflection

        if critique.get("is_sufficient", False):
            break

    memory_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": user_query,
        "attempt_count": len(trace_attempts),
        "is_sufficient": bool(trace_attempts[-1]["critique"].get("is_sufficient", False)) if trace_attempts else False,
        "final_reflection": final_reflection,
        "last_guidance": trace_attempts[-1]["critique"].get("guidance", "") if trace_attempts else "",
        "issues": trace_attempts[-1]["critique"].get("issues", []) if trace_attempts else [],
    }
    _append_memory(config.memory_path, memory_entry)

    return {
        "query": user_query,
        "answer": final_answer,
        "contexts": final_contexts,
        "trace": {
            "memory_hits": len(episodic_memory),
            "attempts": trace_attempts,
            "memory_entry": memory_entry,
        },
    }


def demo() -> None:
    queries = [
        "How much did TechCorp's Cloud Services revenue grow in Q3 2024, and what was their stated investment into AI?",
        "What is the key driver of BioPharma's projected revenue uplift, and what is its cash reserve?",
        "How will the recent interest rate hike impact high-growth tech stocks, and what is the inflation outlook?",
        "Tell me about OncoMend.",
    ]

    for query in queries:
        print("=" * 80)
        print(f"QUERY: {query}")
        result = run_financial_reflexion(query)
        print(result["answer"])
        print("\nTrace summary:")
        for attempt in result["trace"]["attempts"]:
            print(
                f"- attempt={attempt['attempt']} top_k={attempt['top_k']} "
                f"contexts={attempt['contexts']} sufficient={attempt['critique'].get('is_sufficient')}"
            )


if __name__ == "__main__":
    demo()
