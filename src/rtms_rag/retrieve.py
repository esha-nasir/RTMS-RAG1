import math
import os
import re
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

from .yandex_embed import get_embedding

LOCAL_CORPUS_ONLY = os.getenv("LOCAL_CORPUS_ONLY", "0").strip() == "1"
HYBRID_RETRIEVAL_ENABLED = os.getenv("HYBRID_RETRIEVAL_ENABLED", "1").strip() == "1"
LEXICAL_RERANK_ENABLED = os.getenv("LEXICAL_RERANK_ENABLED", "1").strip() == "1"
RERANK_CANDIDATE_MULTIPLIER = max(1, int(os.getenv("RERANK_CANDIDATE_MULTIPLIER", "4")))
RERANK_MIN_CANDIDATES = max(1, int(os.getenv("RERANK_MIN_CANDIDATES", "10")))
LEXICAL_CANDIDATE_K = max(1, int(os.getenv("LEXICAL_CANDIDATE_K", "10")))
SEMANTIC_WEIGHT = float(os.getenv("RERANK_SEMANTIC_WEIGHT", "0.65"))
LEXICAL_WEIGHT = float(os.getenv("RERANK_LEXICAL_WEIGHT", "0.35"))
EXACT_PHRASE_BONUS = float(os.getenv("RERANK_EXACT_PHRASE_BONUS", "0.2"))
QUOTED_TEXT_BONUS = float(os.getenv("RERANK_QUOTED_TEXT_BONUS", "0.35"))
CASE_MATCH_BONUS = float(os.getenv("RERANK_CASE_MATCH_BONUS", "0.25"))
FILE_MATCH_BONUS = float(os.getenv("RERANK_FILE_MATCH_BONUS", "0.2"))
LOCAL_WINDOW_RERANK_ENABLED = os.getenv("LOCAL_WINDOW_RERANK_ENABLED", "1").strip() == "1"
LOCAL_WINDOW_CHARS = max(200, int(os.getenv("LOCAL_WINDOW_CHARS", "700")))
LOCAL_WINDOW_OVERLAP = max(0, int(os.getenv("LOCAL_WINDOW_OVERLAP", "150")))
DOCUMENT_SPAN_RERANK_ENABLED = os.getenv("DOCUMENT_SPAN_RERANK_ENABLED", "1").strip() == "1"
DOCUMENT_SPAN_SOURCE_LIMIT = max(1, int(os.getenv("DOCUMENT_SPAN_SOURCE_LIMIT", "8")))
DOCUMENT_SENTENCE_RERANK_ENABLED = os.getenv("DOCUMENT_SENTENCE_RERANK_ENABLED", "1").strip() == "1"
LLM_RERANK_ENABLED = os.getenv("LLM_RERANK_ENABLED", "0").strip() == "1"
LLM_RERANK_TOP_N = max(2, int(os.getenv("LLM_RERANK_TOP_N", "6")))
LLM_RERANK_MAX_CHARS = max(160, int(os.getenv("LLM_RERANK_MAX_CHARS", "700")))
ROOT = Path(__file__).resolve().parent
DEFAULT_LEXICAL_CORPUS_PATH = str(ROOT / "data" / "benchmark_from_chunks" / "all_chunks.jsonl")
LEXICAL_CORPUS_PATH = os.getenv("LEXICAL_CORPUS_PATH", DEFAULT_LEXICAL_CORPUS_PATH).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _extract_quoted_fragments(query: str) -> list[str]:
    return [frag.strip().lower() for frag in re.findall(r'"([^"]+)"', query or "") if frag.strip()]


def _extract_case_markers(query: str) -> list[str]:
    lowered = (query or "").lower()
    patterns = [
        r"\b(?:w\.p\.|c\.a\.|s\.l\.p\.|r\.p\.)\s*no\.?\s*[\w./-]+",
        r"\bcase\s*no\.?\s*[\w./-]+",
    ]
    markers = []
    for pattern in patterns:
        markers.extend(match.group(0).strip() for match in re.finditer(pattern, lowered))
    return list(dict.fromkeys(markers))


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]]) -> list[float]:
    if not query_tokens or not docs_tokens:
        return [0.0 for _ in docs_tokens]

    avgdl = sum(len(tokens) for tokens in docs_tokens) / max(len(docs_tokens), 1)
    if avgdl <= 0:
        return [0.0 for _ in docs_tokens]

    query_counts = Counter(query_tokens)
    df = Counter()
    for tokens in docs_tokens:
        for token in set(tokens):
            df[token] += 1

    k1 = 1.5
    b = 0.75
    total_docs = len(docs_tokens)
    scores = []
    for tokens in docs_tokens:
        tf = Counter(tokens)
        doc_len = max(len(tokens), 1)
        score = 0.0
        for token, qf in query_counts.items():
            if token not in tf:
                continue
            freq = tf[token]
            idf = math.log(1 + (total_docs - df[token] + 0.5) / (df[token] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (freq * (k1 + 1) / denom) * qf
        scores.append(score)
    return scores


@lru_cache(maxsize=1)
def _load_lexical_corpus() -> list[dict]:
    path = Path(LEXICAL_CORPUS_PATH)
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "doc_id": row.get("doc_id"),
                    "text": text,
                    "file_name": row.get("file_name", ""),
                    "case_no": row.get("case_no", ""),
                    "chunk_id": row.get("chunk_id"),
                    "source_file_path": row.get("source_file_path", ""),
                    "url": row.get("url", ""),
                }
            )
    return rows


def _candidate_bonus(
    query: str,
    text: str,
    ctx_file_name: str = "",
    file_name: str | None = None,
    case_markers: list[str] | None = None,
    quoted_fragments: list[str] | None = None,
) -> float:
    lowered_query = (query or "").lower()
    lowered_text = (text or "").lower()
    quoted_fragments = quoted_fragments or []
    case_markers = case_markers or []
    bonus = 0.0

    if file_name and ctx_file_name == file_name:
        bonus += FILE_MATCH_BONUS
    if ctx_file_name and ctx_file_name.lower() in lowered_query:
        bonus += FILE_MATCH_BONUS
    if any(fragment and fragment in lowered_text for fragment in quoted_fragments):
        bonus += QUOTED_TEXT_BONUS
    if lowered_query.strip() and lowered_query.strip() in lowered_text:
        bonus += EXACT_PHRASE_BONUS
    if any(marker and marker in lowered_text for marker in case_markers):
        bonus += CASE_MATCH_BONUS
    return bonus


def _split_into_windows(text: str, window_chars: int, overlap_chars: int) -> list[str]:
    clean = str(text or "")
    if not clean.strip():
        return []

    blocks = [part.strip() for part in re.split(r"\n\s*\n", clean) if part.strip()]
    if len(blocks) > 1:
        windows = []
        current = ""
        for block in blocks:
            if not current:
                current = block
                continue
            if len(current) + 2 + len(block) <= window_chars:
                current += "\n\n" + block
            else:
                windows.append(current)
                current = block
        if current:
            windows.append(current)
        return windows

    stride = max(1, window_chars - overlap_chars)
    windows = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + window_chars)
        snippet = clean[start:end].strip()
        if snippet:
            windows.append(snippet)
        if end >= len(clean):
            break
        start += stride
    return windows


def _sentence_windows(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if not clean:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if not sentences:
        return []
    windows = []
    for idx, sentence in enumerate(sentences):
        windows.append(sentence)
        if idx + 1 < len(sentences):
            windows.append(f"{sentence} {sentences[idx + 1]}")
    return windows


def _question_focus_terms(query: str) -> set[str]:
    q = (query or "").lower()
    focus = set()
    if any(phrase in q for phrase in ["who can see", "who sees", "visible to", "who can view"]):
        focus.update(
            {
                "visible",
                "public",
                "users",
                "buyers",
                "sellers",
                "share",
                "shared",
                "disclose",
                "disclosed",
                "profile",
                "published",
                "publish",
                "communications",
            }
        )
    if any(phrase in q for phrase in ["what information", "what data", "store about me", "collect about me"]):
        focus.update(
            {
                "collect",
                "collected",
                "gather",
                "gathered",
                "record",
                "recorded",
                "information",
                "data",
                "usage",
                "communications",
                "device",
                "browser",
                "ip",
                "geo",
                "location",
            }
        )
    if any(phrase in q for phrase in ["hire workers", "tasks i hire", "workers for"]):
        focus.update({"workers", "buyers", "sellers", "users", "publish", "published", "profile", "gig", "communications"})
    if any(phrase in q for phrase in ["share with", "third parties", "sell", "rent"]):
        focus.update({"share", "shared", "third", "parties", "sell", "rent", "disclose", "disclosed"})
    return focus


def _natural_sort_key(value) -> list[object]:
    parts = re.split(r"(\d+)", str(value or ""))
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _context_rank_score(ctx: dict) -> float:
    return float(
        ctx.get("rerank_score")
        or ctx.get("lexical_candidate_score")
        or ctx.get("score")
        or 0.0
    )


def _extract_json_object(text: str) -> dict:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text or "")
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _truncate_for_rerank(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _llm_rerank_contexts(query: str, contexts: list[dict], top_k: int) -> list[dict]:
    if not LLM_RERANK_ENABLED or len(contexts) <= top_k:
        return contexts[:top_k]

    try:
        from rag import _call_completion
    except Exception:
        return contexts[:top_k]

    candidate_limit = min(len(contexts), max(top_k, LLM_RERANK_TOP_N))
    candidates = contexts[:candidate_limit]
    candidate_blocks = []
    for idx, ctx in enumerate(candidates, start=1):
        file_name = str(ctx.get("file_name", "") or "").strip()
        chunk_id = str(ctx.get("chunk_id", "") or "").strip()
        label_parts = []
        if file_name:
            label_parts.append(file_name)
        if chunk_id:
            label_parts.append(f"chunk={chunk_id}")
        label = f" ({', '.join(label_parts)})" if label_parts else ""
        snippet = _truncate_for_rerank(ctx.get("text", ""), LLM_RERANK_MAX_CHARS)
        candidate_blocks.append(f"{idx}.{label}\n{snippet}")

    prompt = (
        "You are reranking evidence snippets for legal question answering.\n"
        "Pick the snippets that most directly answer the question.\n"
        "Prefer exact answer-bearing clauses over broader background text.\n"
        "Return JSON only with keys:\n"
        '- indices: array of 1-based candidate numbers ordered best to worst\n'
        '- notes: short string\n\n'
        f"Question: {query}\n\n"
        "Candidates:\n"
        f"{chr(10).join(candidate_blocks)}\n\n"
        f"Return up to {top_k} indices."
    )

    try:
        raw = _call_completion(prompt, temperature=0.0, max_tokens=120, json_mode=True)
    except Exception:
        return contexts[:top_k]

    parsed = _extract_json_object(raw)
    indices = parsed.get("indices", [])
    if isinstance(indices, int):
        indices = [indices]
    if not isinstance(indices, list):
        indices = []

    selected: list[dict] = []
    seen: set[int] = set()
    for idx in indices:
        try:
            pos = int(idx)
        except Exception:
            continue
        if pos < 1 or pos > candidate_limit or pos in seen:
            continue
        seen.add(pos)
        item = dict(candidates[pos - 1])
        item["retrieval_channel"] = f"{item.get('retrieval_channel', 'candidate')}+llm_rerank"
        item["llm_rerank_notes"] = str(parsed.get("notes", "")).strip()
        selected.append(item)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for pos, ctx in enumerate(candidates, start=1):
            if pos in seen:
                continue
            selected.append(ctx)
            if len(selected) >= top_k:
                break
    return selected[:top_k]


def _select_best_document_contexts(
    contexts: list[dict],
    file_name: str | None = None,
    source_file_path: str | None = None,
) -> list[dict]:
    if not contexts:
        return []

    if source_file_path:
        matched = [
            ctx
            for ctx in contexts
            if str(ctx.get("source_file_path", "") or "").strip() == source_file_path
        ]
        if matched:
            matched.sort(key=_context_rank_score, reverse=True)
            return matched

    if file_name:
        matched = [ctx for ctx in contexts if str(ctx.get("file_name", "") or "").strip() == file_name]
        if matched:
            matched.sort(key=_context_rank_score, reverse=True)
            return matched

    grouped: dict[str, list[dict]] = {}
    for ctx in contexts:
        key = str(ctx.get("source_file_path", "") or "").strip() or str(ctx.get("file_name", "") or "").strip()
        if not key:
            key = "__unknown__"
        grouped.setdefault(key, []).append(ctx)

    best_group: list[dict] = []
    best_score = float("-inf")
    for group in grouped.values():
        group.sort(key=_context_rank_score, reverse=True)
        score = sum(_context_rank_score(item) for item in group[: min(3, len(group))])
        if score > best_score:
            best_score = score
            best_group = group
    return best_group


def _collect_document_text(
    contexts: list[dict],
    case_no: str | None = None,
    file_name: str | None = None,
    source_file_path: str | None = None,
) -> str:
    if not contexts:
        return ""

    target_contexts = _select_best_document_contexts(
        contexts,
        file_name=file_name,
        source_file_path=source_file_path,
    )
    target_file_name = str(target_contexts[0].get("file_name", "") or "").strip() if target_contexts else ""
    target_source_file_path = str(target_contexts[0].get("source_file_path", "") or "").strip() if target_contexts else ""
    target_case_no = case_no or str(target_contexts[0].get("case_no", "") or "").strip()

    corpus_rows = _load_lexical_corpus()
    if (target_source_file_path or target_file_name) and corpus_rows:
        matching_rows = []
        for row in corpus_rows:
            row_source_file_path = str(row.get("source_file_path", "") or "").strip()
            row_file = str(row.get("file_name", "") or "").strip()
            row_case = str(row.get("case_no", "") or "").strip()
            if target_source_file_path:
                if row_source_file_path != target_source_file_path:
                    continue
            elif row_file != target_file_name:
                continue
            if target_case_no and row_case and row_case != target_case_no:
                continue
            matching_rows.append(row)
        if matching_rows:
            matching_rows.sort(
                key=lambda row: (
                    _natural_sort_key(row.get("chunk_id")),
                    _natural_sort_key(row.get("id")),
                )
            )
            texts = [str(row.get("text", "") or "").strip() for row in matching_rows if str(row.get("text", "") or "").strip()]
            if texts:
                return "\n\n".join(texts)

    ordered_contexts = sorted(
        target_contexts[:DOCUMENT_SPAN_SOURCE_LIMIT],
        key=lambda ctx: (
            _natural_sort_key(ctx.get("chunk_id")),
            -_context_rank_score(ctx),
        ),
    )
    texts = []
    seen = set()
    for ctx in ordered_contexts:
        text = str(ctx.get("text", "") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return "\n\n".join(texts)


def _document_span_rerank(
    query: str,
    contexts: list[dict],
    top_k: int,
    case_no: str | None = None,
    file_name: str | None = None,
    source_file_path: str | None = None,
) -> list[dict]:
    if not DOCUMENT_SPAN_RERANK_ENABLED or not contexts:
        return contexts[:top_k]

    best_document_contexts = _select_best_document_contexts(
        contexts,
        file_name=file_name,
        source_file_path=source_file_path,
    )
    if not best_document_contexts:
        return contexts[:top_k]

    source_text = _collect_document_text(
        best_document_contexts,
        case_no=case_no,
        file_name=file_name,
        source_file_path=source_file_path,
    )
    if not source_text.strip():
        return contexts[:top_k]

    windows = _sentence_windows(source_text) if DOCUMENT_SENTENCE_RERANK_ENABLED else _split_into_windows(source_text, LOCAL_WINDOW_CHARS, LOCAL_WINDOW_OVERLAP)
    if len(windows) <= 1:
        return best_document_contexts[:top_k]

    docs_tokens = [_tokenize(window) for window in windows]
    query_tokens = _tokenize(query)
    lexical_scores = _bm25_scores(query_tokens, docs_tokens)
    quoted_fragments = _extract_quoted_fragments(query)
    case_markers = _extract_case_markers(query)
    focus_terms = _question_focus_terms(query)
    best_doc = best_document_contexts[0]

    ranked = []
    for idx, window in enumerate(windows):
        candidate_tokens = set(docs_tokens[idx])
        focus_overlap = len(focus_terms & candidate_tokens)
        bonus = _candidate_bonus(
            query=query,
            text=window,
            ctx_file_name=str(best_doc.get("file_name", "") or "").strip(),
            file_name=file_name,
            case_markers=case_markers,
            quoted_fragments=quoted_fragments,
        )
        score = lexical_scores[idx] + bonus + (1.75 * focus_overlap)
        if len(candidate_tokens) <= 8:
            score -= 0.4
        item = dict(best_doc)
        item["text"] = window
        item["chunk_id"] = f"{best_doc.get('chunk_id', 'doc')}::window_{idx}"
        item["id"] = f"{best_doc.get('id', best_doc.get('file_name', 'doc'))}::window_{idx}"
        item["lexical_candidate_score"] = round(score, 6)
        item["retrieval_channel"] = "document_span"
        ranked.append(item)

    ranked.sort(key=lambda item: item.get("lexical_candidate_score", 0.0), reverse=True)
    return ranked[:top_k]


def _local_window_rerank(query: str, contexts: list[dict], top_k: int, file_name: str | None = None) -> list[dict]:
    if not LOCAL_WINDOW_RERANK_ENABLED or not contexts:
        return contexts[:top_k]

    base = contexts[0]
    source_text = str(base.get("text", "") or "")
    if not source_text:
        return contexts[:top_k]

    windows = _split_into_windows(source_text, LOCAL_WINDOW_CHARS, LOCAL_WINDOW_OVERLAP)
    if len(windows) <= 1:
        return contexts[:top_k]

    docs_tokens = [_tokenize(window) for window in windows]
    query_tokens = _tokenize(query)
    lexical_scores = _bm25_scores(query_tokens, docs_tokens)
    quoted_fragments = _extract_quoted_fragments(query)
    case_markers = _extract_case_markers(query)

    ranked = []
    for idx, window in enumerate(windows):
        bonus = _candidate_bonus(
            query=query,
            text=window.lower(),
            ctx_file_name=str(base.get("file_name", "")).strip(),
            file_name=file_name,
            case_markers=case_markers,
            quoted_fragments=quoted_fragments,
        )
        score = lexical_scores[idx] + bonus
        ranked.append((score, idx, window))

    ranked.sort(key=lambda item: item[0], reverse=True)

    reranked_contexts = []
    for score, idx, window in ranked[:top_k]:
        item = dict(base)
        item["text"] = window
        original_chunk_id = str(base.get("chunk_id", ""))
        item["chunk_id"] = f"{original_chunk_id}::window_{idx}"
        item["id"] = f"{base.get('id', '')}::window_{idx}"
        item["lexical_candidate_score"] = round(score, 6)
        item["retrieval_channel"] = "local_window"
        reranked_contexts.append(item)
    return reranked_contexts


def _lexical_candidates(
    query: str,
    top_k: int,
    case_no: str | None = None,
    file_name: str | None = None,
    source_file_path: str | None = None,
) -> list[dict]:
    if not HYBRID_RETRIEVAL_ENABLED:
        return []

    corpus = _load_lexical_corpus()
    if not corpus:
        return []

    filtered_rows = []
    for row in corpus:
        row_case = str(row.get("case_no", "") or "").strip()
        row_file = str(row.get("file_name", "") or "").strip()
        row_source_file_path = str(row.get("source_file_path", "") or "").strip()
        if case_no and row_case != case_no:
            continue
        if source_file_path and row_source_file_path != source_file_path:
            continue
        if file_name and row_file != file_name:
            continue
        filtered_rows.append(row)

    if not filtered_rows:
        return []

    query_tokens = _tokenize(query)
    quoted_fragments = _extract_quoted_fragments(query)
    case_markers = _extract_case_markers(query)
    docs_tokens = [_tokenize(str(row.get("text", ""))) for row in filtered_rows]
    scores = _bm25_scores(query_tokens, docs_tokens)

    ranked: list[dict] = []
    for row, lexical_score in zip(filtered_rows, scores):
        bonus = _candidate_bonus(
            query=query,
            text=str(row.get("text", "")),
            ctx_file_name=str(row.get("file_name", "") or "").strip(),
            file_name=file_name,
            case_markers=case_markers,
            quoted_fragments=quoted_fragments,
        )
        ranked.append(
            {
                "text": row.get("text", ""),
                "file_name": row.get("file_name", ""),
                "folder": "",
                "chunk_id": row.get("chunk_id"),
                "diary_no": "",
                "judgement_type": "",
                "case_no": row.get("case_no", ""),
                "source_file_path": row.get("source_file_path", ""),
                "pet": "",
                "res": "",
                "pet_adv": "",
                "res_adv": "",
                "bench": "",
                "judgement_by": "",
                "judgment_dates": "",
                "url": row.get("url", ""),
                "score": 0.0,
                "id": row.get("id"),
                "embedding": [],
                "lexical_candidate_score": round(lexical_score + bonus, 6),
                "retrieval_channel": "lexical",
            }
        )

    ranked.sort(key=lambda item: item.get("lexical_candidate_score", 0.0), reverse=True)
    return ranked[:top_k]


def _merge_candidates(semantic_contexts: list[dict], lexical_contexts: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for ctx in semantic_contexts:
        key = str(ctx.get("id") or f"{ctx.get('file_name')}::{ctx.get('chunk_id')}")
        merged[key] = dict(ctx, retrieval_channel="semantic")

    for ctx in lexical_contexts:
        key = str(ctx.get("id") or f"{ctx.get('file_name')}::{ctx.get('chunk_id')}")
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(ctx)
            continue
        existing["retrieval_channel"] = "hybrid"
        if not existing.get("text") and ctx.get("text"):
            existing["text"] = ctx.get("text")
        if not existing.get("url") and ctx.get("url"):
            existing["url"] = ctx.get("url")
        existing["lexical_candidate_score"] = ctx.get("lexical_candidate_score", 0.0)

    return list(merged.values())


def _rerank_contexts(query: str, contexts: list[dict], top_k: int, file_name: str | None = None) -> list[dict]:
    if not LEXICAL_RERANK_ENABLED or not contexts:
        return contexts[:top_k]

    query_tokens = _tokenize(query)
    quoted_fragments = _extract_quoted_fragments(query)
    case_markers = _extract_case_markers(query)
    doc_tokens = [_tokenize(str(ctx.get("text", ""))) for ctx in contexts]
    lexical_scores = _bm25_scores(query_tokens, doc_tokens)
    semantic_scores = [float(ctx.get("score", 0.0) or 0.0) for ctx in contexts]
    normalized_lexical = _normalize_scores(lexical_scores)
    normalized_semantic = _normalize_scores(semantic_scores)

    ranked = []
    for idx, ctx in enumerate(contexts):
        text = str(ctx.get("text", "")).lower()
        ctx_file_name = str(ctx.get("file_name", "")).strip()
        bonus = _candidate_bonus(
            query=query,
            text=text,
            ctx_file_name=ctx_file_name,
            file_name=file_name,
            case_markers=case_markers,
            quoted_fragments=quoted_fragments,
        )

        final_score = (
            SEMANTIC_WEIGHT * normalized_semantic[idx]
            + LEXICAL_WEIGHT * normalized_lexical[idx]
            + bonus
        )
        reranked = dict(ctx)
        reranked["semantic_score"] = semantic_scores[idx]
        reranked["lexical_score"] = lexical_scores[idx]
        reranked["rerank_bonus"] = round(bonus, 6)
        reranked["rerank_score"] = round(final_score, 6)
        ranked.append(reranked)

    ranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
    return ranked[:top_k]

def retrieve(
    query: str,
    top_k: int = 5,
    case_no: str | None = None,
    file_name: str | None = None,
    source_file_path: str | None = None,
):
    """
    Retrieve relevant chunks from Pinecone with optional metadata filtering.
    Uses dense retrieval first, then reranks the candidate set with BM25-style
    lexical scoring plus exact-match bonuses for snippet-style legal queries.
    """
    if LOCAL_CORPUS_ONLY:
        candidate_limit = max(top_k, LEXICAL_CANDIDATE_K, (LLM_RERANK_TOP_N if LLM_RERANK_ENABLED else top_k))
        lexical_contexts = _lexical_candidates(
            query=query,
            top_k=candidate_limit,
            case_no=case_no,
            file_name=file_name,
            source_file_path=source_file_path,
        )
        reranked = _rerank_contexts(query, lexical_contexts, top_k=max(candidate_limit, 1), file_name=file_name)
        document_reranked = _document_span_rerank(
            query,
            reranked,
            top_k=candidate_limit,
            case_no=case_no,
            file_name=file_name,
            source_file_path=source_file_path,
        )
        if document_reranked and document_reranked[0].get("retrieval_channel") == "document_span":
            return _llm_rerank_contexts(query, document_reranked[:candidate_limit], top_k=top_k)
        candidate_contexts = _local_window_rerank(query, document_reranked, top_k=candidate_limit, file_name=file_name)
        return _llm_rerank_contexts(query, candidate_contexts, top_k=top_k)

    query_emb = get_embedding(query, kind="query")

    pinecone_filter = None
    if case_no and file_name:
        pinecone_filter = {"$and": [{"case_no": {"$eq": case_no}}, {"file_name": {"$eq": file_name}}]}
    elif case_no:
        pinecone_filter = {"case_no": {"$eq": case_no}}
    elif file_name:
        pinecone_filter = {"file_name": {"$eq": file_name}}

    from .pinecone_setup import get_index

    index = get_index()
    candidate_k = top_k
    if LEXICAL_RERANK_ENABLED:
        candidate_k = max(top_k * RERANK_CANDIDATE_MULTIPLIER, RERANK_MIN_CANDIDATES)
    results = index.query(
        vector=query_emb,
        top_k=candidate_k,
        include_metadata=True,
        filter=pinecone_filter,   # ✅ key change
    )

    contexts = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {}) or {}

        # Fetch embedding along with the text and other metadata
        chunk = {
            "text": metadata.get("text", ""),
            "file_name": metadata.get("file_name", ""),
            "folder": metadata.get("folder", ""),
            "chunk_id": metadata.get("chunk_id", None),
            "diary_no": metadata.get("diary_no", ""),
            "judgement_type": metadata.get("judgement_type", ""),
            "case_no": metadata.get("case_no", ""),
            "source_file_path": metadata.get("source_file_path", ""),
            "pet": metadata.get("pet", ""),
            "res": metadata.get("res", ""),
            "pet_adv": metadata.get("pet_adv", ""),
            "res_adv": metadata.get("res_adv", ""),
            "bench": metadata.get("bench", ""),
            "judgement_by": metadata.get("judgement_by", ""),
            "judgment_dates": metadata.get("judgment_dates", ""),
            "url": metadata.get("url", ""),
            "score": match.get("score", None),
            "id": match.get("id", None),
            # Ensure embedding is part of the context
            "embedding": match.get("values", []),  # Ensure embedding is present
        }

        contexts.append(chunk)

    lexical_contexts = _lexical_candidates(
        query=query,
        top_k=max(top_k, LEXICAL_CANDIDATE_K),
        case_no=case_no,
        file_name=file_name,
        source_file_path=source_file_path,
    )
    merged_contexts = _merge_candidates(contexts, lexical_contexts)
    candidate_limit = max(top_k, LEXICAL_CANDIDATE_K, (LLM_RERANK_TOP_N if LLM_RERANK_ENABLED else top_k))
    reranked = _rerank_contexts(query, merged_contexts, top_k=max(candidate_limit, 1), file_name=file_name)
    document_reranked = _document_span_rerank(
        query,
        reranked,
        top_k=candidate_limit,
        case_no=case_no,
        file_name=file_name,
        source_file_path=source_file_path,
    )
    if document_reranked and document_reranked[0].get("retrieval_channel") == "document_span":
        return _llm_rerank_contexts(query, document_reranked[:candidate_limit], top_k=top_k)
    candidate_contexts = _local_window_rerank(query, document_reranked, top_k=candidate_limit, file_name=file_name)
    return _llm_rerank_contexts(query, candidate_contexts, top_k=top_k)
