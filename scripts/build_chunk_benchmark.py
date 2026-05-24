import argparse
import csv
import json
import os
import re
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "judgements").strip()
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY")

    try:
        from pinecone import Pinecone  # type: ignore
    except Exception:
        from pinecone.pinecone import Pinecone  # type: ignore

    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def _flatten_ids(item) -> list[str]:
    if isinstance(item, str):
        return [item]
    if isinstance(item, dict):
        out = []
        vid = item.get("id")
        if isinstance(vid, str):
            out.append(vid)
        ids = item.get("ids")
        if isinstance(ids, list):
            out.extend([x for x in ids if isinstance(x, str)])
        return out
    if isinstance(item, (list, tuple)):
        return [x for x in item if isinstance(x, str)]
    return []


def list_all_ids(index, page_limit: int, max_ids: int) -> list[str]:
    ids: list[str] = []
    for item in index.list(prefix="", limit=page_limit):
        ids.extend(_flatten_ids(item))
        if max_ids > 0 and len(ids) >= max_ids:
            break
    ids = list(dict.fromkeys(ids))
    if max_ids > 0:
        ids = ids[:max_ids]
    return ids


def chunk_batches(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def first_sentence(text: str, fallback_words: int = 35) -> str:
    clean = normalize_ws(text)
    if not clean:
        return ""
    m = re.search(r"(.+?[.!?])(\s|$)", clean)
    if m:
        sent = m.group(1).strip()
        if len(sent) >= 30:
            return sent
    words = clean.split()
    return " ".join(words[:fallback_words]).strip()


def first_two_sentences(text: str, fallback_words: int = 70) -> str:
    clean = normalize_ws(text)
    if not clean:
        return ""
    matches = re.findall(r".+?[.!?](?=\s|$)", clean)
    if len(matches) >= 2:
        return " ".join(m.strip() for m in matches[:2]).strip()
    words = clean.split()
    return " ".join(words[:fallback_words]).strip()


def middle_phrase(text: str, span_words: int = 14) -> str:
    words = normalize_ws(text).split()
    if not words:
        return ""
    if len(words) <= span_words:
        return " ".join(words)
    start = max(0, (len(words) // 2) - (span_words // 2))
    return " ".join(words[start : start + span_words])


def leading_phrase(text: str, span_words: int = 10) -> str:
    words = normalize_ws(text).split()
    if not words:
        return ""
    return " ".join(words[:span_words]).strip()


def extractive_question(row: dict) -> str:
    quote = middle_phrase(row["text"], span_words=10)
    return f'In {row["file_name"]}, what is stated in relation to: "{quote}"?'


def grounded_question(row: dict) -> str:
    lead = leading_phrase(first_sentence(row["text"]), span_words=8)
    return f'In {row["file_name"]}, what does the judgment say about "{lead}"?'


def reflexion_question(row: dict) -> str:
    quote = middle_phrase(row["text"], span_words=12)
    return f'In {row["file_name"]}, explain the court\'s reasoning related to "{quote}".'


def quality_flags(text: str) -> list[str]:
    t = text or ""
    clean = normalize_ws(t)
    flags: list[str] = []
    if len(clean) < 120:
        flags.append("too_short")
    if "�" in clean:
        flags.append("replacement_char")

    alpha_chars = sum(ch.isalpha() for ch in clean)
    total_chars = max(1, len(clean))
    if alpha_chars / total_chars < 0.45:
        flags.append("low_alpha_ratio")

    weird_chars = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}'\"-_/&%") for ch in clean)
    if weird_chars / total_chars > 0.12:
        flags.append("high_symbol_noise")

    return flags


def _extract_vectors_map(fetch_resp):
    if isinstance(fetch_resp, dict):
        return fetch_resp.get("vectors", {}) or {}
    vectors = getattr(fetch_resp, "vectors", None)
    return vectors or {}


def _vector_metadata(vec) -> dict:
    if isinstance(vec, dict):
        meta = vec.get("metadata", {}) or {}
        return meta if isinstance(meta, dict) else {}
    meta = getattr(vec, "metadata", None)
    if isinstance(meta, dict):
        return meta
    if meta is None:
        return {}
    try:
        return dict(meta)
    except Exception:
        return {}


def build_records(index, ids: list[str]) -> tuple[list[dict], list[dict]]:
    all_rows: list[dict] = []
    good_rows: list[dict] = []

    for batch in chunk_batches(ids, 50):
        resp = index.fetch(ids=batch)
        vectors = _extract_vectors_map(resp)
        for vid in batch:
            vec = vectors.get(vid, {}) if hasattr(vectors, "get") else {}
            meta = _vector_metadata(vec)
            text = normalize_ws(meta.get("text", ""))
            flags = quality_flags(text)
            row = {
                "id": vid,
                "file_name": meta.get("file_name", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "case_no": meta.get("case_no", ""),
                "url": meta.get("url", ""),
                "text_len": len(text),
                "is_bad": 1 if flags else 0,
                "flags": "|".join(flags),
                "text": text,
            }
            all_rows.append(row)
            if not flags and text:
                good_rows.append(row)

    return all_rows, good_rows


def write_ids(ids: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ids), encoding="utf-8")


def write_audit_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "file_name", "chunk_id", "case_no", "url", "text_len", "is_bad", "flags", "text"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_all_chunks_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            rec = {
                "id": row["id"],
                "file_name": row["file_name"],
                "chunk_id": row["chunk_id"],
                "case_no": row["case_no"],
                "url": row["url"],
                "text": row["text"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_retrieval_benchmark(rows: list[dict], out_path: Path, max_samples: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if count >= max_samples:
                break
            phrase = middle_phrase(row["text"], span_words=14)
            if len(phrase.split()) < 8:
                continue
            rec = {
                "id": f"retrieval_{count+1}",
                "query": f'Find the legal source passage containing this statement: "{phrase}"',
                "gold_vector_id": row["id"],
                "gold_file_name": row["file_name"],
                "gold_chunk_id": row["chunk_id"],
                "gold_case_no": row["case_no"],
                "source_quote": phrase,
                "benchmark_type": "retrieval",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_qa_draft_benchmark(rows: list[dict], out_path: Path, max_samples: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if count >= max_samples:
                break
            answer = first_sentence(row["text"])
            if len(answer) < 30:
                continue
            quote = middle_phrase(row["text"], span_words=10)
            question = (
                f"In {row['file_name']}, what is stated in relation to: "
                f'"{quote}"?'
            )
            rec = {
                "id": f"qa_{count+1}",
                "question": question,
                "gold_answer": answer,
                "acceptable_answers": [answer, first_two_sentences(row["text"])],
                "gold_sources": [row["id"]],
                "file_name": row["file_name"],
                "chunk_id": row["chunk_id"],
                "case_no": row["case_no"],
                "benchmark_type": "qa_draft",
                "task_type": "extractive_qa",
                "needs_exact_span": True,
                "supporting_evidence": row["text"],
                "needs_manual_review": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_qa_extractive_benchmark(rows: list[dict], out_path: Path, max_samples: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if count >= max_samples:
                break
            answer = first_sentence(row["text"])
            if len(answer) < 30:
                continue
            rec = {
                "id": f"extractive_{count+1}",
                "question": extractive_question(row),
                "gold_answer": answer,
                "acceptable_answers": [answer],
                "gold_sources": [row["id"]],
                "file_name": row["file_name"],
                "chunk_id": row["chunk_id"],
                "case_no": row["case_no"],
                "benchmark_type": "qa_extractive",
                "task_type": "extractive_qa",
                "needs_exact_span": True,
                "supporting_evidence": row["text"],
                "needs_manual_review": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_qa_grounded_benchmark(rows: list[dict], out_path: Path, max_samples: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if count >= max_samples:
                break
            first = first_sentence(row["text"])
            second = first_two_sentences(row["text"])
            if len(first) < 30 or len(second) < 50:
                continue
            rec = {
                "id": f"grounded_{count+1}",
                "question": grounded_question(row),
                "gold_answer": second,
                "acceptable_answers": [second, first],
                "gold_sources": [row["id"]],
                "file_name": row["file_name"],
                "chunk_id": row["chunk_id"],
                "case_no": row["case_no"],
                "benchmark_type": "qa_grounded",
                "task_type": "grounded_qa",
                "needs_exact_span": False,
                "supporting_evidence": row["text"],
                "needs_manual_review": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_qa_reflexion_benchmark(rows: list[dict], out_path: Path, max_samples: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if count >= max_samples:
                break
            answer = first_two_sentences(row["text"])
            if len(answer) < 90:
                continue
            rec = {
                "id": f"reflexion_{count+1}",
                "question": reflexion_question(row),
                "gold_answer": answer,
                "acceptable_answers": [answer, first_sentence(row["text"])],
                "gold_sources": [row["id"]],
                "file_name": row["file_name"],
                "chunk_id": row["chunk_id"],
                "case_no": row["case_no"],
                "benchmark_type": "qa_reflexion",
                "task_type": "revision_needed",
                "needs_exact_span": False,
                "supporting_evidence": row["text"],
                "needs_manual_review": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Export Pinecone IDs, audit chunks, and build benchmark drafts.")
    parser.add_argument("--env-file", default=str(Path(__file__).resolve().parents[1] / ".env"))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "benchmark_from_chunks"))
    parser.add_argument("--max-ids", type=int, default=0, help="0 = all ids")
    parser.add_argument("--page-limit", type=int, default=99, help="Pinecone list page limit (<100)")
    parser.add_argument("--max-retrieval-samples", type=int, default=200)
    parser.add_argument("--max-qa-samples", type=int, default=200)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    index = get_pinecone_index()

    ids = list_all_ids(index, page_limit=args.page_limit, max_ids=args.max_ids)
    if not ids:
        raise RuntimeError("No vector IDs found in index.")

    all_rows, good_rows = build_records(index, ids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_path = out_dir / "all_vector_ids.txt"
    audit_path = out_dir / "chunk_audit.csv"
    all_chunks_path = out_dir / "all_chunks.jsonl"
    retrieval_path = out_dir / "retrieval_benchmark.jsonl"
    qa_path = out_dir / "qa_draft_benchmark.jsonl"
    qa_extractive_path = out_dir / "qa_extractive_benchmark.jsonl"
    qa_grounded_path = out_dir / "qa_grounded_benchmark.jsonl"
    qa_reflexion_path = out_dir / "qa_reflexion_benchmark.jsonl"
    summary_path = out_dir / "summary.json"

    write_ids(ids, ids_path)
    write_audit_csv(all_rows, audit_path)
    write_all_chunks_jsonl(good_rows, all_chunks_path)
    retrieval_n = write_retrieval_benchmark(good_rows, retrieval_path, args.max_retrieval_samples)
    qa_n = write_qa_draft_benchmark(good_rows, qa_path, args.max_qa_samples)
    qa_extractive_n = write_qa_extractive_benchmark(good_rows, qa_extractive_path, args.max_qa_samples)
    qa_grounded_n = write_qa_grounded_benchmark(good_rows, qa_grounded_path, args.max_qa_samples)
    qa_reflexion_n = write_qa_reflexion_benchmark(good_rows, qa_reflexion_path, args.max_qa_samples)

    bad_n = sum(r["is_bad"] for r in all_rows)
    summary = {
        "index_name": os.getenv("PINECONE_INDEX_NAME", "judgements"),
        "total_ids": len(ids),
        "audited_chunks": len(all_rows),
        "good_chunks": len(good_rows),
        "bad_chunks": bad_n,
        "bad_rate": round((bad_n / len(all_rows)) if all_rows else 0.0, 4),
        "retrieval_benchmark_samples": retrieval_n,
        "qa_draft_benchmark_samples": qa_n,
        "qa_extractive_benchmark_samples": qa_extractive_n,
        "qa_grounded_benchmark_samples": qa_grounded_n,
        "qa_reflexion_benchmark_samples": qa_reflexion_n,
        "files": {
            "all_vector_ids": str(ids_path),
            "chunk_audit_csv": str(audit_path),
            "all_chunks_jsonl": str(all_chunks_path),
            "retrieval_benchmark_jsonl": str(retrieval_path),
            "qa_draft_benchmark_jsonl": str(qa_path),
            "qa_extractive_benchmark_jsonl": str(qa_extractive_path),
            "qa_grounded_benchmark_jsonl": str(qa_grounded_path),
            "qa_reflexion_benchmark_jsonl": str(qa_reflexion_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
