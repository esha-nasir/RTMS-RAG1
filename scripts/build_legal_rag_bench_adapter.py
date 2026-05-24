import argparse
import json
import re
from pathlib import Path


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def extract_span(text: str, span) -> str:
    if not isinstance(span, list) or len(span) != 2:
        return ""
    try:
        start = max(0, int(span[0]))
        end = max(start, int(span[1]))
    except Exception:
        return ""
    return normalize_ws(text[start:end])


def first_sentence(text: str) -> str:
    clean = normalize_ws(text)
    if not clean:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)
    return parts[0].strip()


def file_doc_id(dataset_name: str, relative_file_path: str) -> str:
    return f"{dataset_name}::{relative_file_path}"


def chunk_record_id(dataset_name: str, relative_file_path: str, chunk_index: int) -> str:
    return f"{dataset_name}::{relative_file_path}::chunk_{chunk_index}"


def sentence_record_id(dataset_name: str, relative_file_path: str, sentence_index: int, window_size: int) -> str:
    return f"{dataset_name}::{relative_file_path}::sent_{sentence_index}_w{window_size}"


def collect_benchmark_paths(benchmarks_dir: Path, selected: list[str]) -> list[Path]:
    paths = []
    for name in selected:
        path = benchmarks_dir / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"Missing benchmark file: {path}")
        paths.append(path)
    return paths


def benchmark_profile(dataset_name: str) -> dict:
    if dataset_name == "privacy_qa":
        return {
            "task_type": "grounded_qa",
            "needs_exact_span": False,
        }
    return {
        "task_type": "extractive_qa",
        "needs_exact_span": True,
    }


def build_chunks_for_document(
    dataset_name: str,
    relative_file_path: str,
    source_text: str,
    chunk_chars: int,
    overlap_chars: int,
) -> list[dict]:
    text = source_text or ""
    if not text:
        return []
    stride = max(1, chunk_chars - overlap_chars)
    chunks: list[dict] = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk_text = normalize_ws(text[start:end])
        if chunk_text:
            chunks.append(
                {
                    "id": chunk_record_id(dataset_name, relative_file_path, chunk_index),
                    "doc_id": file_doc_id(dataset_name, relative_file_path),
                    "file_name": dataset_name,
                    "chunk_id": f"{relative_file_path}::chunk_{chunk_index}",
                    "case_no": "",
                    "url": "",
                    "text": chunk_text,
                    "source_file_path": relative_file_path,
                    "char_start": start,
                    "char_end": end,
                }
            )
        if end >= len(text):
            break
        start += stride
        chunk_index += 1
    return chunks


def split_sentences_with_spans(text: str) -> list[tuple[int, int, str]]:
    clean_text = str(text or "")
    if not clean_text:
        return []

    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", clean_text):
        start, end = match.span()
        snippet = normalize_ws(clean_text[start:end])
        if snippet:
            spans.append((start, end, snippet))
    if spans:
        return spans

    fallback = normalize_ws(clean_text)
    return [(0, len(clean_text), fallback)] if fallback else []


def build_sentence_units_for_document(
    dataset_name: str,
    relative_file_path: str,
    source_text: str,
) -> list[dict]:
    sentences = split_sentences_with_spans(source_text)
    rows: list[dict] = []
    for idx, _ in enumerate(sentences):
        for window_size in (1, 2):
            if idx + window_size > len(sentences):
                continue
            window_sentences = sentences[idx : idx + window_size]
            window_start = window_sentences[0][0]
            window_end = window_sentences[-1][1]
            text = normalize_ws(" ".join(part[2] for part in window_sentences))
            if not text:
                continue
            rows.append(
                {
                    "id": sentence_record_id(dataset_name, relative_file_path, idx, window_size),
                    "doc_id": file_doc_id(dataset_name, relative_file_path),
                    "file_name": dataset_name,
                    "chunk_id": f"{relative_file_path}::sent_{idx}_w{window_size}",
                    "case_no": "",
                    "url": "",
                    "text": text,
                    "source_file_path": relative_file_path,
                    "char_start": window_start,
                    "char_end": window_end,
                }
            )
    return rows


def chunk_contains_span(chunk: dict, span) -> bool:
    if not isinstance(span, list) or len(span) != 2:
        return False
    try:
        span_start = int(span[0])
        span_end = int(span[1])
    except Exception:
        return False
    chunk_start = int(chunk.get("char_start", 0))
    chunk_end = int(chunk.get("char_end", 0))
    return span_start < chunk_end and span_end > chunk_start


def build_from_legalbenchrag_layout(
    legalbenchrag_dir: Path,
    selected_benchmarks: list[str],
    output_dir: Path,
    chunk_chars: int,
    overlap_chars: int,
    sentence_mode: bool = False,
) -> dict:
    corpus_dir = legalbenchrag_dir / "corpus"
    benchmarks_dir = legalbenchrag_dir / "benchmarks"
    if not corpus_dir.exists():
        raise SystemExit(f"Missing corpus directory: {corpus_dir}")
    if not benchmarks_dir.exists():
        raise SystemExit(f"Missing benchmarks directory: {benchmarks_dir}")

    benchmark_paths = collect_benchmark_paths(benchmarks_dir, selected_benchmarks)

    corpus_rows_by_id: dict[str, dict] = {}
    chunks_by_doc_id: dict[str, list[dict]] = {}
    qa_rows: list[dict] = []
    qa_grounded_rows: list[dict] = []
    qa_extractive_rows: list[dict] = []
    retrieval_rows: list[dict] = []
    skipped_rows = 0

    for benchmark_path in benchmark_paths:
        dataset_name = benchmark_path.stem
        profile = benchmark_profile(dataset_name)
        payload = load_json(benchmark_path)
        tests = payload.get("tests", [])
        if not isinstance(tests, list):
            continue

        for idx, item in enumerate(tests, 1):
            if not isinstance(item, dict):
                skipped_rows += 1
                continue

            question = normalize_ws(str(item.get("query", "")))
            snippets = item.get("snippets", [])
            if not question or not isinstance(snippets, list) or not snippets:
                skipped_rows += 1
                continue

            gold_sources: list[str] = []
            gold_answers: list[str] = []
            supporting_evidence_parts: list[str] = []
            primary_gold_text = ""
            primary_span = None
            primary_source_file_path = ""

            for snippet in snippets:
                if not isinstance(snippet, dict):
                    continue
                relative_file_path = str(snippet.get("file_path", "")).strip()
                if not relative_file_path:
                    continue

                source_path = corpus_dir / relative_file_path
                if not source_path.exists():
                    continue

                source_text = safe_read_text(source_path)
                doc_id = file_doc_id(dataset_name, relative_file_path)
                if doc_id not in chunks_by_doc_id:
                    if sentence_mode:
                        doc_chunks = build_sentence_units_for_document(
                            dataset_name=dataset_name,
                            relative_file_path=relative_file_path,
                            source_text=source_text,
                        )
                    else:
                        doc_chunks = build_chunks_for_document(
                            dataset_name=dataset_name,
                            relative_file_path=relative_file_path,
                            source_text=source_text,
                            chunk_chars=chunk_chars,
                            overlap_chars=overlap_chars,
                        )
                    chunks_by_doc_id[doc_id] = doc_chunks
                    for chunk in doc_chunks:
                        corpus_rows_by_id[str(chunk["id"])] = chunk

                answer = normalize_ws(str(snippet.get("answer", "")))
                span = snippet.get("span")
                span_text = extract_span(source_text, span)
                if answer:
                    gold_answers.append(answer)
                if span_text:
                    gold_answers.append(span_text)
                    supporting_evidence_parts.append(span_text)
                elif answer:
                    supporting_evidence_parts.append(answer)

                if not primary_gold_text:
                    primary_gold_text = span_text or answer or normalize_ws(source_text[:1200])
                    primary_span = span if isinstance(span, list) and len(span) == 2 else None
                    primary_source_file_path = relative_file_path

                matched_chunk_ids = [
                    str(chunk.get("id", ""))
                    for chunk in chunks_by_doc_id.get(doc_id, [])
                    if chunk_contains_span(chunk, span)
                ]
                if matched_chunk_ids:
                    gold_sources.extend(matched_chunk_ids)
                else:
                    fallback_chunks = chunks_by_doc_id.get(doc_id, [])
                    if fallback_chunks:
                        gold_sources.append(str(fallback_chunks[0].get("id", "")))

            gold_sources = list(dict.fromkeys([x for x in gold_sources if x]))
            gold_answers = [x for x in dict.fromkeys([x for x in gold_answers if x])]
            supporting_evidence = " || ".join(dict.fromkeys([x for x in supporting_evidence_parts if x]))
            if not gold_sources or not gold_answers:
                skipped_rows += 1
                continue

            primary_gold_id = gold_sources[0]
            qa_id = f"{dataset_name}_qa_{idx}"
            combined_gold_answer = " || ".join(gold_answers)
            acceptable_answers = list(dict.fromkeys(gold_answers + [first_sentence(primary_gold_text)]))
            row = {
                "id": qa_id,
                "question": question,
                "gold_answer": combined_gold_answer,
                "acceptable_answers": [x for x in acceptable_answers if x],
                "gold_sources": gold_sources,
                "file_name": dataset_name,
                "chunk_id": primary_gold_id.split("::", 2)[-1],
                "case_no": "",
                "gold_chunk_text": primary_gold_text,
                "gold_vector_id": primary_gold_id,
                "gold_source_file_path": primary_source_file_path,
                "gold_span": primary_span,
                "supporting_evidence": supporting_evidence or primary_gold_text,
                "task_type": profile["task_type"],
                "needs_exact_span": profile["needs_exact_span"],
                "benchmark_type": f"qa_public_{dataset_name}",
            }
            qa_rows.append(row)
            if profile["task_type"] == "grounded_qa":
                qa_grounded_rows.append(row)
            else:
                qa_extractive_rows.append(row)

            retrieval_rows.append(
                {
                    "id": f"{dataset_name}_retrieval_{idx}",
                    "query": question,
                    "gold_vector_id": primary_gold_id,
                    "gold_file_name": dataset_name,
                    "gold_chunk_id": primary_gold_id.split("::", 2)[-1],
                    "gold_case_no": "",
                    "source_quote": gold_answers[0],
                    "gold_source_file_path": primary_source_file_path,
                    "gold_span": primary_span,
                    "benchmark_type": f"retrieval_public_{dataset_name}",
                }
            )

    corpus_rows = list(corpus_rows_by_id.values())
    if not corpus_rows:
        raise SystemExit("No corpus rows were built from LegalBench-RAG.")
    if not qa_rows:
        raise SystemExit("No QA rows were built from LegalBench-RAG.")

    write_jsonl(output_dir / "all_chunks.jsonl", corpus_rows)
    write_jsonl(output_dir / "qa_benchmark.jsonl", qa_rows)
    write_jsonl(output_dir / "qa_extractive_benchmark.jsonl", qa_extractive_rows)
    write_jsonl(output_dir / "qa_grounded_benchmark.jsonl", qa_grounded_rows)
    write_jsonl(output_dir / "retrieval_benchmark.jsonl", retrieval_rows)

    summary = {
        "input_dir": str(legalbenchrag_dir),
        "selected_benchmarks": selected_benchmarks,
        "output_dir": str(output_dir),
        "sentence_mode": sentence_mode,
        "corpus_rows": len(corpus_rows),
        "qa_rows": len(qa_rows),
        "qa_grounded_rows": len(qa_grounded_rows),
        "qa_extractive_rows": len(qa_extractive_rows),
        "retrieval_rows": len(retrieval_rows),
        "skipped_rows": skipped_rows,
        "files": {
            "all_chunks_jsonl": str(output_dir / "all_chunks.jsonl"),
            "qa_benchmark_jsonl": str(output_dir / "qa_benchmark.jsonl"),
            "qa_extractive_benchmark_jsonl": str(output_dir / "qa_extractive_benchmark.jsonl"),
            "qa_grounded_benchmark_jsonl": str(output_dir / "qa_grounded_benchmark.jsonl"),
            "retrieval_benchmark_jsonl": str(output_dir / "retrieval_benchmark.jsonl"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Adapt LegalBench-RAG into the local evaluator format.")
    parser.add_argument("--legalbenchrag-dir", default=str(root / "LegalBench-RAG"))
    parser.add_argument("--output-dir", default=str(root / "data" / "legalbenchrag_eval"))
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["maud", "cuad", "privacy_qa", "contractnli"],
        help="Benchmark files under LegalBench-RAG/benchmarks without the .json suffix.",
    )
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument(
        "--sentence-mode",
        action="store_true",
        help="Build sentence and sentence-pair retrieval units instead of fixed character chunks.",
    )
    args = parser.parse_args()

    summary = build_from_legalbenchrag_layout(
        legalbenchrag_dir=Path(args.legalbenchrag_dir),
        selected_benchmarks=args.benchmarks,
        output_dir=Path(args.output_dir),
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        sentence_mode=args.sentence_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
