import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SEVERE_BENCHMARK_FLAGS = {
    "missing_gold_chunk",
    "missing_gold_text",
    "gold_answer_not_in_gold_chunk",
    "gold_answer_too_short",
    "gold_answer_header_heavy",
    "gold_answer_footnote_heavy",
    "gold_chunk_header_heavy",
    "gold_chunk_too_short",
    "gold_chunk_low_alpha_ratio",
    "gold_chunk_high_symbol_noise",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize(text: str) -> str:
    return normalize_ws(text).lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))


def token_f1(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    sa = set(ta)
    sb = set(tb)
    common = len(sa & sb)
    if common == 0:
        return 0.0
    precision = common / len(sa)
    recall = common / len(sb)
    return 2 * precision * recall / (precision + recall)


def extract_quoted_phrase(question: str) -> str:
    match = re.search(r'"([^"]+)"', question or "")
    return normalize_ws(match.group(1)) if match else ""


def alpha_ratio(text: str) -> float:
    clean = normalize_ws(text)
    if not clean:
        return 0.0
    alpha_chars = sum(ch.isalpha() for ch in clean)
    return alpha_chars / max(1, len(clean))


def symbol_noise_ratio(text: str) -> float:
    clean = normalize_ws(text)
    if not clean:
        return 0.0
    weird = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}'\"-_/&%") for ch in clean)
    return weird / max(1, len(clean))


def header_heavy(text: str) -> bool:
    clean = normalize_ws(text)
    if not clean:
        return False
    head = normalize(clean[:400])
    patterns = [
        "in the supreme court",
        "reportable",
        "civil appeal no",
        "criminal appeal no",
        "judgement",
        "judgment",
        "order",
        "petitioner",
        "respondent",
        "appellant",
        "versus",
    ]
    hits = sum(1 for p in patterns if p in head)
    uppercaseish = sum(ch.isupper() for ch in clean[:200]) / max(1, len(clean[:200]))
    return hits >= 4 or (hits >= 3 and uppercaseish > 0.28)


def starts_with_page_noise(text: str) -> bool:
    head = normalize_ws(text)[:80]
    return bool(re.match(r"^\d+\s", head))


def footnote_heavy(text: str) -> bool:
    clean = normalize_ws(text)
    if not clean:
        return False
    head = normalize(clean[:160])
    if re.match(r"^\d+\s*(dated|issued|reported|section|article|rule|part)\b", head):
        return True
    citation_terms = ["dated", "issued", "reported", "part", "page", "section", "rule", "article"]
    term_hits = sum(head.count(term) for term in citation_terms)
    return term_hits >= 2 and len(tokenize(clean)) <= 18


def build_chunk_maps(chunks: list[dict]) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    by_id: dict[str, dict] = {}
    by_file_chunk: dict[tuple[str, str], dict] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        by_id[str(chunk.get("id", ""))] = chunk
        by_file_chunk[(str(chunk.get("file_name", "")), chunk_id)] = chunk
    return by_id, by_file_chunk


def resolve_gold_chunk(row: dict, by_id: dict[str, dict], by_file_chunk: dict[tuple[str, str], dict]) -> dict:
    gold_sources = row.get("gold_sources") or []
    if isinstance(gold_sources, list):
        for source_id in gold_sources:
            source = by_id.get(str(source_id))
            if source:
                return source
    return by_file_chunk.get((str(row.get("file_name", "")), str(row.get("chunk_id", ""))), {})


def benchmark_flags(row: dict, gold_chunk: dict) -> list[str]:
    flags: list[str] = []
    gold_answer = normalize_ws(str(row.get("gold_answer", "")))
    gold_text = normalize_ws(str(gold_chunk.get("text", "")))
    quote = extract_quoted_phrase(str(row.get("question", "")))

    if not gold_chunk:
        flags.append("missing_gold_chunk")
        return flags
    if not gold_text:
        flags.append("missing_gold_text")
        return flags

    if len(tokenize(gold_answer)) < 5 or len(gold_answer) < 32:
        flags.append("gold_answer_too_short")
    if len(gold_text) < 180:
        flags.append("gold_chunk_too_short")
    if header_heavy(gold_answer):
        flags.append("gold_answer_header_heavy")
    if footnote_heavy(gold_answer):
        flags.append("gold_answer_footnote_heavy")
    if header_heavy(gold_text) and int(gold_chunk.get("chunk_id", 0) or 0) <= 1:
        flags.append("gold_chunk_header_heavy")
    if starts_with_page_noise(gold_answer):
        flags.append("gold_answer_page_noise")
    if starts_with_page_noise(gold_text):
        flags.append("gold_chunk_page_noise")

    if alpha_ratio(gold_text) < 0.45:
        flags.append("gold_chunk_low_alpha_ratio")
    if symbol_noise_ratio(gold_text) > 0.12:
        flags.append("gold_chunk_high_symbol_noise")

    if gold_answer and normalize(gold_answer) not in normalize(gold_text):
        if token_f1(gold_answer, gold_text) < 0.35:
            flags.append("gold_answer_not_in_gold_chunk")

    if quote:
        quote_f1 = token_f1(quote, gold_text)
        if quote_f1 < 0.08 and normalize(quote) not in normalize(gold_text):
            flags.append("question_quote_not_in_gold_chunk")
        if len(tokenize(quote)) < 6:
            flags.append("question_quote_too_short")
    else:
        flags.append("missing_question_quote")

    return sorted(set(flags))


def benchmark_status(flags: list[str]) -> str:
    if not flags:
        return "clear"
    if any(flag in SEVERE_BENCHMARK_FLAGS for flag in flags):
        return "exclude"
    return "review"


def classify_prediction_failure(pred: dict, benchmark_row: dict, gold_chunk: dict) -> tuple[str, str]:
    benchmark_row_flags = benchmark_row.get("audit_flags") or []
    if any(flag in SEVERE_BENCHMARK_FLAGS for flag in benchmark_row_flags):
        return "benchmark/gold mismatch", "benchmark row has severe audit flags"

    retrieved_contexts = pred.get("retrieved_contexts") or []
    gold_sources = {str(x) for x in (benchmark_row.get("gold_sources") or pred.get("gold_sources") or [])}
    top_ids = [str(ctx.get("id", "")) for ctx in retrieved_contexts if isinstance(ctx, dict)]
    retrieval_hit = any(ctx_id in gold_sources for ctx_id in top_ids)

    if not retrieval_hit:
        return "retrieval miss", "gold source not present in retrieved contexts"

    predicted_answer = str(pred.get("predicted_answer", ""))
    gold_answer = str(benchmark_row.get("gold_answer", pred.get("gold_answer", "")))
    retrieved_reference = str(pred.get("retrieved_reference", ""))
    gold_text = str(gold_chunk.get("text", ""))

    pred_gold_f1 = token_f1(predicted_answer, gold_answer)
    pred_ref_f1 = max(token_f1(predicted_answer, retrieved_reference), token_f1(predicted_answer, gold_text))
    gold_len = max(1, len(tokenize(gold_answer)))
    pred_len = len(tokenize(predicted_answer))

    if pred_ref_f1 >= 0.35 and pred_gold_f1 < 0.25 and pred_len >= gold_len * 2:
        return "answer too broad", "answer is grounded but much broader than the gold answer"
    if pred_ref_f1 >= 0.25:
        return "grounded but wrong", "answer overlaps retrieved evidence but not the gold answer"
    return "grounded but wrong", "retrieval hit, but answer does not match the gold answer"


def audit_benchmark_rows(qa_rows: list[dict], by_id: dict[str, dict], by_file_chunk: dict[tuple[str, str], dict]) -> list[dict]:
    audited_rows: list[dict] = []
    for row in qa_rows:
        gold_chunk = resolve_gold_chunk(row, by_id, by_file_chunk)
        flags = benchmark_flags(row, gold_chunk)
        status = benchmark_status(flags)
        audited = dict(row)
        audited["gold_chunk_text"] = gold_chunk.get("text", "")
        audited["gold_vector_id"] = gold_chunk.get("id", "")
        audited["audit_flags"] = flags
        audited["audit_status"] = status
        audited_rows.append(audited)
    return audited_rows


def audit_predictions(
    prediction_rows: list[dict],
    benchmark_by_id: dict[str, dict],
    chunk_by_id: dict[str, dict],
    chunk_by_file_chunk: dict[tuple[str, str], dict],
) -> list[dict]:
    audited_predictions: list[dict] = []
    for pred in prediction_rows:
        benchmark_row = benchmark_by_id.get(str(pred.get("id", "")), {})
        gold_chunk = resolve_gold_chunk(benchmark_row or pred, chunk_by_id, chunk_by_file_chunk)
        record = dict(pred)
        if bool(pred.get("is_correct", False)):
            record["failure_category"] = ""
            record["failure_reason"] = ""
        else:
            category, reason = classify_prediction_failure(record, benchmark_row or pred, gold_chunk)
            record["failure_category"] = category
            record["failure_reason"] = reason
        audited_predictions.append(record)
    return audited_predictions


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def summarize_benchmark(rows: list[dict]) -> dict:
    statuses = Counter(str(row.get("audit_status", "")) for row in rows)
    flags = Counter(flag for row in rows for flag in row.get("audit_flags", []))
    return {
        "total_rows": len(rows),
        "clear_rows": statuses.get("clear", 0),
        "review_rows": statuses.get("review", 0),
        "exclude_rows": statuses.get("exclude", 0),
        "top_flags": dict(flags.most_common(20)),
    }


def summarize_predictions(rows: list[dict]) -> dict:
    failures = [row for row in rows if not bool(row.get("is_correct", False))]
    categories = Counter(str(row.get("failure_category", "")) for row in failures if row.get("failure_category"))
    return {
        "total_predictions": len(rows),
        "incorrect_predictions": len(failures),
        "failure_categories": dict(categories),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Audit and clean the local legal benchmark, and optionally classify QA misses.")
    parser.add_argument("--benchmark-dir", default=str(root / "data" / "benchmark_from_chunks"))
    parser.add_argument("--predictions", default="", help="Optional path to local_benchmark_predictions.jsonl")
    parser.add_argument("--output-dir", default=str(root / "benchmark_audit_outputs"))
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_path = benchmark_dir / "qa_draft_benchmark.jsonl"
    chunks_path = benchmark_dir / "all_chunks.jsonl"

    qa_rows = load_jsonl(qa_path)
    chunk_rows = load_jsonl(chunks_path)
    chunk_by_id, chunk_by_file_chunk = build_chunk_maps(chunk_rows)

    audited_benchmark = audit_benchmark_rows(qa_rows, chunk_by_id, chunk_by_file_chunk)
    benchmark_by_id = {str(row.get("id", "")): row for row in audited_benchmark}
    cleaned_benchmark = [row for row in audited_benchmark if row.get("audit_status") == "clear"]

    write_jsonl(output_dir / "qa_benchmark_audit.jsonl", audited_benchmark)
    write_jsonl(output_dir / "qa_benchmark_clean.jsonl", cleaned_benchmark)
    write_csv(
        output_dir / "qa_benchmark_audit.csv",
        audited_benchmark,
        [
            "id",
            "audit_status",
            "file_name",
            "chunk_id",
            "gold_vector_id",
            "gold_answer",
            "question",
            "audit_flags",
        ],
    )

    summary = {"benchmark": summarize_benchmark(audited_benchmark)}

    if args.predictions:
        prediction_rows = load_jsonl(Path(args.predictions))
        audited_predictions = audit_predictions(
            prediction_rows,
            benchmark_by_id,
            chunk_by_id,
            chunk_by_file_chunk,
        )
        write_jsonl(output_dir / "predictions_audit.jsonl", audited_predictions)
        write_csv(
            output_dir / "predictions_failures.csv",
            [row for row in audited_predictions if not bool(row.get("is_correct", False))],
            [
                "id",
                "file_name",
                "is_correct",
                "failure_category",
                "failure_reason",
                "question",
                "gold_answer",
                "predicted_answer",
            ],
        )
        summary["predictions"] = summarize_predictions(audited_predictions)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
