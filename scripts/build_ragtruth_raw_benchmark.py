import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def infer_prompt(task_type: str, source_name: str) -> str:
    task = normalize_ws(task_type).lower()
    source = normalize_ws(source_name)
    if "summary" in task:
        return "Summarize the following news."
    if "qa" in task or "question" in task:
        return f"Answer the question using the following source from {source or 'RAGTruth'}."
    return f"Respond using only the following source from {source or 'RAGTruth'}."


def normalize_label_type(value: str) -> str:
    return normalize_ws(value).lower().replace(" ", "_")


def build_row(response_row: dict, source_row: dict) -> dict:
    labels = response_row.get("labels", []) or []
    hallucinated_spans = [
        normalize_ws(label.get("text", ""))
        for label in labels
        if normalize_ws(label.get("text", ""))
    ]
    hallucination_types = sorted(
        {
            normalize_label_type(label.get("label_type", ""))
            for label in labels
            if normalize_label_type(label.get("label_type", ""))
        }
    )
    source_name = normalize_ws(source_row.get("source", "RAGTruth")) or "RAGTruth"
    raw_task = normalize_ws(source_row.get("task_type", "Summary")) or "Summary"
    context = normalize_ws(source_row.get("source_info", ""))

    return {
        "id": str(response_row.get("id", "")),
        "question": infer_prompt(raw_task, source_name),
        "fixed_context": context,
        "reference_response": normalize_ws(response_row.get("response", "")),
        "response_has_hallucination": bool(labels),
        "hallucinated_spans": hallucinated_spans,
        "hallucination_types": hallucination_types,
        "source_name": source_name,
        "raw_task": raw_task,
        "task_type": "grounded_qa",
        "benchmark_type": "ragtruth_fixed_context",
        "split": normalize_ws(response_row.get("split", "")),
        "model": normalize_ws(response_row.get("model", "")),
        "quality": normalize_ws(response_row.get("quality", "")),
        "source_id": str(response_row.get("source_id", "")),
        "raw_labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-context RAGTruth benchmark files from raw source_info/response JSONL files.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source-info-file", default=str(Path("/Users/eshanasir/RAGTruth/dataset/source_info.jsonl")))
    parser.add_argument("--response-file", default=str(Path("/Users/eshanasir/RAGTruth/dataset/response.jsonl")))
    parser.add_argument("--output-dir", default=str(root))
    parser.add_argument("--train-name", default="ragtruth_train.jsonl")
    parser.add_argument("--test-name", default="ragtruth_test.jsonl")
    parser.add_argument("--all-name", default="ragtruth_all.jsonl")
    args = parser.parse_args()

    source_rows = load_jsonl(Path(args.source_info_file))
    response_rows = load_jsonl(Path(args.response_file))
    source_by_id = {
        str(row.get("source_id", "")).strip(): row
        for row in source_rows
        if str(row.get("source_id", "")).strip()
    }

    adapted_rows: list[dict] = []
    skipped_missing_source = 0
    skipped_empty_context = 0

    for response_row in response_rows:
        source_id = str(response_row.get("source_id", "")).strip()
        if not source_id or source_id not in source_by_id:
            skipped_missing_source += 1
            continue
        row = build_row(response_row, source_by_id[source_id])
        if not row["fixed_context"] or not row["reference_response"]:
            skipped_empty_context += 1
            continue
        adapted_rows.append(row)

    train_rows = [row for row in adapted_rows if row.get("split") == "train"]
    test_rows = [row for row in adapted_rows if row.get("split") == "test"]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / args.all_name, adapted_rows)
    write_jsonl(output_dir / args.train_name, train_rows)
    write_jsonl(output_dir / args.test_name, test_rows)

    summary = {
        "source_info_file": str(Path(args.source_info_file)),
        "response_file": str(Path(args.response_file)),
        "output_dir": str(output_dir),
        "rows_written": len(adapted_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "skipped_missing_source": skipped_missing_source,
        "skipped_empty_context": skipped_empty_context,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
