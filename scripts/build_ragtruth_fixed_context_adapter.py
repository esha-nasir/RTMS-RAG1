import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def get_field(row: dict, path: str, default=None):
    current = row
    for part in str(path or "").split("."):
        if not part:
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped == "[]":
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
    return [value]


def infer_task_type(raw_task: str, default_task_type: str) -> str:
    task = str(raw_task or "").strip().lower()
    if not task:
        return default_task_type
    if "summ" in task:
        return "grounded_qa"
    if "data" in task and "text" in task:
        return "grounded_qa"
    if "qa" in task or "question" in task:
        return "grounded_qa"
    return default_task_type


def infer_hallucination_label(row: dict, label_field: str, spans: list[str]) -> bool | None:
    response_label = get_field(row, label_field, None)
    if isinstance(response_label, bool):
        return response_label
    if isinstance(response_label, (int, float)):
        return bool(response_label)
    if isinstance(response_label, str):
        lowered = response_label.strip().lower()
        if lowered in {"yes", "true", "hallucinated", "has_hallucination", "1"}:
            return True
        if lowered in {"no", "false", "grounded", "0"}:
            return False
    if spans:
        return True
    processed = get_field(row, "hallucination_labels_processed", None)
    if isinstance(processed, dict):
        try:
            return any(bool(v) for v in processed.values())
        except Exception:
            return None
    return False


def infer_hallucination_types(row: dict, types_field: str) -> list[str]:
    explicit = [normalize_ws(item) for item in as_list(get_field(row, types_field, [])) if normalize_ws(item)]
    if explicit:
        return explicit
    processed = get_field(row, "hallucination_labels_processed", None)
    if isinstance(processed, dict):
        return [str(k) for k, v in processed.items() if v]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt RAGTruth-style data into a fixed-context benchmark format.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input-file", required=True, help="Path to raw RAGTruth JSON or JSONL data.")
    parser.add_argument("--output-dir", default=str(root / "data" / "ragtruth_fixed_context"))
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--context-field", default="context")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--task-field", default="task")
    parser.add_argument("--label-field", default="response_has_hallucination")
    parser.add_argument("--spans-field", default="hallucinated_spans")
    parser.add_argument("--types-field", default="hallucination_types")
    parser.add_argument("--source-field", default="source_name")
    parser.add_argument("--default-task-type", default="grounded_qa")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")

    if input_path.suffix.lower() == ".jsonl":
        raw_rows = load_jsonl(input_path)
    else:
        payload = load_json(input_path)
        if isinstance(payload, list):
            raw_rows = payload
        elif isinstance(payload, dict):
            raw_rows = payload.get("data") or payload.get("rows") or payload.get("examples") or []
        else:
            raw_rows = []

    adapted_rows: list[dict] = []
    for idx, row in enumerate(raw_rows, 1):
        if not isinstance(row, dict):
            continue
        question = normalize_ws(get_field(row, args.question_field, ""))
        context = normalize_ws(get_field(row, args.context_field, "") or get_field(row, "fixed_context", ""))
        reference_response = normalize_ws(
            get_field(row, args.response_field, "") or get_field(row, "reference_response", "")
        )
        raw_task = normalize_ws(get_field(row, args.task_field, ""))
        source_name = normalize_ws(get_field(row, args.source_field, "RAGTruth"))
        hallucinated_spans = [normalize_ws(item) for item in as_list(get_field(row, args.spans_field, [])) if normalize_ws(item)]
        hallucination_types = infer_hallucination_types(row, args.types_field)
        if not question or not context:
            continue

        response_has_hallucination = infer_hallucination_label(row, args.label_field, hallucinated_spans)

        adapted_rows.append(
            {
                "id": str(get_field(row, args.id_field, f"ragtruth_{idx}")),
                "question": question,
                "fixed_context": context,
                "reference_response": reference_response,
                "response_has_hallucination": response_has_hallucination,
                "hallucinated_spans": hallucinated_spans,
                "hallucination_types": hallucination_types,
                "source_name": source_name or "RAGTruth",
                "raw_task": raw_task,
                "task_type": infer_task_type(raw_task, args.default_task_type),
                "benchmark_type": "ragtruth_fixed_context",
                "needs_manual_review": True,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "ragtruth_fixed_context_benchmark.jsonl"
    write_jsonl(benchmark_path, adapted_rows)

    summary = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "rows_written": len(adapted_rows),
        "positive_labels": sum(1 for row in adapted_rows if row.get("response_has_hallucination") is True),
        "negative_labels": sum(1 for row in adapted_rows if row.get("response_has_hallucination") is False),
        "benchmark_file": str(benchmark_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
