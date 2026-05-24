import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_prediction_map(path: Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    return {str(row.get("id", "")): row for row in rows if str(row.get("id", "")).strip()}


def _mark_answer(row: dict) -> str:
    answer = str(row.get("predicted_answer", "") or "").strip()
    if not answer:
        return ""
    correctness = "Correct" if bool(row.get("is_correct", False)) else "Incorrect"
    grounding = "Hallucinated" if bool(row.get("is_hallucinated", False)) else "Grounded"
    return f"{answer}\n[{correctness}; {grounding}]"


def _escape_md(text: str) -> str:
    clean = str(text or "").replace("\n", "<br>")
    return clean.replace("|", "\\|")


def parse_prediction_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"Invalid --prediction value: {spec}. Expected LABEL=/path/to/predictions.jsonl")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Invalid --prediction label in: {spec}")
    return label, Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a markdown comparison table across multiple prediction files.")
    parser.add_argument("--benchmark-file", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="Column label and prediction file in the form LABEL=/abs/path/to/local_benchmark_predictions.jsonl",
    )
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    benchmark_rows = load_jsonl(Path(args.benchmark_file))
    prediction_specs = [parse_prediction_spec(spec) for spec in args.prediction]

    header = ["Question", "Gold Answer"] + [label for label, _ in prediction_specs]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    prediction_maps = [(label, load_prediction_map(path)) for label, path in prediction_specs]

    for row in benchmark_rows:
        row_id = str(row.get("id", ""))
        cells = [
            _escape_md(row.get("question", "")),
            _escape_md(row.get("gold_answer", "")),
        ]
        for _, prediction_map in prediction_maps:
            cells.append(_escape_md(_mark_answer(prediction_map.get(row_id, {}))))
        lines.append("| " + " | ".join(cells) + " |")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
