import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a fixed-context QReCC benchmark file from saved evaluation predictions."
    )
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.predictions_file))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "gold_answer": row.get("gold_answer", ""),
                "acceptable_answers": [],
                "fixed_context": row.get("fixed_context", ""),
                "supporting_evidence": row.get("fixed_context", ""),
                "benchmark_type": "qrecc_fixed_context",
                "task_type": "qa_generation",
                "gold_retrieval_ids": [],
            }
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in output_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
