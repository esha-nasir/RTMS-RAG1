import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_lines(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    for i, row in enumerate(rows, 1):
        lines.append(f"[{i}] ID: {row.get('id', '')}")
        lines.append(f"Question: {row.get('question', '')}")
        if "gold_answer" in row:
            lines.append(f"Gold Answer: {row.get('gold_answer', '')}")
        lines.append(f"Predicted Answer: {row.get('predicted_answer', '')}")
        if "gold_has_hallucination" in row:
            lines.append(f"Gold Hallucination: {row.get('gold_has_hallucination')}")
        if "predicted_has_hallucination" in row:
            lines.append(f"Predicted Hallucination: {row.get('predicted_has_hallucination')}")
        qa_score = row.get("qa_score")
        if isinstance(qa_score, dict) and qa_score:
            lines.append(f"Answer F1: {qa_score.get('best_answer_f1')}")
            lines.append(f"Exact Match: {qa_score.get('exact_match')}")
        retrieval_proxy = row.get("retrieval_proxy")
        if isinstance(retrieval_proxy, dict) and retrieval_proxy:
            lines.append(f"Retrieval Proxy Hit: {retrieval_proxy.get('hit')}")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed-context predictions JSONL into a readable text file.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file) if args.output_file else input_path.with_name("questions_answers_readable.txt")

    rows = load_jsonl(input_path)
    output_path.write_text("\n".join(build_lines(rows)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
