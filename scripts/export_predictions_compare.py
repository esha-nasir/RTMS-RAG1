import argparse
import csv
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


def index_by_id(rows: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        indexed[str(row.get("id", ""))] = row
    return indexed


def fmt_score(row: dict) -> str:
    qa_score = row.get("qa_score")
    if not isinstance(qa_score, dict) or not qa_score:
        return ""
    return f"Answer F1={qa_score.get('best_answer_f1')} | Exact Match={qa_score.get('exact_match')}"


def fmt_hall(row: dict) -> str:
    parts = []
    if "gold_has_hallucination" in row:
        parts.append(f"Gold Hall={row.get('gold_has_hallucination')}")
    if "predicted_has_hallucination" in row:
        parts.append(f"Pred Hall={row.get('predicted_has_hallucination')}")
    return " | ".join(parts)


def fmt_retrieval(row: dict) -> str:
    retrieval = row.get("retrieval_proxy")
    if not isinstance(retrieval, dict) or not retrieval:
        return ""
    return f"Retrieval Proxy Hit={retrieval.get('hit')}"


def scalar_qa_score(row: dict, key: str):
    qa_score = row.get("qa_score")
    if not isinstance(qa_score, dict):
        return ""
    return qa_score.get(key, "")


def scalar_retrieval_hit(row: dict):
    retrieval = row.get("retrieval_proxy")
    if not isinstance(retrieval, dict):
        return ""
    return retrieval.get("hit", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a side-by-side readable comparison of two fixed-context prediction JSONL files.")
    parser.add_argument("--left-file", required=True)
    parser.add_argument("--right-file", required=True)
    parser.add_argument("--left-label", default="Left")
    parser.add_argument("--right-label", default="Right")
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    left_rows = load_jsonl(Path(args.left_file))
    right_rows = load_jsonl(Path(args.right_file))
    left_by_id = index_by_id(left_rows)
    right_by_id = index_by_id(right_rows)

    ordered_ids = []
    seen = set()
    for row in left_rows + right_rows:
        row_id = str(row.get("id", ""))
        if row_id in seen:
            continue
        seen.add(row_id)
        ordered_ids.append(row_id)

    output_path = Path(args.output_file)

    if output_path.suffix.lower() == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "question",
                    "gold_answer",
                    f"{args.left_label.lower()}_answer",
                    f"{args.left_label.lower()}_answer_f1",
                    f"{args.left_label.lower()}_exact_match",
                    f"{args.left_label.lower()}_gold_hallucination",
                    f"{args.left_label.lower()}_predicted_hallucination",
                    f"{args.left_label.lower()}_retrieval_proxy_hit",
                    f"{args.right_label.lower()}_answer",
                    f"{args.right_label.lower()}_answer_f1",
                    f"{args.right_label.lower()}_exact_match",
                    f"{args.right_label.lower()}_gold_hallucination",
                    f"{args.right_label.lower()}_predicted_hallucination",
                    f"{args.right_label.lower()}_retrieval_proxy_hit",
                ],
            )
            writer.writeheader()
            for row_id in ordered_ids:
                left = left_by_id.get(row_id, {})
                right = right_by_id.get(row_id, {})
                writer.writerow(
                    {
                        "id": row_id,
                        "question": left.get("question") or right.get("question") or "",
                        "gold_answer": left.get("gold_answer") or right.get("gold_answer") or "",
                        f"{args.left_label.lower()}_answer": left.get("predicted_answer", ""),
                        f"{args.left_label.lower()}_answer_f1": scalar_qa_score(left, "best_answer_f1"),
                        f"{args.left_label.lower()}_exact_match": scalar_qa_score(left, "exact_match"),
                        f"{args.left_label.lower()}_gold_hallucination": left.get("gold_has_hallucination", ""),
                        f"{args.left_label.lower()}_predicted_hallucination": left.get("predicted_has_hallucination", ""),
                        f"{args.left_label.lower()}_retrieval_proxy_hit": scalar_retrieval_hit(left),
                        f"{args.right_label.lower()}_answer": right.get("predicted_answer", ""),
                        f"{args.right_label.lower()}_answer_f1": scalar_qa_score(right, "best_answer_f1"),
                        f"{args.right_label.lower()}_exact_match": scalar_qa_score(right, "exact_match"),
                        f"{args.right_label.lower()}_gold_hallucination": right.get("gold_has_hallucination", ""),
                        f"{args.right_label.lower()}_predicted_hallucination": right.get("predicted_has_hallucination", ""),
                        f"{args.right_label.lower()}_retrieval_proxy_hit": scalar_retrieval_hit(right),
                    }
                )
        print(output_path)
        return

    lines: list[str] = []
    for idx, row_id in enumerate(ordered_ids, 1):
        left = left_by_id.get(row_id, {})
        right = right_by_id.get(row_id, {})
        question = left.get("question") or right.get("question") or ""
        gold_answer = left.get("gold_answer") or right.get("gold_answer") or ""

        lines.append(f"[{idx}] ID: {row_id}")
        lines.append(f"Question: {question}")
        if gold_answer:
            lines.append(f"Gold Answer: {gold_answer}")
        lines.append(f"{args.left_label} Answer: {left.get('predicted_answer', '')}")
        left_meta = " | ".join(part for part in [fmt_score(left), fmt_hall(left), fmt_retrieval(left)] if part)
        if left_meta:
            lines.append(f"{args.left_label} Metrics: {left_meta}")
        lines.append(f"{args.right_label} Answer: {right.get('predicted_answer', '')}")
        right_meta = " | ".join(part for part in [fmt_score(right), fmt_hall(right), fmt_retrieval(right)] if part)
        if right_meta:
            lines.append(f"{args.right_label} Metrics: {right_meta}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
