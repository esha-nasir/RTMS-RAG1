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


def build_rows(benchmark_rows: list[dict], no_rag: dict[str, dict], no_memory: dict[str, dict], memory: dict[str, dict]) -> list[str]:
    lines = [
        "| Question | Gold Answer | Answer Without RAG | Answer With RAG, No Memory | Answer With RAG, With Memory |",
        "|---|---|---|---|---|",
    ]
    for row in benchmark_rows:
        row_id = str(row.get("id", ""))
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(row.get("question", "")),
                    _escape_md(row.get("gold_answer", "")),
                    _escape_md(_mark_answer(no_rag.get(row_id, {}))),
                    _escape_md(_mark_answer(no_memory.get(row_id, {}))),
                    _escape_md(_mark_answer(memory.get(row_id, {}))),
                ]
            )
            + " |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a markdown comparison table across no-RAG, no-memory, and memory QA predictions.")
    parser.add_argument("--benchmark-file", required=True)
    parser.add_argument("--no-rag-predictions", required=True)
    parser.add_argument("--no-memory-predictions", required=True)
    parser.add_argument("--memory-predictions", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    benchmark_rows = load_jsonl(Path(args.benchmark_file))
    no_rag_map = load_prediction_map(Path(args.no_rag_predictions))
    no_memory_map = load_prediction_map(Path(args.no_memory_predictions))
    memory_map = load_prediction_map(Path(args.memory_predictions))

    table_lines = build_rows(benchmark_rows, no_rag_map, no_memory_map, memory_map)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
