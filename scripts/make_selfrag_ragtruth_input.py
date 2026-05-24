import argparse
import json
from pathlib import Path


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RAGTruth fixed-context examples to Self-RAG ASQA-style input."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=301)
    parser.add_argument("--ndocs", type=int, default=3)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    converted: list[dict] = []
    for row in rows:
        question = normalize_ws(row.get("question", ""))
        evidence = normalize_ws(
            row.get("fixed_context", "")
            or row.get("supporting_evidence", "")
            or row.get("gold_chunk_text", "")
        )
        if not question or not evidence:
            continue

        docs = [
            {
                "title": normalize_ws(row.get("source_name", "RAGTruth")),
                "text": evidence,
            }
        ]
        while len(docs) < max(1, args.ndocs):
            docs.append({"title": "Empty", "text": ""})

        converted.append(
            {
                "id": str(row.get("id", len(converted))),
                "question": question,
                "docs": docs[: args.ndocs],
                "reference_response": normalize_ws(row.get("reference_response", "")),
                "response_has_hallucination": bool(row.get("response_has_hallucination", False)),
                "hallucinated_spans": row.get("hallucinated_spans", []) or [],
                "source_name": normalize_ws(row.get("source_name", "RAGTruth")),
            }
        )

    write_jsonl(Path(args.output), converted)
    print(f"Wrote {len(converted)} Self-RAG examples to {args.output}")


if __name__ == "__main__":
    main()
