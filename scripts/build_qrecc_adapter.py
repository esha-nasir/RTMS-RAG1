import argparse
import json
from pathlib import Path


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def parse_id_parts(raw_id: str) -> tuple[str, int | None]:
    text = normalize_ws(raw_id)
    if "_" not in text:
        return text, None
    conversation_id, _, turn_str = text.partition("_")
    try:
        return conversation_id, int(turn_str)
    except ValueError:
        return conversation_id, None


def flatten_evidence(evidence: dict) -> str:
    parts: list[str] = []
    if not isinstance(evidence, dict):
        return ""
    for source, spans in evidence.items():
        if isinstance(spans, list):
            for span in spans:
                text = normalize_ws(span)
                if text:
                    parts.append(text)
        else:
            text = normalize_ws(spans)
            if text:
                parts.append(text)
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SCAI-QReCC JSON into the JSONL schema used by run_multi_codebase_rag_table.py.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-type", default="grounded_qa")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"Expected a JSON list in {input_path}")

    if args.limit > 0:
        rows = rows[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            raw_id = str(row.get("ID") or f"qrecc_{idx}")
            conversation_id, turn_id = parse_id_parts(raw_id)
            question = normalize_ws(row.get("Question", ""))
            answers = row.get("Answers", [])
            gold_answer = normalize_ws(answers[0] if isinstance(answers, list) and answers else "")
            evidence = row.get("Evidence", {})
            evidence_text = flatten_evidence(evidence)
            evidence_sources = [normalize_ws(source) for source in evidence.keys()] if isinstance(evidence, dict) else []
            benchmark_row = {
                "id": raw_id,
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "question": question,
                "original_question": question,
                "rewrite_target": "",
                "gold_answer": gold_answer,
                "acceptable_answers": [normalize_ws(item) for item in answers[1:] if normalize_ws(item)],
                "evidence_sources": [item for item in evidence_sources if item],
                "gold_retrieval_ids": [item for item in evidence_sources if item],
                "raw_evidence": evidence if isinstance(evidence, dict) else {},
                "supporting_evidence": evidence_text,
                "fixed_context": evidence_text,
                "file_name": "QReCC",
                "case_no": "",
                "task_type": args.task_type,
                "benchmark_type": "qrecc_fixed_context",
                "adapter_notes": "Raw SCAI-QReCC does not expose standalone rewrite targets in this file; rewrite_target is left blank.",
            }
            handle.write(json.dumps(benchmark_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
