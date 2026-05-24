import argparse
import json
import re
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize(text)))


def claim_reappears(claim: str, text: str, threshold: float) -> bool:
    claim_norm = normalize(claim)
    text_norm = normalize(text)
    if not claim_norm or not text_norm:
        return False
    if claim_norm in text_norm:
        return True
    claim_tokens = tokens(claim_norm)
    text_tokens = tokens(text_norm)
    if not claim_tokens:
        return False
    overlap = len(claim_tokens & text_tokens) / len(claim_tokens)
    return overlap >= threshold


def trace_payload(row: dict) -> dict:
    trace = row.get("trace") or {}
    return trace.get("trace") or trace


def later_stage_texts(row: dict, first_attempt_index: int) -> dict[str, str]:
    trace = trace_payload(row)
    attempts = trace.get("attempts") or []
    answer_parts = []
    reflection_parts = []
    memory_parts = []

    for attempt in attempts[first_attempt_index + 1 :]:
        answer_parts.append(attempt.get("answer", ""))
        reflection_parts.append(attempt.get("reflection", ""))
        multi = attempt.get("multi_level_reflection") or {}
        reflection_parts.extend(str(v) for v in multi.values())
        skeptic = attempt.get("skeptic") or {}
        reflection_parts.append(skeptic.get("guidance", ""))
        reflection_parts.extend(str(v) for v in skeptic.get("unsupported_claims", []) or [])

    answer_parts.append(row.get("predicted_answer", ""))
    episode = trace.get("episode_reflection") or {}
    reflection_parts.extend(str(v) for v in episode.values())
    memory = trace.get("memory_entry") or {}
    memory_parts.append(memory.get("exact_answer_span", ""))
    for key in ("final_reflection", "last_guidance"):
        reflection_parts.append(memory.get(key, ""))
    multi_memory = memory.get("multi_level_reflection") or {}
    reflection_parts.extend(str(v) for v in multi_memory.values())

    return {
        "answer": "\n".join(part for part in answer_parts if part),
        "reflection": "\n".join(part for part in reflection_parts if part),
        "memory": "\n".join(part for part in memory_parts if part),
        "all": "\n".join(part for part in answer_parts + reflection_parts + memory_parts if part),
    }


def compute_hpr(path: Path, threshold: float) -> dict:
    records = 0
    records_with_initial_unsupported = 0
    initial_claims = 0
    repeated_answer_claims = 0
    repeated_reflection_claims = 0
    repeated_memory_claims = 0
    repeated_any_claims = 0

    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        records += 1
        row = json.loads(line)
        trace = trace_payload(row)
        attempts = trace.get("attempts") or []

        first_idx = None
        first_claims = []
        for idx, attempt in enumerate(attempts):
            skeptic = attempt.get("skeptic") or {}
            claims = [str(c).strip() for c in skeptic.get("unsupported_claims", []) or [] if str(c).strip()]
            if claims:
                first_idx = idx
                first_claims = claims
                break

        if first_idx is None:
            continue

        records_with_initial_unsupported += 1
        initial_claims += len(first_claims)
        later_text = later_stage_texts(row, first_idx)
        for claim in first_claims:
            in_answer = claim_reappears(claim, later_text["answer"], threshold=threshold)
            in_reflection = claim_reappears(claim, later_text["reflection"], threshold=threshold)
            in_memory = claim_reappears(claim, later_text["memory"], threshold=threshold)
            if in_answer:
                repeated_answer_claims += 1
            if in_reflection:
                repeated_reflection_claims += 1
            if in_memory:
                repeated_memory_claims += 1
            if in_answer or in_reflection or in_memory:
                repeated_any_claims += 1

    hpr_answer = repeated_answer_claims / initial_claims if initial_claims else None
    hpr_reflection = repeated_reflection_claims / initial_claims if initial_claims else None
    hpr_memory = repeated_memory_claims / initial_claims if initial_claims else None
    hpr_any = repeated_any_claims / initial_claims if initial_claims else None
    return {
        "prediction_file": str(path),
        "records": records,
        "records_with_initial_unsupported": records_with_initial_unsupported,
        "initial_unsupported_claims": initial_claims,
        "repeated_answer_claims": repeated_answer_claims,
        "repeated_reflection_claims": repeated_reflection_claims,
        "repeated_memory_claims": repeated_memory_claims,
        "repeated_any_claims": repeated_any_claims,
        "hpr_answer": hpr_answer,
        "hpr_reflection": hpr_reflection,
        "hpr_memory": hpr_memory,
        "hpr_any": hpr_any,
        "match_threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Hallucination Propagation Rate from temporal RAG traces.")
    parser.add_argument("prediction_file", help="Path to fixed_context_predictions.jsonl")
    parser.add_argument("--threshold", type=float, default=0.75, help="Token-overlap threshold for repeated claims.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    result = compute_hpr(Path(args.prediction_file), threshold=args.threshold)
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
