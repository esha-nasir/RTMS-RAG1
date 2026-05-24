import argparse
import json
import os
import sys
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
RAGTRUTH_MEMORY_PATH = str(ROOT / "ragtruth_fixed_context_memory.jsonl")
RAGTRUTH_MEMORY_STATS_PATH = str(ROOT / "ragtruth_fixed_context_memory.stats.jsonl")


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


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


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth and pred)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if not truth and pred)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth and not pred)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if not truth and not pred)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
    }


def spans_to_char_intervals(text: str, spans: list[str]) -> list[tuple[int, int]]:
    normalized_text = normalize_ws(text)
    lowered = normalized_text.lower()
    intervals: list[tuple[int, int]] = []
    for span in spans:
        snippet = normalize_ws(span).lower()
        if not snippet:
            continue
        start = lowered.find(snippet)
        if start >= 0:
            intervals.append((start, start + len(snippet)))
    return intervals


def interval_set(intervals: list[tuple[int, int]]) -> set[int]:
    chars: set[int] = set()
    for start, end in intervals:
        chars.update(range(max(start, 0), max(start, end)))
    return chars


def char_level_span_metrics(reference_text: str, gold_spans: list[str], predicted_spans: list[str]) -> dict:
    gold_chars = interval_set(spans_to_char_intervals(reference_text, gold_spans))
    pred_chars = interval_set(spans_to_char_intervals(reference_text, predicted_spans))
    tp = len(gold_chars & pred_chars)
    fp = len(pred_chars - gold_chars)
    fn = len(gold_chars - pred_chars)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gold_char_count": len(gold_chars),
        "pred_char_count": len(pred_chars),
    }


def build_context_record(row: dict) -> list[dict]:
    evidence_text = normalize_ws(
        row.get("fixed_context", "")
        or row.get("supporting_evidence", "")
        or row.get("gold_chunk_text", "")
    )
    return [
        {
            "id": row.get("id", ""),
            "file_name": row.get("source_name", "RAGTruth"),
            "chunk_id": "fixed_context",
            "case_no": "",
            "url": "",
            "text": evidence_text,
        }
    ]


def detect_response_hallucination(rag, question: str, answer: str, evidence: str) -> tuple[bool, str]:
    if hasattr(rag, "detect_hallucination"):
        result = rag.detect_hallucination(
            question=question,
            candidate_answer=answer,
            evidence=evidence,
            benchmark_family="ragtruth",
        )
        return bool(result.get("has_hallucination", False)), json.dumps(result, ensure_ascii=False)

    prompt = (
        "You are a hallucination detector for retrieval-augmented generation.\n"
        "Use only the provided evidence.\n"
        "Return exactly one token: Yes or No.\n"
        "Yes means the answer contains unsupported or contradictory information.\n"
        "No means the answer is fully supported by the evidence.\n\n"
        f"Question: {question}\n"
        f"Candidate Answer: {answer}\n"
        f"Evidence: {evidence}\n\n"
        "Output:"
    )
    raw = rag._call_completion(prompt, temperature=0.0, max_tokens=5)
    head = normalize_ws(raw[:20]).lower()
    is_hallucinated = head.startswith("yes")
    return is_hallucinated, raw


def detect_hallucinated_spans(rag, question: str, answer: str, evidence: str) -> tuple[list[str], str]:
    prompt = (
        "You are a span-level hallucination detector for retrieval-augmented generation.\n"
        "Use only the provided evidence.\n"
        "Identify unsupported or contradictory spans in the candidate answer.\n"
        "Return only JSON with key \"hallucination_list\" mapped to a list of exact spans copied from the candidate answer.\n"
        "If there is no hallucination, return {\"hallucination_list\": []}.\n\n"
        f"Question: {question}\n"
        f"Candidate Answer: {answer}\n"
        f"Evidence: {evidence}\n"
    )
    raw = rag._call_completion(prompt, temperature=0.0, max_tokens=220, json_mode=True)
    try:
        parsed = json.loads(raw)
        values = parsed.get("hallucination_list", [])
        if isinstance(values, list):
            return [normalize_ws(item) for item in values if normalize_ws(item)], raw
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw or "")
    if match:
        try:
            parsed = json.loads(match.group(0))
            values = parsed.get("hallucination_list", [])
            if isinstance(values, list):
                return [normalize_ws(item) for item in values if normalize_ws(item)], raw
        except Exception:
            pass
    return [], raw


def generate_from_fixed_context(rag, row: dict, use_reflexion: bool, use_memory: bool, max_iters: int) -> tuple[str, dict]:
    question = str(row.get("question", ""))
    contexts = build_context_record(row)
    context_text = rag._build_context_text(contexts)
    primary_context_text = rag._build_primary_context_text(contexts)
    task_type = str(row.get("task_type", "grounded_qa") or "grounded_qa")
    strict_extractive = task_type == "extractive_qa"
    constrained_grounding = rag._use_constrained_grounding_mode(
        task_type=task_type,
        file_name=str(row.get("source_name", "RAGTruth")),
        case_no="RAGTruth",
    )

    if not use_reflexion:
        if strict_extractive:
            answer_text = rag._extractive_span_answer(question, primary_context_text)
        else:
            prompt = rag._answer_prompt(
                question,
                context_text,
                memory_text="",
                reflection_hint="",
                primary_context_text=primary_context_text,
                strict_extractive=False,
                constrained_grounding=constrained_grounding,
            )
            answer_text = rag._call_completion(
                prompt,
                temperature=0.0 if constrained_grounding else 0.2,
                max_tokens=900,
            )
        return answer_text, {"mode": "fixed_context_direct", "memory_hits": 0, "attempts": []}

    episodic = rag._memory_for_query(
        question,
        top_n=3,
        case_no=None,
        file_name=str(row.get("source_name", "RAGTruth")),
        task_type=task_type,
    ) if use_memory else []
    memory_text = rag._format_memory_snippets(episodic)

    trace_attempts = []
    final_answer = ""
    reflection_hint = ""
    for i in range(1, max(1, max_iters) + 1):
        if strict_extractive:
            answer_text = rag._extractive_span_answer(question, primary_context_text)
        else:
            prompt = rag._answer_prompt(
                question,
                context_text,
                memory_text=memory_text,
                reflection_hint=reflection_hint,
                primary_context_text=primary_context_text,
                strict_extractive=False,
                constrained_grounding=constrained_grounding,
            )
            answer_text = rag._call_completion(
                prompt,
                temperature=0.0 if constrained_grounding else 0.2,
                max_tokens=900,
            )

        critique = rag._run_critic(
            question,
            answer_text,
            context_text,
            strict_extractive=strict_extractive,
            task_type=task_type,
        )
        if rag._should_preserve_abstention(answer_text, contexts, strict_extractive):
            critique["is_sufficient"] = True
            critique["issues"] = []
            critique["guidance"] = "Preserve abstention because the fixed context is insufficient."

        reflection = ""
        multi_level_reflection = {
            "retrieval_level": "",
            "answer_level": "",
            "combined_guidance": critique.get("guidance", "").strip(),
        }
        if not critique.get("is_sufficient", False):
            if not rag.REFLEXION_FAST_MODE and rag.REFLEXION_ENABLE_SELF_REFLECTION:
                reflection = rag._call_completion(
                    rag._reflect_prompt(question, answer_text, critique),
                    temperature=0.1,
                    max_tokens=220,
                )
            if not rag.REFLEXION_FAST_MODE and rag.REFLEXION_ENABLE_MULTI_LEVEL_REFLECTION:
                multi_level_reflection = rag._build_multi_level_reflection(question, answer_text, contexts, critique)
            reflection_hint = "\n".join(
                part
                for part in [
                    critique.get("guidance", "").strip(),
                    multi_level_reflection.get("retrieval_level", "").strip(),
                    multi_level_reflection.get("answer_level", "").strip(),
                    reflection.strip(),
                ]
                if part
            ).strip()

        trace_attempts.append(
            {
                "iter": i,
                "answer": answer_text,
                "critique": critique,
                "reflection": reflection,
                "multi_level_reflection": multi_level_reflection,
                "context_count": 1,
            }
        )
        final_answer = answer_text
        if critique.get("is_sufficient", False):
            break

    final_success = rag._is_grounding_sufficient(
        final_answer,
        trace_attempts[-1]["critique"] if trace_attempts else {},
        strict_extractive=strict_extractive,
        task_type=task_type,
    ) if trace_attempts else False
    if use_memory:
        rag._update_memory_outcomes(episodic, is_success=final_success, path=RAGTRUTH_MEMORY_STATS_PATH)
        episode_reflection = rag._build_episode_reflection(question, trace_attempts)
        memory_entry = {
            "question": question,
            "task_type": task_type,
            "case_no": "",
            "file_name": str(row.get("source_name", "RAGTruth")),
            "is_sufficient": final_success,
            "origin_success": final_success,
            "success_count": 1 if final_success else 0,
            "failure_count": 0 if final_success else 1,
            "alpha": rag.REFLEXION_MEMORY_PRIOR_ALPHA + (1 if final_success else 0),
            "beta": rag.REFLEXION_MEMORY_PRIOR_BETA + (0 if final_success else 1),
            "hallucination_risk": 0.0 if final_success else 1.0,
            "final_reflection": episode_reflection.get("summary", ""),
            "last_guidance": episode_reflection.get("next_action", ""),
            "issues": trace_attempts[-1]["critique"].get("issues", []) if trace_attempts else [],
            "multi_level_reflection": {
                "retrieval_level": trace_attempts[-1].get("multi_level_reflection", {}).get("retrieval_level", "") if trace_attempts else "",
                "answer_level": trace_attempts[-1].get("multi_level_reflection", {}).get("answer_level", "") if trace_attempts else "",
                "episode_level": episode_reflection.get("summary", ""),
            },
        }
        rag._append_memory(memory_entry, path=RAGTRUTH_MEMORY_PATH)

    trace = {
        "mode": "fixed_context_reflexion",
        "memory_hits": len(episodic),
        "selected_memories": [
            {
                "memory_id": item.get("memory_id", ""),
                "score": item.get("memory_score", 0.0),
                "question": item.get("question", ""),
                "success_count": item.get("success_count", 0),
                "failure_count": item.get("failure_count", 0),
            }
            for item in episodic
        ],
        "attempts": trace_attempts,
    }
    return final_answer, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAGTruth-style fixed-context data separately from the legal corpus pipeline.")
    parser.add_argument("--benchmark-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["detect", "generate"], default="generate")
    parser.add_argument("--use-reflexion", action="store_true")
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("REFLEXION_MEMORY_PATH", RAGTRUTH_MEMORY_PATH)
    os.environ.setdefault("REFLEXION_MEMORY_STATS_PATH", RAGTRUTH_MEMORY_STATS_PATH)
    os.environ["REFLEXION_BENCHMARK_NO_RETRIEVAL"] = "1"
    os.environ["REFLEXION_DISABLE_MEMORY"] = "1" if args.disable_memory else "0"

    import rtms_rag.rag as rag  # imported after env setup on purpose

    rows = load_jsonl(Path(args.benchmark_file))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit(f"No rows loaded from {args.benchmark_file}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true: list[bool] = []
    y_pred: list[bool] = []
    span_metric_rows: list[dict] = []
    records: list[dict] = []
    generated_hallucinations = 0

    for row in rows:
        question = str(row.get("question", ""))
        context = normalize_ws(
            row.get("fixed_context", "")
            or row.get("supporting_evidence", "")
            or row.get("gold_chunk_text", "")
        )
        if args.mode == "detect":
            candidate = normalize_ws(row.get("reference_response", ""))
            if not candidate:
                continue
            predicted_hallucinated, detector_raw = detect_response_hallucination(rag, question, candidate, context)
            predicted_spans, span_raw = detect_hallucinated_spans(rag, question, candidate, context)
            truth = row.get("response_has_hallucination")
            if truth is not None:
                y_true.append(bool(truth))
                y_pred.append(predicted_hallucinated)
            span_metric = char_level_span_metrics(candidate, row.get("hallucinated_spans", []), predicted_spans)
            span_metric_rows.append(span_metric)
            records.append(
                {
                    "id": row.get("id", ""),
                    "mode": "detect",
                    "question": question,
                    "candidate_answer": candidate,
                    "response_has_hallucination": truth,
                    "predicted_has_hallucination": predicted_hallucinated,
                    "gold_spans": row.get("hallucinated_spans", []),
                    "predicted_spans": predicted_spans,
                    "detector_raw": detector_raw,
                    "span_raw": span_raw,
                    "span_metrics": span_metric,
                }
            )
        else:
            answer, trace = generate_from_fixed_context(
                rag,
                row,
                use_reflexion=args.use_reflexion,
                use_memory=not args.disable_memory,
                max_iters=args.max_iters,
            )
            predicted_hallucinated, detector_raw = detect_response_hallucination(rag, question, answer, context)
            predicted_spans, span_raw = detect_hallucinated_spans(rag, question, answer, context)
            if predicted_hallucinated:
                generated_hallucinations += 1
            records.append(
                {
                    "id": row.get("id", ""),
                    "mode": "generate",
                    "question": question,
                    "generated_answer": answer,
                    "predicted_has_hallucination": predicted_hallucinated,
                    "predicted_spans": predicted_spans,
                    "detector_raw": detector_raw,
                    "span_raw": span_raw,
                    "reflexion_trace": trace,
                    "source_name": row.get("source_name", "RAGTruth"),
                }
            )

    predictions_path = output_dir / "ragtruth_fixed_context_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "config": {
            "mode": args.mode,
            "use_reflexion": args.use_reflexion,
            "disable_memory": args.disable_memory,
            "max_iters": args.max_iters,
            "max_samples": args.max_samples,
        },
        "total_records": len(records),
    }
    if args.mode == "detect":
        summary["response_level"] = classification_metrics(y_true, y_pred) if y_true else {}
        if span_metric_rows:
            summary["span_level"] = {
                "precision": round(sum(row["precision"] for row in span_metric_rows) / len(span_metric_rows), 4),
                "recall": round(sum(row["recall"] for row in span_metric_rows) / len(span_metric_rows), 4),
                "f1": round(sum(row["f1"] for row in span_metric_rows) / len(span_metric_rows), 4),
            }
    else:
        summary["generation"] = {
            "hallucination_rate": round(safe_div(generated_hallucinations, len(records)), 4),
            "non_hallucination_rate": round(safe_div(len(records) - generated_hallucinations, len(records)), 4),
        }

    (output_dir / "ragtruth_fixed_context_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
