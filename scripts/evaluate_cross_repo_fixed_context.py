import argparse
import inspect
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]

BACKEND_REPOS = {
    "vanilla": Path("/Users/eshanasir/vanillaRAG"),
    "selfrefine": Path("/Users/eshanasir/selfRefineAgentRAG"),
    "reflexion": Path("/Users/eshanasir/reflexionAgentRAG"),
    "temporal": Path("/Users/eshanasir/ReflexionTemporalMemorySuccessAgentRAG"),
}


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_text(text: str) -> str:
    return normalize_ws(str(text or "")).lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_text(text))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def token_f1(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    sa = set(ta)
    sb = set(tb)
    common = len(sa & sb)
    if common == 0:
        return 0.0
    precision = common / len(sa)
    recall = common / len(sb)
    return 2 * precision * recall / (precision + recall)


def classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth and pred)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if not truth and pred)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth and not pred)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if not truth and not pred)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "accuracy": round(accuracy, 4),
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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(value: str) -> str:
    cleaned = (value or "default").strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", cleaned)
    return cleaned or "default"


def memory_file_for(benchmark_name: str, task_type: str) -> Path:
    memory_dir = Path((os.getenv("REFLEXION_MEMORY_DIR", "") or ".").strip())
    return memory_dir / f"{safe_name(benchmark_name)}.{safe_name(task_type)}.memory.jsonl"


def trace_memory_entry(trace: dict) -> dict:
    if not isinstance(trace, dict):
        return {}
    inner = trace.get("trace") if isinstance(trace.get("trace"), dict) else trace
    memory_entry = inner.get("memory_entry") if isinstance(inner, dict) else None
    return memory_entry if isinstance(memory_entry, dict) else {}


def update_memory_from_external_eval(
    trace: dict,
    benchmark_name: str,
    task_type: str,
    success: bool,
    external_eval: dict,
) -> bool:
    memory_entry = trace_memory_entry(trace)
    memory_id = str(memory_entry.get("memory_id", "")).strip()
    if not memory_id:
        return False
    memory_path = memory_file_for(benchmark_name, task_type)
    if not memory_path.exists():
        return False

    rows = load_jsonl(memory_path)
    changed = False
    for row in rows:
        if str(row.get("memory_id", "")).strip() != memory_id:
            continue
        skeptic = row.get("skeptic", {}) or {}
        unsupported_count = int(skeptic.get("unsupported_claim_count", 0) or 0)
        if unsupported_count <= 0:
            unsupported_count = len(skeptic.get("unsupported_claims", []) or [])
        prior_risk = float(row.get("hallucination_risk", 0.0) or 0.0)
        skeptic_risk = 0.0
        if bool(skeptic.get("has_hallucination", False)) or unsupported_count > 0:
            skeptic_risk = max(skeptic_risk, 0.35 + 0.05 * min(unsupported_count, 3))
        if str(skeptic.get("verdict", "") or "").strip().lower() in {"reject", "abstain"}:
            skeptic_risk = max(skeptic_risk, 0.8)
        row["external_eval"] = dict(external_eval)
        row["external_eval"]["source"] = "evaluate_cross_repo_fixed_context"
        row["is_sufficient"] = bool(success)
        row["origin_success"] = bool(success)
        row["success_count"] = 1 if success else 0
        row["failure_count"] = 0 if success else 1
        row["alpha"] = 2.0 if success else 1.0
        row["beta"] = 1.0 if success else 2.0
        eval_risk = 1.0 if not success or bool(external_eval.get("generated_has_hallucination", False)) else 0.0
        row["hallucination_risk"] = max(prior_risk, skeptic_risk, eval_risk)
        changed = True
        break

    if changed:
        write_jsonl(memory_path, rows)
    return changed


def build_context_record(row: dict) -> list[dict]:
    raw_evidence = row.get("raw_evidence", {})
    if isinstance(raw_evidence, dict) and raw_evidence:
        contexts: list[dict] = []
        for index, (source_id, spans) in enumerate(raw_evidence.items(), start=1):
            parts: list[str] = []
            if isinstance(spans, list):
                parts = [normalize_ws(span) for span in spans if normalize_ws(span)]
            else:
                text = normalize_ws(spans)
                if text:
                    parts = [text]
            joined = "\n".join(parts).strip()
            if not joined:
                continue
            normalized_source = normalize_ws(source_id)
            contexts.append(
                {
                    "id": normalized_source or f"{row.get('id', '')}::source::{index}",
                    "file_name": row.get("source_name", "FixedContext"),
                    "chunk_id": normalized_source or f"fixed_context_{index}",
                    "case_no": "",
                    "url": normalized_source,
                    "text": joined,
                }
            )
        if contexts:
            return contexts
    evidence_text = normalize_ws(
        row.get("fixed_context", "")
        or row.get("supporting_evidence", "")
        or row.get("gold_chunk_text", "")
    )
    return [
        {
            "id": row.get("id", ""),
            "file_name": row.get("source_name", "FixedContext"),
            "chunk_id": "fixed_context",
            "case_no": "",
            "url": "",
            "text": evidence_text,
        }
    ]


def infer_benchmark_family(rows: list[dict], benchmark_file: Path) -> str:
    name = benchmark_file.name.lower()
    if "ragtruth" in name:
        return "ragtruth"
    if "qrecc" in name:
        return "qrecc"
    for row in rows[:5]:
        benchmark_type = str(row.get("benchmark_type", "")).lower()
        if "ragtruth" in benchmark_type:
            return "ragtruth"
        if "qrecc" in benchmark_type:
            return "qrecc"
    return "generic_fixed_context"


def answer_targets(row: dict) -> list[str]:
    targets: list[str] = []
    for key in ("gold_answer", "reference_response"):
        value = normalize_ws(row.get(key, ""))
        if value:
            targets.append(value)
    for item in row.get("acceptable_answers", []) or []:
        value = normalize_ws(item)
        if value:
            targets.append(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for target in targets:
        norm = normalize_text(target)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(target)
    return deduped


def score_answer(answer: str, row: dict) -> dict:
    targets = answer_targets(row)
    if not targets:
        return {
            "best_answer_f1": 0.0,
            "exact_match": False,
            "has_target": False,
        }
    best_f1 = max(token_f1(answer, target) for target in targets)
    answer_norm = normalize_text(answer)
    exact_match = any(answer_norm == normalize_text(target) for target in targets)
    return {
        "best_answer_f1": round(best_f1, 4),
        "exact_match": exact_match,
        "has_target": True,
    }


def context_match_strings(ctx: dict) -> set[str]:
    values = set()
    for key in ("id", "chunk_id", "file_name", "url"):
        value = normalize_ws(ctx.get(key, ""))
        if value:
            values.add(value)
    return values


def gold_retrieval_hit(row: dict, used_contexts: list[dict]) -> bool:
    gold_ids = {
        normalize_ws(item)
        for item in (row.get("gold_retrieval_ids", []) or [])
        if normalize_ws(item)
    }
    if not gold_ids:
        return False
    for ctx in used_contexts:
        if gold_ids & context_match_strings(ctx):
            return True
    return False


def qrecc_metric_summary(rows: list[dict], predictions: list[dict]) -> dict:
    scored = [pred for pred in predictions if pred.get("qa_score", {}).get("has_target")]
    total = len(scored)
    mean_f1 = safe_div(sum(float(pred["qa_score"]["best_answer_f1"]) for pred in scored), total)
    exact_match_rate = safe_div(sum(1 for pred in scored if pred["qa_score"]["exact_match"]), total)
    return {
        "answer_f1": round(mean_f1, 4),
        "exact_match_rate": round(exact_match_rate, 4),
        "scored_records": total,
        "native_retrieval_metrics_available": False,
        "native_rewrite_metrics_available": False,
        "notes": (
            "This fixed-context QReCC adapter supports conversational QA quality metrics. "
            "Native QReCC retrieval/rewrite metrics require rewrite targets and retrieval ids, "
            "which are not present in this adapted JSONL."
        ),
    }


def qrecc_retrieval_proxy_summary(predictions: list[dict]) -> dict:
    scored = [pred for pred in predictions if pred.get("retrieval_proxy", {}).get("available")]
    total = len(scored)
    hits = sum(1 for pred in scored if pred["retrieval_proxy"]["hit"])
    return {
        "available": bool(total),
        "scored_records": total,
        "retrieval_hit_rate": round(safe_div(hits, total), 4),
        "notes": (
            "This is a fixed-context source-hit proxy using preserved gold evidence ids. "
            "It is not native QReCC retrieval ranking evaluation such as Recall@k or MRR."
        ),
    }


def import_module_from_repo(repo_root: Path, module_name: str) -> ModuleType:
    module_path = repo_root / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def clear_repo_modules(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    remove = []
    for name, module in sys.modules.items():
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            if Path(module_file).resolve().is_relative_to(repo_root):
                remove.append(name)
        except Exception:
            continue
    for name in remove:
        sys.modules.pop(name, None)


def import_backend(repo_root: Path) -> tuple[ModuleType, ModuleType]:
    repo_root = repo_root.resolve()
    clear_repo_modules(repo_root)
    sys.path.insert(0, str(repo_root))
    try:
        retrieve_module = import_module_from_repo(repo_root, "retrieve")
        rag_module = import_module_from_repo(repo_root, "rag")
        return rag_module, retrieve_module
    finally:
        if sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)


def install_fixed_context_retrieve(
    retrieve_module: ModuleType,
    contexts: list[dict],
    rag_module: ModuleType | None = None,
):
    original_retrieve = retrieve_module.retrieve
    original_rag_retrieve = getattr(rag_module, "retrieve", None) if rag_module is not None else None

    def fixed_retrieve(
        query: str,
        top_k: int = 5,
        case_no: str | None = None,
        file_name: str | None = None,
        source_file_path: str | None = None,
        **kwargs,
    ):
        _ = (query, top_k, case_no, file_name, source_file_path, kwargs)
        if top_k and top_k > 0:
            return [dict(ctx) for ctx in contexts[:top_k]]
        return [dict(ctx) for ctx in contexts]

    retrieve_module.retrieve = fixed_retrieve
    if rag_module is not None and hasattr(rag_module, "retrieve"):
        rag_module.retrieve = fixed_retrieve
    return original_retrieve, original_rag_retrieve


def detect_response_hallucination(
    backend: str,
    question: str,
    answer: str,
    evidence: str,
    benchmark_family: str,
) -> dict:
    repo_root = BACKEND_REPOS[backend]
    detector_repo_root = repo_root
    rag_module, _ = import_backend(repo_root)
    try:
        if hasattr(rag_module, "detect_hallucination"):
            result = rag_module.detect_hallucination(
                question=question,
                candidate_answer=answer,
                evidence=evidence,
                benchmark_family=benchmark_family,
            )
            if isinstance(result, dict):
                result = dict(result)
                result["has_hallucination"] = bool(result.get("has_hallucination", False))
                return result

        completion_fn = getattr(rag_module, "_call_completion", None)
        if completion_fn is None:
            # Some baseline repos only implement answer generation. Use the
            # temporal repo's shared detector so all backends are judged
            # consistently.
            clear_repo_modules(repo_root)
            detector_repo_root = BACKEND_REPOS["temporal"]
            rag_module, _ = import_backend(detector_repo_root)
            if hasattr(rag_module, "detect_hallucination"):
                result = rag_module.detect_hallucination(
                    question=question,
                    candidate_answer=answer,
                    evidence=evidence,
                    benchmark_family=benchmark_family,
                )
                if isinstance(result, dict):
                    result = dict(result)
                    result["has_hallucination"] = bool(result.get("has_hallucination", False))
                    result["reason"] = (
                        result.get("reason", "")
                        or "Shared temporal detector used because backend has no detector."
                    )
                    return result
            completion_fn = getattr(rag_module, "_call_completion", None)
        if completion_fn is None:
            raise AttributeError(
                f"No hallucination detector or _call_completion() available for backend {backend!r}"
            )

        raw = completion_fn(
            _legacy_detection_prompt(question, answer, evidence),
            temperature=0.0,
            max_tokens=5,
        )
        return {
            "has_hallucination": normalize_ws(raw[:20]).lower().startswith("yes"),
            "unsupported_spans": [],
            "supported_spans": [],
            "reason": "Legacy Yes/No detector used because backend has no detect_hallucination().",
            "confidence": 0.5,
            "raw_output": raw,
        }
    finally:
        clear_repo_modules(repo_root)
        if detector_repo_root != repo_root:
            clear_repo_modules(detector_repo_root)


def call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return fn(**supported)


def normalize_generation_result(result) -> tuple[str, list[dict], dict]:
    if isinstance(result, tuple):
        if len(result) >= 3:
            return str(result[0]), list(result[1] or []), dict(result[2] or {})
        if len(result) == 2:
            return str(result[0]), list(result[1] or []), {}
        if len(result) == 1:
            return str(result[0]), [], {}
    return str(result or ""), [], {}


def _legacy_detection_prompt(question: str, answer: str, evidence: str) -> str:
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
    return prompt


def detection_target_for_row(row: dict, requested_target: str) -> tuple[str, str]:
    if requested_target == "generated_answer":
        return "generated_answer", ""
    reference_response = normalize_ws(row.get("reference_response", "") or row.get("gold_answer", ""))
    if requested_target == "reference_response":
        return "reference_response", reference_response
    if isinstance(row.get("response_has_hallucination"), bool) and reference_response:
        return "reference_response", reference_response
    return "generated_answer", ""


def skeptic_hallucination_signal(trace: dict) -> bool | None:
    if not isinstance(trace, dict):
        return None
    if isinstance(trace.get("trace"), dict):
        trace = trace["trace"]
    attempts = trace.get("attempts", [])
    if not isinstance(attempts, list) or not attempts:
        return None
    last_attempt = attempts[-1]
    if not isinstance(last_attempt, dict):
        return None
    skeptic = last_attempt.get("skeptic", {})
    if not isinstance(skeptic, dict):
        return None
    value = skeptic.get("has_hallucination")
    if isinstance(value, bool):
        return value
    return None


def run_backend(
    backend: str,
    question: str,
    contexts: list[dict],
    use_reflexion: bool,
    disable_memory: bool,
    max_iters: int,
    top_k: int,
    benchmark_name: str,
) -> tuple[str, dict]:
    repo_root = BACKEND_REPOS[backend]
    previous_disable_memory = os.environ.get("REFLEXION_DISABLE_MEMORY")

    if disable_memory:
        os.environ["REFLEXION_DISABLE_MEMORY"] = "1"
    else:
        os.environ.pop("REFLEXION_DISABLE_MEMORY", None)

    rag_module = None
    retrieve_module = None
    original_retrieve = None
    original_rag_retrieve = None
    try:
        rag_module, retrieve_module = import_backend(repo_root)
        if hasattr(rag_module, "REFLEXION_DISABLE_MEMORY"):
            rag_module.REFLEXION_DISABLE_MEMORY = bool(disable_memory)

        original_retrieve, original_rag_retrieve = install_fixed_context_retrieve(
            retrieve_module,
            contexts,
            rag_module=rag_module,
        )

        if backend in {"reflexion", "temporal"} and use_reflexion:
            answer, used_contexts, trace = normalize_generation_result(call_with_supported_kwargs(
                rag_module.generate_answer_with_reflexion,
                question=question,
                top_k=top_k,
                case_no=None,
                file_name=str(contexts[0].get("file_name", "FixedContext")) if contexts else None,
                max_iters=max_iters,
                task_type="grounded_qa",
                benchmark_name=benchmark_name,
            ))
            return answer, {"contexts": used_contexts, "trace": trace}

        answer, used_contexts, meta = normalize_generation_result(call_with_supported_kwargs(
            rag_module.generate_answer,
            question=question,
            top_k=top_k,
            case_no=None,
            file_name=str(contexts[0].get("file_name", "FixedContext")) if contexts else None,
            task_type="grounded_qa",
            benchmark_name=benchmark_name,
        ))
        return answer, {"contexts": used_contexts, "meta": meta}
    finally:
        if retrieve_module is not None and original_retrieve is not None:
            retrieve_module.retrieve = original_retrieve
        if rag_module is not None and original_rag_retrieve is not None:
            rag_module.retrieve = original_rag_retrieve
        if previous_disable_memory is None:
            os.environ.pop("REFLEXION_DISABLE_MEMORY", None)
        else:
            os.environ["REFLEXION_DISABLE_MEMORY"] = previous_disable_memory
        clear_repo_modules(repo_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a repo backend on a fixed-context benchmark like RAGTruth.")
    parser.add_argument("--backend", choices=sorted(BACKEND_REPOS.keys()), required=True)
    parser.add_argument("--benchmark-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-reflexion", action="store_true")
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--hallucination-source",
        choices=("final_answer", "skeptic", "auto"),
        default="final_answer",
        help=(
            "How to derive generation hallucination rate: judge the final answer, "
            "use the temporal skeptic signal, or auto-select the fair cross-backend "
            "final-answer judge while recording skeptic as a diagnostic when present."
        ),
    )
    parser.add_argument(
        "--detection-target",
        choices=("auto", "reference_response", "generated_answer"),
        default="auto",
        help=(
            "Which answer to compare against gold hallucination labels. "
            "auto uses reference_response when response_has_hallucination is present, "
            "otherwise it uses the generated answer."
        ),
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_file)
    rows = load_jsonl(benchmark_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    benchmark_family = infer_benchmark_family(rows, benchmark_path)

    predictions = []
    hallucinated = 0
    generation_source_counts: dict[str, int] = {}
    gold_labels: list[bool] = []
    pred_labels: list[bool] = []
    for row in rows:
        contexts = build_context_record(row)
        answer, trace = run_backend(
            backend=args.backend,
            question=str(row.get("question", "")),
            contexts=contexts,
            use_reflexion=args.use_reflexion,
            disable_memory=args.disable_memory,
            max_iters=args.max_iters,
            top_k=args.top_k,
            benchmark_name=benchmark_family,
        )
        skeptic_signal = skeptic_hallucination_signal(trace)
        evidence = normalize_ws(
            row.get("fixed_context", "")
            or row.get("supporting_evidence", "")
            or row.get("gold_chunk_text", "")
        )
        if args.hallucination_source == "skeptic" and skeptic_signal is not None:
            generated_has_hallucination = skeptic_signal
            generated_detection = {"source": "skeptic", "has_hallucination": skeptic_signal}
        else:
            generated_detection = detect_response_hallucination(
                args.backend,
                str(row.get("question", "")),
                answer,
                evidence,
                benchmark_family,
            )
            generated_detection["source"] = "detector"
            generated_has_hallucination = bool(generated_detection.get("has_hallucination", False))
        generation_source = str(generated_detection.get("source", "unknown"))
        generation_source_counts[generation_source] = generation_source_counts.get(generation_source, 0) + 1

        detection_target, detection_candidate = detection_target_for_row(row, args.detection_target)
        if detection_target == "generated_answer":
            detection_candidate = answer
            detection_result = generated_detection
        else:
            detection_result = detect_response_hallucination(
                args.backend,
                str(row.get("question", "")),
                detection_candidate,
                evidence,
                benchmark_family,
            )
            detection_result["source"] = "detector"
        predicted_has_hallucination = bool(detection_result.get("has_hallucination", False))
        qa_score = score_answer(answer, row)
        success_f1_threshold = float(os.getenv("REFLEXION_EXTERNAL_MEMORY_SUCCESS_F1", "0.35"))
        external_memory_success = not generated_has_hallucination
        if qa_score.get("has_target"):
            external_memory_success = external_memory_success and (
                bool(qa_score.get("exact_match", False))
                or float(qa_score.get("best_answer_f1", 0.0)) >= success_f1_threshold
            )
        update_memory_from_external_eval(
            trace,
            benchmark_name=benchmark_family,
            task_type="grounded_qa",
            success=external_memory_success,
            external_eval={
                "has_target": bool(qa_score.get("has_target", False)),
                "best_answer_f1": float(qa_score.get("best_answer_f1", 0.0)),
                "exact_match": bool(qa_score.get("exact_match", False)),
                "generated_has_hallucination": bool(generated_has_hallucination),
                "predicted_has_hallucination": bool(predicted_has_hallucination),
                "success_f1_threshold": success_f1_threshold,
            },
        )
        used_contexts = trace.get("contexts", []) if isinstance(trace, dict) else []
        retrieval_available = bool(row.get("gold_retrieval_ids"))
        retrieval_hit = gold_retrieval_hit(row, used_contexts) if retrieval_available else False
        truth_value = row.get("response_has_hallucination")
        if isinstance(truth_value, bool):
            gold_labels.append(truth_value)
            pred_labels.append(predicted_has_hallucination)
        hallucinated += int(generated_has_hallucination)
        predictions.append(
            {
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "gold_answer": row.get("gold_answer", row.get("reference_response", "")),
                "gold_has_hallucination": truth_value,
                "predicted_answer": answer,
                "generated_has_hallucination": generated_has_hallucination,
                "predicted_has_hallucination": predicted_has_hallucination,
                "detection_target": detection_target,
                "detection_candidate": detection_candidate,
                "detection_result": detection_result,
                "generated_detection_result": generated_detection,
                "skeptic_has_hallucination": skeptic_signal,
                "qa_score": qa_score,
                "retrieval_proxy": {
                    "available": retrieval_available,
                    "hit": retrieval_hit,
                },
                "fixed_context": row.get("fixed_context", ""),
                "backend": args.backend,
                "trace": trace,
            }
        )

    total = len(rows)
    summary = {
        "config": {
            "backend": args.backend,
            "benchmark_family": benchmark_family,
            "use_reflexion": args.use_reflexion,
            "disable_memory": args.disable_memory,
            "max_iters": args.max_iters,
            "max_samples": args.max_samples,
            "top_k": args.top_k,
            "hallucination_source": args.hallucination_source,
            "detection_target": args.detection_target,
        },
        "total_records": total,
        "generation": {
            "hallucination_rate": round((hallucinated / total) if total else 0.0, 4),
            "non_hallucination_rate": round(((total - hallucinated) / total) if total else 0.0, 4),
            "source_counts": generation_source_counts,
        },
    }
    notes = []
    if args.hallucination_source == "auto":
        notes.append(
            "hallucination_source=auto uses the final-answer detector for fair cross-backend "
            "generation-rate comparisons; temporal skeptic values are still written per row."
        )
    if args.detection_target == "auto" and any(
        isinstance(row.get("response_has_hallucination"), bool)
        and normalize_ws(row.get("reference_response", "") or row.get("gold_answer", ""))
        for row in rows
    ):
        notes.append(
            "detection_target=auto evaluates reference_response when gold response hallucination "
            "labels are present, so hallucination_detection can be identical across backends."
        )
    if notes:
        summary["notes"] = notes
    if benchmark_family == "qrecc":
        summary["qa_quality"] = qrecc_metric_summary(rows, predictions)
        summary["retrieval_proxy"] = qrecc_retrieval_proxy_summary(predictions)
    if gold_labels and len(gold_labels) == len(pred_labels):
        summary["hallucination_detection"] = classification_metrics(gold_labels, pred_labels)
        summary["hallucination_detection"]["gold_positive_rate"] = round(sum(gold_labels) / len(gold_labels), 4)
        summary["hallucination_detection"]["predicted_positive_rate"] = round(sum(pred_labels) / len(pred_labels), 4)
    elif benchmark_family == "qrecc":
        summary["hallucination_detection"] = {
            "available": False,
            "notes": (
                "Response-level precision/recall/F1 requires gold hallucination labels. "
                "This adapted QReCC file does not include gold hallucination annotations, "
                "so only generation hallucination rate is reported."
            ),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fixed_context_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "fixed_context_predictions.jsonl", predictions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
