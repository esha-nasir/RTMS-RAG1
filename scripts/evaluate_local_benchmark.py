import argparse
import json
import os
import sys
import re
from dataclasses import dataclass
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

GENERATORS = None
JUDGE_FN = None
RETRIEVE_FN = None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def token_f1(a: str, b: str) -> float:
    ta = normalize(a).split()
    tb = normalize(b).split()
    if not ta or not tb:
        return 0.0
    sa = set(ta)
    sb = set(tb)
    common = len(sa & sb)
    if common == 0:
        return 0.0
    p = common / len(sa)
    r = common / len(sb)
    return 2 * p * r / (p + r)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def safe_name(value: str) -> str:
    cleaned = (value or "default").strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", cleaned)
    return cleaned or "default"


def memory_file_for(benchmark_name: str, task_type: str) -> Path:
    memory_dir = Path((os.getenv("REFLEXION_MEMORY_DIR", "") or ".").strip())
    return memory_dir / f"{safe_name(benchmark_name)}.{safe_name(task_type)}.memory.jsonl"


def update_memory_from_external_eval(trace: dict, benchmark_name: str, task_type: str, success: bool, eval_flags: dict) -> bool:
    if not isinstance(trace, dict):
        return False
    memory_entry = trace.get("memory_entry")
    if not isinstance(memory_entry, dict):
        return False
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
        row["external_eval"] = {
            "is_correct": bool(eval_flags.get("is_correct", False)),
            "is_grounded": bool(eval_flags.get("is_grounded", False)),
            "is_hallucinated": bool(eval_flags.get("is_hallucinated", False)),
            "is_abstained": bool(eval_flags.get("is_abstained", False)),
            "source": "evaluate_local_benchmark",
        }
        row["is_sufficient"] = bool(success)
        row["origin_success"] = bool(success)
        row["success_count"] = 1 if success else 0
        row["failure_count"] = 0 if success else 1
        row["alpha"] = 2.0 if success else 1.0
        row["beta"] = 1.0 if success else 2.0
        eval_risk = 1.0 if (not success or bool(eval_flags.get("is_hallucinated", False))) else 0.0
        row["hallucination_risk"] = max(prior_risk, skeptic_risk, eval_risk)
        changed = True
        break

    if changed:
        write_jsonl(memory_path, rows)
    return changed


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
            if "=" not in line:
                continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_generators():
    global GENERATORS
    if GENERATORS is None:
        from rtms_rag.rag import generate_answer, generate_answer_with_reflexion

        GENERATORS = (generate_answer, generate_answer_with_reflexion)
    return GENERATORS


def get_judge_fn():
    global JUDGE_FN
    if JUDGE_FN is None:
        from rtms_rag.rag import _call_completion

        JUDGE_FN = _call_completion
    return JUDGE_FN


def get_retrieve_fn():
    global RETRIEVE_FN
    if RETRIEVE_FN is None:
        from rtms_rag.retrieve import retrieve

        RETRIEVE_FN = retrieve
    return RETRIEVE_FN


def call_rag(
    question: str,
    use_reflexion: bool,
    max_iters: int,
    top_k: int,
    case_no: str | None = None,
    file_name: str | None = None,
    source_file_path: str | None = None,
    disable_retrieval: bool = False,
    task_type: str = "qa_generation",
    benchmark_name: str = "default",
):
    generate_answer, generate_answer_with_reflexion = get_generators()
    previous_disable = os.environ.get("REFLEXION_BENCHMARK_NO_RETRIEVAL")
    rag_module = None
    previous_module_disable = None
    if disable_retrieval:
        import rtms_rag.rag as rag_module  # type: ignore

        previous_module_disable = getattr(rag_module, "REFLEXION_BENCHMARK_NO_RETRIEVAL", False)
    if disable_retrieval:
        os.environ["REFLEXION_BENCHMARK_NO_RETRIEVAL"] = "1"
        if rag_module is not None:
            rag_module.REFLEXION_BENCHMARK_NO_RETRIEVAL = True
    if use_reflexion:
        try:
            return generate_answer_with_reflexion(
                question,
                top_k=top_k,
                case_no=case_no,
                file_name=file_name,
                source_file_path=source_file_path,
                max_iters=max_iters,
                task_type=task_type,
                benchmark_name=benchmark_name,
            )
        finally:
            if disable_retrieval:
                if previous_disable is None:
                    os.environ.pop("REFLEXION_BENCHMARK_NO_RETRIEVAL", None)
                else:
                    os.environ["REFLEXION_BENCHMARK_NO_RETRIEVAL"] = previous_disable
                if rag_module is not None and previous_module_disable is not None:
                    rag_module.REFLEXION_BENCHMARK_NO_RETRIEVAL = previous_module_disable
    try:
        answer, contexts = generate_answer(
            question,
            top_k=top_k,
            case_no=case_no,
            file_name=file_name,
            source_file_path=source_file_path,
            task_type=task_type,
            benchmark_name=benchmark_name,
        )
        return answer, contexts, {}
    finally:
        if disable_retrieval:
            if previous_disable is None:
                os.environ.pop("REFLEXION_BENCHMARK_NO_RETRIEVAL", None)
            else:
                os.environ["REFLEXION_BENCHMARK_NO_RETRIEVAL"] = previous_disable
            if rag_module is not None and previous_module_disable is not None:
                rag_module.REFLEXION_BENCHMARK_NO_RETRIEVAL = previous_module_disable


def judge_prompt(question: str, answer: str, gold_answer: str, evidence: str, acceptable_answers: list[str] | None = None) -> str:
    acceptable_block = ""
    if acceptable_answers:
        cleaned = [str(item).strip() for item in acceptable_answers if str(item).strip()]
        if cleaned:
            acceptable_block = f"Acceptable Answers: {json.dumps(cleaned, ensure_ascii=False)}\n"
    return (
        "You are evaluating a legal RAG answer.\n"
        "Use only the provided evidence and gold answer.\n"
        "Treat the acceptable answers as valid paraphrase/coverage targets when deciding correctness.\n"
        "Return ONLY JSON with keys:\n"
        '- is_correct: true or false\n'
        '- is_grounded: true or false\n'
        '- is_hallucinated: true or false\n'
        '- is_abstained: true or false\n'
        '- notes: short string\n\n'
        f"Question: {question}\n"
        f"Candidate Answer: {answer}\n"
        f"Gold Answer: {gold_answer}\n"
        f"{acceptable_block}"
        f"Evidence: {evidence}\n"
    )


def extract_json_object(text: str) -> dict:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text or "")
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def is_abstention_answer(answer: str) -> bool:
    normalized = normalize(answer)
    if not normalized:
        return False
    abstention_prefixes = (
        "insufficient context",
        "there is no relevant statement",
        "the provided context does not state",
        "the context does not state",
        "not stated in the provided context",
        "not stated in the context",
        "cannot be determined from the provided context",
        "cannot be determined from the context",
        "no relevant statement in the provided context",
        "no relevant statement in the context",
    )
    return normalized.startswith(abstention_prefixes)


def judge_qa_answer(question: str, answer: str, gold_answer: str, evidence: str, acceptable_answers: list[str] | None = None) -> dict:
    judge = get_judge_fn()
    raw = judge(
        judge_prompt(question, answer, gold_answer, evidence, acceptable_answers=acceptable_answers),
        temperature=0.0,
        max_tokens=260,
    )
    parsed = extract_json_object(raw)
    if parsed:
        grounded = bool(parsed.get("is_grounded", False))
        correct = bool(parsed.get("is_correct", False))
        hallucinated = bool(parsed.get("is_hallucinated", False))
        abstained = bool(parsed.get("is_abstained", False))
        return {
            "is_correct": correct,
            "is_grounded": grounded,
            "is_hallucinated": hallucinated,
            "is_abstained": abstained,
            "notes": str(parsed.get("notes", "")).strip(),
        }
    abstained = is_abstention_answer(answer)
    return {
        "is_correct": False,
        "is_grounded": False,
        "is_hallucinated": not abstained,
        "is_abstained": abstained,
        "notes": "",
    }


def refine_eval_flags(answer: str, contexts: list[dict], eval_flags: dict) -> dict:
    refined = dict(eval_flags)
    abstained = is_abstention_answer(answer) or bool(refined.get("is_abstained", False))
    has_context = bool(contexts)

    refined["is_abstained"] = abstained

    # Empty retrieval cannot support a grounded answer.
    if not has_context:
        refined["is_grounded"] = False
        if not abstained and normalize(answer):
            refined["is_hallucinated"] = True
        else:
            refined["is_hallucinated"] = False

    # Abstentions should not count as grounded or hallucinated.
    if abstained:
        refined["is_grounded"] = False
        refined["is_hallucinated"] = False

    # A correct answer should not be tagged as hallucinated.
    if refined.get("is_correct", False):
        refined["is_hallucinated"] = False

    return refined


@dataclass
class EvalResult:
    total: int
    correct: int
    parse_failures: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class QaMetrics:
    total: int = 0
    correct: int = 0
    grounded: int = 0
    hallucinated: int = 0
    abstained: int = 0
    retrieval_hit: int = 0
    span_hit: int = 0
    supported_but_wrong: int = 0
    parse_failures: int = 0

    def to_summary(self) -> dict:
        total = self.total or 0
        return {
            "total": total,
            "correct": self.correct,
            "accuracy": round((self.correct / total) if total else 0.0, 4),
            "retrieval_hit": self.retrieval_hit,
            "retrieval_hit_rate": round((self.retrieval_hit / total) if total else 0.0, 4),
            "span_hit": self.span_hit,
            "span_hit_rate": round((self.span_hit / total) if total else 0.0, 4),
            "grounded": self.grounded,
            "grounded_rate": round((self.grounded / total) if total else 0.0, 4),
            "hallucinated": self.hallucinated,
            "hallucination_rate": round((self.hallucinated / total) if total else 0.0, 4),
            "abstained": self.abstained,
            "abstention_rate": round((self.abstained / total) if total else 0.0, 4),
            "supported_but_wrong": self.supported_but_wrong,
            "supported_but_wrong_rate": round((self.supported_but_wrong / total) if total else 0.0, 4),
            "parse_failures": self.parse_failures,
        }


def build_reference_text(contexts: list[dict]) -> str:
    parts = []
    for ctx in contexts:
        text = str(ctx.get("text", "")).strip()
        if not text:
            continue
        file_name = str(ctx.get("file_name", "")).strip()
        chunk_id = ctx.get("chunk_id")
        header_bits = []
        if file_name:
            header_bits.append(file_name)
        if chunk_id not in (None, ""):
            header_bits.append(f"chunk {chunk_id}")
        header = f"[{' | '.join(header_bits)}]\n" if header_bits else ""
        parts.append(f"{header}{text}")
    return "\n\n".join(parts)


def _context_match_strings(ctx: dict) -> set[str]:
    values = set()
    for value in (ctx.get("id"), ctx.get("chunk_id"), ctx.get("file_name"), ctx.get("source_file_path"), ctx.get("doc_id")):
        text = str(value or "").strip()
        if text:
            values.add(text)

    file_name = str(ctx.get("file_name", "")).strip()
    chunk_id = str(ctx.get("chunk_id", "")).strip()
    if file_name and chunk_id:
        values.add(f"{file_name}::{chunk_id}")
        values.add(f"._{file_name}_{chunk_id}")
    return values


def _matches_source_file_path(ctx: dict, source_file_path: str) -> bool:
    source_file_path = str(source_file_path or "").strip()
    if not source_file_path:
        return False
    ctx_source = str(ctx.get("source_file_path", "") or "").strip()
    ctx_chunk = str(ctx.get("chunk_id", "") or "").strip()
    ctx_id = str(ctx.get("id", "") or "").strip()
    return (
        ctx_source == source_file_path
        or source_file_path in ctx_chunk
        or source_file_path in ctx_id
    )


def _matches_gold_chunk(ctx: dict, gold_vector_id: str, gold_chunk_id: str) -> bool:
    gold_vector_id = str(gold_vector_id or "").strip()
    gold_chunk_id = str(gold_chunk_id or "").strip()
    ctx_id = str(ctx.get("id", "") or "").strip()
    ctx_chunk = str(ctx.get("chunk_id", "") or "").strip()
    if gold_vector_id and (ctx_id == gold_vector_id or ctx_id.startswith(f"{gold_vector_id}::window_")):
        return True
    if gold_chunk_id and (
        ctx_chunk.endswith(f"::{gold_chunk_id}")
        or f"::{gold_chunk_id}::window_" in ctx_chunk
        or ctx_id.endswith(f"::{gold_chunk_id}")
        or f"::{gold_chunk_id}::window_" in ctx_id
    ):
        return True
    return False


def has_gold_source_hit(row: dict, contexts: list[dict]) -> bool:
    gold_sources = {
        str(source).strip()
        for source in row.get("gold_sources", [])
        if str(source).strip()
    }

    gold_chunk_id = str(row.get("chunk_id", "")).strip()
    gold_file_name = str(row.get("file_name", "")).strip()
    gold_vector_id = str(row.get("gold_vector_id", "")).strip()
    gold_source_file_path = str(row.get("gold_source_file_path", "") or "").strip()
    if gold_chunk_id:
        gold_sources.add(gold_chunk_id)
    if gold_file_name and not gold_source_file_path:
        gold_sources.add(gold_file_name)
    if gold_file_name and gold_chunk_id:
        gold_sources.add(f"{gold_file_name}::{gold_chunk_id}")
        gold_sources.add(f"._{gold_file_name}_{gold_chunk_id}")
    if gold_source_file_path:
        gold_sources.add(gold_source_file_path)

    if not gold_sources:
        return False

    for ctx in contexts:
        if gold_source_file_path and not _matches_source_file_path(ctx, gold_source_file_path):
            continue
        if _matches_gold_chunk(ctx, gold_vector_id, gold_chunk_id):
            return True
        if gold_sources & _context_match_strings(ctx):
            return True
    return False


def has_gold_span_hit(row: dict, contexts: list[dict], f1_threshold: float = 0.5) -> bool:
    gold_span_text = normalize(str(row.get("gold_chunk_text", "") or ""))
    gold_source_file_path = str(row.get("gold_source_file_path", "") or "").strip()
    if not gold_span_text:
        return False

    for ctx in contexts:
        ctx_text = normalize(str(ctx.get("text", "") or ""))
        if not ctx_text:
            continue
        ctx_chunk_id = str(ctx.get("chunk_id", "") or "").strip()
        if gold_source_file_path and gold_source_file_path not in ctx_chunk_id:
            continue
        if gold_span_text in ctx_text or ctx_text in gold_span_text:
            return True
        if token_f1(ctx_text, gold_span_text) >= f1_threshold:
            return True
    return False


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_chunk_suffix(vector_id: str) -> int | None:
    m = re.search(r"_(\d+)$", vector_id or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def evaluate_retrieval(rows: list[dict], top_k: int) -> tuple[EvalResult, EvalResult]:
    total = 0
    correct_strict = 0
    correct_soft = 0
    parse_failures = 0
    retrieve = get_retrieve_fn()

    for row in rows:
        total += 1
        q = row.get("query", "")
        gold_file = str(row.get("gold_file_name", "")).strip()
        gold_vector_id = str(row.get("gold_vector_id", row.get("gold_chunk_id", ""))).strip()
        gold_chunk_id = str(row.get("gold_chunk_id", "")).strip()
        gold_case_no = str(row.get("gold_case_no", "")).strip() or None
        gold_source_file_path = str(row.get("gold_source_file_path", "") or "").strip() or None

        try:
            contexts = retrieve(
                q,
                top_k=top_k,
                case_no=gold_case_no,
                file_name=gold_file or None,
                source_file_path=gold_source_file_path,
            )
        except Exception:
            parse_failures += 1
            continue

        strict_hit = False
        soft_hit = False
        gold_chunk_idx = parse_chunk_suffix(gold_vector_id)
        for c in contexts:
            file_name = str(c.get("file_name", "")).strip()
            chunk_id = str(c.get("chunk_id", "")).strip()
            source_ok = True
            if gold_source_file_path:
                source_ok = _matches_source_file_path(c, gold_source_file_path)
            elif gold_file:
                source_ok = file_name == gold_file

            if source_ok and _matches_gold_chunk(c, gold_vector_id, gold_chunk_id):
                strict_hit = True
                soft_hit = True
                break

            if source_ok and gold_chunk_idx is not None:
                try:
                    cidx = int(chunk_id)
                except Exception:
                    cidx = None
                if cidx is not None and abs(cidx - gold_chunk_idx) <= 1:
                    soft_hit = True

        if strict_hit:
            correct_strict += 1
        if soft_hit:
            correct_soft += 1

    return (
        EvalResult(total=total, correct=correct_strict, parse_failures=parse_failures),
        EvalResult(total=total, correct=correct_soft, parse_failures=parse_failures),
    )


def evaluate_qa(
    rows: list[dict],
    top_k: int,
    f1_threshold: float,
    use_reflexion: bool,
    max_iters: int,
    judge_mode: bool,
    collect_predictions: bool,
    disable_retrieval: bool,
    benchmark_name: str,
) -> tuple[QaMetrics, list[dict]]:
    metrics = QaMetrics()
    prediction_records = []

    for row in rows:
        metrics.total += 1
        q = row.get("question", "")
        gold = row.get("gold_answer", "")
        acceptable_answers = [
            str(item).strip()
            for item in row.get("acceptable_answers", [])
            if str(item).strip()
        ]
        evidence = row.get("gold_chunk_text", "")

        answer, contexts, trace = call_rag(
            q,
            use_reflexion=use_reflexion,
            max_iters=max_iters,
            top_k=top_k,
            case_no=(row.get("case_no") or None),
            file_name=(row.get("file_name") or None),
            source_file_path=(row.get("gold_source_file_path") or None),
            disable_retrieval=disable_retrieval,
            task_type=str(row.get("task_type") or "qa_generation"),
            benchmark_name=benchmark_name,
        )
        ans = normalize(answer)
        g = normalize(gold)
        if not ans:
            metrics.parse_failures += 1

        eval_flags = {
            "is_correct": False,
            "is_grounded": False,
            "is_hallucinated": False,
            "is_abstained": is_abstention_answer(answer),
            "notes": "",
        }
        if ans and judge_mode:
            eval_flags = judge_qa_answer(q, answer, gold, evidence, acceptable_answers=acceptable_answers)
        elif ans:
            is_correct = False
            candidates = [g] + [normalize(item) for item in acceptable_answers]
            if any(candidate and (candidate[:120] in ans or ans[:120] in candidate) for candidate in candidates):
                is_correct = True
            elif any(candidate and token_f1(ans, candidate) >= f1_threshold for candidate in candidates):
                is_correct = True
            grounded = is_correct and not is_abstention_answer(answer)
            eval_flags = {
                "is_correct": is_correct,
                "is_grounded": grounded,
                "is_hallucinated": bool(ans) and not grounded and not is_abstention_answer(answer),
                "is_abstained": is_abstention_answer(answer),
                "notes": "",
            }
        eval_flags = refine_eval_flags(answer, contexts, eval_flags)
        external_memory_success = (
            bool(eval_flags.get("is_correct", False))
            and bool(eval_flags.get("is_grounded", False))
            and not bool(eval_flags.get("is_hallucinated", False))
        )
        update_memory_from_external_eval(
            trace,
            benchmark_name=benchmark_name,
            task_type=str(row.get("task_type") or "qa_generation"),
            success=external_memory_success,
            eval_flags=eval_flags,
        )
        retrieval_hit = has_gold_source_hit(row, contexts)
        span_hit = has_gold_span_hit(row, contexts)
        supported_but_wrong = eval_flags["is_grounded"] and not eval_flags["is_correct"]

        if eval_flags["is_correct"]:
            metrics.correct += 1
        if retrieval_hit:
            metrics.retrieval_hit += 1
        if span_hit:
            metrics.span_hit += 1
        if eval_flags["is_grounded"]:
            metrics.grounded += 1
        if eval_flags["is_hallucinated"]:
            metrics.hallucinated += 1
        if eval_flags["is_abstained"]:
            metrics.abstained += 1
        if supported_but_wrong:
            metrics.supported_but_wrong += 1

        if collect_predictions:
            retrieved_reference = build_reference_text(contexts)
            prediction_records.append(
                {
                    "id": row.get("id", ""),
                    "question": q,
                    "gold_answer": gold,
                    "acceptable_answers": acceptable_answers,
                    "predicted_answer": answer,
                    "is_correct": eval_flags["is_correct"],
                    "is_grounded": eval_flags["is_grounded"],
                    "is_hallucinated": eval_flags["is_hallucinated"],
                    "is_abstained": eval_flags["is_abstained"],
                    "retrieval_hit": retrieval_hit,
                    "span_hit": span_hit,
                    "supported_but_wrong": supported_but_wrong,
                    "judge_notes": eval_flags.get("notes", ""),
                    "use_reflexion": use_reflexion,
                    "max_iters": max_iters,
                    "file_name": row.get("file_name", ""),
                    "case_no": row.get("case_no", ""),
                    "gold_sources": row.get("gold_sources", []),
                    "gold_reference": evidence,
                    "retrieved_reference": retrieved_reference,
                    "retrieved_contexts": contexts,
                    "reflexion_trace": trace,
                }
            )

    return metrics, prediction_records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate reflexionAgentRAG on corpus-derived local legal benchmark.")
    parser.add_argument("--benchmark-dir", default=str(root / "data" / "benchmark_from_chunks"))
    parser.add_argument("--qa-benchmark-file", default="qa_draft_benchmark.jsonl")
    parser.add_argument("--retrieval-benchmark-file", default="retrieval_benchmark.jsonl")
    parser.add_argument("--output-dir", default=str(root / "eval_outputs_local_benchmark"))
    parser.add_argument("--env-file", default=str(root / ".env"))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qa-f1-threshold", type=float, default=0.35)
    parser.add_argument("--use-reflexion", action="store_true")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--judge-qa", action="store_true", help="Use an evidence-based judge instead of overlap/F1")
    parser.add_argument("--qa-only", action="store_true", help="Run only QA evaluation and skip retrieval evaluation.")
    parser.add_argument("--retrieval-only", action="store_true", help="Run only retrieval evaluation and skip QA evaluation.")
    parser.add_argument("--export-predictions", action="store_true", help="Write QA prediction records to the output directory.")
    parser.add_argument("--disable-retrieval", action="store_true", help="Run QA generation without retrieving any evidence.")
    parser.add_argument(
        "--benchmark-name",
        default="default",
        help="Logical benchmark/domain name used to isolate episodic memory files.",
    )
    args = parser.parse_args()

    if args.qa_only and args.retrieval_only:
        raise SystemExit("Choose only one of --qa-only or --retrieval-only.")

    load_env_file(Path(args.env_file))
    benchmark_dir = Path(args.benchmark_dir)
    local_corpus_path = benchmark_dir / "all_chunks.jsonl"
    if local_corpus_path.exists():
        os.environ.setdefault("LOCAL_CORPUS_ONLY", "1")
        os.environ.setdefault("LEXICAL_CORPUS_PATH", str(local_corpus_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows = load_jsonl(benchmark_dir / args.retrieval_benchmark_file)
    qa_rows = load_jsonl(benchmark_dir / args.qa_benchmark_file)

    if not args.qa_only and not retrieval_rows:
        raise SystemExit(
            f"No retrieval benchmark rows loaded from {benchmark_dir / args.retrieval_benchmark_file}. "
            "Check the file path or benchmark file name."
        )
    if not args.retrieval_only and not qa_rows:
        raise SystemExit(
            f"No QA benchmark rows loaded from {benchmark_dir / args.qa_benchmark_file}. "
            "Check the file path or benchmark file name."
        )

    if args.max_samples > 0:
        retrieval_rows = retrieval_rows[: args.max_samples]
        qa_rows = qa_rows[: args.max_samples]

    retrieval_strict = retrieval_soft = None
    qa_result = None
    qa_predictions = []
    if not args.qa_only:
        retrieval_strict, retrieval_soft = evaluate_retrieval(retrieval_rows, top_k=args.top_k)
    if not args.retrieval_only:
        qa_result, qa_predictions = evaluate_qa(
            qa_rows,
            top_k=args.top_k,
            f1_threshold=args.qa_f1_threshold,
            use_reflexion=args.use_reflexion,
            max_iters=args.max_iters,
            judge_mode=args.judge_qa,
            collect_predictions=args.export_predictions,
            disable_retrieval=args.disable_retrieval,
            benchmark_name=args.benchmark_name,
        )

    summary = {
        "config": {
            "use_reflexion": args.use_reflexion,
            "max_iters": args.max_iters,
            "top_k": args.top_k,
            "judge_qa": args.judge_qa,
            "qa_only": args.qa_only,
            "retrieval_only": args.retrieval_only,
            "disable_retrieval": args.disable_retrieval,
            "benchmark_name": args.benchmark_name,
            "qa_benchmark_file": args.qa_benchmark_file,
            "retrieval_benchmark_file": args.retrieval_benchmark_file,
        }
    }

    if retrieval_strict is not None and retrieval_soft is not None:
        summary["retrieval_strict"] = {
            "total": retrieval_strict.total,
            "correct": retrieval_strict.correct,
            "accuracy": round(retrieval_strict.accuracy, 4),
            "parse_failures": retrieval_strict.parse_failures,
        }
        summary["retrieval_soft"] = {
            "total": retrieval_soft.total,
            "correct": retrieval_soft.correct,
            "accuracy": round(retrieval_soft.accuracy, 4),
            "parse_failures": retrieval_soft.parse_failures,
        }

    if qa_result is not None:
        summary["qa_draft"] = qa_result.to_summary()

    if args.export_predictions and not args.retrieval_only:
        write_jsonl(output_dir / "local_benchmark_predictions.jsonl", qa_predictions)

    (output_dir / "local_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
