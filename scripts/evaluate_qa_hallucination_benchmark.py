import argparse
import json
import math
import os
import sys
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
GENERATORS = None
DETECTOR_FN = None
YES_NO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)
STRICT_LABEL_RE = re.compile(r"^\s*(?:answer\s*[:\-]\s*)?(yes|no)\s*[.!]?\s*$", flags=re.IGNORECASE)


def parse_yes_no(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    # Prefer strict labels from the first few non-empty lines.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in lines[:4]:
        m = STRICT_LABEL_RE.match(ln)
        if m:
            return m.group(1).capitalize()

    # Fallback: inspect only the leading segment before citations/metadata.
    lead = raw[:120]
    lead = re.split(r"citations?:", lead, flags=re.IGNORECASE)[0]
    m = YES_NO_RE.search(lead)
    if m:
        return m.group(1).capitalize()

    # Semantic fallback for verbose outputs that omit explicit Yes/No labels.
    lowered = raw.lower()
    body = re.split(r"citations?:", lowered, flags=re.IGNORECASE)[0]

    yes_patterns = [
        "contains a hallucination",
        "is a hallucination",
        "unsupported",
        "not supported by the evidence",
        "contradiction",
        "factual error",
    ]
    no_patterns = [
        "supported by the evidence",
        "supported by evidence",
        "fully supported",
        "grounded in the evidence",
        "factual and grounded",
    ]

    yes_hit = any(p in body for p in yes_patterns)
    no_hit = any(p in body for p in no_patterns)
    if yes_hit and not no_hit:
        return "Yes"
    if no_hit and not yes_hit:
        return "No"
    return ""


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


def estimate_tokens(text: str) -> int:
    """
    Lightweight token estimate for LLM prompts/responses.
    Rough rule of thumb: ~1 token ~= 4 characters for English text.
    """
    clean = (text or "").strip()
    if not clean:
        return 0
    return max(1, math.ceil(len(clean) / 4))


def get_generators():
    global GENERATORS
    if GENERATORS is None:
        from rtms_rag.rag import generate_answer, generate_answer_with_reflexion

        GENERATORS = (generate_answer, generate_answer_with_reflexion)
    return GENERATORS


def get_detector_fn():
    global DETECTOR_FN
    if DETECTOR_FN is None:
        from rtms_rag.rag import _call_completion

        DETECTOR_FN = _call_completion
    return DETECTOR_FN


def detector_prompt(question: str, candidate_answer: str, evidence: str) -> str:
    return (
        "You are a hallucination detector for legal QA.\n"
        "Use only the provided evidence.\n"
        "Output exactly one token: Yes or No.\n"
        "Do not add explanation, citations, punctuation, or extra text.\n"
        "Yes = the candidate answer contains hallucination, contradiction, or unsupported facts.\n"
        "No = the candidate answer is supported by the evidence.\n\n"
        f"Question: {question}\n"
        f"Candidate Answer: {candidate_answer}\n"
        f"Evidence: {evidence}\n\n"
        "Output:"
    )


def call_rag(prompt: str, use_reflexion: bool, max_iters: int, top_k: int = 1) -> str:
    detector = get_detector_fn()
    tries = max(1, int(max_iters)) if use_reflexion else 1
    current_prompt = prompt
    last_output = ""
    for _ in range(tries):
        last_output = detector(current_prompt, temperature=0.0, max_tokens=5)
        if parse_yes_no(last_output):
            return last_output
        current_prompt = (
            f"{prompt}\n\nYour previous output was invalid. "
            "Return exactly one token: Yes or No."
        )
    return last_output


@dataclass
class EvalResult:
    total: int
    correct: int
    parse_failures: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def classification_metrics(y_true: list[str], y_pred: list[str], positive_label: str) -> dict:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == positive_label and pred == positive_label)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != positive_label and pred == positive_label)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == positive_label and pred != positive_label)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth != positive_label and pred != positive_label)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)

    return {
        "positive_label": positive_label,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "support": int(sum(1 for truth in y_true if truth == positive_label)),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
    }


def evaluate(rows: list[dict], output_dir: Path, use_reflexion: bool, max_iters: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    grounded_total = 0
    grounded_correct = 0
    hallucinated_total = 0
    hallucinated_correct = 0
    parse_failures = 0
    estimated_prompt_tokens_total = 0
    estimated_output_tokens_total = 0

    for row in rows:
        question = str(row.get("question", ""))
        evidence = str(row.get("gold_chunk_text", "")).strip()

        pairs = [
            ("grounded_answer", "No"),
            ("hallucinated_answer", "Yes"),
        ]

        for field, truth in pairs:
            candidate = str(row.get(field, "")).strip()
            if not candidate:
                continue

            prompt = detector_prompt(question, candidate, evidence)
            model_output = call_rag(prompt, use_reflexion=use_reflexion, max_iters=max_iters, top_k=1)
            estimated_prompt_tokens = estimate_tokens(prompt)
            estimated_output_tokens = estimate_tokens(model_output)
            estimated_prompt_tokens_total += estimated_prompt_tokens
            estimated_output_tokens_total += estimated_output_tokens
            pred = parse_yes_no(model_output)
            if not pred:
                parse_failures += 1
                pred = ""

            y_true.append(truth)
            y_pred.append(pred)

            if truth == "No":
                grounded_total += 1
                if pred == truth:
                    grounded_correct += 1
            else:
                hallucinated_total += 1
                if pred == truth:
                    hallucinated_correct += 1

            records.append(
                {
                    "id": row.get("id", ""),
                    "pair_type": field,
                    "hallucination_type": row.get("hallucination_type", "") if field == "hallucinated_answer" else "",
                    "question": question,
                    "ground_truth": truth,
                    "prediction": pred,
                    "raw_output": model_output,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "estimated_output_tokens": estimated_output_tokens,
                    "estimated_total_tokens": estimated_prompt_tokens + estimated_output_tokens,
                }
            )

    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    result = EvalResult(total=len(y_true), correct=correct, parse_failures=parse_failures)

    predictions_path = output_dir / "qa_hallucination_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "config": {
            "use_reflexion": use_reflexion,
            "max_iters": max_iters,
        },
        "overall": {
            "total": result.total,
            "correct": result.correct,
            "accuracy": round(result.accuracy, 4),
            "parse_failures": result.parse_failures,
        },
        "token_estimate": {
            "prompt_tokens": estimated_prompt_tokens_total,
            "output_tokens": estimated_output_tokens_total,
            "total_tokens": estimated_prompt_tokens_total + estimated_output_tokens_total,
            "avg_tokens_per_example": round(
                (estimated_prompt_tokens_total + estimated_output_tokens_total) / result.total, 2
            )
            if result.total
            else 0.0,
        },
        "grounded_answer": {
            "total": grounded_total,
            "correct": grounded_correct,
            "accuracy": round((grounded_correct / grounded_total), 4) if grounded_total else 0.0,
        },
        "hallucinated_answer": {
            "total": hallucinated_total,
            "correct": hallucinated_correct,
            "accuracy": round((hallucinated_correct / hallucinated_total), 4) if hallucinated_total else 0.0,
        },
        "hallucination_detection": classification_metrics(y_true, y_pred, positive_label="Yes"),
        "grounded_detection": classification_metrics(y_true, y_pred, positive_label="No"),
    }

    summary_path = output_dir / "qa_hallucination_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reflexionAgentRAG on legal qa_hallucination_benchmark.jsonl."
    )
    parser.add_argument(
        "--benchmark-file",
        default=str(ROOT / "data" / "benchmark_from_chunks" / "qa_hallucination_benchmark.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "eval_outputs_qa_hallucination_reflexion"),
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--use-reflexion", action="store_true")
    parser.add_argument("--max-iters", type=int, default=3)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    rows = load_jsonl(Path(args.benchmark_file))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    summary = evaluate(rows, Path(args.output_dir), use_reflexion=args.use_reflexion, max_iters=args.max_iters)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
