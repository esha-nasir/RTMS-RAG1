import argparse
import csv
import json
import math
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rtms_rag.rag import _call_completion


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def load_tsv(task_dir: Path) -> list[dict]:
    for candidate in ("test.tsv", "test_old.tsv", "train.tsv", "train_old.tsv"):
        path = task_dir / candidate
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                return [dict(row) for row in reader]
    raise FileNotFoundError(f"No TSV found in {task_dir}")


def choose_text_field(rows: list[dict]) -> str:
    if not rows:
        return "text"
    preferred = ("text", "statement", "premise", "input", "document", "passage")
    keys = rows[0].keys()
    for key in preferred:
        if key in keys:
            return key
    for key in keys:
        if key.lower() != "label":
            return key
    return "text"


def choose_label_field(rows: list[dict]) -> str:
    if not rows:
        return "label"
    preferred = ("label", "answer", "target", "gold")
    keys = rows[0].keys()
    for key in preferred:
        if key in keys:
            return key
    return "label"


def extract_labels(rows: list[dict], label_field: str) -> list[str]:
    labels = []
    for row in rows:
        label = str(row.get(label_field, "")).strip()
        if label:
            labels.append(label)
    return sorted(dict.fromkeys(labels))


def extract_label(answer_text: str, labels: list[str]) -> str:
    ans_norm = normalize_text(answer_text)
    if not ans_norm:
        return "Unknown"

    label_norm_pairs = [(label, normalize_text(label)) for label in labels]
    for label, label_norm in label_norm_pairs:
        if ans_norm == label_norm:
            return label
    for label, label_norm in label_norm_pairs:
        if re.search(rf"\b{re.escape(label_norm)}\b", ans_norm):
            return label
    for label, label_norm in label_norm_pairs:
        if label_norm and label_norm in ans_norm:
            return label
    return "Unknown"


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    by_label_total = Counter(y_true)
    by_label_correct = Counter()
    for truth, pred in zip(y_true, y_pred):
        if truth == pred:
            by_label_correct[truth] += 1
    recalls = []
    for label, total in by_label_total.items():
        if total <= 0:
            continue
        recalls.append(by_label_correct[label] / total)
    return sum(recalls) / len(recalls) if recalls else 0.0


def build_prompt(task_name: str, text: str, labels: list[str]) -> str:
    joined = ", ".join(labels)
    return (
        f"You are solving a LegalBench task: {task_name}.\n"
        f"Choose exactly one label from: {joined}.\n"
        "Reply with only the label, no explanation.\n\n"
        f"Text:\n{text}"
    )


def build_reflexion_critique_prompt(task_name: str, text: str, labels: list[str], answer: str) -> str:
    joined = ", ".join(labels)
    return (
        "You are critiquing a LegalBench classification answer.\n"
        f"Task: {task_name}\n"
        f"Allowed labels: {joined}\n"
        f"Previous answer: {answer}\n\n"
        "Return ONLY JSON with keys:\n"
        '- is_sufficient: true or false\n'
        '- issues: array of short strings\n'
        '- guidance: short string\n\n'
        f"Text:\n{text}\n"
    )


def extract_json_object(text: str) -> dict:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def predict_label(task_name: str, text: str, labels: list[str]) -> str:
    prompt = build_prompt(task_name, text, labels)
    answer = _call_completion(prompt, temperature=0.0, max_tokens=32, json_mode=False)
    return extract_label(answer, labels)


def predict_label_with_reflexion(task_name: str, text: str, labels: list[str], max_iters: int) -> tuple[str, list[dict]]:
    trace = []
    guidance = ""
    final_label = "Unknown"
    for iteration in range(1, max_iters + 1):
        base_prompt = build_prompt(task_name, text, labels)
        if guidance:
            base_prompt += f"\n\nRevision guidance:\n{guidance}"
        answer = _call_completion(base_prompt, temperature=0.0, max_tokens=32, json_mode=False)
        label = extract_label(answer, labels)
        final_label = label

        critique_prompt = build_reflexion_critique_prompt(task_name, text, labels, label)
        critique_raw = _call_completion(
            critique_prompt,
            temperature=0.0,
            max_tokens=160,
            json_mode=False,
        )
        critique = extract_json_object(critique_raw)
        is_sufficient = bool(critique.get("is_sufficient", iteration == max_iters))
        guidance = str(critique.get("guidance", "")).strip()
        trace.append(
            {
                "iter": iteration,
                "label": label,
                "is_sufficient": is_sufficient,
                "issues": critique.get("issues", []),
                "guidance": guidance,
            }
        )
        if is_sufficient:
            break
    return final_label, trace


def evaluate_task(task_dir: Path, task_name: str, use_reflexion: bool, max_iters: int, max_samples: int) -> dict:
    rows = load_tsv(task_dir)
    if max_samples > 0:
        rows = rows[:max_samples]

    text_field = choose_text_field(rows)
    label_field = choose_label_field(rows)
    labels = extract_labels(rows, label_field)
    if not labels:
        raise SystemExit(f"Could not infer labels for task {task_name}")

    y_true = []
    y_pred = []
    traces = []

    for row in rows:
        text = str(row.get(text_field, "")).strip()
        truth = str(row.get(label_field, "")).strip()
        if not text or not truth:
            continue
        if use_reflexion:
            pred, trace = predict_label_with_reflexion(task_name, text, labels, max_iters=max_iters)
            traces.append({"text": text[:300], "truth": truth, "pred": pred, "trace": trace})
        else:
            pred = predict_label(task_name, text, labels)
        y_true.append(truth)
        y_pred.append(pred)

    total = len(y_true)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    accuracy = correct / total if total else 0.0
    bal_acc = balanced_accuracy(y_true, y_pred)

    per_label = {}
    by_label_total = Counter(y_true)
    by_label_correct = Counter()
    for truth, pred in zip(y_true, y_pred):
        if truth == pred:
            by_label_correct[truth] += 1
    for label, total_count in by_label_total.items():
        per_label[label] = {
            "total": total_count,
            "correct": by_label_correct[label],
            "recall": round(by_label_correct[label] / total_count if total_count else 0.0, 4),
        }

    return {
        "task_name": task_name,
        "text_field": text_field,
        "label_field": label_field,
        "labels": labels,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "per_label": per_label,
        "examples": traces[:10],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run LegalBench task benchmark against the local LLM stack.")
    parser.add_argument("--legalbench-dir", default=str(root / "LegalBench"))
    parser.add_argument("--task", required=True, help="Task directory name under the LegalBench root.")
    parser.add_argument("--use-reflexion", action="store_true")
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--output-dir", default=str(root / "eval_outputs_legalbench"))
    args = parser.parse_args()

    task_dir = Path(args.legalbench_dir) / args.task
    if not task_dir.exists():
        raise SystemExit(f"Missing LegalBench task directory: {task_dir}")

    result = evaluate_task(
        task_dir=task_dir,
        task_name=args.task,
        use_reflexion=args.use_reflexion,
        max_iters=args.max_iters,
        max_samples=args.max_samples,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.task}_{'reflexion' if args.use_reflexion else 'baseline'}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
