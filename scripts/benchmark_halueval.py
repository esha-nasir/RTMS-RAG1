import argparse
import json
import os
import sys
import random
import re
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

YES_NO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)
STRICT_LABEL_RE = re.compile(r"^\s*(?:answer\s*[:\-]\s*)?(yes|no)\s*[.!]?\s*$", flags=re.IGNORECASE)
_GENERATORS = None
_DETECTOR_FN = None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def parse_yes_no(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in lines[:4]:
        m = STRICT_LABEL_RE.match(ln)
        if m:
            return m.group(1).capitalize()

    lead = raw[:120]
    lead = re.split(r"citations?:", lead, flags=re.IGNORECASE)[0]
    m = YES_NO_RE.search(lead)
    if m:
        return m.group(1).capitalize()

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


def load_json_objects(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_detector_fn():
    global _DETECTOR_FN
    if _DETECTOR_FN is None:
        from rtms_rag.rag import _call_completion

        _DETECTOR_FN = _call_completion
    return _DETECTOR_FN


def call_rag(prompt: str, use_reflexion: bool, max_iters: int) -> str:
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


def prompt_qa(question: str, candidate_answer: str) -> str:
    return (
        "You are a hallucination detector.\n"
        "Given a question and candidate answer, output exactly one token: Yes or No.\n"
        "Do not add explanation, citations, punctuation, or extra text.\n"
        "Yes = contains hallucination / factual error.\n"
        "No = answer is factual and grounded.\n\n"
        f"Question: {question}\n"
        f"Candidate Answer: {candidate_answer}\n\n"
        "Output:"
    )


def prompt_dialogue(dialogue_history: str, candidate_response: str) -> str:
    return (
        "You are a hallucination detector for dialogue responses.\n"
        "Output exactly one token: Yes or No.\n"
        "Do not add explanation, citations, punctuation, or extra text.\n"
        "Yes = contains hallucination / factual error.\n"
        "No = factual and grounded.\n\n"
        f"Dialogue History: {dialogue_history}\n"
        f"Candidate Response: {candidate_response}\n\n"
        "Output:"
    )


def prompt_summary(document: str, candidate_summary: str) -> str:
    return (
        "You are a hallucination detector for summaries.\n"
        "Output exactly one token: Yes or No.\n"
        "Do not add explanation, citations, punctuation, or extra text.\n"
        "Yes = summary contains hallucination.\n"
        "No = summary is factual relative to the document.\n\n"
        f"Document: {document}\n"
        f"Candidate Summary: {candidate_summary}\n\n"
        "Output:"
    )


def prompt_general(query: str, response: str) -> str:
    return (
        "You are a hallucination detector.\n"
        "Output exactly one token: Yes or No.\n"
        "Do not add explanation, citations, punctuation, or extra text.\n"
        "Yes = the response contains a hallucination.\n"
        "No = the response does not contain hallucination.\n\n"
        f"User Query: {query}\n"
        f"Response: {response}\n\n"
        "Output:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HaluEval benchmark against reflexionAgentRAG (in-process)")
    parser.add_argument("--halu-root", default="/Users/eshanasir/haluEval", help="Path to HaluEval root")
    parser.add_argument("--n", type=int, default=100, help="Max source rows per task (0=all)")
    parser.add_argument("--tasks", default="qa,dialogue,summarization,general", help="Comma list of tasks")
    parser.add_argument("--use-reflexion", action="store_true", help="Enable reflexion loop")
    parser.add_argument("--max-iters", type=int, default=2, help="Reflexion max iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="", help="Optional output dir for predictions/summary")
    parser.add_argument("--env-file", default=str(Path(__file__).resolve().parents[1] / ".env"), help="Env file path")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    # Benchmark defaults: avoid legal-corpus retrieval and memory growth/cost.
    os.environ.setdefault("REFLEXION_BENCHMARK_NO_RETRIEVAL", "1")
    os.environ.setdefault("REFLEXION_DISABLE_MEMORY", "1")
    random.seed(args.seed)
    data_dir = Path(args.halu_root) / "data"

    task_files = {
        "qa": data_dir / "qa_data.json",
        "dialogue": data_dir / "dialogue_data.json",
        "summarization": data_dir / "summarization_data.json",
        "general": data_dir / "general_data.json",
    }

    selected_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    summary = {}
    all_records = []

    for task in selected_tasks:
        if task not in task_files:
            print(f"Skipping unknown task: {task}")
            continue
        rows = load_json_objects(task_files[task])
        if args.n > 0:
            rows = rows[: args.n]

        y_true = []
        y_pred = []
        parse_failures = 0

        for row in rows:
            if task == "qa":
                pairs = [
                    (prompt_qa(row["question"], row["right_answer"]), "No"),
                    (prompt_qa(row["question"], row["hallucinated_answer"]), "Yes"),
                ]
            elif task == "dialogue":
                pairs = [
                    (prompt_dialogue(row["dialogue_history"], row["right_response"]), "No"),
                    (prompt_dialogue(row["dialogue_history"], row["hallucinated_response"]), "Yes"),
                ]
            elif task == "summarization":
                pairs = [
                    (prompt_summary(row["document"], row["right_summary"]), "No"),
                    (prompt_summary(row["document"], row["hallucinated_summary"]), "Yes"),
                ]
            else:
                truth = "Yes" if normalize(str(row.get("hallucination", row.get("hallucination_label", "")))) == "yes" else "No"
                pairs = [(prompt_general(row["user_query"], row["chatgpt_response"]), truth)]

            for prompt, truth in pairs:
                answer = call_rag(prompt, args.use_reflexion, args.max_iters)
                pred = parse_yes_no(answer)
                if not pred:
                    parse_failures += 1
                    pred = ""

                y_true.append(truth)
                y_pred.append(pred)

                all_records.append(
                    {
                        "task": task,
                        "ground_truth": truth,
                        "prediction": pred,
                        "raw_output": answer,
                    }
                )

        total = len(y_true)
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        accuracy = (correct / total) if total else 0.0

        summary[task] = {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "parse_failures": parse_failures,
        }

    print(json.dumps(summary, indent=2))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary_path = out / "halueval_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        pred_path = out / "halueval_predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
