import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    for key, value in _load_env_file(ROOT / ".env").items():
        env.setdefault(key, value)
    return env


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _memory_hit_summary(predictions_path: Path) -> dict:
    rows = _load_jsonl(predictions_path)
    total = len(rows)
    hit_count = 0
    hit_sum = 0
    prompt_first_count = 0
    for row in rows:
        trace = row.get("trace", {})
        inner = trace.get("trace") if isinstance(trace, dict) else {}
        if not isinstance(inner, dict):
            inner = {}
        hits = int(inner.get("memory_hits", 0) or 0)
        hit_sum += hits
        if hits > 0:
            hit_count += 1
        attempts = inner.get("attempts", [])
        if isinstance(attempts, list) and attempts:
            if bool(attempts[0].get("memory_guided_prompt_first", False)):
                prompt_first_count += 1
    return {
        "records": total,
        "records_with_memory_hits": hit_count,
        "memory_hit_record_rate": round(hit_count / total, 4) if total else 0.0,
        "total_memory_hits": hit_sum,
        "memory_guided_prompt_first_records": prompt_first_count,
    }


def _ablation_env(memory_dir: Path, base_env: dict[str, str], component_mode: str) -> dict[str, str]:
    env = dict(base_env)
    env.update(
        {
            "REFLEXION_MEMORY_DIR": str(memory_dir),
            # Enable benchmark memory so QReCC/RAGTruth fixed-context runs can
            # actually exercise the memory path.
            "REFLEXION_DISABLE_BENCHMARK_MEMORY": "0",
            "REFLEXION_QRECC_ALLOW_MEMORY_SAVE": "1",
            "REFLEXION_RAGTRUTH_ALLOW_MEMORY_SAVE": "1",
            "REFLEXION_MEMORY_GUIDED_PROMPT_FIRST": "1",
        }
    )

    if component_mode == "memory_only":
        env.update(
            {
                # Switch off non-memory components that otherwise dominate the
                # answer and hide the raw memory effect.
                "REFLEXION_ENABLE_SKEPTIC": "0",
                "REFLEXION_ENABLE_SELF_REFLECTION": "0",
                "REFLEXION_ENABLE_MULTI_LEVEL_REFLECTION": "0",
                "REFLEXION_ENABLE_EPISODE_REFLECTION": "0",
                "REFLEXION_FAST_MODE": "1",
                # Let memory enter as regular past reflection text, not only as
                # conservative process guidance.
                "REFLEXION_RAGTRUTH_FORCE_GUIDANCE_ONLY_MEMORY": "0",
                "REFLEXION_GROUNDED_FORCE_GUIDANCE_ONLY_MEMORY": "0",
                # Make the retrieval threshold permissive enough for an
                # isolation ablation to observe memory impact without editing
                # rag.py.
                "REFLEXION_GROUNDED_MIN_MEMORY_SCORE": "0.05",
                "REFLEXION_GROUNDED_MIN_MEMORY_CONTENT_OVERLAP": "0",
                "REFLEXION_GROUNDED_MAX_MEMORY_HALLUCINATION_RISK": "1.0",
                "REFLEXION_GROUNDED_TOP_N": "3",
            }
        )
    elif component_mode == "full_no_skeptic":
        env.update(
            {
                "REFLEXION_ENABLE_SKEPTIC": "0",
                "REFLEXION_ENABLE_SELF_REFLECTION": "1",
                "REFLEXION_ENABLE_MULTI_LEVEL_REFLECTION": "1",
                "REFLEXION_ENABLE_EPISODE_REFLECTION": "1",
                "REFLEXION_FAST_MODE": "0",
            }
        )
    elif component_mode == "full":
        env.update(
            {
                "REFLEXION_ENABLE_SKEPTIC": "1",
                "REFLEXION_ENABLE_SELF_REFLECTION": "1",
                "REFLEXION_ENABLE_MULTI_LEVEL_REFLECTION": "1",
                "REFLEXION_ENABLE_EPISODE_REFLECTION": "1",
                "REFLEXION_FAST_MODE": "0",
            }
        )
    else:
        raise ValueError(f"Unsupported component mode: {component_mode}")
    return env


def _evaluate(
    *,
    benchmark_file: Path,
    output_dir: Path,
    memory_dir: Path,
    disable_memory: bool,
    max_samples: int,
    top_k: int,
    max_iters: int,
    hallucination_source: str,
    detection_target: str,
    component_mode: str,
) -> None:
    env = _ablation_env(memory_dir, _base_env(), component_mode=component_mode)
    if disable_memory:
        env["REFLEXION_DISABLE_MEMORY"] = "1"
    else:
        env["REFLEXION_DISABLE_MEMORY"] = "0"

    cmd = [
        sys.executable,
        "evaluate_cross_repo_fixed_context.py",
        "--backend",
        "temporal",
        "--use-reflexion",
        "--benchmark-file",
        str(benchmark_file),
        "--output-dir",
        str(output_dir),
        "--max-iters",
        str(max_iters),
        "--top-k",
        str(top_k),
        "--hallucination-source",
        hallucination_source,
        "--detection-target",
        detection_target,
    ]
    if max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])
    if disable_memory:
        cmd.append("--disable-memory")
    _run(cmd, env)


def _write_comparison(output_dir: Path, no_memory_dir: Path, with_memory_dir: Path) -> None:
    no_mem = _load_summary(no_memory_dir / "fixed_context_summary.json")
    with_mem = _load_summary(with_memory_dir / "fixed_context_summary.json")

    no_gen = no_mem.get("generation", {})
    with_gen = with_mem.get("generation", {})
    no_qa = no_mem.get("qa_quality", {})
    with_qa = with_mem.get("qa_quality", {})

    comparison = {
        "no_memory_summary": str(no_memory_dir / "fixed_context_summary.json"),
        "with_memory_summary": str(with_memory_dir / "fixed_context_summary.json"),
        "generation_hallucination_rate": {
            "no_memory": no_gen.get("hallucination_rate"),
            "with_memory": with_gen.get("hallucination_rate"),
            "delta_with_minus_no": (
                round(float(with_gen.get("hallucination_rate", 0.0)) - float(no_gen.get("hallucination_rate", 0.0)), 4)
                if no_gen and with_gen
                else None
            ),
        },
        "generation_non_hallucination_rate": {
            "no_memory": no_gen.get("non_hallucination_rate"),
            "with_memory": with_gen.get("non_hallucination_rate"),
            "delta_with_minus_no": (
                round(float(with_gen.get("non_hallucination_rate", 0.0)) - float(no_gen.get("non_hallucination_rate", 0.0)), 4)
                if no_gen and with_gen
                else None
            ),
        },
        "qrecc_answer_f1": {
            "no_memory": no_qa.get("answer_f1"),
            "with_memory": with_qa.get("answer_f1"),
            "delta_with_minus_no": (
                round(float(with_qa.get("answer_f1", 0.0)) - float(no_qa.get("answer_f1", 0.0)), 4)
                if no_qa and with_qa
                else None
            ),
        },
        "qrecc_exact_match_rate": {
            "no_memory": no_qa.get("exact_match_rate"),
            "with_memory": with_qa.get("exact_match_rate"),
            "delta_with_minus_no": (
                round(float(with_qa.get("exact_match_rate", 0.0)) - float(no_qa.get("exact_match_rate", 0.0)), 4)
                if no_qa and with_qa
                else None
            ),
        },
        "memory_usage": {
            "no_memory": _memory_hit_summary(no_memory_dir / "fixed_context_predictions.jsonl"),
            "with_memory": _memory_hit_summary(with_memory_dir / "fixed_context_predictions.jsonl"),
        },
    }
    (output_dir / "memory_only_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a memory-only ablation for ReflexionTemporalMemorySuccessAgentRAG. "
            "All major non-memory Reflexion components are disabled so the comparison "
            "isolates the impact of temporal memory."
        )
    )
    parser.add_argument(
        "--benchmark-file",
        default="data/qrecc_fixed_context_100_last50.jsonl",
        help="Fixed-context benchmark JSONL to evaluate.",
    )
    parser.add_argument(
        "--build-memory-file",
        default="data/qrecc_fixed_context_100_first50.jsonl",
        help=(
            "Optional fixed-context JSONL used to warm/build memory before the comparison. "
            "Use an empty string to skip warmup and use an existing memory dir."
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--memory-dir", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--build-max-samples", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument(
        "--component-mode",
        choices=("memory_only", "full_no_skeptic", "full"),
        default="memory_only",
        help=(
            "memory_only disables skeptic/reflection extras; full_no_skeptic "
            "enables reflection components but leaves skeptic off; full enables "
            "reflection components and skeptic."
        ),
    )
    parser.add_argument(
        "--hallucination-source",
        choices=("final_answer", "skeptic", "auto"),
        default="final_answer",
    )
    parser.add_argument(
        "--detection-target",
        choices=("auto", "reference_response", "generated_answer"),
        default="generated_answer",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_memory_only_ablation")
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else ROOT / "eval_clean_runs" / run_id
    memory_dir = Path(args.memory_dir).expanduser() if args.memory_dir else output_dir / "memory_store"
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    benchmark_file = Path(args.benchmark_file)
    if not benchmark_file.is_absolute():
        benchmark_file = ROOT / benchmark_file

    build_memory_file = Path(args.build_memory_file) if str(args.build_memory_file).strip() else None
    if build_memory_file is not None and not build_memory_file.is_absolute():
        build_memory_file = ROOT / build_memory_file

    if build_memory_file is not None:
        _evaluate(
            benchmark_file=build_memory_file,
            output_dir=output_dir / "00_build_memory",
            memory_dir=memory_dir,
            disable_memory=False,
            max_samples=args.build_max_samples,
            top_k=args.top_k,
            max_iters=args.max_iters,
            hallucination_source=args.hallucination_source,
            detection_target=args.detection_target,
            component_mode=args.component_mode,
        )

    no_memory_dir = output_dir / "01_no_memory_all_extra_components_off"
    with_memory_dir = output_dir / "02_with_temporal_memory_all_extra_components_off"
    _evaluate(
        benchmark_file=benchmark_file,
        output_dir=no_memory_dir,
        memory_dir=memory_dir,
        disable_memory=True,
        max_samples=args.max_samples,
        top_k=args.top_k,
        max_iters=args.max_iters,
        hallucination_source=args.hallucination_source,
        detection_target=args.detection_target,
        component_mode=args.component_mode,
    )
    _evaluate(
        benchmark_file=benchmark_file,
        output_dir=with_memory_dir,
        memory_dir=memory_dir,
        disable_memory=False,
        max_samples=args.max_samples,
        top_k=args.top_k,
        max_iters=args.max_iters,
        hallucination_source=args.hallucination_source,
        detection_target=args.detection_target,
        component_mode=args.component_mode,
    )

    _write_comparison(output_dir, no_memory_dir, with_memory_dir)
    print(f"\nWrote memory-only ablation to: {output_dir}")
    print(f"Memory store: {memory_dir}")
    print(f"Comparison: {output_dir / 'memory_only_comparison.json'}")


if __name__ == "__main__":
    main()
