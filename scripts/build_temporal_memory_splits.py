from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def qrecc_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    try:
        turn_id = int(row.get("turn_id", 0) or 0)
    except (TypeError, ValueError):
        turn_id = 0
    return turn_id, str(row.get("id", ""))


def legal_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    try:
        step = int(row.get("episode_step", 0) or 0)
    except (TypeError, ValueError):
        step = 0
    try:
        chunk_id = int(row.get("chunk_id", 0) or 0)
    except (TypeError, ValueError):
        chunk_id = 0
    return step, chunk_id, str(row.get("id", ""))


def split_index(size: int, build_fraction: float) -> int:
    index = int(round(size * build_fraction))
    return min(max(index, 1), size - 1)


def build_qrecc_episode_split(
    input_files: list[Path],
    output_dir: Path,
    min_episode_size: int,
    build_fraction: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for input_file in input_files:
        for row in load_jsonl(input_file):
            conversation_id = str(row.get("conversation_id") or "").strip()
            if conversation_id:
                grouped[conversation_id].append(dict(row))

    build_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for conversation_id, rows in sorted(grouped.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        rows = sorted(rows, key=qrecc_sort_key)
        if len(rows) < min_episode_size:
            continue
        split_at = split_index(len(rows), build_fraction)
        episode_id = f"qrecc_conversation_{conversation_id}"
        file_name = f"QReCC_{episode_id}"
        case_no = episode_id
        for phase, target, phase_rows in (
            ("build", build_rows, rows[:split_at]),
            ("test", test_rows, rows[split_at:]),
        ):
            for step, row in enumerate(phase_rows, start=1):
                enriched = dict(row)
                enriched["episode_id"] = episode_id
                enriched["episode_phase"] = phase
                enriched["episode_step"] = int(row.get("turn_id", step) or step)
                enriched["episode_size"] = len(rows)
                enriched["temporal_group_key"] = episode_id
                enriched["file_name"] = file_name
                enriched["case_no"] = case_no
                enriched["benchmark_type"] = "qrecc_episode_temporal"
                enriched["temporal_benchmark_type"] = "qrecc_episode_memory_grounded_qa"
                target.append(enriched)
        manifest.append(
            {
                "episode_id": episode_id,
                "conversation_id": conversation_id,
                "total_turns": len(rows),
                "build_turns": split_at,
                "test_turns": len(rows) - split_at,
                "source_ids": [str(row.get("id", "")) for row in rows],
            }
        )

    build_path = output_dir / "qrecc_episode_build.jsonl"
    test_path = output_dir / "qrecc_episode_test.jsonl"
    write_jsonl(build_path, build_rows)
    write_jsonl(test_path, test_rows)
    summary = {
        "kind": "qrecc",
        "input_files": [str(path) for path in input_files],
        "build_file": str(build_path),
        "test_file": str(test_path),
        "episodes": len(manifest),
        "build_rows": len(build_rows),
        "test_rows": len(test_rows),
        "manifest": manifest,
    }
    (output_dir / "qrecc_episode_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_legal_episode_split(
    input_file: Path,
    output_dir: Path,
    min_episode_size: int,
    build_fraction: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(input_file):
        episode_id = str(row.get("episode_id") or row.get("temporal_group_key") or row.get("file_name") or "").strip()
        if episode_id:
            grouped[episode_id].append(dict(row))

    build_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for episode_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=legal_sort_key)
        if len(rows) < min_episode_size:
            continue
        split_at = split_index(len(rows), build_fraction)
        for phase, target, phase_rows in (
            ("build", build_rows, rows[:split_at]),
            ("test", test_rows, rows[split_at:]),
        ):
            for row in phase_rows:
                enriched = dict(row)
                enriched["episode_phase"] = phase
                enriched["temporal_benchmark_type"] = "legal_episode_memory_grounded_qa"
                target.append(enriched)
        manifest.append(
            {
                "episode_id": episode_id,
                "total_turns": len(rows),
                "build_turns": split_at,
                "test_turns": len(rows) - split_at,
                "source_ids": [str(row.get("id", "")) for row in rows],
            }
        )

    build_path = output_dir / "legal_episode_build.jsonl"
    test_path = output_dir / "legal_episode_test.jsonl"
    write_jsonl(build_path, build_rows)
    write_jsonl(test_path, test_rows)
    summary = {
        "kind": "legal",
        "input_file": str(input_file),
        "build_file": str(build_path),
        "test_file": str(test_path),
        "episodes": len(manifest),
        "build_rows": len(build_rows),
        "test_rows": len(test_rows),
        "manifest": manifest,
    }
    (output_dir / "legal_episode_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal memory train/test splits.")
    parser.add_argument("--kind", choices=["qrecc", "legal"], required=True)
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-episode-size", type=int, default=3)
    parser.add_argument("--build-fraction", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.kind == "qrecc":
        summary = build_qrecc_episode_split(
            [Path(path) for path in args.input_files],
            output_dir,
            min_episode_size=max(2, args.min_episode_size),
            build_fraction=args.build_fraction,
        )
    else:
        if len(args.input_files) != 1:
            raise SystemExit("legal split expects exactly one --input-files path")
        summary = build_legal_episode_split(
            Path(args.input_files[0]),
            output_dir,
            min_episode_size=max(2, args.min_episode_size),
            build_fraction=args.build_fraction,
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
