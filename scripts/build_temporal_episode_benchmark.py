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
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalized_group_key(row: dict[str, Any], group_by: str) -> str:
    if group_by == "case_no":
        value = str(row.get("case_no") or "").strip()
        if value:
            return value
    return str(row.get("file_name") or row.get("case_no") or row.get("source_name") or "unknown").strip()


def sort_key(row: dict[str, Any]) -> tuple[int, str]:
    chunk_id = row.get("chunk_id")
    if isinstance(chunk_id, int):
        return (chunk_id, str(row.get("id", "")))
    if isinstance(chunk_id, str):
        digits = "".join(ch for ch in chunk_id if ch.isdigit())
        if digits:
            return (int(digits), str(row.get("id", "")))
    return (10**9, str(row.get("id", "")))


def episode_label(index: int) -> str:
    return f"episode_{index:04d}"


def build_episodes(
    rows: list[dict[str, Any]],
    group_by: str,
    min_episode_size: int,
    max_episode_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalized_group_key(row, group_by)].append(dict(row))

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )

    episodic_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    episode_index = 1

    for group_key, group_rows in ordered_groups:
        if len(group_rows) < min_episode_size:
            continue
        group_rows = sorted(group_rows, key=sort_key)
        for start in range(0, len(group_rows), max_episode_size):
            chunk = group_rows[start : start + max_episode_size]
            if len(chunk) < min_episode_size:
                continue
            current_episode_id = episode_label(episode_index)
            manifest_rows.append(
                {
                    "episode_id": current_episode_id,
                    "group_key": group_key,
                    "group_by": group_by,
                    "size": len(chunk),
                    "source_ids": [str(row.get("id", "")).strip() for row in chunk],
                }
            )
            for step_index, row in enumerate(chunk, start=1):
                enriched = dict(row)
                enriched["episode_id"] = current_episode_id
                enriched["episode_step"] = step_index
                enriched["episode_size"] = len(chunk)
                enriched["temporal_group_key"] = group_key
                enriched["benchmark_type"] = str(row.get("benchmark_type") or "temporal_episode")
                enriched["temporal_benchmark_type"] = "episodic_temporal_grounded_qa"
                episodic_rows.append(enriched)
            episode_index += 1

    return episodic_rows, manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an episode-ordered temporal benchmark from an existing JSONL benchmark."
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--manifest-file", default="")
    parser.add_argument("--group-by", choices=["file_name", "case_no"], default="file_name")
    parser.add_argument("--min-episode-size", type=int, default=3)
    parser.add_argument("--max-episode-size", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    manifest_path = Path(args.manifest_file) if args.manifest_file else output_path.with_suffix(".manifest.json")

    rows = load_jsonl(input_path)
    episodic_rows, manifest_rows = build_episodes(
        rows,
        group_by=args.group_by,
        min_episode_size=max(2, args.min_episode_size),
        max_episode_size=max(2, args.max_episode_size),
    )
    if not episodic_rows:
        raise SystemExit("No episodes were built. Try lowering --min-episode-size or changing --group-by.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, episodic_rows)

    manifest = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "group_by": args.group_by,
        "min_episode_size": max(2, args.min_episode_size),
        "max_episode_size": max(2, args.max_episode_size),
        "total_rows": len(rows),
        "episodic_rows": len(episodic_rows),
        "episodes": len(manifest_rows),
        "episode_manifest": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
