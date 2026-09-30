from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download


# ==========================================
# 1. CONFIGURATION
# ==========================================
ORIGINAL_REPO = "poolvarine/robomimic-mh-lift-image-dense"
NEW_REPO = "poolvarine/robomimic-mh-lift-image-dense-negshift-success5"
HF_TOKEN = os.environ.get("HF_TOKEN")

MAX_DENSE_REWARD = 1.25 / 2.25
SUCCESS_BONUS = 0.0
SUCCESS_REWARD_OVERRIDE = 5.0
ZERO_TOLERANCE = 1e-12


def _snap_near_zero(value: float, *, zero_tolerance: float) -> float:
    value = float(value)
    if abs(value) <= float(zero_tolerance):
        return 0.0
    return value


def _transform_dense_reward(
    reward: float,
    *,
    is_success: bool,
    max_dense_reward: float,
    success_bonus: float,
    success_reward_override: float | None,
    zero_tolerance: float,
) -> float:
    shifted_reward = _snap_near_zero(float(reward) - float(max_dense_reward), zero_tolerance=zero_tolerance)

    if not is_success:
        return shifted_reward

    if success_reward_override is not None:
        return _snap_near_zero(float(success_reward_override), zero_tolerance=zero_tolerance)

    if success_bonus:
        shifted_reward += float(success_bonus)

    return _snap_near_zero(shifted_reward, zero_tolerance=zero_tolerance)


def require_token() -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Export a Hugging Face write token before running this script.")
    return HF_TOKEN


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=4)
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dump_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def trim_and_transform_episode(
    parquet_path: Path,
    *,
    max_dense_reward: float,
    success_bonus: float,
    success_reward_override: float | None,
    zero_tolerance: float,
) -> tuple[int, int, int]:
    df = pd.read_parquet(parquet_path)
    original_rows = len(df)

    done_mask = df["next.done"].astype(bool)
    if done_mask.any():
        first_done_pos = int(done_mask.to_numpy().argmax())
        df = df.iloc[: first_done_pos + 1].copy()
    else:
        df = df.copy()

    df["next.reward"] = [
        _transform_dense_reward(
            reward,
            is_success=bool(is_done),
            max_dense_reward=max_dense_reward,
            success_bonus=success_bonus,
            success_reward_override=success_reward_override,
            zero_tolerance=zero_tolerance,
        )
        for reward, is_done in zip(df["next.reward"].tolist(), df["next.done"].tolist(), strict=True)
    ]

    transformed_success_rows = int(done_mask.sum())
    df.to_parquet(parquet_path, index=False)
    return original_rows, len(df), transformed_success_rows


def update_info_json(info_path: Path, total_frames: int) -> None:
    info = load_json(info_path)
    info["total_frames"] = total_frames
    dump_json(info_path, info)


def update_episodes_jsonl(episodes_path: Path, episode_lengths: dict[int, int]) -> None:
    records = load_jsonl(episodes_path)
    for record in records:
        episode_index = int(record["episode_index"])
        if episode_index not in episode_lengths:
            raise KeyError(f"Episode {episode_index} present in episodes.jsonl but missing parquet output")
        record["length"] = int(episode_lengths[episode_index])
    dump_jsonl(episodes_path, records)


def update_readme(
    readme_path: Path,
    *,
    source_repo: str,
    target_repo: str,
    max_dense_reward: float,
    success_bonus: float,
    success_reward_override: float | None,
    total_frames: int,
    removed_frames: int,
    info: dict,
) -> None:
    original_text = readme_path.read_text(encoding="utf-8")
    if original_text.startswith("## Derived Dataset Note\n"):
        _, _, remainder = original_text.partition("\n---\n")
        if remainder:
            original_text = "---\n" + remainder

    success_text = (
        f"Set terminal success rewards to exactly {success_reward_override:g}."
        if success_reward_override is not None
        else f"Added success bonus of {success_bonus:g} after the dense-reward shift on terminal rows."
    )

    summary_lines = [
        "## Derived Dataset Note",
        "",
        f"This dataset was derived from `{source_repo}`.",
        "",
        "Applied preprocessing:",
        "- Trimmed each episode after the first `next.done == true` row (inclusive).",
        f"- Shifted every `next.reward` by subtracting `MAX_DENSE_REWARD = {max_dense_reward:.16f}`.",
        f"- Snapped numerically tiny values to exactly `0.0` using tolerance `{ZERO_TOLERANCE:g}`.",
        f"- {success_text}",
        f"- Updated metadata to reflect the new total frame count: {total_frames}.",
        f"- Removed trailing post-success rows: {removed_frames}.",
        "",
        f"Target repo: `{target_repo}`.",
        "",
    ]
    summary_text = "\n".join(summary_lines)

    info_marker = "[meta/info.json](meta/info.json):\n```json\n"
    if info_marker in original_text:
        prefix, _, remainder = original_text.partition(info_marker)
        _, closing, suffix = remainder.partition("\n```")
        if closing:
            rendered_info = json.dumps(info, indent=4)
            original_text = prefix + info_marker + rendered_info + "\n```" + suffix

    if original_text.startswith("---\n"):
        _, _, remainder = original_text[4:].partition("\n---\n")
        if remainder:
            frontmatter = "---\n" + original_text[4:].split("\n---\n", 1)[0] + "\n---\n"
            body = remainder
            new_text = frontmatter + "\n" + summary_text + body
        else:
            new_text = summary_text + original_text
    else:
        new_text = summary_text + original_text

    readme_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    token = require_token()
    api = HfApi(token=token)

    print(f"1. Downloading source dataset {ORIGINAL_REPO} locally...")
    with tempfile.TemporaryDirectory(prefix="custom_dense_lerobot_") as temp_dir:
        working_dir = Path(temp_dir) / "dataset_copy"
        snapshot_download(
            repo_id=ORIGINAL_REPO,
            repo_type="dataset",
            local_dir=str(working_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"   -> Downloaded to {working_dir}")

        print(f"2. Ensuring target dataset repo exists: {NEW_REPO}")
        api.create_repo(repo_id=NEW_REPO, repo_type="dataset", private=False, exist_ok=True)
        print("   -> Target repo is ready")

        print("3. Trimming trailing done rows and transforming rewards in each parquet...")
        data_dir = working_dir / "data"
        parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"No parquet files found under {data_dir}")

        total_original_frames = 0
        total_final_frames = 0
        total_success_rows = 0
        episode_lengths: dict[int, int] = {}

        for parquet_path in parquet_files:
            episode_index = int(parquet_path.stem.split("_")[-1])
            original_rows, final_rows, success_rows = trim_and_transform_episode(
                parquet_path,
                max_dense_reward=MAX_DENSE_REWARD,
                success_bonus=SUCCESS_BONUS,
                success_reward_override=SUCCESS_REWARD_OVERRIDE,
                zero_tolerance=ZERO_TOLERANCE,
            )
            total_original_frames += original_rows
            total_final_frames += final_rows
            total_success_rows += success_rows
            episode_lengths[episode_index] = final_rows

        removed_frames = total_original_frames - total_final_frames
        print(
            f"   -> Processed {len(parquet_files)} parquet files, removed {removed_frames} trailing rows, "
            f"and transformed {total_success_rows} terminal reward rows"
        )

        print("4. Updating metadata files...")
        info_path = working_dir / "meta" / "info.json"
        episodes_path = working_dir / "meta" / "episodes.jsonl"
        readme_path = working_dir / "README.md"

        update_info_json(info_path, total_final_frames)
        info = load_json(info_path)
        update_episodes_jsonl(episodes_path, episode_lengths)
        update_readme(
            readme_path,
            source_repo=ORIGINAL_REPO,
            target_repo=NEW_REPO,
            max_dense_reward=MAX_DENSE_REWARD,
            success_bonus=SUCCESS_BONUS,
            success_reward_override=SUCCESS_REWARD_OVERRIDE,
            total_frames=total_final_frames,
            removed_frames=removed_frames,
            info=info,
        )
        print("   -> Updated meta/info.json, meta/episodes.jsonl, and README.md")

        print(f"5. Uploading processed dataset to {NEW_REPO}...")
        commit_message = (
            f"Trim trailing done rows and shift rewards by MAX_DENSE_REWARD={MAX_DENSE_REWARD:.6f}"
        )
        if SUCCESS_REWARD_OVERRIDE is not None:
            commit_message += f" with success override {SUCCESS_REWARD_OVERRIDE:g}"
        elif SUCCESS_BONUS:
            commit_message += f" and success bonus {SUCCESS_BONUS:g}"
        api.upload_folder(
            folder_path=str(working_dir),
            repo_id=NEW_REPO,
            repo_type="dataset",
            commit_message=commit_message,
        )
        print("   -> Upload complete")

        codebase_version = str(info.get("codebase_version", "")).strip()
        if codebase_version:
            print(f"6. Ensuring dataset tag '{codebase_version}' exists for LeRobot revision checks...")
            try:
                api.create_tag(repo_id=NEW_REPO, tag=codebase_version, repo_type="dataset")
                print(f"   -> Created tag: {codebase_version}")
            except Exception as exc:
                print(f"   -> Tag creation skipped: {exc}")
        else:
            print("6. Skipping tag creation: codebase_version missing in meta/info.json")

        print("\nDone.")
        print(f"Source repo kept unchanged: {ORIGINAL_REPO}")
        print(f"Target repo updated: {NEW_REPO}")
        print(f"Final total frames: {total_final_frames}")
        print(f"Removed trailing frames: {removed_frames}")
        print(f"Transformed terminal reward rows: {total_success_rows}")


if __name__ == "__main__":
    main()