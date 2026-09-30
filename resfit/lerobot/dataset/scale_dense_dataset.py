import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download


# ==========================================
# 1. CONFIGURATION
# ==========================================
ORIGINAL_REPO = "poolvarine/robomimic-mh-can-image-dense"
NEW_REPO = "poolvarine/robomimic-mh-can-image-dense-scaled-100"
HF_TOKEN = os.environ.get("HF_TOKEN")

SUCCESS_REWARD_SCALING_FACTOR = 100.0


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


def trim_and_scale_episode(parquet_path: Path, scale_factor: float) -> tuple[int, int, int]:
    df = pd.read_parquet(parquet_path)
    original_rows = len(df)

    done_mask = df["next.done"].astype(bool)
    if done_mask.any():
        first_done_pos = int(done_mask.to_numpy().argmax())
        df = df.iloc[: first_done_pos + 1].copy()
    else:
        df = df.copy()

    success_mask = df["next.done"].astype(bool)
    scaled_rows = int(success_mask.sum())
    if scaled_rows:
        df.loc[success_mask, "next.reward"] = df.loc[success_mask, "next.reward"] * scale_factor

    df.to_parquet(parquet_path, index=False)
    return original_rows, len(df), scaled_rows


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
    scale_factor: float,
    total_frames: int,
    removed_frames: int,
    info: dict,
) -> None:
    original_text = readme_path.read_text(encoding="utf-8")
    if original_text.startswith("## Derived Dataset Note\n"):
        _, _, remainder = original_text.partition("\n---\n")
        if remainder:
            original_text = "---\n" + remainder

    summary_lines = [
        "## Derived Dataset Note",
        "",
        f"This dataset was derived from `{source_repo}`.",
        "",
        "Applied preprocessing:",
        "- Trimmed each episode after the first `next.done == true` row (inclusive).",
        f"- Multiplied `next.reward` by {scale_factor:g} only on rows where `next.done == true`.",
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
    with tempfile.TemporaryDirectory(prefix="scaled_lerobot_") as temp_dir:
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

        print("3. Trimming trailing done rows and scaling terminal rewards in each parquet...")
        data_dir = working_dir / "data"

        # Support both layouts:
        #   (a) chunk-based: data/chunk-000/episode_000000.parquet (one per episode)
        #   (b) flat/train split: data/train-*.parquet (all episodes in one file)
        parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
        is_chunked = bool(parquet_files)

        if not is_chunked:
            parquet_files = sorted(data_dir.glob("*.parquet"))

        if not parquet_files:
            raise RuntimeError(f"No parquet files found under {data_dir}")

        total_original_frames = 0
        total_final_frames = 0
        total_scaled_rows = 0
        episode_lengths: dict[int, int] = {}

        if is_chunked:
            # One parquet per episode — process individually
            for parquet_path in parquet_files:
                episode_index = int(parquet_path.stem.split("_")[-1])
                original_rows, final_rows, scaled_rows = trim_and_scale_episode(
                    parquet_path,
                    SUCCESS_REWARD_SCALING_FACTOR,
                )
                total_original_frames += original_rows
                total_final_frames += final_rows
                total_scaled_rows += scaled_rows
                episode_lengths[episode_index] = final_rows
        else:
            # Single (or few) parquet file(s) with all episodes — process by episode_index groups
            for parquet_path in parquet_files:
                df = pd.read_parquet(parquet_path)
                total_original_frames += len(df)
                processed_dfs = []

                for ep_idx, ep_df in df.groupby("episode_index", sort=True):
                    ep_df = ep_df.copy()
                    original_ep_rows = len(ep_df)

                    done_mask = ep_df["next.done"].astype(bool)
                    if done_mask.any():
                        first_done_pos = int(done_mask.to_numpy().argmax())
                        ep_df = ep_df.iloc[: first_done_pos + 1].copy()

                    success_mask = ep_df["next.done"].astype(bool)
                    scaled_count = int(success_mask.sum())
                    if scaled_count:
                        ep_df.loc[success_mask, "next.reward"] = (
                            ep_df.loc[success_mask, "next.reward"] * SUCCESS_REWARD_SCALING_FACTOR
                        )
                    total_scaled_rows += scaled_count
                    total_final_frames += len(ep_df)
                    episode_lengths[int(ep_idx)] = len(ep_df)
                    processed_dfs.append(ep_df)

                result_df = pd.concat(processed_dfs, ignore_index=True)
                # Re-assign sequential index column if present
                if "index" in result_df.columns:
                    result_df["index"] = range(len(result_df))
                result_df.to_parquet(parquet_path, index=False)

        removed_frames = total_original_frames - total_final_frames
        print(
            f"   -> Processed {len(parquet_files)} parquet file(s), removed {removed_frames} trailing rows, "
            f"and scaled {total_scaled_rows} terminal reward rows"
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
            scale_factor=SUCCESS_REWARD_SCALING_FACTOR,
            total_frames=total_final_frames,
            removed_frames=removed_frames,
            info=info,
        )
        print("   -> Updated meta/info.json, meta/episodes.jsonl, and README.md")

        print(f"5. Uploading processed dataset to {NEW_REPO}...")
        api.upload_folder(
            folder_path=str(working_dir),
            repo_id=NEW_REPO,
            repo_type="dataset",
            commit_message=(
                f"Trim trailing done rows and scale terminal rewards by {SUCCESS_REWARD_SCALING_FACTOR:g}x"
            ),
        )
        print("   -> Upload complete")

        codebase_version = str(info.get("codebase_version", "")).strip()
        if codebase_version:
            print(f"6. Ensuring dataset tag '{codebase_version}' exists for LeRobot revision checks...")
            try:
                api.create_tag(repo_id=NEW_REPO, tag=codebase_version, repo_type="dataset")
                print(f"   -> Created tag: {codebase_version}")
            except Exception as exc:
                # Tag may already exist if this script was run before.
                print(f"   -> Tag creation skipped: {exc}")
        else:
            print("6. Skipping tag creation: codebase_version missing in meta/info.json")

        print("\nDone.")
        print(f"Source repo kept unchanged: {ORIGINAL_REPO}")
        print(f"Target repo updated: {NEW_REPO}")
        print(f"Final total frames: {total_final_frames}")
        print(f"Removed trailing frames: {removed_frames}")
        print(f"Scaled terminal reward rows: {total_scaled_rows}")


if __name__ == "__main__":
    main()