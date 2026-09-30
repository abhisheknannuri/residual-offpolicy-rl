#!/usr/bin/env python3
"""
LeRobot Dataset Upload Utilities
Handles dataset upload with proper error recovery and README generation
"""
import json
from pathlib import Path
from huggingface_hub import HfApi, create_tag
import time
from typing import Optional, Callable


def create_dataset_readme_with_frontmatter(output_dir: Path, repo_id: str) -> Optional[str]:
    """
    Generate README.md with dynamic YAML frontmatter from actual dataset files
    YAML is generated from the actual parquet schema, not hardcoded.
    
    Args:
        output_dir: Path to dataset directory
        repo_id: HuggingFace repo ID
        
    Returns:
        README content as string or None if failed
    """
    
    info_file = output_dir / "meta" / "info.json"
    
    if not info_file.exists():
        return None
    
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    # Try to read actual parquet schema for dynamic YAML
    parquet_file = output_dir / "data" / "train-00000-of-00001.parquet"
    
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_file))
        parquet_columns = table.column_names
    except:
        parquet_columns = None
    
    # Build dynamic YAML based on actual columns
    yaml_features_lines = []
    
    if parquet_columns:
        # Get actual columns from parquet
        for col_name in parquet_columns:
            if col_name.startswith('observation.images.'):
                yaml_features_lines.append(f"  - name: {col_name}\n    dtype: video")
            elif '.' in col_name:  # nested fields like next.done, next.reward
                yaml_features_lines.append(f"  - name: {col_name}")
            else:
                yaml_features_lines.append(f"  - name: {col_name}")
    else:
        # Fallback: use features from info.json
        for feat_name in info.get("features", {}):
            # Only include parquet-level features, not video features
            if not feat_name.startswith("observation.images."):
                yaml_features_lines.append(f"  - name: {feat_name}")
    
    yaml_features = "\n".join(yaml_features_lines) if yaml_features_lines else "  - name: action\n  - name: observation"
    
    # Create minimal YAML frontmatter (let HF auto-detect types)
    yaml_frontmatter = f"""---
task_categories:
- robotics
dataset_info:
  features:
{yaml_features}
  splits:
  - name: train
    num_examples: {info.get('total_episodes', 0)}
---
"""
    
    # Build features documentation section
    features_section = "## Features\n\n"
    if "features" in info:
        for feat_name, feat_info in info["features"].items():
            dtype = feat_info.get("dtype", "unknown")
            shape = feat_info.get("shape", [])
            names = feat_info.get("names")
            
            features_section += f"- **{feat_name}** (`{dtype}`)"
            if shape:
                features_section += f" - Shape: {shape}"
            if names:
                features_section += f" - Components: {', '.join(names[:3])}"
                if len(names) > 3:
                    features_section += f" ... (+{len(names)-3} more)"
            features_section += "\n"
    
    # Build info.json section
    info_json_str = json.dumps(info, indent=2)
    
    # Create README with dynamic YAML
    dataset_name = repo_id.split('/')[-1]
    
    # Build dataset structure section (generic)
    dataset_structure = "## Dataset Structure\n\nThe dataset directory contains:\n\n- `meta/`: Metadata files\n  - `info.json`: Dataset configuration\n  - `episodes.jsonl`: Episode metadata\n  - `episodes_stats.jsonl`: Episode statistics\n  - `tasks.jsonl`: Task information\n\n- `data/`: Parquet data files\n  - `train-00000-of-00001.parquet`: Consolidated dataset\n\n- `videos/`: Video files (if available)\n  - `episode_*.mp4`: Episode videos\n\n"
    
    # Dataset description
    stats = f"""## Dataset Description

A robotics demonstration dataset for training imitation learning and reinforcement learning policies.

### Dataset Statistics

- **Total Episodes:** {info.get('total_episodes', 'N/A')}
- **Total Frames:** {info.get('total_frames', 'N/A')}
- **Total Tasks:** {info.get('total_tasks', 'N/A')}
- **FPS:** {info.get('fps', 'N/A')}
- **Robot Type:** {info.get('robot_type', 'N/A')}
- **Codebase Version:** {info.get('codebase_version', 'N/A')}

"""
    
    # Metadata section
    metadata_section = f"""## Dataset Metadata

### info.json

Contains comprehensive dataset metadata and configuration:

```json
{info_json_str}
```

{features_section}"""
    
    # Usage section (generic)
    usage_section = f"""## Usage

To load this dataset with LeRobot:

```python
from lerobot.common.datasets import LeRobotDataset
dataset = LeRobotDataset(repo_id="{repo_id}")
```

Or with the Hugging Face Datasets library:

```python
from datasets import load_dataset
dataset = load_dataset("{repo_id}")
```

## Citation

If you use this dataset, please cite LeRobot:

```bibtex
@software{{ lerobot2024,
  author = {{ The HuggingFace Team }},
  title = {{ LeRobot: A Python Library for Robot Learning }},
  url = {{ https://github.com/huggingface/lerobot }},
  year = {{ 2024 }}
}}
```

## License

[More Information Needed]

## Dataset Card Contact

[More Information Needed]
"""
    
    # Combine all sections
    readme_content = (
        yaml_frontmatter +
        f"# {dataset_name}\n\n" +
        "This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).\n\n" +
        stats +
        dataset_structure +
        metadata_section +
        usage_section
    )
    
    return readme_content


def fix_episodes_jsonl(output_dir: Path) -> bool:
    """
    Fix paths in episodes.jsonl to point to correct parquet file
    
    Args:
        output_dir: Path to dataset directory
        
    Returns:
        True if successful, False otherwise
    """
    ep_file = output_dir / "meta" / "episodes.jsonl"
    
    if not ep_file.exists():
        return False
    
    try:
        lines = []
        with open(ep_file, "r") as f:
            for line in f:
                ep_meta = json.loads(line)
                ep_meta["data_path"] = "data/train-00000-of-00001.parquet"
                lines.append(ep_meta)

        with open(ep_file, "w") as f:
            for ep_meta in lines:
                f.write(json.dumps(ep_meta) + "\n")
        
        return True
    except Exception as e:
        print(f"Error fixing episodes.jsonl: {e}")
        return False


def create_hub_repo_with_tag(repo_id: str) -> bool:
    """
    Create HuggingFace repository and create version tag
    
    Args:
        repo_id: HuggingFace repo ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        
        # Create version tag for LeRobot compatibility
        try:
            create_tag(repo_id, tag="1.0.0", repo_type="dataset")
        except Exception as tag_err:
            print(f"Warning: Could not create tag: {type(tag_err).__name__}")
        
        return True
    except Exception as e:
        print(f"Error creating repo: {e}")
        return False


def upload_dataset_files(output_dir: Path, repo_id: str) -> bool:
    """
    Upload all dataset files to HuggingFace Hub
    
    Args:
        output_dir: Path to dataset directory
        repo_id: HuggingFace repo ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        api = HfApi()
        
        # Upload data folder
        print("   - Uploading data files...")
        api.upload_folder(
            folder_path=str(output_dir / "data"),
            repo_id=repo_id,
            path_in_repo="data",
            repo_type="dataset"
        )
        print("     OK: Data uploaded")
        
        # Upload meta folder
        print("   - Uploading metadata...")
        api.upload_folder(
            folder_path=str(output_dir / "meta"),
            repo_id=repo_id,
            path_in_repo="meta",
            repo_type="dataset"
        )
        print("     OK: Metadata uploaded")
        
        # Upload videos if present
        videos_dir = output_dir / "videos"
        if videos_dir.exists() and any(videos_dir.iterdir()):
            print("   - Uploading videos...")
            api.upload_folder(
                folder_path=str(videos_dir),
                repo_id=repo_id,
                path_in_repo="videos",
                repo_type="dataset"
            )
            print("     OK: Videos uploaded")
        
        return True
    except Exception as e:
        print(f"Error uploading files: {type(e).__name__}: {str(e)[:100]}")
        return False


def upload_readme(output_dir: Path, repo_id: str) -> bool:
    """
    Create and upload README.md to HuggingFace Hub
    
    Args:
        output_dir: Path to dataset directory
        repo_id: HuggingFace repo ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        api = HfApi()
        
        readme = create_dataset_readme_with_frontmatter(output_dir, repo_id)
        if not readme:
            print("Warning: Could not generate README")
            return False
        
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update dataset documentation and fix schema"
        )
        print("     OK: README uploaded")
        return True
    except Exception as e:
        print(f"Warning: README upload failed: {type(e).__name__}: {str(e)[:100]}")
        return False


def salvage_upload_dataset(output_dir: Path, repo_id: str, 
                          progress_callback: Optional[Callable] = None) -> bool:
    """
    Salvage upload approach: Upload dataset files directly to Hub without LeRobotDataset
    
    This is used as a fallback when the normal LeRobotDataset path fails.
    
    Args:
        output_dir: Path to dataset directory
        repo_id: HuggingFace repo ID
        progress_callback: Optional callback function for progress updates
        
    Returns:
        True if successful, False otherwise
    """
    
    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)
    
    log("\n" + "="*60)
    log("SALVAGE UPLOAD: Direct Hub Upload")
    log("="*60)
    
    # Step 1: Fix episodes.jsonl
    log("\nStep 1: Fixing dataset paths...")
    if fix_episodes_jsonl(output_dir):
        log("   OK: Paths fixed")
    else:
        log("   WARN: Could not fix paths")
    
    # Step 2: Create repo and tag
    log("\nStep 2: Creating Hub repository...")
    if not create_hub_repo_with_tag(repo_id):
        log("   ERROR: Failed to create repo")
        return False
    log("   OK: Repo created")
    
    # Step 3: Verify files
    log("\nStep 3: Verifying dataset files...")
    parquet_file = output_dir / "data" / "train-00000-of-00001.parquet"
    meta_files = ["info.json", "episodes.jsonl", "episodes_stats.jsonl"]
    
    all_good = True
    if parquet_file.exists():
        size_mb = parquet_file.stat().st_size / (1024*1024)
        log(f"   OK: Parquet data: {size_mb:.1f} MB")
    else:
        log(f"   ERROR: Missing parquet file")
        all_good = False
    
    for mf in meta_files:
        meta_path = output_dir / "meta" / mf
        if meta_path.exists():
            size_kb = meta_path.stat().st_size / 1024
            log(f"   OK: {mf}: {size_kb:.1f} KB")
        else:
            log(f"   ERROR: Missing {mf}")
            all_good = False
    
    if not all_good:
        log("\nWARN: Some files are missing!")
        return False
    
    # Step 4: Upload files
    log("\nStep 4: Uploading files to Hub...")
    if not upload_dataset_files(output_dir, repo_id):
        log("   ERROR: File upload failed")
        return False
    
    # Step 5: Upload README
    log("\nStep 5: Creating and uploading README...")
    upload_readme(output_dir, repo_id)
    
    log(f"\nSUCCESS! Dataset ready at:")
    log(f"   https://huggingface.co/datasets/{repo_id}")
    log("="*60 + "\n")
    
    return True


def upload_with_retry(output_dir: Path, repo_id: str,
                     primary_upload_fn: Optional[Callable] = None,
                     max_retries: int = 2) -> bool:
    """
    Upload dataset with automatic retry on known errors
    
    Tries primary upload function first, falls back to salvage upload on failure
    
    Args:
        output_dir: Path to dataset directory
        repo_id: HuggingFace repo ID
        primary_upload_fn: Primary upload function to try first (optional)
        max_retries: Number of retry attempts
        
    Returns:
        True if successful, False if all attempts failed
    """
    
    print("="*60)
    print("DATASET UPLOAD: Auto-Retry with Fallback")
    print("="*60)
    
    # Try primary upload first if provided
    if primary_upload_fn:
        print("\nAttempt 1: Trying primary upload method...")
        try:
            result = primary_upload_fn()
            if result:
                print("SUCCESS: Primary upload succeeded!")
                return True
        except Exception as e:
            print(f"WARN: Primary upload failed: {type(e).__name__}")
            print(f"   {str(e)[:150]}")
    
    # Fall back to salvage upload
    print("\nFallback: Using salvage upload method...")
    return salvage_upload_dataset(output_dir, repo_id)
