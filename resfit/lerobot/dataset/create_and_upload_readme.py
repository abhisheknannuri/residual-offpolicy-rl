#!/usr/bin/env python3
"""
Create and upload a proper README.md for LeRobot datasets on HuggingFace
"""
import json
from pathlib import Path
from huggingface_hub import HfApi

def create_dataset_readme(output_dir, repo_id):
    """Generate README.md content from dataset info.json"""
    
    output_dir = Path(output_dir)
    info_file = output_dir / "meta" / "info.json"
    
    if not info_file.exists():
        print(f"❌ Warning: {info_file} not found")
        return None
    
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    # Build features documentation
    features_section = "## Features\n\n"
    if "features" in info:
        for feat_name, feat_info in info["features"].items():
            dtype = feat_info.get("dtype", "unknown")
            shape = feat_info.get("shape", [])
            names = feat_info.get("names")
            
            features_section += f"- **{feat_name}** (`{dtype}`)\n"
            features_section += f"  - Shape: {shape}\n"
            if names:
                features_section += f"  - Components: {', '.join(names)}\n"
            features_section += "\n"
    
    # Build info.json section
    info_json_str = json.dumps(info, indent=2)
    
    # Create README content
    dataset_name = repo_id.split('/')[-1]
    readme_content = f"""# {dataset_name}

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

A robotics demonstration dataset for training imitation learning and reinforcement learning policies.

### Dataset Statistics

- **Total Episodes:** {info.get('total_episodes', 'N/A')}
- **Total Frames:** {info.get('total_frames', 'N/A')}
- **Total Tasks:** {info.get('total_tasks', 'N/A')}
- **FPS:** {info.get('fps', 'N/A')}
- **Robot Type:** {info.get('robot_type', 'N/A')}
- **Codebase Version:** {info.get('codebase_version', 'N/A')}

## Dataset Structure

```
meta/
  ├── info.json              # Dataset metadata and configuration
  ├── episodes.jsonl         # Episode indices and metadata
  ├── episodes_stats.jsonl   # Per-episode statistics
  └── tasks.jsonl            # Task information

data/
  └── train-00000-of-00001.parquet  # Consolidated dataset in Parquet format

videos/
  └── episode_*.mp4          # Video recordings (if available)
```

## Dataset Metadata

### info.json

Contains comprehensive dataset metadata and configuration:

```json
{info_json_str}
```

{features_section}

## Usage

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
@software{{lerobot2024,
  author = {{The HuggingFace Team}},
  title = {{LeRobot: A Python Library for Robot Learning}},
  url = {{https://github.com/huggingface/lerobot}},
  year = {{2024}}
}}
```

## License

[More Information Needed]

## Dataset Card Contact

[More Information Needed]
"""
    
    return readme_content


def main():
    repo_id = "poolvarine/robomimic-mh-transport-image-dense"
    output_dir = Path("/home/qte9489/personal_abhi/.cache/huggingface/lerobot/poolvarine/robomimic-mh-transport-image-dense")
    
    print("=" * 70)
    print("📝 Creating and Uploading README.md")
    print("=" * 70)
    
    # Step 1: Generate README
    print("\n📄 Step 1: Generating README.md...")
    readme = create_dataset_readme(output_dir, repo_id)
    
    if not readme:
        print("❌ Failed to generate README")
        return
    
    # Step 2: Save README locally
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    readme_size_kb = readme_path.stat().st_size / 1024
    print(f"✅ README created locally: {readme_path}")
    print(f"   Size: {readme_size_kb:.1f} KB")
    
    # Step 3: Upload README to Hub
    print("\n⬆️  Step 2: Uploading README.md to Hub...")
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset documentation"
        )
        print(f"✅ README uploaded to {repo_id}")
        print(f"\n🌐 View on Hub: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"⚠️  Upload failed: {type(e).__name__}")
        print(f"   {str(e)[:200]}")
        return
    
    print("\n" + "=" * 70)
    print("✨ SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
