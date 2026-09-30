"""
Script to create a proper README.md for LeRobot datasets and upload to Hub
"""
import json
from pathlib import Path

def create_dataset_readme(output_dir, repo_id):
    """Create a README.md file for HuggingFace Hub display"""
    
    output_dir = Path(output_dir)
    info_file = output_dir / "meta" / "info.json"
    
    if not info_file.exists():
        print(f"Warning: {info_file} not found")
        return None
    
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    # Build features documentation
    features_doc = ""
    if "features" in info:
        features_doc = "### Features\n\n"
        for feat_name, feat_info in info["features"].items():
            dtype = feat_info.get("dtype", "unknown")
            shape = feat_info.get("shape", [])
            features_doc += f"- **{feat_name}** (`{dtype}`): shape={shape}\n"
    
    # Build README content
    readme_content = f"""---
dataset_info:
  features:
  - name: action
    dtype: float32
  - name: observation
    sequence:
    - name: state
      dtype: float32
    - name: images
      sequence:
      - name: agentview
        dtype: image
      - name: robot0_eye_in_hand
        dtype: image
  - name: next
    struct:
    - name: done
      dtype: bool
  - name: timestamp
    dtype: float32
  splits:
  - name: train
    num_bytes: {info.get('total_frames', 'N/A')}
    num_examples: {info.get('total_episodes', 'N/A')}
---

# LeRobot Dataset: {repo_id.split('/')[-1]}

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

A robotics demonstration dataset for training imitation learning and reinforcement learning policies.

**Dataset Statistics:**
- **Total Episodes:** {info.get('total_episodes', 'N/A')}
- **Total Frames:** {info.get('total_frames', 'N/A')}
- **Total Tasks:** {info.get('total_tasks', 'N/A')}
- **FPS:** {info.get('fps', 'N/A')}
- **Robot Type:** {info.get('robot_type', 'N/A')}
- **Codebase Version:** {info.get('codebase_version', 'N/A')}

## Dataset Structure

### Directory Layout
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

### metadata/info.json

Contains comprehensive dataset metadata:

```json
{{"
    "codebase_version": "{info.get('codebase_version', 'v2.1')}",
    "robot_type": "{info.get('robot_type', 'robomimic')}",
    "total_episodes": {info.get('total_episodes', 300)},
    "total_frames": {info.get('total_frames', 62756)},
    "fps": {info.get('fps', 20)},
    "splits": {json.dumps(info.get('splits', {}), indent=6).replace(chr(10), chr(10)+"    ")}
"""

# Add features section if available
if "features" in info:
    readme_content += """,
    "features": {
"""
    for feat_name, feat_info in info["features"].items():
        readme_content += f'        "{feat_name}": {json.dumps(feat_info, indent=8).replace(chr(10), chr(10)+"            ")},\n'
    readme_content = readme_content.rstrip(',\n') + "\n    }\n"

readme_content += """}
```

## Features

The dataset contains the following features:

"""

if "features" in info:
    for feat_name, feat_info in info["features"].items():
        dtype = feat_info.get("dtype", "unknown")
        shape = feat_info.get("shape", [])
        names = feat_info.get("names")
        
        readme_content += f"- **{feat_name}** (`{dtype}`)\n"
        readme_content += f"  - Shape: {shape}\n"
        if names:
            readme_content += f"  - Components: {', '.join(names)}\n"
        readme_content += "\n"

readme_content += """
## Usage

To load this dataset with LeRobot:

```python
from lerobot.common.datasets import LeRobotDataset

dataset = LeRobotDataset(repo_id="%s")
```

Or with the Hugging Face Datasets library:

```python
from datasets import load_dataset

dataset = load_dataset("%s")
```

## Citation

If you use this dataset, please cite LeRobot:

```bibtex
@software{lerobot2024,
  author = {The HuggingFace Team},
  title = {LeRobot: A Python Library for Robot Learning},
  url = {https://github.com/huggingface/lerobot},
  year = {2024}
}
```

## License

[More Information Needed]

## Dataset Card Contact

[More Information Needed]
""" % (repo_id, repo_id)
    
    return readme_content

# Test
if __name__ == "__main__":
    output_dir = Path("/home/qte9489/personal_abhi/.cache/huggingface/lerobot/poolvarine/robomimic-mh-transport-image-dense")
    repo_id = "poolvarine/robomimic-mh-transport-image-dense"
    
    readme = create_dataset_readme(output_dir, repo_id)
    if readme:
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        print(f"✅ README created: {readme_path}")
        print(f"Size: {len(readme)} bytes")
        print("\n--- Preview ---")
        print(readme[:500])
    else:
        print("❌ Failed to create README")
