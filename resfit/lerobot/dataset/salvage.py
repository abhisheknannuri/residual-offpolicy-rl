#!/usr/bin/env python3
"""
LeRobot Dataset Salvage Script
Recovers and uploads datasets that failed during creation or upload.

This script is used as a recovery tool when the normal dataset creation
fails, using direct HuggingFace Hub upload without LeRobotDataset validation.
"""

from pathlib import Path
from lerobot_upload_utils import salvage_upload_dataset

# Configuration - Edit these to match your dataset
repo_id = "poolvarine/robomimic-mh-transport-image-dense"
output_dir = Path("/home/qte9489/personal_abhi/.cache/huggingface/lerobot/poolvarine/robomimic-mh-transport-image-dense")

if __name__ == "__main__":
    success = salvage_upload_dataset(output_dir, repo_id)
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"Dataset is live at: https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"\n❌ Salvage upload failed. Check the errors above.")
        exit(1)