import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def load_and_compare(repo1, repo2):
    print(f"Loading {repo1}...")
    ds1 = LeRobotDataset(repo1)
    print(f"Loading {repo2}...")
    ds2 = LeRobotDataset(repo2)
    
    print("\n" + "="*60)
    print(" LEVEL 1: HIGH-LEVEL METADATA & SCHEMA")
    print("="*60)
    
    # 1. Compare total frames and episodes
    print(f"{repo1}: {ds1.num_frames} frames, {ds1.num_episodes} episodes, {ds1.fps} FPS")
    print(f"{repo2}: {ds2.num_frames} frames, {ds2.num_episodes} episodes, {ds2.fps} FPS")
    
    if ds1.num_frames != ds2.num_frames:
        print("\n[WARNING] Datasets have different total frames. The underlying HDF5 filter/mask might be different.")

    # 2. Compare Feature Shapes and Dtypes
    print("\n--- Feature Schema Comparison ---")
    keys1 = set(ds1.features.keys())
    keys2 = set(ds2.features.keys())
    
    all_keys = keys1.union(keys2)
    for k in sorted(all_keys):
        if k not in keys1:
            print(f"Key '{k}' is MISSING in {repo1}")
        elif k not in keys2:
            print(f"Key '{k}' is MISSING in {repo2}")
        else:
            f1, f2 = ds1.features[k], ds2.features[k]
            if f1["shape"] != f2["shape"] or f1["dtype"] != f2["dtype"]:
                print(f"[MISMATCH] {k}:")
                print(f"  {repo1}: shape={f1['shape']}, dtype={f1['dtype']}")
                print(f"  {repo2}: shape={f2['shape']}, dtype={f2['dtype']}")
            else:
                print(f"[MATCH] {k}: shape={f1['shape']}, dtype={f1['dtype']}")

    print("\n" + "="*60)
    print(" LEVEL 2: STATISTICAL COMPARISON (Normalization bugs hide here)")
    print("="*60)
    # We sample the first 1000 frames to get a statistical profile
    sample_size = min(1000, ds1.num_frames, ds2.num_frames)
    
    # Extract tensors
    actions1 = torch.stack([ds1[i]["action"] for i in range(sample_size)])
    actions2 = torch.stack([ds2[i]["action"] for i in range(sample_size)])
    
    print("\n--- Action Statistics ---")
    print(f"{repo1} Action - Min: {actions1.min().item():.4f}, Max: {actions1.max().item():.4f}, Mean: {actions1.mean().item():.4f}")
    print(f"{repo2} Action - Min: {actions2.min().item():.4f}, Max: {actions2.max().item():.4f}, Mean: {actions2.mean().item():.4f}")
    
    if "observation.state" in ds1.features and "observation.state" in ds2.features:
        states1 = torch.stack([ds1[i]["observation.state"] for i in range(sample_size)])
        states2 = torch.stack([ds2[i]["observation.state"] for i in range(sample_size)])
        print("\n--- State Observation Statistics ---")
        print(f"{repo1} State - Min: {states1.min().item():.4f}, Max: {states1.max().item():.4f}, Mean: {states1.mean().item():.4f}")
        print(f"{repo2} State - Min: {states2.min().item():.4f}, Max: {states2.max().item():.4f}, Mean: {states2.mean().item():.4f}")

    # Image normalization check
    img_key = [k for k in keys1 if "images" in k][0] # Grab first image key
    img1 = ds1[0][img_key]
    img2 = ds2[0][img_key]
    print(f"\n--- Image Normalization ({img_key}) ---")
    print(f"{repo1}: dtype={img1.dtype}, min={img1.float().min().item()}, max={img1.float().max().item()}")
    print(f"{repo2}: dtype={img2.dtype}, min={img2.float().min().item()}, max={img2.float().max().item()}")

    print("\n" + "="*60)
    print(" LEVEL 3: FRAME-BY-FRAME EPISODE 0 DEEP DIVE")
    print("="*60)
    
    # Get frame indices for Episode 0 in both datasets
    ep0_frames1 = [i for i in range(ds1.num_frames) if ds1[i]["episode_index"].item() == 0]
    ep0_frames2 = [i for i in range(ds2.num_frames) if ds2[i]["episode_index"].item() == 0]
    
    print(f"Episode 0 length -> {repo1}: {len(ep0_frames1)}, {repo2}: {len(ep0_frames2)}")
    
    min_len = min(len(ep0_frames1), len(ep0_frames2))
    diff_count = 0
    
    for i in range(min_len):
        idx1, idx2 = ep0_frames1[i], ep0_frames2[i]
        row1, row2 = ds1[idx1], ds2[idx2]
        
        # Check actions
        act_diff = torch.max(torch.abs(row1["action"] - row2["action"])).item()
        
        # Check states
        state_diff = 0.0
        if "observation.state" in row1 and "observation.state" in row2:
            state_diff = torch.max(torch.abs(row1["observation.state"] - row2["observation.state"])).item()
            
        # Check dones
        done1 = row1["next.done"].item()
        done2 = row2["next.done"].item()
        
        if act_diff > 1e-4 or state_diff > 1e-4 or done1 != done2:
            if diff_count == 0:
                print("\nFound mismatches in Episode 0! First mismatch at step", i)
            print(f"Step {i}: Act_Diff={act_diff:.6f}, State_Diff={state_diff:.6f}, Dones=({done1}, {done2})")
            diff_count += 1
            
            if diff_count >= 10:
                print("... limiting output to first 10 mismatches.")
                break
                
    if diff_count == 0:
        print("Episode 0 matches perfectly in actions, states, and done flags!")

if __name__ == "__main__":
    REFERENCE_DATASET = "ankile/robomimic-mh-lift-image"
    MY_DATASET = "poolvarine/robomimic-mh-lift-image-dense"
    
    load_and_compare(REFERENCE_DATASET, MY_DATASET)