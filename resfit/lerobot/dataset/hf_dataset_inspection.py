import numpy as np
import pandas as pd
from datasets import load_dataset

HF_DATASET = "poolvarine/robomimic-mh-lift-image-dense"

print("Loading dataset from Parquet...")
ds = load_dataset(
    "parquet", 
    data_files=f"hf://datasets/{HF_DATASET}/**/*.parquet", 
    split="train"
)

# Convert to a Pandas DataFrame for super fast grouping
df = ds.to_pandas()

# Get the unique episode IDs
unique_episodes = df['episode_index'].unique()
total_episodes = len(unique_episodes)

print(f"\n✅ True Episode Count: {total_episodes}")
print(f"✅ Total Transitions (Frames): {len(df)}")

print("\n--- Inspecting First 5 Episodes ---")

# Iterate through the first 5 unique episode IDs
for ep_id in unique_episodes[:5]:
    # Extract only the rows for this specific episode
    ep_df = df[df['episode_index'] == ep_id].copy()
    
    # Reset the index so the step count for this episode starts at 0, 1, 2...
    ep_df = ep_df.reset_index(drop=True)
    
    ep_length = len(ep_df)
    
    # Find the indices within this episode where done is True and reward is 1.0
    # Using np.isclose for float comparison safety
    reward_1_indices = ep_df.index[np.isclose(ep_df['next.reward'], 1.0)].tolist()
    done_true_indices = ep_df.index[ep_df['next.done'] == True].tolist()
    
    print(f"\nEpisode ID: {ep_id}")
    print(f"  Total Steps: {ep_length}")
    print(f"  'done == True' count : {len(done_true_indices)} | Frame Indices: {done_true_indices}")
    print(f"  'reward == 1' count  : {len(reward_1_indices)} | Frame Indices: {reward_1_indices}")

print("\n... Script Complete.")