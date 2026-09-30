import sys
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def inspect_dataset():
    dataset_id = "ankile/robomimic-mh-can-image"
    print(f"Loading dataset: {dataset_id}")
    dataset = LeRobotDataset(dataset_id)
    
    ep_data = dataset.episode_data_index
    first_ep_start = ep_data["from"][0].item()
    first_ep_end = ep_data["to"][0].item()
    
    print(f"\nEpisode 0: frames {first_ep_start} to {first_ep_end - 1}")
    
    print("\nInspecting the exact boundary where 'next.done' becomes True:")
    for i in range(108, first_ep_end):
        sample = dataset[i]
        action = sample['action']
        done = sample.get('next.done', None)
        
        # Determine if it's False or True
        done_val = done.item() if done is not None else "Missing"
        marker = " <=== SWITCH" if i > 108 and done_val and not prev_done else ""
        
        print(f"Frame {i:3d}: action_sum={action.abs().sum().item():.4f} | next.done={done_val}{marker}")
        prev_done = done_val

if __name__ == "__main__":
    inspect_dataset()
