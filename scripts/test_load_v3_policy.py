#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root and deps/lerobot to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "lerobot"))

import torch
from resfit.lerobot.utils.load_policy import load_policy

def main():
    checkpoint_dir = Path("/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_rabc/checkpoints/005000/pretrained_model/")
    
    print(f"Testing loading checkpoint from: {checkpoint_dir}")
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint dir {checkpoint_dir} does not exist.")
        return
        
    print("Loading policy...")
    policy = load_policy(checkpoint_dir)
    print("Policy loaded successfully!")
    print(f"Policy type: {type(policy)}")
    print(f"Input features/shapes: {policy.config.input_features}")
    print(f"Output features/shapes: {policy.config.output_features}")
    
    # Create a dummy batch to test forward pass
    print("\nCreating dummy batch for forward pass...")
    batch = {}
    
    device = getattr(policy, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Use config-defined input features to build dummy inputs
    for name, ft in policy.config.input_features.items():
        if name.startswith("observation.images."):
            # Batch of size 1
            batch[name] = torch.zeros(1, *ft.shape, device=device)
        elif name == "observation.state":
            batch[name] = torch.zeros(1, *ft.shape, device=device)
            
    # Add dummy action (since VAE ACT expects action during training, or when run in training mode)
    # Put policy in eval mode so VAE latent is zero and action target isn't needed
    policy.eval()
    
    print("Running forward pass (select_action)...")
    with torch.no_grad():
        action = policy.select_action(batch)
        print(f"Success! Action shape: {action.shape}")
        
if __name__ == "__main__":
    main()
