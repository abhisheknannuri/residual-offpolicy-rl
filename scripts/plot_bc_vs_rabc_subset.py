#!/usr/bin/env python3
"""
Custom script to compare BC vs RABC evaluation runs across different dataset subsets (50, 100, 300 episodes).
It hardcodes the directory paths and assigns shared colors for matching dataset sizes and distinct line styles 
for the algorithm types (solid for RABC, dashed for BC).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    # Base directory where all evaluation batch folders are located
    base_dir = Path("outputs/eval")
    
    # Map configurations to styles.
    # Group by episode count for colors, and algorithm for line style.
    configs = [
        {"dir": "policy_bc_50_episode_train", "label": "BC (50 eps)", "color": "blue", "linestyle": "dashed", "marker": "s"},
        {"dir": "policy_rabc_50_episode_train", "label": "RABC (50 eps)", "color": "blue", "linestyle": "solid", "marker": "o"},
        
        {"dir": "policy_bc_100_episode_train", "label": "BC (100 eps)", "color": "green", "linestyle": "dashed", "marker": "s"},
        {"dir": "policy_rabc_100_episode_train", "label": "RABC (100 eps)", "color": "green", "linestyle": "solid", "marker": "o"},
        
        {"dir": "policy_bc_300_episode_train", "label": "BC (300 eps)", "color": "red", "linestyle": "dashed", "marker": "s"},
        {"dir": "policy_rabc_300_episode_train", "label": "RABC (300 eps)", "color": "red", "linestyle": "solid", "marker": "o"},
    ]

    # Set up plots
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    valid_plots = 0

    for cfg in configs:
        eval_dir = base_dir / cfg["dir"]
        if not eval_dir.exists():
            print(f"Warning: Directory {eval_dir} does not exist. Skipping.")
            continue
            
        results = []
        # Find all summary.json files recursively
        for summary_file in eval_dir.rglob("summary.json"):
            try:
                with open(summary_file, "r") as f:
                    data = json.load(f)
                
                ckpt_dir = Path(data.get("checkpoint_dir", ""))
                step_str = ckpt_dir.name
                
                if not step_str.isdigit():
                    if ckpt_dir.parent.name.isdigit():
                        step_str = ckpt_dir.parent.name
                    else:
                        continue
                
                step = int(step_str)
                success_rate = data.get("success_rate", 0.0)
                avg_succ_len = data.get("mean_successful_episode_length", 0.0)
                
                results.append({
                    "step": step,
                    "success_rate": success_rate,
                    "avg_succ_len": avg_succ_len
                })
                
            except Exception as e:
                print(f"Error parsing {summary_file}: {e}")

        if not results:
            print(f"Warning: No valid summary.json files found in {eval_dir}.")
            continue

        valid_plots += 1
        results.sort(key=lambda x: x["step"])

        steps = [r["step"] for r in results]
        success_rates = [r["success_rate"] * 100 for r in results]  # Convert to percentage
        avg_succ_lens = [r["avg_succ_len"] for r in results]
        
        ax1.plot(steps, success_rates, marker=cfg["marker"], linestyle=cfg["linestyle"], color=cfg["color"], label=cfg["label"])
        ax2.plot(steps, avg_succ_lens, marker=cfg["marker"], linestyle=cfg["linestyle"], color=cfg["color"], label=cfg["label"])

    if valid_plots == 0:
        print("No valid data to plot from any of the configured directories.")
        return

    # Plot 1: Success Rate formatting
    ax1.set_title("Success Rate vs Checkpoint Step")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Success Rate (%)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Plot 2: Average Successful Episode Length formatting
    ax2.set_title("Avg Successful Episode Length vs Step")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Episode Length (Steps)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    
    # Save the plot
    output_plot_path = base_dir / "bc_vs_rabc_subset_comparison.png"
    plt.savefig(output_plot_path)
    print(f"Plot successfully generated and saved to: {output_plot_path}")

if __name__ == "__main__":
    main()
