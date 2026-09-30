#!/usr/bin/env python3
"""
Custom script to compare BC, RABC SARM, and RABC TCC evaluation runs on the 300 episodes training dataset.
It hardcodes the directory paths and assigns shared colors and styles:
- BC: Charcoal/Gray (dashed)
- RABC SARM: Deep Blue (solid)
- RABC TCC: Warm Orange (solid)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    # Base directory where all evaluation batch folders are located
    base_dir = Path("outputs/eval")
    
    # Map configurations to styles
    configs = [
        {
            "dir": "dp_bc_300_horizon300",
            "label": "BC Baseline",
            "color": "#555555",      # Charcoal
            "linestyle": "dashed",
            "marker": "s"
        },
        {
            "dir": "dp_rabc_sarm_300_horizon300",
            "label": "RABC SARM",
            "color": "#1f77b4",      # Deep Blue
            "linestyle": "solid",
            "marker": "o"
        },
        {
            "dir": "dp_rabc_tcc_temporal_300_horizon300",
            "label": "RABC TCC (sarm temporal normalization)",
            "color": "#ff7f0e",      # Warm Orange
            "linestyle": "solid",
            "marker": "^"
        },
        {
            "dir": "dp_rabc_tcc_uniform_300_horizon300",
            "label": "RABC TCC (sarm uniform normalization)",
            "color": "#ff7f0e",      # Warm Orange
            "linestyle": "dotted",
            "marker": "^"
        }
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
        
        ax1.plot(
            steps, success_rates, 
            marker=cfg["marker"], 
            linestyle=cfg["linestyle"], 
            color=cfg["color"], 
            label=cfg["label"],
            linewidth=2,
            markersize=6
        )
        ax2.plot(
            steps, avg_succ_lens, 
            marker=cfg["marker"], 
            linestyle=cfg["linestyle"], 
            color=cfg["color"], 
            label=cfg["label"],
            linewidth=2,
            markersize=6
        )

    if valid_plots == 0:
        print("No valid data to plot from any of the configured directories.")
        return

    # Plot 1: Success Rate formatting
    ax1.set_title("Success Rate vs Checkpoint Step (300 Episodes)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Training Step", fontsize=10)
    ax1.set_ylabel("Success Rate (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

    # Plot 2: Average Successful Episode Length formatting
    ax2.set_title("Avg Successful Episode Length vs Step (300 Episodes)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Training Step", fontsize=10)
    ax2.set_ylabel("Episode Length (Steps)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    
    # Save the plot
    output_plot_path = base_dir / "DP_300step_horizon_bc_vs_rabc_sarm_tcc_300.png"
    plt.savefig(output_plot_path, dpi=300)
    print(f"Plot successfully generated and saved to: {output_plot_path}")

if __name__ == "__main__":
    main()
