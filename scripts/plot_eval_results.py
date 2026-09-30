#!/usr/bin/env python3
"""
Plot evaluation results from one or multiple batch evaluation folders.
Usage: python plot_eval_results.py <path_to_batch_eval_dir1> [--strategy separate] [--labels label1 ...]
"""

import argparse
import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

def extract_step(summary_file, data):
    """Extracts step number from JSON data or fallback to the file path structure."""
    ckpt_dir = data.get("checkpoint_dir", "")
    if ckpt_dir:
        parts = Path(ckpt_dir).parts
        for part in reversed(parts):
            match = re.search(r'\d+', part)
            if match:
                return int(match.group())
                
    for part in reversed(summary_file.parts):
        match = re.search(r'\d+', part)
        if match:
            return int(match.group())
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results from summary JSONs.")
    parser.add_argument("eval_dirs", type=str, nargs="+", help="Path(s) to the directory containing batch evaluation results")
    parser.add_argument("--labels", type=str, nargs="+", help="Optional labels for the legend (must match number of directories)")
    parser.add_argument("--strategy", type=str, choices=["latest", "average", "separate"], default="separate",
                        help="Strategy for handling multiple evaluation timestamps per step: 'latest', 'average', or 'separate'")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.eval_dirs):
        print("Error: Number of labels must match the number of evaluation directories.")
        return

    # Set up plots
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # Palette configuration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    line_styles = ['-', '--', ':', '-.']
    
    valid_plots = 0

    for idx, eval_dir_str in enumerate(args.eval_dirs):
        eval_dir = Path(eval_dir_str)
        if not eval_dir.exists():
            print(f"Error: Directory {eval_dir} does not exist. Skipping.")
            continue
            
        # Group entries by step to evaluate duplicates chronologically
        steps_dict = {}
        
        for summary_file in eval_dir.rglob("summary.json"):
            try:
                with open(summary_file, "r") as f:
                    data = json.load(f)
                
                step = extract_step(summary_file, data)
                if step is None:
                    print(f"Warning: Could not parse step from {summary_file}, skipping.")
                    continue
                
                success_rate = data.get("success_rate", 0.0)
                avg_succ_len = data.get("mean_successful_episode_length", 0.0)
                timestamp = summary_file.parent.name  # e.g., '2026-07-12_16-36-28'
                
                entry = {
                    "step": step,
                    "success_rate": success_rate,
                    "avg_succ_len": avg_succ_len,
                    "timestamp": timestamp
                }
                
                if step not in steps_dict:
                    steps_dict[step] = []
                steps_dict[step].append(entry)
                
            except Exception as e:
                print(f"Error reading {summary_file}: {e}")

        if not steps_dict:
            print(f"No valid evaluation results found in {eval_dir}.")
            continue

        base_label = args.labels[idx] if args.labels else eval_dir.name
        base_color = colors[idx % len(colors)]

        # --- STRATEGY: SEPARATE ---
        if args.strategy == "separate":
            # Group curves by their chronological evaluation run index (Run 0, Run 1, Run 2...)
            curves = {} 
            
            for step, entries in steps_dict.items():
                # Sort timestamps chronologically within this checkpoint step
                entries.sort(key=lambda x: x["timestamp"])
                for run_idx, entry in enumerate(entries):
                    if run_idx not in curves:
                        curves[run_idx] = []
                    curves[run_idx].append(entry)
            
            # Plot each run sequence as its own separate continuous line
            for run_idx, run_entries in curves.items():
                run_entries.sort(key=lambda x: x["step"])
                
                steps = [r["step"] for r in run_entries]
                success_rates = [r["success_rate"] * 100 for r in run_entries]
                avg_succ_lens = [r["avg_succ_len"] for r in run_entries]
                
                # Vary line-styles/markers for visibility, keeping base colors aligned per directory
                style = line_styles[run_idx % len(line_styles)]
                marker = markers[run_idx % len(markers)]
                curve_label = f"{base_label} (Run {run_idx + 1})"
                
                ax1.plot(steps, success_rates, marker=marker, linestyle=style, color=base_color, label=curve_label, linewidth=2, markersize=6)
                ax2.plot(steps, avg_succ_lens, marker=marker, linestyle=style, color=base_color, label=curve_label, linewidth=2, markersize=6)
            
            valid_plots += 1

        # --- STRATEGY: LATEST OR AVERAGE ---
        else:
            processed_results = []
            for step, entries in steps_dict.items():
                if args.strategy == "latest":
                    entries.sort(key=lambda x: x["timestamp"])
                    processed_results.append(entries[-1])
                elif args.strategy == "average":
                    avg_sr = sum(e["success_rate"] for e in entries) / len(entries)
                    avg_len = sum(e["avg_succ_len"] for e in entries) / len(entries)
                    processed_results.append({
                        "step": step,
                        "success_rate": avg_sr,
                        "avg_succ_len": avg_len
                    })

            processed_results.sort(key=lambda x: x["step"])
            steps = [r["step"] for r in processed_results]
            success_rates = [r["success_rate"] * 100 for r in processed_results]
            avg_succ_lens = [r["avg_succ_len"] for r in processed_results]
            
            marker = markers[idx % len(markers)]
            ax1.plot(steps, success_rates, marker=marker, linestyle='-', color=base_color, label=base_label, linewidth=2, markersize=6)
            ax2.plot(steps, avg_succ_lens, marker=marker, linestyle='-', color=base_color, label=base_label, linewidth=2, markersize=6)
            valid_plots += 1

    if valid_plots == 0:
        print("No valid data to plot from any of the provided directories.")
        return

    # Plot 1: Success Rate formatting
    ax1.set_title("Success Rate vs Checkpoint Step", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Training Step", fontsize=10)
    ax1.set_ylabel("Success Rate (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

    # Plot 2: Average Successful Episode Length formatting
    ax2.set_title("Avg Successful Episode Length vs Step", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Training Step", fontsize=10)
    ax2.set_ylabel("Episode Length (Steps)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    
    # Save target determination
    if len(args.eval_dirs) > 1:
        output_dir = Path(args.eval_dirs[0]).parent
        output_plot_path = output_dir / f"comparison_evaluation_metrics_{args.strategy}.png"
    else:
        output_dir = Path(args.eval_dirs[0])
        output_plot_path = output_dir / f"evaluation_metrics_{args.strategy}.png"
        
    plt.savefig(output_plot_path, dpi=200)
    print(f"Plot successfully generated and saved to: {output_plot_path}")

if __name__ == "__main__":
    main()