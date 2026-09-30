import wandb

# 1. Provide your workspace details
ENTITY = "nannuriabhi2000-hochschule-schmalkalden"
PROJECT = "robomimic-can-residual-td3-ablation-studies"

# Initialize the API and fetch all runs
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

print(f"Scanning {len(runs)} runs in {ENTITY}/{PROJECT}...")

updated_count = 0

for run in runs:
    # 2. Read the tags you added manually in the UI
    if len(run.tags) > 0:
        
        # In this example, we just take the first tag you added 
        # (e.g. if you tagged it "baseline_test", ui_tag = "baseline_test")
        ui_tag = run.tags[0]
        
        # 3. Add the new config key (does NOT delete existing config)
        run.config["custom_panel_group"] = ui_tag
        
        # 4. Push the safe update back to W&B
        run.update()
        
        print(f"✅ Updated '{run.name}' -> Assigned to group: {ui_tag}")
        updated_count += 1
    else:
        print(f"⏭️ Skipped '{run.name}' (No UI tags found)")

print(f"\nDone! Safely updated {updated_count} runs.")