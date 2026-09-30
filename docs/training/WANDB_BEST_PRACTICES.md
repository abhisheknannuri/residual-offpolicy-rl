# WandB Best Practices for ResFiT Repository

This document outlines the best practices for using WandB (Weights and Biases) in the ResFiT repository, including setup, logging patterns, and metrics.

---

## 1. Setting Up WandB

### Environment Variables

To use WandB, the following environment variables must be set:

```ini
# Required:
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token

# Optional:
CUDA_VISIBLE_DEVICES=0           # Limit to specific GPU
WANDB_ENTITY=your_wandb_username # Default WandB entity
WANDB_PROJECT=resfit             # Default WandB project
```

- **WANDB_API_KEY:** Obtain from [WandB Authorization](https://wandb.ai/authorize).
- **WANDB_ENTITY:** Your WandB username or team name.
- **WANDB_PROJECT:** The project name (default: `resfit`).

### Docker Integration

- WandB cache is stored in `/root/.local/share/wandb/`.
- Use Docker named volume `wandb-cache` for persistent caching.

---

## 2. Logging Metrics

### Metrics Logged to WandB

| Metric | Formula |
|---|---|
| `eval/success_rate` | `mean(successes)` — fraction of episodes that succeeded |
| `eval/mean_return` | `mean(episode_returns)` — mean undiscounted sum of rewards per episode (equals 1.0 for success, 0.0 for failure, since reward is sparse) |
| `eval/mean_successful_episode_length` | Mean number of steps for episodes that succeeded (shorter = better — the agent solved it faster) |

### Training Scripts

#### `scripts/train_bc_act.sh`
- Trains a base BC policy using the ACT architecture.
- Key parameters:
  ```bash
  STEPS=200000
  BATCH_SIZE=128
  EVAL_NUM_ENVS=4
  EVAL_NUM_EPISODES=100
  ```

#### `scripts/train_residual_rl.sh`
- Fine-tunes the frozen BC policy with TD3 residual RL.
- Key parameters:
  ```bash
  BASE_WANDB_ID="dexmg-bc/zp7niccu"
  TOTAL_TIMESTEPS=500000
  ACTION_SCALE=0.2
  UTD=4
  BUFFER_SIZE=80000
  ```

#### `scripts/resume_residual_rl.sh`
- Resumes from a checkpoint.
- Key parameters:
  ```bash
  RESUME_CKPT="run_2026-.../models/latest"
  WANDB_CONTINUE_ID="1wrldnus"
  SEED=1987747100
  ```

---

## 3. Best Practices

1. **Use Descriptive Run Names:**
   - Set `WANDB_RUN_ID` and `WANDB_NAME` for clarity.

2. **Log Key Metrics:**
   - Always log `eval/success_rate`, `eval/mean_return`, and `eval/mean_successful_episode_length`.

3. **Cache Management:**
   - Use Docker named volumes for WandB cache to avoid redundant downloads.

4. **Resume Runs:**
   - Use `WANDB_CONTINUE_ID` to resume interrupted runs.

5. **Environment Isolation:**
   - Use `.env` files to manage secrets and environment variables.

---

## 4. Additional Notes

- **Simulation-Only:** The repository is designed for simulation. Real-world experiments require manual evaluation.
- **Sparse Rewards:** Metrics assume sparse rewards (1.0 for success, 0.0 for failure).

---

For more details, refer to the [../setup/DOCKER_SETUP.md](../setup/DOCKER_SETUP.md) and [../rewards/REWARD_AND_SUCCESS.md](../rewards/REWARD_AND_SUCCESS.md) files in the repository.