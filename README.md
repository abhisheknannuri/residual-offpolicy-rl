# ResFiT

This repository contains the official release of code for paper "Residual Off-Policy RL for Finetuning Behavior Cloning Policies".

Website: https://residual-offpolicy-rl.github.io/

Paper: https://arxiv.org/abs/2509.19301

## Documentation

Full docs are in **[docs/](docs/README.md)** - setup, the BC policy, the RL algorithms, training runbooks, rewards, datasets, and the real-robot stack.

## Getting Started

### Environment Setup

#### 1. Create and Activate Conda Environment

Create a new conda environment with Python 3.10:

```bash
conda create -n residual python=3.10 -y
conda activate residual
```

#### 2. Install Core Dependencies

Install the RL finetuning dependencies:

```bash
./resfit/rl_finetuning/setup_rlpd_robosuite.sh
```

Install additional required packages:

```bash
pip install wandb
pip install draccus==0.10.0 torchrl==0.9.2
pip install hydra-core serial deepdiff matplotlib
```

#### 3. Logging into Hugging Face and wandb

Login to Hugging Face to access dataset and wandb for policy weights saving and loading:

```bash
hf auth login
wandb login
```

#### 4. Fix CUDA Support (if needed)

If you encounter CUDA-related issues, clean out CPU-only installs and reinstall CUDA-enabled packages:

```bash
# Remove CPU-only torchcodec
pip uninstall -y torchcodec

# Install CUDA-enabled wheel for CUDA 12.8
pip install --no-cache-dir torchcodec --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA is enabled:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Launch training

### BC policy training

First we need to train the base BC policy. Taking TwoArmCoffee as an example:

```
python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ankile/dexmg-two-arm-coffee \
    --policy act \
    --steps 200000 \
    --batch_size 128 \
    --wandb_project dexmg-bc \
    --eval_env TwoArmCoffee \
    --rollout_freq 5000 \
    --eval_video_key observation.images.frontview \
    --eval_render_size 224 \
    --eval_num_envs 4 \
    --eval_num_episodes 100 \
    --wandb_enable \
    --no_cleanup
```

After training finished, put the `wandb_project_name/run_id` into the corresponding task config in [residual_td3.py](./resfit/rl_finetuning/config/residual_td3.py).

### Residual RL training

Next we can train our residual RL policy:

```
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_coffee_config \
    base_policy.wandb_id=dexmg-bc/kpid6s4r \
    base_policy.wt_type=best \
    algo.prefetch_batches=4 \
    algo.n_step=5 \
    algo.gamma=0.995 \
    algo.learning_starts=10_000 \
    algo.critic_warmup_steps=10_000 \
    algo.num_updates_per_iteration=4 \
    algo.stddev_max=0.025 \
    algo.stddev_min=0.025 \
    algo.buffer_size=80_000 \
    algo.batch_size=128 \
    algo.sampling_strategy=uniform \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=1e-6 \
    offline_data.num_episodes=250 \
    wandb.project=dexmg-coffee \
    wandb.name=resfit \
    wandb.group=resfit \
    headless=true \
    eval_num_envs=4 \
    debug=false


# Default steps is total_timesteps=500_000 for coffee config. Original base config (RLPDAlgoConfig) is 300_000


python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_coffee_config \
    base_policy.wandb_id=dexmg-bc/zp7niccu base_policy.wt_type=best \
    algo.prefetch_batches=4 algo.n_step=5 algo.gamma=0.995 \
    algo.learning_starts=10_000 algo.critic_warmup_steps=10_000 algo.total_timesteps=500_000 \
    algo.num_updates_per_iteration=4 algo.stddev_max=0.025 algo.stddev_min=0.025 \
    algo.buffer_size=80_000 algo.batch_size=128 algo.sampling_strategy=uniform \
    agent.actor.action_scale=0.2 agent.actor_lr=1e-6 \
    offline_data.num_episodes=100 wandb.project=dexmg-coffee \
    wandb.name=resfit wandb.group=resfit headless=true eval_num_envs=4 \
    no_cleanup=true save_freq=10000 debug=false 


python resfit/rl_finetuning/scripts/train_residual_td3.py     --config-name=residual_td3_coffee_config     base_policy.wandb_id=dexmg-bc/zp7niccu base_policy.wt_type=best     algo.prefetch_batches=4 algo.n_step=5 algo.gamma=0.995     algo.learning_starts=10_000 algo.critic_warmup_steps=10_000 algo.total_timesteps=500_000     algo.num_updates_per_iteration=4 algo.stddev_max=0.025 algo.stddev_min=0.025     algo.buffer_size=80_000 algo.batch_size=128 algo.sampling_strategy=uniform     agent.actor.action_scale=0.2 agent.actor_lr=1e-6     offline_data.num_episodes=100 wandb.project=dexmg-coffee     wandb.name=resfit wandb.group=resfit headless=false eval_num_envs=4     no_cleanup=true save_freq=10000 debug=false




python resfit/rl_finetuning/scripts/train_residual_td3.py \
  --config-name=residual_td3_coffee_config \
  base_policy.wandb_id=dexmg-bc/kpid6s4r \
  base_policy.wt_type=best \
  offline_data.num_episodes=100 \
  seed=1987747100 \
  resume_ckpt=run_2026-02-22_21-00-24_resfit__2026-02-22_21-00-22__TwoArmCoffee_n5_utd4_buf80000_off100ep_lr1e-06__seed1987747100/models/latest \
  wandb.continue_run_id=1wrldnus

```





### If you get any issues

```sh
# 1. Clear out any traces of the old broken torch packages
pip uninstall torch torchvision torchaudio -y
conda uninstall torch torchvision torchaudio -y
pip cache purge

# 2. Install the PyTorch build matched to CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

then run 
```sh
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```