#!/usr/bin/env bash
# ============================================================================
# train_bc_lift.sh — Train a BC (Behavior Cloning) policy for Lift (Panda)
# ============================================================================
# Trains an ACT base policy on the Robosuite Lift task (single Franka Panda arm
# picking up a cube). This must be done BEFORE residual RL fine-tuning.
#
# Task:   Lift  (robosuite)
# Robot:  Franka Panda (single arm, 7-DOF + parallel-jaw gripper)
# Action: 7D  — Δpos(3) + Δrot_axis_angle(3) + gripper(1)
#
# Usage:
#   # Inside container:
#   bash scripts/train_bc_lift.sh
#
#   # From host (with docker compose):
#   docker compose run --rm --name qte9489-resfit-train train bash scripts/train_bc_lift.sh
#
# After training:
#   - Best checkpoint: $CACHE_DIR/bc_run_<timestamp>/best/
#   - WandB run ID: check WandB dashboard → use as base_policy.wandb_id in RL
#   - Put "wandb_project/run_id" into residual RL config
# ============================================================================
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Pre-converted LeRobot datasets on HuggingFace (by ankile):
#
#   ankile/robomimic-mh-lift-image   — 300 eps, multi-human (6 operators, mixed quality)
#   ankile/robomimic-ph-lift-image   — 200 eps, proficient-human (1 skilled operator)
#   ankile/robomimic-mg-lift-image   — ~1500 eps, machine-generated (RL checkpoints)
#
# MH = multi-human = standard benchmark for residual RL (diverse, mixed quality)
# PH = proficient-human = clean data, good for upper-bound baselines
# MG = machine-generated = largest but lowest quality
#
# The MH dataset contains:
#   - 300 episodes, ~31K frames total
#   - 2 cameras: agentview (84x84), robot0_eye_in_hand (84x84)
#   - State: 9D (eef_pos[3] + eef_quat[4] + gripper_qpos[2])
#   - Actions: 7D (delta_pos[3] + delta_rot[3] + gripper[1])
#   - FPS: 20 Hz
#
# To use a local dataset instead of HuggingFace, set DATASET to the local path:
#   DATASET="/path/to/my-local-lerobot-dataset"
#
DATASET="poolvarine/robomimic-mh-can-image-dense"

# ── Episode Count ────────────────────────────────────────────────────────────
# BC uses ALL episodes in the dataset for training (all 300 in this case).
# There is no --max_episodes flag in train_bc_dexmg.py.
#
# If you want fewer episodes, create a smaller dataset during conversion:
#   python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
#       --dataset lift_mh_image.hdf5 --output_dir /tmp/lift-100ep --max_episodes 100
#
# For RL later, you CAN control how many episodes go into the offline buffer:
#   offline_data.num_episodes=100  (in the Hydra config / shell script)

# ── How Evaluation Works (no train/test split needed) ────────────────────────
# There is NO held-out test set. ALL 300 episodes are used for training.
# Evaluation is done via LIVE ROLLOUTS in the Robosuite simulator:
#
#   Every ROLLOUT_FREQ steps → spin up EVAL_NUM_EPISODES episodes in
#   the actual Lift environment → run the current policy → measure
#   how many times the robot successfully picks up the cube.
#
# This means evaluation tests real execution performance, not loss on unseen
# data. There's no data overlap concern because the eval environment generates
# fresh randomized initial states every time.
#
# The eval settings (EVAL_ENV, ROLLOUT_FREQ, EVAL_NUM_EPISODES, etc.) are
# configured in the EVALUATION section below.

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CAMERAS                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Which cameras the policy sees during training. Must exist in the dataset.
# If unset (empty string), ALL cameras in the dataset are used.
#
# The ankile/robomimic-mh-lift-image dataset has these 2 cameras:
#   agentview            — third-person workspace view
#   robot0_eye_in_hand   — wrist-mounted camera on the gripper
#
# Other cameras available in Robosuite Lift (need custom dataset to use):
#   frontview            — front-facing camera
#   sideview             — side view
#   birdview             — top-down view
#
# Examples:
#   POLICY_CAMERAS="agentview robot0_eye_in_hand"  ← both cameras (default)
#   POLICY_CAMERAS="agentview"                     ← third-person only
#   POLICY_CAMERAS="robot0_eye_in_hand"            ← wrist camera only
#   POLICY_CAMERAS=""                              ← use all in dataset
#
POLICY_CAMERAS="agentview robot0_eye_in_hand"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATE OBSERVATIONS (proprioceptive)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# The state vector is determined by the DATASET, not the training script.
# You cannot change state composition at training time — it's baked in during
# dataset conversion (see convert_robomimic_to_lerobot.py get_expected_low_dim_keys).
#
# ┌────────────────────────────────────────────────────────────────────────┐
# │  ankile/robomimic-mh-lift-image state — 9D (current dataset)         │
# │                                                                        │
# │  robot0_eef_pos       [3D]  — end-effector XYZ position               │
# │  robot0_eef_quat      [4D]  — end-effector quaternion orientation      │
# │  robot0_gripper_qpos  [2D]  — gripper finger joint positions           │
# │                               (symmetric: +q and -q of same joint)     │
# │  Total: 9D                                                             │
# └────────────────────────────────────────────────────────────────────────┘
#
# To get a 16D state vector (with joint positions), you need a CUSTOM DATASET.
# See the "CUSTOM DATASET" section at the bottom of this file.
#
# ┌────────────────────────────────────────────────────────────────────────┐
# │  Desired 16D state (requires custom dataset + code changes)          │
# │                                                                        │
# │  robot0_eef_pos       [3D]  — end-effector XYZ position               │
# │  robot0_eef_quat      [4D]  — end-effector quaternion orientation      │
# │  robot0_joint_pos     [7D]  — 7 joint angles (shoulder to wrist)       │
# │  robot0_gripper_qpos  [2D]  — gripper finger joint positions           │
# │  Total: 16D                                                            │
# └────────────────────────────────────────────────────────────────────────┘
#
# ┌────────────────────────────────────────────────────────────────────────┐
# │  ALL available low-dim obs keys in Robosuite Lift (Panda)            │
# │  (what you can include when making a custom dataset)                   │
# │                                                                        │
# │  ROBOT STATE:                                                          │
# │    robot0_eef_pos          [3D]   EE position (world frame)            │
# │    robot0_eef_quat         [4D]   EE quaternion orientation            │
# │    robot0_joint_pos        [7D]   joint angles (7 revolute joints)     │
# │    robot0_joint_vel        [7D]   joint angular velocities             │
# │    robot0_gripper_qpos     [2D]   gripper finger positions (+/-)       │
# │    robot0_gripper_qvel     [2D]   gripper finger velocities            │
# │    robot0_eef_vel_lin      [3D]   EE linear velocity  (if available)   │
# │    robot0_eef_vel_ang      [3D]   EE angular velocity (if available)   │
# │                                                                        │
# │  OBJECT STATE:                                                         │
# │    object                  [14D]  cube pos(3) + quat(4) + vel(7)       │
# │    Can_pos / cube_pos      [3D]   object position (task-dependent)     │
# │    Can_quat / cube_quat    [4D]   object orientation                   │
# │                                                                        │
# │  Note: Not all keys exist in all HDF5 files. Check with:              │
# │    python -c "import h5py; f=h5py.File('image.hdf5','r'); \           │
# │               print(list(f['data/demo_0/obs'].keys()))"               │
# └────────────────────────────────────────────────────────────────────────┘
#
# Set to "--disable_proprioceptive_obs" to train vision-only (no state input):
DISABLE_PROPRIO=""
# DISABLE_PROPRIO="--disable_proprioceptive_obs"   # ← uncomment for vision-only

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  POLICY                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

POLICY="act"                              # Policy architecture
                                          # Options: "act", "diffusion", "vqbet", "tdmpc"
                                          # "act" (Action Chunking Transformer) is the tested default

# ── ACT-specific hyperparameters (override defaults via JSON) ────────────
# Uncomment POLICY_KWARGS to customize. Defaults are sensible for Lift.
# Full list of ACTConfig fields: resfit/lerobot/policies/act/configuration_act.py
#
POLICY_KWARGS=""
# POLICY_KWARGS='{"dim_model": 256, "n_heads": 8, "n_layers": 6}'  # ← default ACT arch
# POLICY_KWARGS='{"dim_model": 512, "n_heads": 8, "n_layers": 8}'  # ← larger model
# POLICY_KWARGS='{"chunk_size": 20, "n_action_steps": 20}'         # ← action chunking (set by script)
# POLICY_KWARGS='{"optimizer_lr": 1e-4}'                            # ← learning rate override

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

STEPS=25000                              # Total optimization steps
                                          # Lift is simpler than Coffee — 100K usually suffices
                                          # 50K for quick tests, 200K for thorough training
BATCH_SIZE=64                            # Batch size. 128 fits 16GB GPU, 256 on 48GB
GRAD_CLIP_NORM=10.0                       # Gradient clipping norm (default 10.0)
NUM_WORKERS=4                             # Dataloader workers (4 is good default)
SEED=""                                   # Random seed. "" = random, or set e.g. SEED="42"
DEVICE="cuda"                             # "cuda" or "cpu"

# ── Logging & Checkpoints ───────────────────────────────────────────────
LOG_FREQ=100                              # Print + WandB log every N steps
SAVE_FREQ=5000                            # Save checkpoint every N steps
                                          # 5K gives 20 checkpoints over 100K steps

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION (rollouts in Robosuite simulator)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

EVAL_ENV="Can"                           # Robosuite environment name. Must match dataset task.
                                          # This is used for rollout evaluation during training.
ROLLOUT_FREQ=5000                         # Run eval rollouts every N steps
                                          # 5K = decent balance of eval frequency vs speed
                                          # 2500 for more granular, 10000 for faster training
EVAL_NUM_ENVS=4                           # Parallel eval environments
                                          # 4 = safe on 16GB, 8-16 on 48GB GPU
EVAL_NUM_EPISODES=50                      # Episodes per evaluation
                                          # 50 = reasonable for Lift (short horizon)
                                          # 100 for final/rigorous runs, 20 for quick sanity
EVAL_CAMERA_SIZE=84                       # Camera resolution for eval rollouts (match dataset)
EVAL_VIDEO_KEY="observation.images.agentview"   # Camera for recorded eval videos
EVAL_RENDER_SIZE=224                      # High-res video recording (pixels)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IMAGE RESIZE                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Resize training images to this square size (e.g. 84).
# Use when your dataset has a different resolution than you want to train at.
# Example: dataset is 256×256 but you want to train at 84×84.
# Leave empty ("") to use the native dataset resolution.
#
# When set, eval_camera_size is automatically matched unless you override it.
# And make sure the EVAL_CAMERA_SIZE matches the IMAGE_SIZE to avoid resolution mismatch during eval rollouts.
#
# IMAGE_SIZE=""
IMAGE_SIZE=84                           # ← uncomment to resize to 84×84

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WANDB LOGGING                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

WANDB_PROJECT="resfit-robomimic-can-bc"         # WandB project name
WANDB_ENABLE="--wandb_enable"             # Set to "" to disable WandB logging
# WANDB_ENABLE=""                         # ← uncomment to disable WandB
WANDB_ENTITY=""                           # WandB entity (team). "" = your default entity

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RESUME (from a previous run)                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

RESUME_CKPT=""                            # Path to checkpoint dir to resume from
                                          # e.g., "$CACHE_DIR/bc_run_.../best"
RESUME_RUN_ID=""                          # WandB run ID to resume (grabs 'latest' artifact)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLEANUP                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# NO_CLEANUP="--no_cleanup"                 # Keep all checkpoint files after training
                                          # Remove this flag to auto-delete intermediate checkpoints
NO_CLEANUP=""                           # ← uncomment to auto-cleanup

# ============================================================================
# Build command
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BC Policy Training — ACT on Lift (Franka Panda)            "
echo "║  Dataset: ${DATASET}                                        "
echo "║  Steps: ${STEPS}  |  Batch: ${BATCH_SIZE}  |  Eval envs: ${EVAL_NUM_ENVS}"
echo "║  Cameras: ${POLICY_CAMERAS:-all in dataset}                 "
echo "║  WandB: ${WANDB_PROJECT}                                    "
echo "╚══════════════════════════════════════════════════════════════╝"

CMD=(
    python resfit/lerobot/scripts/train_bc_dexmg.py
    --dataset "${DATASET}"
    --policy "${POLICY}"
    --steps "${STEPS}"
    --batch_size "${BATCH_SIZE}"
    --grad_clip_norm "${GRAD_CLIP_NORM}"
    --num_workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --log_freq "${LOG_FREQ}"
    --save_freq "${SAVE_FREQ}"
    --wandb_project "${WANDB_PROJECT}"
    --eval_env "${EVAL_ENV}"
    --rollout_freq "${ROLLOUT_FREQ}"
    --eval_camera_size "${EVAL_CAMERA_SIZE}"
    --eval_video_key "${EVAL_VIDEO_KEY}"
    --eval_render_size "${EVAL_RENDER_SIZE}"
    --eval_num_envs "${EVAL_NUM_ENVS}"
    --eval_num_episodes "${EVAL_NUM_EPISODES}"
)

# Optional flags (only add if non-empty)
[[ -n "${WANDB_ENABLE}" ]]      && CMD+=("${WANDB_ENABLE}")
[[ -n "${NO_CLEANUP}" ]]        && CMD+=("${NO_CLEANUP}")
[[ -n "${SEED}" ]]              && CMD+=(--seed "${SEED}")
[[ -n "${WANDB_ENTITY}" ]]      && CMD+=(--wandb_entity "${WANDB_ENTITY}")
[[ -n "${RESUME_CKPT}" ]]       && CMD+=(--resume_ckpt "${RESUME_CKPT}")
[[ -n "${RESUME_RUN_ID}" ]]     && CMD+=(--resume_run_id "${RESUME_RUN_ID}")
[[ -n "${POLICY_KWARGS}" ]]     && CMD+=(--policy_kwargs "${POLICY_KWARGS}")
[[ -n "${DISABLE_PROPRIO}" ]]   && CMD+=("${DISABLE_PROPRIO}")
[[ -n "${POLICY_CAMERAS}" ]]    && CMD+=(--policy_cameras ${POLICY_CAMERAS})
[[ -n "${IMAGE_SIZE}" ]]        && CMD+=(--image_size "${IMAGE_SIZE}")

# Execute
"${CMD[@]}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ BC training on Lift complete."
echo "  Check WandB for the run ID: ${WANDB_PROJECT}"
echo "  Use for residual RL: base_policy.wandb_id=${WANDB_PROJECT}/<run_id>"
echo "════════════════════════════════════════════════════════════════"

# ============================================================================
# ============================================================================
#
#  APPENDIX: CREATING A CUSTOM DATASET WITH 16D STATE
#
#  The pre-built ankile/robomimic-mh-lift-image has 9D state (no joint_pos).
#  To get 16D state (eef_pos + eef_quat + joint_pos + gripper_qpos), you need
#  to create your own dataset. Here's the full recipe:
#
#  ── Step 1: Download the raw HDF5 ──────────────────────────────────────
#
#    cd deps/robomimic
#    python robomimic/scripts/download_datasets.py \
#        --tasks lift --dataset_types mh --hdf5_types raw
#
#    # This gives you: datasets/lift/mh/demo_v15.hdf5
#
#  ── Step 2: Extract observations with desired cameras ──────────────────
#
#    python robomimic/scripts/dataset_states_to_obs.py \
#        --dataset datasets/lift/mh/demo_v15.hdf5 \
#        --output_name image_custom.hdf5 \
#        --done_mode 2 \
#        --camera_names agentview robot0_eye_in_hand \
#        --camera_height 84 \
#        --camera_width 84
#
#    # This renders images and extracts ALL low-dim obs into the HDF5.
#    # The resulting file will have joint_pos, joint_vel, eef_pos, etc.
#
#  ── Step 3: Edit the conversion script to include joint_pos ────────────
#
#    In resfit/lerobot/dataset/convert_robomimic_to_lerobot.py, find the
#    get_expected_low_dim_keys() function (around line 253) and change the
#    panda_low_dim_keys to include robot0_joint_pos:
#
#      panda_low_dim_keys = [
#          "robot0_eef_pos",          # 3D
#          "robot0_eef_quat",         # 4D
#          "robot0_joint_pos",        # 7D  ← ADD THIS
#          "robot0_gripper_qpos",     # 2D
#          # "robot0_joint_vel",      # 7D  ← uncomment if you also want velocities
#          # "robot0_gripper_qvel",   # 2D
#          # "object",                # 14D ← cube pos+quat+vel (gives oracle info)
#      ]
#
#    ALSO edit the environment wrapper to match (so rollout eval works):
#    In resfit/dexmg/environments/dexmg.py, find _get_expected_low_dim_keys()
#    (around line 448) and make the same change to panda_low_dim_keys_single:
#
#      panda_low_dim_keys_single = [
#          "robot0_eef_pos",
#          "robot0_eef_quat",
#          "robot0_joint_pos",        # ← ADD THIS (must match dataset)
#          "robot0_gripper_qpos",
#      ]
#
#  ── Step 4: Convert to LeRobot format ──────────────────────────────────
#
#    cd ../..  # back to repo root
#    python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
#        --dataset deps/robomimic/datasets/lift/mh/image_custom.hdf5 \
#        --output_dir ~/lerobot_datasets/lift-mh-16d \
#        --max_episodes 300
#
#    # To use only 200 episodes:  --max_episodes 200
#    # To create train/test split: --train_ratio 0.9 (90% train, 10% test)
#    # To upload to HF Hub:        --repo_id your-username/lift-mh-16d
#
#  ── Step 5: Use the custom dataset ─────────────────────────────────────
#
#    Just change DATASET at the top of this script:
#      DATASET="~/lerobot_datasets/lift-mh-16d"
#
#    Or if you uploaded to HF:
#      DATASET="your-username/lift-mh-16d"
#
# ============================================================================
