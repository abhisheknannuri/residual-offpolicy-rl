# PickAndInsertCube — ACT BC Evaluation Notes

## Task

Pick the red cube and insert it precisely into the correct insertion hole.

## Evaluation Protocol

- **15 fixed cube positions** — randomized once (by tossing the cube and marking each landing spot), then reused in the same order (1-15) for every checkpoint's evaluation, so results are comparable across checkpoints.
- **Robot start position:** identical every episode.
- **Placement fixture position:** fixed (removing/refitting the fixture between sessions introduces very minor variation, treated as fixed for comparison purposes).
- **Two evaluation passes per checkpoint:**
  - **Pass A — Autonomous only:** zero human intervention.
  - **Pass B — Light intervention allowed:** brief human guidance only when the policy is visibly stuck.
- **Sub-stages tracked per position** (both passes):
  1. **Approach/Align** — gripper approaches and orients to match the cube
  2. **Grasp/Lift** — grasps and lifts the cube
  3. **Transport** — carries the cube to the placement fixture
  4. **Place** — precise insertion into the correct hole
- Max Env steps (Episode Horizon) is `500` for eveluation of BC
## Legend

| Symbol | Meaning |
|---|---|
| ✅ | Success, autonomous |
| 🟡 | Success, required human intervention |
| ❌ | Failed |
| — | Not reached (an earlier stage failed) |

---

## 1. Baseline Training

### Action Space: Delta Joint

- **Dataset:** `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint` (server copy of this exact laptop dataset)
- **Checkpoints:** `/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_bc_PickAndInsertCubeStation1_merged_deltaJoint/checkpoints/{005000..080000, step 5000}`
- **Training command:**
  ```sh
  cd /home/qte9489/Research/LEROBOT/lerobot/
  uv run lerobot-train \
    --config_path act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml \
    --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/PickAndInsertCube_Station1_merged_deltajoint/ \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/policy_bc_PickAndInsertCubeStation1_merged_deltaJoint \
    --batch_size=64 \
    --num_workers=8 \
    --steps=80000 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --eval_freq=0 \
    --log_freq=100 \
    --wandb.enable=true \
    --job_name=act_bc_pickandinsertcube_real \
    --wandb.project=Trossen_PickAndInsertCube_Real \
    --wandb.notes="ACT BC baseline on real PickAndInsertCube Station 1 merged delta joint, 256x256" \
    --wandb.disable_artifact=true \
    --policy.chunk_size=20 \
    --policy.n_action_steps=20 \
    --policy.use_vae=false \
    --policy.optimizer_lr=1e-4
  ```

#### Checkpoint `005000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `010000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `015000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `020000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `025000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `030000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `035000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `//home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/14-27-30`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 14:27 to 14:38
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_142755/`
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `040000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/14-13-42`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 14:13 to 14:24
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_141357/`
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `045000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/13-59-52`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 13:59 to 14:11
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_140015/`
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `050000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/13-46-33`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 13:46 to 13:58
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_134711/`
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `055000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/13-33-12`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 13:33 to 13:45
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_133353/`
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `060000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/13-18-57`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 13:18 to 13:32
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_131926/`
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success 2/15 · intervened-success __/15 · failed __/15

#### Checkpoint `065000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/13-04-33`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 13:00 to 13:16
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_130526/`
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success 0/15 · intervened-success __/15 · failed __/15

#### Checkpoint `070000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/11-56-56`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 11:56 to 12:10
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_115731/`
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `075000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/11-42-42`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 11:42 to 11:55
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_114405/`
- 
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success 0/15 · intervened-success __/15 · failed __/15

#### Checkpoint `080000`

**Pass A — Autonomous only** (no human intervention)
- log paths: `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/logs/infer_app/2026-09-02/11-16-04`
- vides recorded on robot `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_videos/2026-09-02` but from time 11:16 to 11:35
- Top-View recordings in server are at `sftp://10.213.245.105/home/qte9489/Research/IntelReasense/realsense_utils/VideoRecordings/Base_BC_delta_action_evaluations/20260902_112104/`
| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success 0/15 · intervened-success __/15 · failed __/15


### Action Space: Relative Delta

*(not yet evaluated — see [training_prompts.md](../trossen_real/training_prompts.md) "ACT training with relative delta actions" for the prepared datasets/commands. Fill in once a checkpoint exists.)*


---

## 2. Iterative Training (Iteration 1)

### Action Space: Delta Joint

- **Dataset:** `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/iterativeTraining/PickAndInsertCube_TS1_ForIter1_v3` (teleop + Iteration-1 infer data merged)
- **Checkpoints:** `/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_bc_PickAndInsertCube_TS1_ForIter1_v3_deltaJoint/checkpoints/{005000..080000, step 5000}`
- **Training command:**
  ```sh
  cd /home/qte9489/Research/LEROBOT/lerobot/
  uv run lerobot-train \
    --config_path act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml \
    --dataset.repo_id=/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/iterativeTraining/PickAndInsertCube_TS1_ForIter1_v3 \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/policy_bc_PickAndInsertCube_TS1_ForIter1_v3_deltaJoint \
    --batch_size=64 \
    --num_workers=8 \
    --steps=80000 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --eval_freq=0 \
    --log_freq=100 \
    --wandb.enable=true \
    --job_name=act_bc_pickandinsertcube_real \
    --wandb.project=Trossen_PickAndInsertCube_Real \
    --wandb.notes="ACT BC IterTrain1 on real PickAndInsertCube Station 1 merged delta joint, 256x256" \
    --wandb.disable_artifact=true \
    --policy.chunk_size=20 \
    --policy.n_action_steps=20 \
    --policy.use_vae=false \
    --policy.optimizer_lr=1e-4
  ```

#### Checkpoint `005000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `010000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `015000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `020000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `025000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `030000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `035000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `040000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `045000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `050000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `055000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `060000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `065000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `070000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `075000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15

#### Checkpoint `080000`

**Pass A — Autonomous only** (no human intervention)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |
| 11 |  |  |  |  |  |
| 12 |  |  |  |  |  |
| 13 |  |  |  |  |  |
| 14 |  |  |  |  |  |
| 15 |  |  |  |  |  |

**Pass B — Light intervention allowed** (brief guidance only when visibly stuck)

| Pos | Approach/Align | Grasp/Lift | Transport | Place | Final | Notes |
|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 11 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 13 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |

**Summary:** autonomous success __/15 · intervened-success __/15 · failed __/15


### Action Space: Relative Delta

*(not yet evaluated — see [training_prompts.md](../trossen_real/training_prompts.md) "ACT training with relative delta actions" for the prepared datasets/commands. Fill in once a checkpoint exists.)*

