#!/usr/bin/env bash
# =============================================================================
# PickCubeAndInsert Station3 - iterative training 1 (2026-10-16)
#
# Builds the two v3 delta-joint datasets for the second ACT training round:
#   A  PickCubeAndInsert_TS3_Iter1_deltajoint               Trial1 + iter1 rollouts
#   B  PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint  A + recovery segments
#
# Train both with the SAME command and hyper-parameters as Trial1, only the
# dataset changes (see README.md "Iterative training 1").
#
# RECORD OF WHAT WAS RUN. Every step writes a NEW directory; delete outputs
# first to redo. Same three environments as prepare_dataset.sh (never `uv`).
# =============================================================================
set -euo pipefail

REPO=/home/qte9489/personal_abhi/temp/residual-offpolicy-rl
DU=/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil
REPO_PY="${REPO}/.venv/bin/python"
DU_PY="${DU}/.venv/bin/python"

D="${REPO}/trossen_real/datasets"
IT="${D}/iterativeTraining/TrossenStation3"
INFER="${IT}/InferData/20261014_113503_PickCubeAndInsert_iter1"            # 51 eps, ACT rollouts + HG-DAgger interventions, abs joint
RECOV="${IT}/RecoveryData/20261014_155431_PickCubeAndInsert_RecoveryEpisodes" # 51 eps, teleop: deliberate failed insert, then recovery
TRIAL1="${D}/PickCubeAndInsert_Station3_Trial1_deltajoint_old"               # 178 eps, v2.1 copy of the Trial1 delta dataset
OUT="${IT}/ACT_Iter1"
mkdir -p "${OUT}"
cd "${REPO}"

# ── 0. Remove the RL tab's .rl_bak backups from data/ ────────────────────────
# The visualizer's RL (reward) tab left episode_*.parquet.rl_bak files next to
# the parquets. lerobot 0.1.0 loads EVERY file in data/, so they are read as
# extra rows and the load fails with "timestamps unexpectedly violate the
# tolerance". Moved (not deleted) to <dataset>/.rl_backups/:
#   mkdir -p <ds>/.rl_backups/data/chunk-000 && mv <ds>/data/chunk-000/*.rl_bak <ds>/.rl_backups/data/chunk-000/

# ── 1. Schema: match Trial1 ──────────────────────────────────────────────────
# Trial1 has 2 cameras (wrists). The new data also has cam_specialist (added
# for SARM) and the infer data has observation.intervened/policy_action. The
# merge takes the UNION of schemas and would give Trial1 episodes a
# cam_specialist they don't have, so these are dropped during the delta
# conversion (--drop-features). ACT keeps the same 2-camera input as Trial1.

# ── 2. iter1 rollouts -> delta joint, without the 3 failures ────────────────
# filter_episodes --dry-run discards exactly 3, 13, 34 (reward pedal never
# pressed); --episodes does the same selection in this one pass.
"${REPO_PY}" -m trossen_real.scripts.convert_to_delta_joint_dataset \
  --input-root  "${INFER}" \
  --output-root "${OUT}/iter1_48eps_deltajoint" \
  --episodes "$(python3 -c "print(','.join(str(i) for i in range(51) if i not in (3, 13, 34)))")" \
  --drop-features observation.images.cam_specialist,observation.intervened,observation.policy_action
# -> 48 episodes, 23707 frames

# ── 3. recovery episodes -> delta joint, recovery part only ─────────────────
# Each recovery episode is a teleop demo of: approach, grasp, DELIBERATELY
# misaligned insert, release (cube tipped/crooked on the plate), recovery.
# Whole episodes would teach ACT the deliberate failure, so only the frames
# from 10 frames (0.5 s) after the failed release are kept: ACT sees
# "cube unseated on the plate -> recover", never the failure itself.
# Failed release = first gripper opening (state[6] > 0.02) while in stage 3
# ("cube being inserted"), from the SARM stage labels; 47/51 episodes have
# one (4 without: 2, 11, 18, 35 - dropped). Checked visually at 8 cut points.
# KEEP_FROM = "recovery_episode:first_kept_frame,..." (SARM episode = 48 + i).
KEEP_FROM="0:423,1:359,3:513,4:315,5:342,6:476,7:287,8:313,9:340,10:372,12:345,13:270,14:261,15:248,16:302,17:264,19:316,20:299,21:334,22:245,23:280,24:275,25:367,26:271,27:346,28:267,29:320,30:302,31:345,32:320,33:292,34:342,36:274,37:346,38:335,39:253,40:320,41:257,42:304,43:276,44:230,45:322,46:263,47:260,48:320,49:305,50:257"
"${REPO_PY}" -m trossen_real.scripts.convert_to_delta_joint_dataset \
  --input-root  "${RECOV}" \
  --output-root "${OUT}/recovery_47eps_fromFailedRelease_deltajoint" \
  --episodes "$(echo "${KEEP_FROM}" | tr ',' '\n' | cut -d: -f1 | paste -sd,)" \
  --keep-from "${KEEP_FROM}" \
  --drop-features observation.images.cam_specialist
# -> 47 episodes, 13497 frames (14673 frames before the cut dropped)

# ── 4. Merge (v2.1) ──────────────────────────────────────────────────────────
# Task strings differ per source; force Trial1's.
"${REPO_PY}" "${DU}/tools/merge_v21_datasets.py" \
  --dataset "${TRIAL1}" \
  --dataset "${OUT}/iter1_48eps_deltajoint" \
  --force-task "PickCubeAndInsertStation3" \
  --output  "${OUT}/PickCubeAndInsert_TS3_Iter1_deltajoint"
"${REPO_PY}" "${DU}/tools/merge_v21_datasets.py" \
  --dataset "${TRIAL1}" \
  --dataset "${OUT}/iter1_48eps_deltajoint" \
  --dataset "${OUT}/recovery_47eps_fromFailedRelease_deltajoint" \
  --force-task "PickCubeAndInsertStation3" \
  --output  "${OUT}/PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint"

# ── 5. v2.1 -> v3 ────────────────────────────────────────────────────────────
# Renames each v2.1 input to <name>_old and writes v3 at the original name.
cd "${DU}"
"${DU_PY}" tools/convert_v21_to_v3.py --repo-id PickCubeAndInsert_TS3_Iter1_deltajoint              --root "${OUT}"
"${DU_PY}" tools/convert_v21_to_v3.py --repo-id PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint --root "${OUT}"
cd "${REPO}"

echo "Done. Copy to the server's DATA_DIR and train (README.md):"
echo "  ${OUT}/PickCubeAndInsert_TS3_Iter1_deltajoint"
echo "  ${OUT}/PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint"
