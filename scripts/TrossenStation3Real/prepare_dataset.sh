#!/usr/bin/env bash
# =============================================================================
# PickCubeAndInsert Station3 Trial1 - dataset preparation (2026-09-28)
#
# Turns the 5 raw teleop sessions into the 3 v3 datasets ACT trains on:
#   abs joint | delta joint | relative (chunk-anchor delta)
#
# THIS IS A RECORD OF WHAT WAS ALREADY RUN. It is re-runnable, but every step
# writes a NEW dataset directory - re-running over existing outputs will fail
# or overwrite. Delete the outputs first if you really want to redo it.
#
# NEVER use `uv` on this machine - it breaks the repo venv. Every command below
# calls an interpreter directly. Three different environments are involved:
#   REPO_PY : this repo's .venv          - trossen_real.scripts.*   (lerobot 0.1.0)
#   DU_PY   : DatasetUtil/.venv          - merge + v2.1->v3         (lerobot 0.4.4, needed for v3)
#   LR_BIN  : GeneralistRewardModels/... - lerobot-edit-dataset
# Using the wrong one fails: the repo venv's lerobot 0.1.0 cannot write v3.
# =============================================================================
set -euo pipefail

REPO=/home/qte9489/personal_abhi/temp/residual-offpolicy-rl
DU=/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil
LR=/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot

REPO_PY="${REPO}/.venv/bin/python"
DU_PY="${DU}/.venv/bin/python"
LR_EDIT="${LR}/.venv/bin/lerobot-edit-dataset"

D="${REPO}/trossen_real/datasets"
OUT=PickCubeAndInsert_Station3_Trial1

cd "${REPO}"

# ── 0. Raw sessions (v2.1, 180 episodes / 56186 frames total) ────────────────
#   20260924_132455  3 eps  GOOD
#   20260924_155210 60 eps  drop ep 21   (also caught automatically)
#   20260924_164756 60 eps  GOOD
#   20260925_100304  6 eps  drop ep 5    (manual: pedal held through a failure)
#   20260925_100751 51 eps  GOOD

# ── 1. Filter the two sessions with bad episodes ─────────────────────────────
# Writes a NEW re-indexed v2.1 dataset; the raw sessions are never modified.
# Always dry-run first to see the classifier's verdict:
#   ${REPO_PY} -m trossen_real.scripts.filter_episodes \
#     --dataset ${D}/<session> --output /tmp/unused --dry-run
#
# ep21: the automatic classifier discards it on its own (reward pedal never
# held). --exclude 21 is passed anyway so the intent is explicit in this script.
"${REPO_PY}" -m trossen_real.scripts.filter_episodes \
  --dataset "${D}/20260924_155210_PickCubeAndInsertStation3" \
  --output  "${D}/20260924_155210_PickCubeAndInsertStation3_filtered" \
  --exclude 21
# -> 59 kept, 1 discarded (19754 frames)

# ep5: the classifier KEEPS it (reward pedal genuinely held, 563 frames / 248
# done-frames vs ~300/45 for its siblings). Operator marked success on a failed
# attempt - not detectable from next.reward/next.done, so it needs the manual
# override. Same failure mode as Trial1's episodes 75/84.
"${REPO_PY}" -m trossen_real.scripts.filter_episodes \
  --dataset "${D}/20260925_100304_PickCubeAndInsertStation3" \
  --output  "${D}/20260925_100304_PickCubeAndInsertStation3_filtered" \
  --exclude 5
# -> 5 kept, 1 discarded (1611 frames), logged as "MANUAL OVERRIDE (--exclude)"

# ── 2. Merge all 5 into one v2.1 dataset ─────────────────────────────────────
# No --force-task needed: all 5 already share task "PickCubeAndInsertStation3"
# and an identical 16-key feature schema (checked before merging).
"${REPO_PY}" "${DU}/tools/merge_v21_datasets.py" \
  --dataset "${D}/20260924_132455_PickCubeAndInsertStation3" \
  --dataset "${D}/20260924_155210_PickCubeAndInsertStation3_filtered" \
  --dataset "${D}/20260924_164756_PickCubeAndInsertStation3" \
  --dataset "${D}/20260925_100304_PickCubeAndInsertStation3_filtered" \
  --dataset "${D}/20260925_100751_PickCubeAndInsertStation3" \
  --output  "${D}/${OUT}"
# -> 178 episodes, 55177 frames, 356 videos, 1 task (v2.1)

# ── 3. Delta-joint conversion (v2.1 -> v2.1) ─────────────────────────────────
# Joints 0-5 become action-minus-state deltas; the gripper stays absolute.
# Start/end trimming is DISABLED in this script (since 2026-09-01) - see the
# long comments in convert_to_delta_joint_dataset.py. Expect "trimmed 0".
"${REPO_PY}" -m trossen_real.scripts.convert_to_delta_joint_dataset \
  --input-root  "${D}/${OUT}" \
  --output-root "${D}/${OUT}_deltajoint" \
  --dry-run
"${REPO_PY}" -m trossen_real.scripts.convert_to_delta_joint_dataset \
  --input-root  "${D}/${OUT}" \
  --output-root "${D}/${OUT}_deltajoint"
# -> 178 episodes, 55177 frames, 0 frames trimmed

# ── 4. v2.1 -> v3 (both) ─────────────────────────────────────────────────────
# MUST use DU_PY: this repo's venv has lerobot 0.1.0, which cannot write v3.
# Renames each v2.1 input to <name>_old and writes v3 at the original name.
cd "${DU}"
"${DU_PY}" tools/convert_v21_to_v3.py --repo-id "${OUT}"            --root "${D}"
"${DU_PY}" tools/convert_v21_to_v3.py --repo-id "${OUT}_deltajoint" --root "${D}"
cd "${REPO}"

# ── 5. Relative-action stats (from the ABS v3 dataset) ───────────────────────
# Relative training needs ABSOLUTE actions in the dataset - the chunk-anchor
# delta transform happens in the processor at train time. So this is built from
# the abs dataset, NOT from the delta one.
"${LR_EDIT}" \
  --repo_id     "${D}/${OUT}" \
  --new_repo_id "${D}/${OUT}_relativeDeltaTraining" \
  --operation.type recompute_stats \
  --operation.relative_action true \
  --operation.chunk_size 20 \
  --operation.relative_exclude_joints "['gripper']" \
  --operation.num_workers 4
# -> relative_dims=6/7 (excluded=1), mean=0.0098, std=0.0834, q01=-0.2908, q99=0.2470

echo
echo "Done. Three v3 datasets, all 178 episodes / 55177 frames:"
echo "  abs      : ${D}/${OUT}"
echo "  delta    : ${D}/${OUT}_deltajoint"
echo "  relative : ${D}/${OUT}_relativeDeltaTraining"
echo "(_old siblings are the v2.1 backups kept by the v3 converter.)"
