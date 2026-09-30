# 1. real hardware (or mock, for testing)
python -m trossen_real.follower.follower_single_server --config trossen_station2_single --port 5060

# 2. training repo's venv
…/lerobot  main !? is 📦 v0.5.2  v3.10.12 (.venv) 1h58m16s 
❯ cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
uv run python custom_scripts/policy_server.py \
  --checkpoint /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_bc_PickAndInsertCubeStation1_merged_deltaJoint/checkpoints/045000/pretrained_model/ --port 5070 --n-action-steps 10

# 3. this workspace's venv
…/residual-offpolicy-rl  dev/abhi-resfit-setup !? is 📦 v0.1.0  v3.10.20 (.venv) 26m39s 
❯ python -m trossen_real.infer_app.app --port 5080
python -m trossen_real.infer_app.app --port 5080 --log-dir ./infer_logs



python -m trossen_real.scripts.test_real_residual_env \
    --config trossen_station2_single \
    --policy-server-url http://127.0.0.1:5070 \
    --dataset-repo-id 20260805_150824_and_153134_153727_20260807_110711_PickAndInsertCube_delta_old \
    --dataset-root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260805_150824_and_153134_153727_20260807_110711_PickAndInsertCube_delta_old \
    --action-space delta_joint \
    --max-steps 20 --num-episodes 1


--device /dev/input/event7

python DatasetUtil/tools/reward_annotator_ui.py --dataset-path /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260805_150824_and_153134_153727_20260807_110711_PickAndInsertCube_delta_old


(.venv) qte9489@cw011081522:~/personal_abhi/temp/residual-offpolicy-rl$ python /home/qte9489/work/ibrl/identify_pedal.py --device /dev/input/event7
====================================================================
Reading pedal: /dev/input/event7   (VEC VEC USB Footpedal)
Listening for 25 s. Press each pedal now.
Suggested: LEFT x2, pause, MIDDLE x2, pause, RIGHT x2.
====================================================================
  [16:35:15]  PRESS    code=256   name=BTN_0/BTN_MISC
  [16:35:15]  release  code=256   name=BTN_0/BTN_MISC
  [16:35:16]  PRESS    code=258   name=BTN_2
  [16:35:16]  release  code=258   name=BTN_2
  [16:35:17]  PRESS    code=257   name=BTN_1
  [16:35:17]  release  code=257   name=BTN_1
  [16:35:17]  PRESS    code=257   name=BTN_1
  [16:35:17]  release  code=257   name=BTN_1
  [16:35:18]  PRESS    code=257   name=BTN_1
  [16:35:18]  release  code=257   name=BTN_1
  [16:35:18]  PRESS    code=257   name=BTN_1
  [16:35:18]  release  code=257   name=BTN_1
^C
(stopped early)

====================================================================
SUMMARY - distinct pedals seen (by press count):
  code=256   name=BTN_0/BTN_MISC   presses=1
  code=257   name=BTN_1            presses=4
  code=258   name=BTN_2            presses=1
====================================================================
Tell me which physical pedal (left/middle/right) produced each code.
(.venv) qte9489@cw011081522:~/personal_abhi/temp/residual-offpolicy-rl$ pwd
/home/qte9489/personal_abhi/temp/residual-offpolicy-rl


 /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260813_195545_and_200141_and_20260814_142010_PickAndInsertCube_Station1





python -m trossen_real.scripts.convert_to_delta_joint_dataset --input-root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260813_195545_and_200141_and_20260814_142010_PickAndInsertCube_Station1 --output-root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260813_195545_and_200141_and_20260814_142010_PickAndInsertCube_Station1_delta

/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260805_150824_and_153134_153727_20260807_110711_PickAndInsertCube_delta_rewardLabled_v21_old
/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260813_195545_and_200141_and_20260814_142010_PickAndInsertCube_Station1_delta_v21


cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil

source .venv/bin/activate

python tools/merge_v21_datasets.py \
        --dataset_a /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260805_150824_and_153134_153727_20260807_110711_PickAndInsertCube_delta_rewardLabled_v21_old \
        --dataset_b /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260813_195545_and_200141_and_20260814_142010_PickAndInsertCube_Station1_delta_v21 \
        --output /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260805_150824_and_153134_153727_20260807_110711_PickAndInsertCube_delta_rewardLabled_v21_old_PickAndInsertCube_Station1_delta_v21



uv run python tools/convert_v21_to_v3.py --repo-id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260805_150824_and_153134_153727_20260807_110711_PickAndInsertCube_delta_rewardLabled_v21_old_PickAndInsertCube_Station1_delta_v21


cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
uv run lerobot-train \
  --config_path act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml \
  --dataset.repo_id=/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260805_150824_and_153134_153727_20260807_110711_PickAndInsertCube_delta_rewardLabled_v21_old_PickAndInsertCube_Station1_delta_v21 \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/policy_bc_PickAndInsertCubeStation2and1_deltaJoint \
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
  --wandb.notes="ACT BC baseline on real PickAndInsertCube Station 2 and 1, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=25 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4



/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260813_195545_PickAndInsertCube_Station1
/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260813_200141_PickAndInsertCube_Station1

/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_temp1

/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260814_142010_PickAndInsertCube_Station1
/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260826_100134_PickAndInsertCube_Station1

/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_temp2

/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260826_100912_PickAndInsertCube_Station1


python tools/merge_v21_datasets.py \
        --dataset_a /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_temp3 \
        --dataset_b /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260826_100912_PickAndInsertCube_Station1 \
        --output /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged
        
        
        


/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged


python -m trossen_real.scripts.convert_to_delta_joint_dataset --input-root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_old --output-root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint



uv run python tools/convert_v21_to_v3.py --repo-id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint


cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
uv run lerobot-train \
  --config_path act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml \
  --dataset.repo_id=/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint \
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
  
  
  
  

### Server Training:
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



reward labelling using lerobot-dataset-visualizer (the modified version from my github. i.e in /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/lerobot-dataset-visualizer)
do a docker compose up --build

and annoatte teh v3 or v2.1 directly with sparse reward for now and next.done and next.reward will be updated automatically.

❯ cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
uv run python custom_scripts/policy_server.py \
  --checkpoint /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_bc_PickAndInsertCubeStation1_merged_deltaJoint/checkpoints/045000/pretrained_model/ --port 5070 --n-action-steps 12

python -m trossen_real.infer_app.app --port 5080






## RECAP style Iterative Training

python tools/merge_v21_datasets.py \
# NOTE (2026-09-01): the 8 raw sessions below originally had a next.done/
# next.reward sync bug (next.done was narrow - only a single terminal tick -
# instead of matching every frame where next.reward==1; fixed going forward
# in infer_loop.py/dataset_recorder.py). These _donefixed copies are the
# retroactively-corrected versions (trossen_real.scripts.fix_next_done_sync,
# next.done |= next.reward==1, one direction only) - the originals (without
# _donefixed) are the RAW, still-uncorrected recordings, kept for now as a
# fallback, not meant to be used directly anymore.
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_101911_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_102733_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_103451_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_104209_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_105006_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_105352_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_110442_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_110828_PickAndInsertCube_TS1_Iter1_donefixed \
--output /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed



.venv/bin/python -m trossen_real.scripts.filter_episodes \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed --output /tmp/unused --dry-run


.venv/bin/python -m trossen_real.scripts.filter_episodes \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed --output /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed_filtered


firsst trained ACT using this adtaset: /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint for 70k steps in server
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

randomly picked 50k checkoint since all checkpoints performed the same and i.e picking teh cube is not perfect annd some tiems it failed to arraneg it sef proeprly while picking and if arranged properly tehn it picks an dlilfts and transfers to teh insertion hoe proeprly. and insertion is an otehr issue since its is precise task and almost all teh time it fails.


collected 65 episodes of new adtat by doing iner of taht 50K echeckpoints with human corrections.

```sh

python tools/merge_v21_datasets.py \
# NOTE (2026-09-01): the 8 raw sessions below originally had a next.done/
# next.reward sync bug (next.done was narrow - only a single terminal tick -
# instead of matching every frame where next.reward==1; fixed going forward
# in infer_loop.py/dataset_recorder.py). These _donefixed copies are the
# retroactively-corrected versions (trossen_real.scripts.fix_next_done_sync,
# next.done |= next.reward==1, one direction only) - the originals (without
# _donefixed) are the RAW, still-uncorrected recordings, kept for now as a
# fallback, not meant to be used directly anymore.
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_101911_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_102733_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_103451_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_104209_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_105006_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_105352_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_110442_PickAndInsertCube_TS1_Iter1_donefixed \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/20260901_110828_PickAndInsertCube_TS1_Iter1_donefixed \
--output /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed



.venv/bin/python -m trossen_real.scripts.filter_episodes \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed --output /tmp/unused --dry-run


.venv/bin/python -m trossen_real.scripts.filter_episodes \
--dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed --output /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed_filtered

```

Iteration1: 
- dataset (original + newly collected using polciy):
	/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint_old (_old is v2.1 version of v3 PickAndInsertCube_Station1_merged_deltajoint). it is delta joint adtaset - verified 2026-09-01: 0/136 episodes affected by the next.done/next.reward bugs below, no fix needed for this one.
	/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed_filtered (abs joint action dataset, next.done/next.reward-corrected + filtered per the corrected merge/filter commands above - needs to eb converted to delta joint action)

# NOTE (2026-09-01): PickAndInsertCube_TS1_Iter1_absJoint_merged_filtered (no
# _donefixed) has been REMOVED - it had the narrow next.done bug (fixed via
# fix_next_done_sync.py, see the merge/filter commands above) AND its
# delta-converted/final-merged descendants below
# (..._filtered_deltajoint, iterativeTraining/PickAndInsertCube_TS1_ForIter1)
# are now STALE (also hit by the separate end-trim bug in
# convert_to_delta_joint_dataset.py, since fixed - end-trim disabled) - both
# need to be deleted and regenerated from the _donefixed_filtered dataset
# below before being used for anything.
```sh
# converting to delta joint action dataset.

python -m trossen_real.scripts.convert_to_delta_joint_dataset \
--input-root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed_filtered \
--output-root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_deltaJoint_merged_donefixed_filtered

```

tha gives us teh delta joint action dataset but it is still v2.1


- merged v2.1 adtaset is: 

```sh
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil
uv run python tools/merge_v21_datasets.py \
  --dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint_old \
  --dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_deltaJoint_merged_donefixed_filtered \
  --force-task "PickAndInsertCube" \
  --output /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/iterativeTraining/PickAndInsertCube_TS1_ForIter1_v21
# ^ delete the STALE iterativeTraining/PickAndInsertCube_TS1_ForIter1 (built from
# the buggy pre-donefixed/pre-endtrim-fix data) before re-running this.

```


- V3 conversion of that merged dataset.
```sh

cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil
uv run python tools/convert_v21_to_v3.py --repo-id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/iterativeTraining/PickAndInsertCube_TS1_ForIter1_v21
```
- That renamed teh dataset `PickAndInsertCube_TS1_ForIter1_v21` to `PickAndInsertCube_TS1_ForIter1_v21_old`
- and teh dataset currently `PickAndInsertCube_TS1_ForIter1_v21` i `V3` dataset. and i manually renamed the adtaset `PickAndInsertCube_TS1_ForIter1_v21` to `PickAndInsertCube_TS1_ForIter1_v3`


- Copy it to server `TVD29ServerRack`


- Start teh iterative training 1 using 

```sh

cd /home/qte9489/Research/LEROBOT/lerobot/
uv run lerobot-train \
  --config_path act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/PickAndInsertCube_TS1_ForIter1_v3/ \
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
# NOTE (2026-09-01/02): this dataset's meta/stats.json was missing an entry
# for observation.images.cam_right_wrist/cam_left_wrist entirely (see "ACT
# training with relative delta actions" section below) - patched in place
# with a placeholder for completeness. CORRECTION: this was never actually a
# crash risk for real training - GeneralistRewardModels/lerobot's
# datasets/factory.py already guards this (`if key not in dataset.meta.stats:
# dataset.meta.stats[key] = {}` before applying ImageNet stats), confirmed by
# the fact that THIS EXACT dataset (pre-patch) already trained cleanly to
# 80k steps with both cameras as real VISUAL inputs - see checkpoints at
# .../GeneralistRewardModels/lerobot/outputs/train/policy_bc_PickAndInsertCube_TS1_ForIter1_v3_deltaJoint/.
# The KeyError only exists in the unrelated lerobot_v3 clone (see below).
# Still otherwise STALE per the note above (built from
# pre-donefixed/pre-endtrim-fix data) - don't use for new training without
# regenerating from the corrected sources.

## ACT training with relative delta actions

Uses pi0/pi05-style chunk-anchor-relative action deltas
(`delta[t+k] = action[t+k] - state[t]` for every k in a chunk, one anchor
state per chunk - NOT a rolling per-frame diff) for ACT. The training repo
is **`/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot`**
(the one actually used for real training runs, per the policy-server
commands elsewhere in this file - NOT `/home/qte9489/work/external_repos/lerobot_v3`,
a stray, unrelated clone on this machine that this section mistakenly used
at first). That repo's `ACTConfig` ALREADY declares `use_relative_actions`/
`relative_exclude_joints`/`action_feature_names` natively (verified:
`grep -n use_relative_actions src/lerobot/policies/act/configuration_act.py`
finds it at line 137) - no code change needed there at all. (`lerobot_v3`
lacks these fields on `ACTConfig`, which is what caused the confusion - a
field addition made there to work around it was reverted since that repo
isn't used for anything.)

### Dataset stats.json completeness (cosmetic, NOT a crash risk in the real training repo)

`merge_v21_datasets.py`'s stats.json never had ANY entry for video/image
feature keys (it deliberately skips computing real per-pixel stats for them
- reasonable, since that'd mean decoding every video). This looked like a
real bug at first - the OTHER, unrelated `lerobot_v3` clone's
`datasets/factory.py` does `dataset.meta.stats[key][stats_type] =
imagenet_value` with no guard, which really does `KeyError` on a dataset
missing that key.

**But it is NOT a bug in the actual training repo.**
`GeneralistRewardModels/lerobot`'s `datasets/factory.py::make_dataset()`
already guards this:
```python
if key not in dataset.meta.stats:
    dataset.meta.stats[key] = {}
```
before applying ImageNet mean/std - confirmed both by reading the code and
by the fact that `iterativeTraining/PickAndInsertCube_TS1_ForIter1_v3`
(pre-patch, genuinely missing these keys) already trained cleanly to 80k
steps with both cameras as real `VISUAL` policy inputs (see its checkpoints
at `.../GeneralistRewardModels/lerobot/outputs/train/policy_bc_PickAndInsertCube_TS1_ForIter1_v3_deltaJoint/`).

So: `compute_stats_from_parquet()` in `merge_v21_datasets.py` was still
updated to emit a real (c,1,1)-shaped placeholder (matching
`IMAGENET_STATS` in `lerobot/utils/constants.py`) for every video/image
feature, and the already-built datasets below were hand-patched with the
same placeholder (`PickAndInsertCube_Station1_merged`,
`PickAndInsertCube_Station1_merged_relativeDeltaTraining`,
`iterativeTraining/relative/PickAndInsertCube_TS1_ForIter1[_relativeDeltaTraining]`,
`iterativeTraining/PickAndInsertCube_TS1_ForIter1_v3`) - harmless and makes
the metadata complete/consistent - but this was never required for training
in the real repo to work, only for the stray `lerobot_v3` clone that isn't
actually used for anything.

Verified by actually running a 5-step `lerobot-train` sanity check to
completion on both tracks below, from the correct
`GeneralistRewardModels/lerobot` repo (loss moving normally, no errors),
then deleting the sanity run's `--output_dir` afterward.

### Track A - original teleop dataset, relative actions

Source: `trossen_real/datasets/PickAndInsertCube_Station1_merged_old` (v2.1,
136 episodes, fully teleoperated, `next.done`/`next.reward` were already
verified in sync before this - `next.reward` doesn't even exist as a column
here, `next.done` is False everywhere, which is fine for ACT since neither
is a model input).

```sh
# 1. sanity-check next.done/next.reward on the v2.1 source (should drop 0 -
#    it will actually classify all 136 as DISCARD since next.reward never
#    exists/hits 1, which is expected/correct for pure teleop data with no
#    reward pedal - this check is just confirming nothing is inconsistent,
#    not something to act on for a pure-BC dataset):
.venv/bin/python -m trossen_real.scripts.filter_episodes \
  --dataset trossen_real/datasets/PickAndInsertCube_Station1_merged_old \
  --output /tmp/unused --dry-run

# 2. v3 dataset already exists at trossen_real/datasets/PickAndInsertCube_Station1_merged
#    (136 episodes, 48929 frames - confirmed identical content to _old).

# 3. Compute relative-action stats. This OVERWRITES stats.json in place, and
#    that v3 dataset is also used for non-relative training, so write to a
#    new name instead of overwriting:
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
uv run lerobot-edit-dataset \
  --repo_id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged \
  --new_repo_id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_relativeDeltaTraining \
  --operation.type recompute_stats \
  --operation.relative_action true \
  --operation.chunk_size 20 \
  --operation.relative_exclude_joints "['gripper']" \
  --operation.num_workers 4
# -> relative_dims=6/7 (excluded=1), mean=0.0086, std=0.0856, q01=-0.3062, q99=0.2728
# -> trossen_real/datasets/PickAndInsertCube_Station1_merged_relativeDeltaTraining
```

Real training command (Track A, this is what to actually run tonight):
```sh
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
uv run lerobot-train \
  --dataset.repo_id=/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_relativeDeltaTraining \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints="['gripper']" \
  --output_dir=outputs/train/policy_bc_PickAndInsertCube_Station1_merged_relativeDeltaTraining \
  --batch_size=64 \
  --num_workers=8 \
  --steps=80000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=act_bc_pickandinsertcube_real_relative \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="ACT BC baseline on real PickAndInsertCube Station 1 merged, relative (chunk-anchor delta) actions, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4
```

Training in server (Trained in server an dnot in my laptop. but copied the dataset `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_relativeDeltaTraining` to server at `/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/PickAndInsertCube_Station1_merged_relativeDeltaTraining/`)
```sh
cd /home/qte9489/Research/LEROBOT/lerobot
uv run lerobot-train \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/PickAndInsertCube_Station1_merged_relativeDeltaTraining/ \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints="['gripper']" \
  --output_dir=outputs/train/policy_bc_PickAndInsertCube_Station1_merged_relativeDeltaTraining \
  --batch_size=64 \
  --num_workers=8 \
  --steps=80000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=act_bc_pickandinsertcube_real_relative \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="ACT BC baseline on real PickAndInsertCube Station 1 merged, relative (chunk-anchor delta) actions, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4
```


### Track B - iterative (teleop + Iteration1 infer data), relative actions

Sources:
- `trossen_real/datasets/PickAndInsertCube_Station1_merged_old` (v2.1, 136
  episodes, fully teleoperated, no `next.reward` column).
- `infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed_filtered`
  (v2.1, 65 episodes, absolute-joint, next.done/next.reward-corrected +
  filtered - see "Iteration1" section above). Note this is the ABSOLUTE
  joint version, not the delta-joint one used by the earlier non-relative
  Iteration1 pipeline - relative-action training needs absolute actions
  going in, since the relative transform happens in the processor pipeline
  at train time, not baked into the dataset.

```sh
# 1. merge (force everything under one task string):
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil
uv run python tools/merge_v21_datasets.py \
  --dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_old \
  --dataset /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/infer_dataset/PickAndInsertCube_TS1_Iter1_absJoint_merged_donefixed_filtered \
  --output /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/iterativeTraining/relative/PickAndInsertCube_TS1_ForIter1 \
  --force-task "PickAndInsertCube"
# -> 201 episodes, 78150 frames (v2.1). The 136 teleop episodes get
#    next.reward/observation.intervened/observation.policy_action padded
#    with 0.0/False (columns the teleop source never had) - verified real
#    values, not NaN.

# 2. convert to v3 (renames the v2.1 merge to _old, writes v3 at the same
#    name; auto-casts observation.intervened bool->float32 for ACT):
uv run python tools/convert_v21_to_v3.py \
  --repo-id PickAndInsertCube_TS1_ForIter1 \
  --root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/iterativeTraining/relative
# -> iterativeTraining/relative/PickAndInsertCube_TS1_ForIter1_old  (v2.1)
# -> iterativeTraining/relative/PickAndInsertCube_TS1_ForIter1      (v3)

# 3. compute relative-action stats (fresh dataset, no other training
#    depends on it, but the tool always writes to --new_repo_id anyway):
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
uv run lerobot-edit-dataset \
  --repo_id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/iterativeTraining/relative/PickAndInsertCube_TS1_ForIter1 \
  --new_repo_id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/iterativeTraining/relative/PickAndInsertCube_TS1_ForIter1_relativeDeltaTraining \
  --operation.type recompute_stats \
  --operation.relative_action true \
  --operation.chunk_size 20 \
  --operation.relative_exclude_joints "['gripper']" \
  --operation.num_workers 4
# -> relative_dims=6/7 (excluded=1), mean=0.0087, std=0.0766, q01=-0.2787, q99=0.2515
# -> iterativeTraining/relative/PickAndInsertCube_TS1_ForIter1_relativeDeltaTraining
```

Real training command (Track B - only run this after evaluating Track A
tomorrow, per plan):
```sh
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
uv run lerobot-train \
  --dataset.repo_id=/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/iterativeTraining/relative/PickAndInsertCube_TS1_ForIter1_relativeDeltaTraining \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints="['gripper']" \
  --output_dir=outputs/train/policy_bc_PickAndInsertCube_TS1_ForIter1_relativeDeltaTraining \
  --batch_size=64 \
  --num_workers=8 \
  --steps=80000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=act_bc_pickandinsertcube_real_iter1_relative \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="ACT BC IterTrain1 on real PickAndInsertCube (teleop + Iter1 infer data), relative (chunk-anchor delta) actions, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4
```

### Sanity checks run (2026-09-01)

Ran both tracks for 5 real training steps from
`GeneralistRewardModels/lerobot` (the correct training repo - see note
above; an earlier pass mistakenly used an unrelated clone at
`/home/qte9489/work/external_repos/lerobot_v3` and has been redone here)
(`--steps=5 --save_checkpoint=false --wandb.enable=false`, `--output_dir`
pointed at a scratch dir) to confirm the full pipeline (dataset load ->
relative-action processor -> ACT forward/backward) works end-to-end before
committing to the real 80k-step runs:
- Track A: loss 0.741 -> 0.870 over 5 steps (noisy at n=5, not a fit metric -
  the point is it ran with no errors).
- Track B: loss 0.724 -> 0.725 over 5 steps, no errors.
Scratch output dirs deleted afterward - nothing left in either repo's
`outputs/train/` from these sanity runs.

### Separate: reward-annotation port (NOT part of the relative-actions work above)

Also ported manually-annotated `next.reward`/`next.done` labels (annotated
on the TRIMMED delta-joint dataset
`PickAndInsertCube_Station1_merged_deltajoint_old`) back onto the untrimmed
absolute-joint `PickAndInsertCube_Station1_merged_old` dataset, since the
old (now-disabled) start/end trim heuristic meant frame indices didn't line
up 1:1 between the two. Matched by content instead of index/timestamp:
`observation.state` is a byte-exact passthrough in the delta conversion, so
each delta episode's full state sequence must appear as one exact
contiguous block inside its source abs episode (trimming only removes
frames from the edges, never the middle) - verified 136/136 episodes
matched uniquely, all to their own same-numbered episode. Also verified
before applying: all 136 episodes have the delta episode's first frame at
`next.reward==0` (front-trim never cut into a hold) and last frame at
`next.reward==1` (end-trim always cut INTO an active hold) - so front-trimmed
frames are left at the default (reward=0/done=False), and end-trimmed
frames get the hold carried forward (reward=1/done=True), matching this
project's established wide `next.done` invariant.
```sh
.venv/bin/python -m trossen_real.scripts.annotate_reward_from_delta \
  --abs-root trossen_real/datasets/PickAndInsertCube_Station1_merged_old \
  --delta-root trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint_old \
  --output-root trossen_real/datasets/PickAndInsertCube_Station1_merged_annotated
```
-> 136 episodes, 48929 frames, 12832 `next.reward==1`/`next.done==True`
frames (7224 from end-trim carry-forward). `next.reward` column added fresh
(didn't exist in `_old`'s schema at all). Not currently used by anything -
`_old` is untouched, this is a separate artifact for whenever reward-model /
reward-labeled training is needed on this data.

## PickCubeAndInsert_Trial1 (2026-09-03) - abs/delta/relative, ready to train

Real teleop collection using the pedal-driven `next.reward`/`next.done`
feature (see "PEDAL_BEHAVIOR.md") - first dataset collected this way.

- Source (v2.1, raw): `trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1`
  (87 episodes). `next.reward`/`next.done` verified in sync (0 real violations).
- Filtered (v2.1): 4 episodes discarded -> `..._filtered` (83 episodes, 31245 frames).
  - `filter_episodes.py`'s automatic classifier only caught 2 (69, 85 - reward
    pedal never held).
  - 2 more (75, 84) needed a **manual** override: reward pedal was genuinely
    held (passes the automatic check) but the actual insertion failed - not
    detectable from next.reward/next.done/joint-state alone, since those
    columns only reflect operator *intent* at press time, not physical
    task outcome. Added a new `--exclude EP1,EP2,...` flag to
    `filter_episodes.py` for exactly this (forces DISCARD regardless of the
    automatic classification, logged as "MANUAL OVERRIDE" in the report).
  ```sh
  .venv/bin/python -m trossen_real.scripts.filter_episodes \
    --dataset trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1 \
    --output trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1_filtered \
    --exclude 69,75,84,85
  ```
- Delta (v2.1): built from the filtered dataset, same 83 episodes/31245 frames, 0 trim.
  ```sh
  .venv/bin/python -m trossen_real.scripts.convert_to_delta_joint_dataset \
    --input-root trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1_filtered \
    --output-root trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1_filtered_deltajoint
  ```
- V2.1 -> V3 (both), from the DatasetUtil repo:
  ```sh
  cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil
  uv run python tools/convert_v21_to_v3.py --repo-id 20260903_095904_PickCubeAndInsert_Trial1_filtered \
    --root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets
  uv run python tools/convert_v21_to_v3.py --repo-id 20260903_095904_PickCubeAndInsert_Trial1_filtered_deltajoint \
    --root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets
  ```
  Renames each v2.1 input to `..._old` (backup, untouched) and writes the v3
  dataset at the original (no-suffix) name.
  - Found + fixed a NEW gap in `convert_v21_to_v3.py::standardize_v3_dataset()`:
    lerobot's own native v2.1->v3 `convert_dataset()` can silently drop
    video/image keys from `stats.json` entirely, even when the v2.1 input
    already had a placeholder for them (from `merge_v21_datasets.py`'s
    earlier fix). Not a crash risk in the real training repo (`factory.py`
    already guards a missing camera-stats key) but was leaving `stats.json`
    genuinely incomplete - `standardize_v3_dataset()` now re-adds the same
    placeholder for any video/image feature missing after the rename step.
- Relative-action stats (from the abs v3 dataset), from `GeneralistRewardModels/lerobot`:
  ```sh
  cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
  uv run lerobot-edit-dataset \
    --repo_id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1_filtered \
    --new_repo_id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1_filtered_relativeDeltaTraining \
    --operation.type recompute_stats --operation.relative_action true --operation.chunk_size 20 \
    --operation.relative_exclude_joints "['gripper']" --operation.num_workers 4
  ```
  -> relative_dims=6/7 (excluded=1), mean=0.0107, std=0.0734, q01=-0.2661, q99=0.2331.

**Final 3 datasets, all v3, all 83 episodes/31245 frames, verified via real
`LeRobotDataset` loads + a 5-step `lerobot-train` sanity run (loss moving,
no errors) - copy these to the server to train:**
- Absolute: `trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1_filtered`
- Delta joint: `trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1_filtered_deltajoint`
- Relative (train with `--policy.use_relative_actions=true --policy.relative_exclude_joints="['gripper']"`):
  `trossen_real/datasets/20260903_095904_PickCubeAndInsert_Trial1_filtered_relativeDeltaTraining`

**Still open:** episodes 75/84's failure mode (reward pedal held through a
failed insertion) has no reliable automatic detector - root cause is
operator timing (press reward only after visually confirming the cube is
actually seated, not during the release motion), not a data/schema issue.


copied to sever TVD29ServerRack and to these paths:
```plaintext
qte9489@clid2140319:~$ cd Research/DatasetUtils/CubePickAndInsert_TrossenReal/Trial1/
qte9489@clid2140319:~/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Trial1$ ll
total 20
drwxr-xr-x 5 qte9489 ldcint 4096 Sep  3 13:36 ./
drwxr-xr-x 6 qte9489 ldcint 4096 Sep  3 13:35 ../
drwxr-xr-x 5 qte9489 ldcint 4096 Sep  3 13:22 20260903_095904_PickCubeAndInsert_Trial1_filtered/
drwxr-xr-x 5 qte9489 ldcint 4096 Sep  3 13:22 20260903_095904_PickCubeAndInsert_Trial1_filtered_deltajoint/
drwxr-xr-x 5 qte9489 ldcint 4096 Sep  3 13:22 20260903_095904_PickCubeAndInsert_Trial1_filtered_relativeDeltaTraining/
qte9489@clid2140319:~/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Trial1$ 

```

ACT Training sccripts ran in server:
- Absolute Joint Action
```sh
cd /home/qte9489/Research/LEROBOT/lerobot/
uv run lerobot-train \
  --config_path act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Trial1/20260903_095904_PickCubeAndInsert_Trial1_filtered/ \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/ACT_BC_20260903_095904_PickCubeAndInsert_Trial1_filtered \
  --batch_size=64 \
  --num_workers=8 \
  --steps=75000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=act_bc_pickandinsertcube_real \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="ACT BC Trial1 dataset on real PickAndInsertCube Station 1 Abs Joint Action, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4
```

- Delta Joint Action
```sh
cd /home/qte9489/Research/LEROBOT/lerobot/
uv run lerobot-train \
  --config_path act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Trial1/20260903_095904_PickCubeAndInsert_Trial1_filtered_deltajoint/ \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/ACT_BC_20260903_095904_PickCubeAndInsert_Trial1_filtered_deltajoint \
  --batch_size=64 \
  --num_workers=8 \
  --steps=75000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=act_bc_pickandinsertcube_real \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="ACT BC Trial1 dataset on real PickAndInsertCube Station 1 delta Joint Action, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4
```

- Relative Delta Joint Action
```sh
cd /home/qte9489/Research/LEROBOT/lerobot
uv run lerobot-train \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Trial1/20260903_095904_PickCubeAndInsert_Trial1_filtered_relativeDeltaTraining/ \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints="['gripper']" \
  --output_dir=outputs/train/ACT_BC_20260903_095904_PickCubeAndInsert_Trial1_filtered_relativeDeltaTraining \
  --batch_size=64 \
  --num_workers=8 \
  --steps=75000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=act_bc_pickandinsertcube_real_relative \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="ACT BC Trial1 dataset on real PickAndInsertCube Station 1, relative (chunk-anchor delta) actions, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4
```

## Diffusion Policy (DP) support (2026-09-03)

`policy_server.py` used to hardcode `ACTPolicy` - now loads whatever policy
type the checkpoint's own `config.json` says (`get_policy_class()`), and
`_action_queue_len()` handles both ACT's `_action_queue` and Diffusion's
`_queues["action"]` for the chunk-anchor-relative bookkeeping. Verified with
real 100-step training runs + real `policy_server.py` loads on the Trial1
datasets above:

- **Absolute: working.** Trains and serves correctly, dynamic policy
  loading confirmed (`policy class: DiffusionPolicy`).
- **Delta joint: working.** Same.
- **Relative: NOT supported yet - real bug, not just a missing wire-up.**
  - `DiffusionConfig` now has `use_relative_actions`/`relative_exclude_joints`/
    `action_feature_names` (mirrors ACT), but that alone did nothing -
    `processor_diffusion.py` never constructed a `RelativeActionsProcessorStep`
    at all (unlike `processor_act.py`, which does) - fixed, mirrored ACT's
    wiring exactly.
  - That fix then exposed a REAL crash, not a subtle one:
    `RuntimeError: The size of tensor a (8) must match the size of tensor b (2)
    at non-singleton dimension 1`.
  - Root cause, from the actual code: `to_relative_actions()`
    (`src/lerobot/processor/relative_action_processor.py`) hardcodes
    `state: (B, state_dim)` in its own docstring - true for ACT (one
    observation frame per example) but NOT for Diffusion, which stacks
    `n_obs_steps` (default 2 - prev + current) observation frames per
    training example. Confirmed directly in `modeling_diffusion.py::reset()`:
    `_queues[OBS_STATE] = deque(maxlen=config.n_obs_steps)` AND
    `_queues[OBS_IMAGES] = deque(maxlen=config.n_obs_steps)` - the 2-frame
    history applies to BOTH state and images, not state alone.
  - Needs a real design decision (which of the `n_obs_steps` frames is the
    relative anchor - almost certainly the latest one, but unverified for
    both the training-time batched shape AND inference-time `select_action()`
    calls) before touching `to_relative_actions()`'s math - not attempted yet.

Server training commands (absolute + delta only - relative not supported for
DP, see above). DP-specific structural hyperparams (`n_obs_steps`, `horizon`,
`n_action_steps`) use DP's own sensible defaults rather than reusing ACT's
`chunk_size`/`n_action_steps` (different underlying meaning); everything else
(`batch_size`, `steps`, `optimizer_lr`, wandb settings) mirrors the ACT
commands above directly. No `--config_path` (that yaml is ACT-specific).

- Absolute Joint Action
```sh
cd /home/qte9489/Research/LEROBOT/lerobot/
uv run lerobot-train \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Trial1/20260903_095904_PickCubeAndInsert_Trial1_filtered/ \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/DP_BC_20260903_095904_PickCubeAndInsert_Trial1_filtered \
  --batch_size=64 \
  --num_workers=8 \
  --steps=75000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=dp_bc_pickandinsertcube_real \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="Diffusion Policy BC Trial1 dataset on real PickAndInsertCube Station 1 Abs Joint Action, 256x256" \
  --wandb.disable_artifact=true \
  --policy.optimizer_lr=1e-4
```

- Delta Joint Action
```sh
cd /home/qte9489/Research/LEROBOT/lerobot/
uv run lerobot-train \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Trial1/20260903_095904_PickCubeAndInsert_Trial1_filtered_deltajoint/ \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/DP_BC_20260903_095904_PickCubeAndInsert_Trial1_filtered_deltajoint \
  --batch_size=64 \
  --num_workers=8 \
  --steps=75000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=dp_bc_pickandinsertcube_real \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="Diffusion Policy BC Trial1 dataset on real PickAndInsertCube Station 1 delta Joint Action, 256x256" \
  --wandb.disable_artifact=true \
  --policy.optimizer_lr=1e-4
```

## PickCubeAndInsert **Station 3** Trial1 (2026-09-28) - abs/delta/relative, ready to train

5 teleop sessions recorded 2026-09-24/25 on Station 3, merged into one training
set. Scripts + a fuller write-up: `scripts/TrossenStation3Real/`
(`prepare_dataset.sh` reproduces everything below, `README.md` explains it).

**Every command below was actually executed, in this order, on 2026-09-28.**

### Which interpreter (this matters - three different venvs)

Never `uv` on this machine (it breaks the repo venv) - interpreters are called
directly. `uv run` on the SERVER for `lerobot-train` is unchanged from Trial1.

| Shorthand | Interpreter | lerobot | Used for |
|---|---|---|---|
| `REPO_PY` | `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/.venv/bin/python` | 0.1.0 | `trossen_real.scripts.*`, and it also runs `merge_v21_datasets.py` |
| `DU_PY` | `…/RLRewardResearchWS/DatasetUtil/.venv/bin/python` | 0.4.4 | **v2.1 -> v3** (repo venv's 0.1.0 CANNOT do this) |
| `LR_EDIT` | `…/GeneralistRewardModels/lerobot/.venv/bin/lerobot-edit-dataset` | - | relative-action stats |

### Sources (v2.1, 180 episodes / 56186 frames)

| Session | Eps | Verdict |
|---|---|---|
| `20260924_132455_PickCubeAndInsertStation3` | 3 | GOOD |
| `20260924_155210_PickCubeAndInsertStation3` | 60 | drop ep 21 |
| `20260924_164756_PickCubeAndInsertStation3` | 60 | GOOD |
| `20260925_100304_PickCubeAndInsertStation3` | 6 | drop ep 5 |
| `20260925_100751_PickCubeAndInsertStation3` | 51 | GOOD |

Dry-run classification first (writes nothing), on all five:
```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
for d in 20260924_132455 20260924_155210 20260924_164756 20260925_100304 20260925_100751; do
  .venv/bin/python -m trossen_real.scripts.filter_episodes \
    --dataset trossen_real/datasets/${d}_PickCubeAndInsertStation3 \
    --output /tmp/unused --dry-run
done
```
Result: the classifier flags **only ep 21** (of `155210`) on its own. Every
other episode in all five sessions classifies KEEP.

**ep 5 of `20260925_100304` is a manual override.** The classifier KEEPS it
(reward pedal genuinely held -> looks like a success), but it is 563 frames /
248 done-frames vs ~300/45 for its siblings: the operator marked success on a
failed attempt. Not detectable from `next.reward`/`next.done`, which only
record operator intent at press time - identical to Trial1's eps 75/84.

### 1. Filter (writes NEW datasets; raw sessions untouched)
```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
.venv/bin/python -m trossen_real.scripts.filter_episodes \
  --dataset trossen_real/datasets/20260924_155210_PickCubeAndInsertStation3 \
  --output  trossen_real/datasets/20260924_155210_PickCubeAndInsertStation3_filtered \
  --exclude 21
# -> 59 kept, 1 discarded, 19754 frames

.venv/bin/python -m trossen_real.scripts.filter_episodes \
  --dataset trossen_real/datasets/20260925_100304_PickCubeAndInsertStation3 \
  --output  trossen_real/datasets/20260925_100304_PickCubeAndInsertStation3_filtered \
  --exclude 5
# -> 5 kept, 1 discarded, 1611 frames, logged "MANUAL OVERRIDE (--exclude)"
```
(`filter_episodes.py` delegates the reindexing to `merge_v21_datasets.py` via
`--dataset PATH::EP,EP,...`, run with `sys.executable` = the repo venv. So
meta/episode counts/stats are regenerated by the merge tool, not hand-edited.)

### 2. Merge all 5 -> one v2.1 dataset
No `--force-task`: all five already share task `PickCubeAndInsertStation3` and
an identical 16-key feature schema (verified before merging).
```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
D=trossen_real/datasets
.venv/bin/python /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil/tools/merge_v21_datasets.py \
  --dataset $D/20260924_132455_PickCubeAndInsertStation3 \
  --dataset $D/20260924_155210_PickCubeAndInsertStation3_filtered \
  --dataset $D/20260924_164756_PickCubeAndInsertStation3 \
  --dataset $D/20260925_100304_PickCubeAndInsertStation3_filtered \
  --dataset $D/20260925_100751_PickCubeAndInsertStation3 \
  --output  $D/PickCubeAndInsert_Station3_Trial1
# -> 178 episodes, 55177 frames, 356 videos, 1 task (v2.1)
```

### 3. Delta-joint conversion (v2.1 -> v2.1)
```sh
.venv/bin/python -m trossen_real.scripts.convert_to_delta_joint_dataset \
  --input-root  trossen_real/datasets/PickCubeAndInsert_Station3_Trial1 \
  --output-root trossen_real/datasets/PickCubeAndInsert_Station3_Trial1_deltajoint \
  --dry-run
.venv/bin/python -m trossen_real.scripts.convert_to_delta_joint_dataset \
  --input-root  trossen_real/datasets/PickCubeAndInsert_Station3_Trial1 \
  --output-root trossen_real/datasets/PickCubeAndInsert_Station3_Trial1_deltajoint
# -> 178 episodes, 55177 frames, "trimmed 0 start + 0 end" on every episode
```

**On trimming the dead time at the start:** it is NOT trimmed, and that is
deliberate - both trims have been disabled in this script since 2026-09-01
(see the long in-file comments). Start-trim assumed every episode opens with a
still period to calibrate a resting floor against, and on policy-rollout data
it threw away the entire real approach; end-trim was speed-only, unaware of
`next.done`/`next.reward`, and cut the success hold including the true terminal
frame (26/65 episodes in one batch). Trial1 also shipped with 0 trim, so this
dataset is consistent with what has already been trained on. Re-enabling needs
a new heuristic, not a flag flip.

### 4. v2.1 -> v3 (both datasets)
MUST use the DatasetUtil venv - the repo venv's lerobot 0.1.0 cannot write v3.
Each call renames the v2.1 input to `<name>_old` and writes v3 at the same name.
```sh
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil
./.venv/bin/python tools/convert_v21_to_v3.py \
  --repo-id PickCubeAndInsert_Station3_Trial1 \
  --root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets
./.venv/bin/python tools/convert_v21_to_v3.py \
  --repo-id PickCubeAndInsert_Station3_Trial1_deltajoint \
  --root /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets
```
(The abs run re-added the camera-key stats placeholder that lerobot's native
`convert_dataset()` drops; both v3 datasets end up with both camera keys in
`stats.json`.)

### 5. Relative-action stats (built from the ABS v3 dataset, not the delta one)
Relative training needs absolute actions in the dataset - the chunk-anchor
delta transform happens in the processor at train time.
```sh
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
./.venv/bin/lerobot-edit-dataset \
  --repo_id     /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickCubeAndInsert_Station3_Trial1 \
  --new_repo_id /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickCubeAndInsert_Station3_Trial1_relativeDeltaTraining \
  --operation.type recompute_stats \
  --operation.relative_action true \
  --operation.chunk_size 20 \
  --operation.relative_exclude_joints "['gripper']" \
  --operation.num_workers 4
# -> relative_dims=6/7 (excluded=1), mean=0.0098, std=0.0834, q01=-0.2908, q99=0.2470
#    (Trial1 Station1 was mean=0.0107, std=0.0734 - comparable)
```

### 6. Verification (run after the pipeline)
```sh
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot
./.venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
base='/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/'
for n in ['PickCubeAndInsert_Station3_Trial1','PickCubeAndInsert_Station3_Trial1_deltajoint','PickCubeAndInsert_Station3_Trial1_relativeDeltaTraining']:
    d=LeRobotDataset(repo_id=base+n, root=base+n); s=d[0]
    print(n, d.num_episodes, d.num_frames, tuple(s['action'].shape))
"
```
-> all three: v3.0, 178 episodes, 55177 frames, `action`/`observation.state` `(7,)`.

**Final 3 datasets (copy these to the server):**
- Absolute: `trossen_real/datasets/PickCubeAndInsert_Station3_Trial1`
- Delta joint: `trossen_real/datasets/PickCubeAndInsert_Station3_Trial1_deltajoint`
- Relative: `trossen_real/datasets/PickCubeAndInsert_Station3_Trial1_relativeDeltaTraining`

### ACT training (server) - same params as Trial1

Scripts: `scripts/TrossenStation3Real/train_act_{abs,deltajoint,relative}.sh`
(shared settings in `_common_train.sh`; override with env vars, e.g.
`STEPS=50000 DATA_DIR=... ./train_act_abs.sh`). They expand to the commands
below - 75k steps, batch 64, chunk 20 / n_action_steps 20, `use_vae=false`,
lr 1e-4, identical to the Trial1 runs. Relative deliberately has NO
`--config_path`, matching Trial1.

- Absolute Joint Action
```sh
cd /home/qte9489/Research/LEROBOT/lerobot/
uv run lerobot-train \
  --config_path act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Station3_Trial1/PickCubeAndInsert_Station3_Trial1/ \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/ACT_BC_PickCubeAndInsert_Station3_Trial1 \
  --batch_size=64 \
  --num_workers=8 \
  --steps=75000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=act_bc_pickcubeandinsert_station3_abs \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="ACT BC Station3 Trial1 (5 sessions merged, 178 eps) Abs Joint Action, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4
```

- Delta Joint Action
```sh
cd /home/qte9489/Research/LEROBOT/lerobot/
uv run lerobot-train \
  --config_path act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Station3_Trial1/PickCubeAndInsert_Station3_Trial1_deltajoint/ \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/ACT_BC_PickCubeAndInsert_Station3_Trial1_deltajoint \
  --batch_size=64 \
  --num_workers=8 \
  --steps=75000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=act_bc_pickcubeandinsert_station3_delta \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="ACT BC Station3 Trial1 (5 sessions merged, 178 eps) Delta Joint Action, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4
```

- Relative Delta Joint Action
```sh
cd /home/qte9489/Research/LEROBOT/lerobot
uv run lerobot-train \
  --dataset.repo_id=/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Station3_Trial1/PickCubeAndInsert_Station3_Trial1_relativeDeltaTraining/ \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints="['gripper']" \
  --output_dir=outputs/train/ACT_BC_PickCubeAndInsert_Station3_Trial1_relativeDeltaTraining \
  --batch_size=64 \
  --num_workers=8 \
  --steps=75000 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=act_bc_pickcubeandinsert_station3_relative \
  --wandb.project=Trossen_PickAndInsertCube_Real \
  --wandb.notes="ACT BC Station3 Trial1 (5 sessions merged, 178 eps) relative (chunk-anchor delta) actions, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.use_vae=false \
  --policy.optimizer_lr=1e-4
```

### Notes / still open
- No local `lerobot-train` sanity run was done for these three (unlike Trial1) -
  only real `LeRobotDataset` loads. Run a 5-step `--steps=5 --save_checkpoint=false
  --wandb.enable=false` job before committing to the 75k runs if you want the
  same level of assurance.
- `20260925_100751` has a root-owned `meta/lerobot_rl_labels.json`
  (`{"episodes": {}}` - empty) and `episodes_stats.jsonl.rl_bak` from an RL
  labelling tool run on 2026-09-25. Empty, and the merge reads
  `episodes.jsonl`/parquet, so the merged output is unaffected.
- Disk was at 99% (13 GB free) after this; the new artifacts total ~1.1 GB.
  `PickCubeAndInsert_Station3_Trial1_old` (213 MB) and `..._deltajoint_old`
  (191 MB) are the v2.1 backups - deletable once the v3 outputs are trusted.
- ep 5 / `20260925_100304`'s failure mode (reward pedal held through a failed
  insertion) still has no automatic detector - same open item as Trial1's
  eps 75/84.
