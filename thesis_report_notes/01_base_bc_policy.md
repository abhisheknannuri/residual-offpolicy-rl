# Base BC Policy — Sim/CAN Residual RL: Architecture Spec

Scope: which module is loaded as the frozen base policy for SIM residual RL on the
robosuite `Can` (PickPlaceCan) task, its full forward pass, ground-truth hyperparameters
from the actual local checkpoint the user trained in the external LeRobot repo, and
proof that it is frozen during RL fine-tuning.

All line numbers are current as of the working tree at
`/home/qte9489/personal_abhi/temp/residual-offpolicy-rl` (git branch
`dev/abhi-resfit-setup`, several files with **uncommitted changes** per `git status`:
`resfit/dexmg/environments/dexmg.py`, `resfit/lerobot/policies/diffusion/modeling_diffusion.py`,
`resfit/lerobot/utils/load_policy.py`, `scripts/train_bc_can.sh`). Everything below was read
from the on-disk (uncommitted) state, which is what actually executes.

---

## A) Which base-policy module class is actually loaded and run

**Answer: `ACTPolicy`** from `resfit/lerobot/policies/act/modeling_act.py` (ResFiT's
vendored/forked copy of LeRobot's ACT implementation) — **not** `DiffusionPolicy`. This
holds both for the W&B-hosted path and for the local-checkpoint path the user asked about.

### Code path, traced end to end

1. **Shell script → Hydra CLI arg.** `scripts/train_residual_rl_can.sh:814-815` and
   `scripts/train_residual_rl_can_sparse.sh:830-831`:
   ```bash
   # Local base-policy checkpoint (takes priority over W&B)
   [[ -n "${BASE_LOCAL_PATH}" ]] && CMD+=(base_policy.local_path="${BASE_LOCAL_PATH}")
   ```
   `BASE_LOCAL_PATH` is documented in both scripts (`train_residual_rl_can.sh:85-91`,
   `train_residual_rl_can_sparse.sh:85-91`) as pointing at a LeRobot `pretrained_model`
   dir, with the exact example path
   `.../GeneralistRewardModels/lerobot/outputs/train/policy_rabc/checkpoints/050000/pretrained_model`.
   In both checked-in scripts `BASE_LOCAL_PATH=""` by default (must be set by the user to activate
   the local path).

2. **Hydra config dataclass.** `resfit/rl_finetuning/config/residual_td3.py:65-75`:
   ```python
   @dataclass
   class BasePolicyConfig:
       wandb_id: str = "TODO"
       wt_type: str = "best"
       wt_version: str = "latest"
       # ... Takes priority over wandb_id.
       local_path: str | None = None
   ```

3. **Training script resolves local vs. W&B.** `resfit/rl_finetuning/scripts/train_residual_td3.py:312-332`:
   ```python
   else:
       assert "base_policy" in cfg, "Base policy configuration is required"
       base_local_path = cfg.base_policy.get("local_path", None)
       if base_local_path:
           # Local checkpoint takes priority over W&B download.
           policy_dir = resolve_local_policy_dir(base_local_path)
           logger.info(f"Loading base policy from LOCAL checkpoint: {policy_dir} (W&B download skipped)")
       else:
           policy_dir, _ = download_policy_from_wandb(
               cfg.base_policy.wandb_id, step=cfg.base_policy.wt_type,
               artifact_version=cfg.base_policy.wt_version,
           )
           logger.info(f"Loading base policy from W&B artifact: {cfg.base_policy.wandb_id} -> {policy_dir}")

       base_policy = load_policy(policy_dir)
       base_policy.to(device)
       base_policy.eval()
       eval_base_policy = load_policy(policy_dir)
       eval_base_policy.to(device)
       eval_base_policy.eval()

       base_cfg = base_policy.config
       if isinstance(base_cfg, ACTConfig):
           cfg.actor_name = "residual_act"
       else:
           ...  # (diffusion branch, not exercised for the Can/ACT checkpoint)
   ```

4. **`resolve_local_policy_dir`** — `resfit/lerobot/utils/load_policy.py:65-90`. Resolves a
   local dir to wherever `config.json` lives (accepts the dir itself, or a
   `policy/`/`pretrained_model/` subfolder, or a recursive glob fallback). This directly
   supports the exact layout the user described:
   `.../checkpoints/025000/pretrained_model/config.json`.

5. **`load_policy` picks the class from `config.json["type"]`** —
   `resfit/lerobot/utils/load_policy.py:93-260`:
   ```python
   policy_name_field = str(cfg_dict.get("type", "")).lower()
   if "diffusion" in policy_name_field:
       ...
       policy = DiffusionPolicy.from_pretrained(policy_dir)
   elif "use_vae" in cfg_dict or policy_name_field == "act":
       policy = ACTPolicy.from_pretrained(policy_dir)
   else:
       raise ValueError(f"Unknown policy type: {policy_name_field}")
   ```
   For the checkpoint at
   `.../lerobot/outputs/train/policy_rabc_100/checkpoints/025000/pretrained_model/config.json`
   (read directly, see §C), `"type": "act"` → **`ACTPolicy.from_pretrained(...)` is called.**

6. `load_policy` then filters `config.json` down to an allow-listed key set matching
   ResFiT's `ACTConfig` (`load_policy.py:143-193`) and — because the external repo's newer
   LeRobot checkpoint format stores normalization stats in a separate
   pre/post-processor pipeline (`policy_preprocessor.json` /
   `policy_postprocessor.json` + `*_normalizer_processor.safetensors` /
   `*_unnormalizer_processor.safetensors`, not baked into `ACTPolicy` itself) — manually
   copies those stats into `policy.normalize_inputs` / `policy.normalize_targets` /
   `policy.unnormalize_outputs` buffers (`load_policy.py:202-258`). This is a bridge: it
   lets a checkpoint produced by the *external* repo's newer processor-based LeRobot be
   loaded into ResFiT's older, self-contained `ACTPolicy` (see §C for how the two ACT
   implementations otherwise differ).

7. **`eval_policy_can.py`** (standalone eval script, not part of the RL training loop but
   asked about explicitly) uses the identical resolution + load logic inline at
   `scripts/eval_policy_can.py:270-283` (`config.json` / `policy/` / `pretrained_model/`
   subdir resolution, then `load_policy(policy_dir)`), then
   `resfit.dexmg.environments.dexmg.create_vectorized_env` / `VectorizedEnvWrapper` for
   rollouts. Same `ACTPolicy` class.

### Confirms / corrects `../docs/rewards/REWARD_MODEL_INTEGRATION.md`

`REWARD_MODEL_INTEGRATION.md:211` (repo root, an existing untracked doc from a prior
session) claims local ACT-checkpoint loading was implemented via
`resolve_local_policy_dir` in `load_policy.py` and `BasePolicyConfig.local_path` in
`residual_td3.py`. **This is confirmed accurate against the current on-disk code** — the
"local takes priority over W&B" feature is fully implemented, not just planned, exactly
as the doc describes.

---

## B) Full forward pass of `ACTPolicy` / `ACT`

Ground-truth numbers below use the **actual checkpoint** the user has locally
(`.../policy_rabc_100/checkpoints/025000/pretrained_model/config.json`, read in full in
§C) mapped onto the sim `Can` task in `resfit/dexmg/environments/dexmg.py`. This
supersedes the TwoArmCoffee/GR1 example numbers in the existing `../docs/policies/ACT_ARCHITECTURE.md`
(see "Discrepancies" section — that doc's *numeric* walkthrough is for a different task,
though its architectural description of the ACT module graph is otherwise accurate).

### Inputs (Can / PickPlaceCan, single Panda arm)

- `observation.state`: **9D** = `robot0_eef_pos`(3) + `robot0_eef_quat`(4) +
  `robot0_gripper_qpos`(2), assembled in
  `resfit/dexmg/environments/dexmg.py:515-524` (`_get_expected_low_dim_keys`,
  `panda_low_dim_keys_single`) and confirmed by the checkpoint's
  `input_features.observation.state.shape == [9]` (config.json).
- Cameras: **2** — `observation.images.agentview`, `observation.images.robot0_eye_in_hand`,
  from `dexmg.py:463-473` (`_get_expected_image_keys`, `panda_image_keys_single`), matching
  the checkpoint's two VISUAL input features.
- Action: **7D** = Δpos(3) + Δaxis-angle rot(3) + gripper(1) — `output_features.action.shape
  == [7]` in config.json; robosuite `OSC_POSE` delta controller for Panda.
- Env image resolution actually fed to the policy during RL training: **84×84** —
  `create_vectorized_env` is called at `train_residual_td3.py:526-540` **without** passing
  `camera_size`, so it uses the `camera_size: int = 84` default
  (`resfit/dexmg/environments/dexmg.py:90`, propagated to `robosuite.make(camera_heights=84,
  camera_widths=84, ...)` at `dexmg.py:196-197`).

### Module list, forward order (from `resfit/lerobot/policies/act/modeling_act.py`, class `ACT`)

Using this checkpoint's hyperparameters: `n_obs_steps=1`, `chunk_size=20`,
`n_action_steps=20`, `dim_model=512`, `n_heads=8`, `dim_feedforward=3200`,
`n_encoder_layers=4`, `n_decoder_layers=1`, `use_vae=False`, `latent_dim=32`,
`vision_backbone="resnet18"`, `pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"`.

1. **Normalization** (outside `ACT`, in `ACTPolicy.select_action` /
   `ACTPolicy.forward`, `modeling_act.py:154-259` in ResFiT's copy): `self.normalize_inputs`
   (MEAN_STD, per `normalization_mapping`) applied to `observation.state` and both image
   keys. Buffer values for this checkpoint are loaded post-hoc from
   `policy_preprocessor_step_3_normalizer_processor.safetensors` by `load_policy.py:203-224`.
2. **Vision backbone** (`ACT.__init__`, `modeling_act.py:419-429`): one **shared**
   `torchvision.models.resnet18` per config (`getattr(torchvision.models,
   config.vision_backbone)(...)`), `weights=ResNet18_Weights.IMAGENET1K_V1`
   (**ImageNet-pretrained**), `norm_layer=FrozenBatchNorm2d` (BN stats frozen from
   ImageNet, never updated by SGD even in `.train()` mode). Only `layer4` is kept via
   `IntermediateLayerGetter(..., return_layers={"layer4": "feature_map"})`. Applied
   independently to each of the 2 camera images in a Python loop
   (`modeling_act.py:557-576`), i.e. the **same weights** process both cameras (shared
   backbone, not per-camera).
   - Input per camera: `(B, 3, 84, 84)`.
   - Output: `(B, 512, 3, 3)` feature map (per the ResNet18 stride math for 84×84 input;
     same math as `../docs/policies/ACT_ARCHITECTURE.md` §3, which is architecture-correct even though its
     numeric example task differs).
3. **Image token projection**: `Conv2d(512, 512, kernel_size=1)`
   (`encoder_img_feat_input_proj`, `modeling_act.py:442-443`) + 2D sinusoidal positional
   embedding (`ACTSinusoidalPositionEmbedding2d(dim_model // 2 = 256)`,
   `modeling_act.py:452`, `770-819`) → reshaped `(B,512,3,3) → (9, B, 512)` per camera
   (`modeling_act.py:568-569`). 2 cameras × 9 = **18 image tokens** for this task (not 27 —
   that was a 3-camera TwoArmCoffee example in the existing doc).
4. **State token**: `Linear(9, 512)` (`encoder_robot_state_input_proj`,
   `modeling_act.py:438`) → 1 token.
5. **Latent token** (no VAE, `use_vae=False`): zeros `(B, 32)` →
   `Linear(32, 512)` (`encoder_latent_input_proj`, `modeling_act.py:441`) → 1 token. (VAE
   encoder branch, `modeling_act.py:394-417, 496-537`, is entirely skipped at inference and
   at training since `use_vae=False` for this checkpoint.)
6. **Transformer encoder input**: concatenated `[latent(1), state(1), img_tokens(18)]` =
   **20 tokens × 512-dim**, `torch.stack` at `modeling_act.py:578-579`.
7. **Transformer encoder** (`ACTEncoder`, `n_encoder_layers=4`,
   `modeling_act.py:604-617, 620-656`): 4× post-norm self-attention layers, 8 heads,
   FFN `512→3200→512` (ReLU, `feedforward_activation="relu"`), dropout 0.1. Output:
   `(20, B, 512)`.
8. **Transformer decoder** (`ACTDecoder`, `n_decoder_layers=1`,
   `modeling_act.py:659-677, 680-749`): decoder input is a **zero tensor**
   `(chunk_size=20, B, 512)` with learned positional embeddings
   `decoder_pos_embed = nn.Embedding(20, 512)` (`modeling_act.py:456`, one embedding per
   future action step). Self-attn among the 20 queries, then cross-attn to the 20-token
   encoder output, then FFN (`512→3200→512`). `n_decoder_layers=1` for this checkpoint
   (matches the well-known original-ACT off-by-one bug that this LeRobot port
   intentionally reproduces). Output: `(20, B, 512)`.
9. **Action head**: `Linear(512, 7)` (`action_head`, `modeling_act.py:459`). Output:
   `(B, 20, 7)` — 20-step action chunk, 7D per step.
10. **Un-normalization** (`ACTPolicy.select_action`,
    `modeling_act.py:189/210` in ResFiT's copy): `self.unnormalize_outputs` (MEAN_STD)
    maps back to raw action units. Stats loaded from
    `policy_postprocessor_step_0_unnormalizer_processor.safetensors` by
    `load_policy.py:225-258`.

### Text/task conditioning

**NOT FOUND.** Searched `modeling_act.py` (both repos) and `configuration_act.py`: no text
encoder, no CLIP, no language conditioning anywhere in the ACT forward pass. The only
place a task string appears in the whole pipeline is the *external stage-aware reward
model* client (`REWARD_TASK_PROMPT="pick up the can and place it in the bin"` in
`scripts/train_residual_rl_can.sh:685`), which is unrelated to the base BC policy — it
conditions SARM/TCC reward-model inference, not the ACT policy.

### Vision encoder — frozen/trained status

- **At BC training time** (external lerobot repo): `pretrained_backbone_weights =
  "ResNet18_Weights.IMAGENET1K_V1"` — ImageNet-pretrained init, then presumably
  fine-tuned jointly with the rest of ACT via the standard `optimizer_lr_backbone=1e-05`
  (`train_config.json`, lower LR than the rest of the network at
  `optimizer.lr=0.0001`) — i.e. the ResNet18 backbone parameters are **not frozen** during
  BC training, just trained at a smaller LR, per `ACTPolicy.get_optim_params()`
  (`resfit/lerobot/policies/act/modeling_act.py:79-92`, same logic in both repos).
  `FrozenBatchNorm2d` (`modeling_act.py:424`, `torchvision.ops.misc.FrozenBatchNorm2d`)
  keeps only the BatchNorm running statistics fixed at their ImageNet values; it does not
  freeze conv weights.
- **During residual RL** (this repo): the entire `ACTPolicy` (backbone included) is
  frozen — see §D for the exact proof.

### Proprioceptive state path

Raw concat of 3 robosuite obs keys into a single 9D vector (`dexmg.py:515-524`, listed
above), fed through **one** `Linear(9, 512)` projection (`modeling_act.py:438`) — no
separate per-component encoder.

### Action chunking / consumption at rollout

- `chunk_size = n_action_steps = 20` for this checkpoint (config.json). Per
  `configuration_act.py` validation (`resfit/lerobot/policies/act/configuration_act.py:154-157`),
  `n_action_steps` must be `<= chunk_size`; here they're equal, so the **entire** predicted
  chunk is consumed — no temporal ensembling (`temporal_ensemble_coeff: null` in
  config.json) and **no receding-horizon re-query within a chunk**.
- Consumption mechanism: **per-environment `deque`** action queue,
  `resfit/lerobot/policies/act/modeling_act.py:130-219` (`_ensure_action_queues`,
  `select_action`). `select_action(batch)` is called **every env step**
  (`resfit/rl_finetuning/wrappers/residual_env_wrapper.py:102, 149`), but the queue check
  `envs_needing_chunk = [idx for idx, q in enumerate(self._action_queues) if len(q) ==
  0]` (`modeling_act.py:205`) means the actual 512-dim transformer forward pass —
  ResNet18 + encoder + decoder — runs **only once every 20 env steps** per env; the other
  19 calls just pop the next queued action, no compute.
- **Query cadence relative to env steps: 1 forward pass every `n_action_steps = 20` env
  steps** (open-loop chunk execution, not receding-horizon re-planning every step). At
  `control_freq=20` Hz (`dexmg.py:194`) this is 1 Hz effective re-planning rate.

### Where the base action is consumed in the residual-RL loop

`resfit/rl_finetuning/wrappers/residual_env_wrapper.py:100-168`
(`BasePolicyVecEnvWrapper.reset`/`.step`):
```python
with torch.no_grad():
    base_action = self.base_policy.select_action(raw_obs)
base_naction = self.action_scaler.scale(base_action)
...
combined_naction = self._last_base_naction + residual_naction
env_action = self.action_scaler.unscale(combined_naction)
raw_obs, reward, terminated, truncated, info = self.vec_env.step(env_action)
```
This matches the pipeline diagram already documented in the shell-script headers
(`scripts/train_residual_rl_can.sh:34-41`).

---

## C) Checkpoint config.json ground truth (external LeRobot repo checkpoint)

Read directly from
`/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_rabc_100/checkpoints/025000/pretrained_model/config.json`
and the sibling `train_config.json` in the same directory.

| Field | Value | Source |
|---|---|---|
| `type` | `"act"` | `config.json:2` |
| `n_obs_steps` | `1` | `config.json:3` |
| `input_features` | `observation.state` (9D), `observation.images.agentview` (3×256×256), `observation.images.robot0_eye_in_hand` (3×256×256) | `config.json:4-27` |
| `output_features` | `action` (7D) | `config.json:28-35` |
| `chunk_size` | `20` | `config.json:38` |
| `n_action_steps` | `20` | `config.json:39` |
| `normalization_mapping` | VISUAL/STATE/ACTION all `MEAN_STD` | `config.json:40-44` |
| `vision_backbone` | `"resnet18"` | `config.json:45` |
| `pretrained_backbone_weights` | `"ResNet18_Weights.IMAGENET1K_V1"` | `config.json:46` |
| `replace_final_stride_with_dilation` | `false` | `config.json:47` |
| `pre_norm` | `false` (post-norm residual blocks) | `config.json:48` |
| `dim_model` | `512` | `config.json:49` |
| `n_heads` | `8` | `config.json:50` |
| `dim_feedforward` | `3200` | `config.json:51` |
| `feedforward_activation` | `"relu"` | `config.json:52` |
| `n_encoder_layers` | `4` | `config.json:53` |
| `n_decoder_layers` | `1` | `config.json:54` |
| `use_vae` | `false` | `config.json:55` |
| `latent_dim` | `32` | `config.json:56` |
| `n_vae_encoder_layers` | `4` (unused, `use_vae=false`) | `config.json:57` |
| `temporal_ensemble_coeff` | `null` | `config.json:58` |
| `dropout` | `0.1` | `config.json:59` |
| `kl_weight` | `10.0` (unused, `use_vae=false`) | `config.json:60` |
| `optimizer_lr` / `optimizer_lr_backbone` | `1e-4` / `1e-5` | `config.json:61-63` |

**Training-time data note (important discrepancy inside the checkpoint metadata
itself):** `train_config.json:107-123` declares
`dataset.image_transforms.enable=true` with a single `resize` transform to
`[84, 84]` applied with `weight: 1.0` (i.e. always applied, since it's the only
transform and `max_num_transforms=1`). So although `config.json`'s
`input_features.*.shape` says `[3, 256, 256]` (the *raw* dataset resolution before
the transform), the network was **actually trained on 84×84 images** — matching the
sim's 84×84 render size used at RL time (`dexmg.py:90`). Both `config.json` (declared
256×256) and the true effective training resolution (84×84, per `train_config.json`'s
transform pipeline) are reported here since they disagree; the 84×84 figure is what
actually executed through the ResNet and is the one that matters for architecture/shape
tracing.

Other training-run metadata from `train_config.json` (not architecture, but useful
context): dataset root
`.../DatasetUtil/PickPlaceCan/SARM-robosuite-can-mh-stages_v3` (100 episodes, listed
explicitly), `steps=50000`, `batch_size=64`, AdamW (`lr=1e-4, weight_decay=1e-4,
grad_clip_norm=10.0`), and a **RA-BC (risk-aware BC) sample-weighting scheme**
(`sample_weighting.type="rabc"`, `kappa=0.18`, `head_mode="sparse"`,
`train_config.json:247-254`) — this is a non-uniform per-sample loss weighting during
BC training, not part of the ACT forward pass itself but relevant if the thesis
discusses how this specific base policy was trained. The 025000-step checkpoint (of a
50000-step run) is the one the user pointed at; not evaluated here for success rate.

### ACT implementation cross-check: same code, forked and diverged around normalization/vectorization

`src/lerobot/policies/act/modeling_act.py` (external repo, current upstream-style
LeRobot) vs. `resfit/lerobot/policies/act/modeling_act.py` (ResFiT's vendored copy):

- **Core `ACT`, `ACTEncoder`, `ACTEncoderLayer`, `ACTDecoder`, `ACTDecoderLayer`,
  `ACTSinusoidalPositionEmbedding2d`, `create_sinusoidal_pos_embedding` classes are
  essentially byte-identical** — same math, same layer order, same variable names, same
  DETR-style zero-tensor decoder input, same off-by-one `n_decoder_layers` bug preserved
  intentionally. This is the same fork lineage, not a reimplementation.
- **Where they diverge (the `ACTPolicy` wrapper, not the `ACT` network):**
  - External repo (`modeling_act.py:42-190`): normalization is **not** part of the
    policy class — it's handled by an external "processor" pipeline
    (`policy_preprocessor.json`/`policy_postprocessor.json` +
    `*_normalizer_processor.safetensors`), and `forward()` supports a `reduction="none"`
    mode returning per-sample losses (used for the RA-BC weighting scheme,
    `modeling_act.py:150-170`, `197-200` local numbering `forward(..., reduction=...)`).
    `select_action` uses one **global** `deque` action queue (single-env assumption,
    `modeling_act.py:93-98, 100-123`).
  - ResFiT's copy (`resfit/lerobot/policies/act/modeling_act.py:64-77, 154-231`):
    normalization (`Normalize`/`Unnormalize` from `lerobot.common.policies.normalize`) is
    built directly into `ACTPolicy.__init__` and called explicitly inside
    `select_action`/`forward`. `select_action` maintains a **list of per-environment
    deques** (`_ensure_action_queues`, `modeling_act.py:130-141`) and supports
    `reset(env_ids=...)` for partial resets — required for the vectorized-env residual RL
    setup (`resfit/rl_finetuning/wrappers/residual_env_wrapper.py:98, 156` calls
    `base_policy.reset(env_ids=reset_ids)` on a subset of terminated envs). No RA-BC
    `reduction="none"` support (not needed — ResFiT never trains the base policy).
  - `load_policy.py`'s manual copy of normalizer-processor safetensors into
    `policy.normalize_inputs`/`normalize_targets`/`unnormalize_outputs` buffers
    (`load_policy.py:202-258`) is precisely the adapter that bridges these two divergent
    checkpoint/normalization conventions, confirming this is a deliberate compatibility
    shim rather than two independently-developed models.

---

## D) Is the base policy frozen during residual RL fine-tuning?

**Yes — frozen.** Evidence, all file:line:

1. **`.eval()` set once at load and never toggled back.**
   `resfit/rl_finetuning/scripts/train_residual_td3.py:327-332`:
   ```python
   base_policy = load_policy(policy_dir)
   base_policy.to(device)
   base_policy.eval()
   eval_base_policy = load_policy(policy_dir)
   eval_base_policy.to(device)
   eval_base_policy.eval()
   ```
   Grepped the rest of `train_residual_td3.py` and
   `resfit/rl_finetuning/wrappers/residual_env_wrapper.py` for any `.train()` call on
   `base_policy`/`eval_base_policy` — **none found**.

2. **All calls into the base policy are wrapped in `torch.no_grad()` at the call site**,
   in addition to `select_action` itself being `@torch.no_grad` in both ACT
   implementations (`resfit/lerobot/policies/act/modeling_act.py:154`,
   `.../lerobot/src/lerobot/policies/act/modeling_act.py:100`):
   - `resfit/rl_finetuning/wrappers/residual_env_wrapper.py:101-102` (reset) and
     `:148-149` (step):
     ```python
     with torch.no_grad():
         base_action = self.base_policy.select_action(raw_obs)
     ```
   - `resfit/rl_finetuning/scripts/train_residual_td3.py:986`: same pattern
     (`base_action = base_policy.select_action(raw_obs)`, called inside the
     offline-buffer-population helper which is itself under a `torch.no_grad()`-guarded
     `select_action`).

3. **Base policy parameters never appear in any optimizer.** The only optimizers
   constructed anywhere in the RL agent are in
   `resfit/rl_finetuning/off_policy/rl/q_agent.py:111-113`:
   ```python
   self.encoder_opt = torch.optim.AdamW(self.encoders.parameters(), lr=self.cfg.critic_lr)
   self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.cfg.critic_lr)
   self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.actor_lr)
   ```
   `self.encoders` (`q_agent.py:62`, `_build_encoders`) is a **freshly constructed**
   `nn.ModuleList` of `VitEncoder` instances
   (`resfit/rl_finetuning/off_policy/networks/encoder.py:12-27`) — a **separate ViT
   vision encoder**, architecturally unrelated to and never initialized from the base
   ACT policy's ResNet18. Grepped `q_agent.py`, `actor.py`, `critic.py`, and
   `common_utils/utils.py` for any reference to `base_policy` —
   **none found in any of them**; the frozen `ACTPolicy` instance is passed only to
   `BasePolicyVecEnvWrapper` (env-side) and never touches `QAgent`/`Actor`/`Critic`
   construction or their optimizers.
   - **Important scoping note for the thesis figure:** the residual actor/critic do
     **not** share or fine-tune the base ACT policy's vision backbone. They run their
     own independent `VitEncoder` over the same camera observations. That encoder *is*
     trainable by default (`FREEZE_ENCODER="false"` in both shell scripts, wired to
     `agent.freeze_encoder` → `resfit/rl_finetuning/off_policy/rl/q_agent.py:104-108`:
     `if getattr(self.cfg, "freeze_encoder", False): for param in
     self.encoders.parameters(): param.requires_grad = False`). This
     `requires_grad_(False)` toggle exists only for the RL agent's *own* encoder, not
     for the base ACT policy — do not conflate the two when drawing the figure.

4. **No explicit `requires_grad_(False)` call on the base ACT policy anywhere** —
   searched the whole repo (`grep -rn "base_policy" resfit/rl_finetuning/off_policy/`,
   `train_residual_td3.py`, `residual_env_wrapper.py`). It isn't needed: because (a) it's
   never in an optimizer's param groups and (b) every call site is `torch.no_grad()`-guarded
   on top of `select_action` already being `@torch.no_grad`, no gradient ever reaches its
   parameters regardless of their `requires_grad` flag. Functionally frozen by
   construction rather than by an explicit flag flip.

**Conclusion for D:** the base BC policy is frozen for the entire residual-RL run —
no `.train()` call, no gradient path (no-grad at every call site plus an eval() that's
never undone), and structurally excluded from all three RL optimizers
(`encoder_opt`, `critic_opt`, `actor_opt`).

---

## Discrepancies with existing docs

1. **`../docs/policies/ACT_ARCHITECTURE.md` numeric example is for a different task than sim/CAN.**
   The doc's "Complete Dimension Trace" (`ACT_ARCHITECTURE.md:497-552`) and running
   example throughout use **TwoArmCoffee / GR1 humanoid**: 3 cameras, 36D state, 24D
   action, 27 image tokens. The residual-RL Can checkpoint the user actually has locally
   is **2 cameras, 9D state, 7D action, 18 image tokens, single Panda arm**. The
   *architectural* description (module graph, encoder/decoder structure, positional
   embeddings) in that doc is accurate and was cross-checked directly against
   `resfit/lerobot/policies/act/modeling_act.py` in this investigation — only the
   concrete tensor-shape numbers are task-specific and don't transfer to Can. Section B
   above gives the Can-specific numbers.

2. **`../docs/rewards/REWARD_MODEL_INTEGRATION.md` — confirmed, not stale**, on the one point this task
   asked about: the "local checkpoint takes priority over W&B" feature
   (`REWARD_MODEL_INTEGRATION.md:200, 211`) is fully implemented in the current
   (uncommitted) `resfit/lerobot/utils/load_policy.py` and
   `resfit/rl_finetuning/config/residual_td3.py`, matching the doc's description exactly
   (verified in §A above). No contradiction found here.

3. **Shell-script header comments are copy-pasted from the Lift script and are stale for
   the Can task specifically.** `scripts/train_residual_rl_can.sh:1-20` and
   `scripts/train_residual_rl_can_sparse.sh:1-20` both literally header themselves as
   `"train_residual_rl_lift.sh — Residual RL fine-tuning (TD3) for Lift (Panda)"` and
   claim `"Horizon: 100 steps (hardcoded in dexmg.py for 'Lift' task)"`. Both scripts
   actually set `CONFIG_NAME="residual_td3_can_config"` and run the `Can`
   (`PickPlaceCan`) task, whose horizon is **200**, not 100
   (`resfit/dexmg/environments/dexmg.py:132-145`, `{"Lift": 100, "PickPlaceCan": 200,
   ...}`). The 9D-state / 84×84-image / 20Hz claims in those same headers happen to be
   correct for Can too (same robot family as Lift), but the horizon figure and the
   "Lift" framing are copy-paste artifacts, not accurate documentation of what the
   script actually runs.

4. **`../docs/rewards/REWARD_MODEL_INTEGRATION.md`'s embedded prompt transcript (§1, informal notes, not
   the doc's own authored claims)** asserts training happens in a single (non-vectorized)
   sim env, with vectorized envs used only for periodic eval. This is corroborated by the
   current code (`NUM_ENVS=1` in both shell scripts with comment `"training environments
   (must be 1 due to n-step implementation)"`, `scripts/train_residual_rl_can.sh:621-622`;
   `EVAL_NUM_ENVS=10` used for `create_vectorized_env` at eval time). Included here for
   completeness even though it wasn't the focus of this investigation.

---

## NOT FOUND

- Any text/language/CLIP conditioning in the ACT forward pass (searched
  `resfit/lerobot/policies/act/modeling_act.py`,
  `resfit/lerobot/policies/act/configuration_act.py`, and the external repo's
  equivalents — no such component exists in ACT; the only text conditioning anywhere in
  the pipeline is the unrelated external SARM/TCC reward-model task prompt).
- Any explicit `requires_grad_(False)` call on the base ACT policy's parameters
  specifically (freezing is achieved structurally — see §D point 4 — not via an explicit
  flag).
- Evaluation metrics (success rate, etc.) for the `policy_rabc_100/checkpoints/025000`
  checkpoint itself — out of scope for this investigation (architecture-only), and no
  eval log/summary.json was located alongside that checkpoint directory.
