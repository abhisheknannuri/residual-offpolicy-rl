# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig, RLPDAlgoConfig, RLPDDexmgConfig


@dataclass
class OfflineDataConfig:
    name: str = "ankile/robomimic-mh-can-image"
    num_episodes: int | None = 300
    # Offline data action labeling options
    use_base_policy_for_base_actions: bool = True
    # Normalization safeguards
    min_action_range: float = 1e-1  # Minimum range for any action dimension to prevent normalization blow-up
    min_state_std: float = 1e-1  # Minimum std for any state dimension to prevent normalization blow-up
    # Resize dataset images to this square size (e.g. 84) before storing in
    # the offline replay buffer.  Set to null/None to use native resolution.
    # Must match the resolution the BC base policy was trained on.
    image_size: int | None = None

    # ------------------------------------------------------------------
    # External per-transition reward source (sidecar parquet)
    # ------------------------------------------------------------------
    # Optional path to a parquet produced by scripts/label_offline_pbrs.py (or any
    # labeler). When set, the offline replay-buffer reward for each transition is
    # read from ``reward_column`` keyed by the LeRobot global frame ``index`` and
    # is treated exactly like the dataset's ``next.reward`` (source-frame
    # convention: value at frame i is the reward of transition i -> i+1). Takes
    # priority over ``reward_shaping``/sparse. Generic: works for PBRS today and
    # any future labeled reward key.
    reward_parquet: str | None = None
    reward_column: str = "reward_pbrs"

    # ------------------------------------------------------------------
    # Local dataset + on-the-fly feature-key remapping
    # ------------------------------------------------------------------
    # Local dataset root. When set, the offline dataset is loaded from this path
    # (LeRobot v2.1 format) instead of downloading ``name`` from the Hub.
    root: str | None = None
    # Named feature-key remap preset (see resfit/rl_finetuning/datasets/remapped_lerobot.py
    # REMAP_PRESETS), e.g. "sarm_v21" to load the SARM Can v2.1 dataset whose keys are
    # agentview-images-rgb / state / actions instead of the resfit-standard names.
    remap_preset: str | None = None


@dataclass
class WandBConfig:
    project: str = "robomimic-can-residual-td3"
    mode: str = "online"
    entity: str | None = None
    notes: str | None = None
    continue_run_id: str | None = None
    name: str | None = None
    group: str | None = None


@dataclass
class BasePolicyConfig:
    wandb_id: str = "TODO"
    wt_type: str = "best"
    wt_version: str = "latest"
    # Local checkpoint directory. When set (non-null), the base policy is loaded
    # from this local path and W&B download is skipped entirely. The path may point
    # at the directory containing config.json, or a parent that has a
    # `policy/` or `pretrained_model/` subfolder (e.g. a LeRobot
    # `.../checkpoints/050000/pretrained_model` directory). Takes priority over wandb_id.
    local_path: str | None = None


@dataclass
class RewardModelConfig:
    """External stage-aware reward-model (SARM / TCC) integration.

    When ``enabled`` is True the training env's reward is replaced by a
    Potential-Based Reward Shaping (PBRS) signal driven by an external reward
    model served over HTTP:  ``r = r_sparse + gamma*phi(s') - phi(s)`` where the
    potential ``phi`` is a function of the (monotonic, server-gated) task stage.

    The reward server is queried every ``query_every_k`` env steps; the ``k``
    most-recent frames are scored in a single batched request and passed through
    server-side hysteresis (upgrade a stage only after ``hysteresis_k`` consecutive
    high-confidence predictions). Requires ``algo.terminated_only_bootstrap=True``
    so the shaped potentials telescope correctly under n-step returns.
    """

    enabled: bool = False
    backend: str = "sarm"  # "sarm" | "tcc" (informational; server decides the model)
    server_url: str = "http://127.0.0.1:8001"
    request_timeout_s: float = 20.0
    # Fail fast at startup if the server /health check does not pass.
    require_server: bool = True

    # Natural-language task prompt sent to the (CLIP-conditioned) reward model.
    task_prompt: str = "pick up the can and place it in the bin"

    # Rollout query cadence: query the server every k env steps (20Hz control ->
    # k=5 == 4Hz). The k most-recent frames are batched into one request.
    query_every_k: int = 5

    # Client-side rolling raw-frame history (chronological, most-recent last) sent
    # to the server, which performs all model-specific windowing/sampling. For SARM
    # (n_obs_steps=8, frame_gap=5) the server needs (8-1)*5+1=36 back frames per
    # anchor; with k=5 anchors the union is ~40 frames.
    client_history_len: int = 40
    image_key: str = "observation.images.agentview"
    # Vertically flip camera frames before sending to the reward model. Robosuite
    # obs frames can be upside-down relative to the reward model's training frames;
    # set True if a quick frame comparison shows a vertical flip is needed.
    image_vflip: bool = False

    # PBRS parameters -------------------------------------------------------
    num_stages: int = 4
    # Per-stage potentials (index = stage). Small values keep the critic scale
    # well below the sparse success reward (=1.0) for ResFiT/Q-ensemble stability.
    potentials: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15)
    # Discount used in the PBRS shaping term. When None, falls back to algo.gamma.
    pbrs_gamma: float | None = None
    # Keep the environment's sparse success reward as the base term of the PBRS
    # signal (r = r_sparse + gamma*phi' - phi). If False, pure shaping only.
    keep_sparse_term: bool = True

    # Reward mode ----------------------------------------------------------
    # "pbrs"      -> potential-based reward shaping (policy-invariant; telescopes
    #                under n-step). Uses potentials / pbrs_gamma / keep_sparse_term.
    # "milestone" -> ratchet one-time stage payouts + a terminal success bonus
    #                (NOT policy-invariant; does NOT telescope). Ablation to test
    #                whether a non-PBRS stage-aware dense reward helps residual RL.
    #                Ignores potentials / pbrs_gamma.
    reward_mode: str = "pbrs"  # "pbrs" | "milestone"
    # Paid ONCE on the first monotonic entry into each stage (index = stage;
    # index 0 = start, never paid). Skipped stages are summed (order-independent).
    milestone_payouts: tuple[float, ...] = (0.0, 0.1, 0.1, 0.1)
    # Terminal reward on TRUE simulator success (env sparse reward > 0). Dominant
    # so the agent can't farm the stage-3 "placing" payout without succeeding.
    # Default keeps max return = 1.0 (0.1*3 + 0.7) — safe even for v_max=1.0 critics.
    milestone_success_bonus: float = 0.7

    # Server-side gating (advertised to the server per request) -------------
    hysteresis_k: int = 5           # consecutive high-conf predictions to upgrade a stage
    conf_threshold: float = 0.80    # min stage confidence to count toward an upgrade
    monotonic: bool = True          # never downgrade the stage within an episode

    # Offline dataset labeling ---------------------------------------------
    # New parquet key written by scripts/label_offline_pbrs.py (does not override
    # existing rewards) and, when set, read by the offline replay-buffer loader.
    offline_key: str | None = None



@dataclass
class ResidualTD3AlgoConfig(RLPDAlgoConfig):
    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    # Number of critic-only updates before training the actor
    critic_warmup_steps: int = 10_000
    # How often (in critic-warmup steps) to save a checkpoint during the critic-warmup
    # phase - separate from offline_save_freq (which only applies to the offline-RL
    # phase below) since critic-warmup previously saved no checkpoints at all.
    critic_warmup_save_freq: int = 5_000

    # ------------------------------------------------------------------
    # Online warm-up (random-policy) buffer collection -------------------
    # ------------------------------------------------------------------
    # How often (in transitions added to the ONLINE buffer during the live
    # random/base-policy warm-up collection loop, before learning_starts is
    # reached) to persist the online buffer to online_cache_dir. Previously
    # this buffer was only ever written to disk once, after the *entire*
    # warm-up loop finished - so a lost robot connection mid-collection lost
    # all progress. 0 disables periodic saving (dump only happens at the end,
    # the old behavior).
    online_warmup_save_freq: int = 2_000

    # ------------------------------------------------------------------
    # Random action exploration -----------------------------------------
    # ------------------------------------------------------------------
    # Scale for random action noise during initial exploration phase
    # Actions are sampled as: rand_actions = torch.rand(...) * 2 * random_action_noise_scale - random_action_noise_scale
    random_action_noise_scale: Any = 0.2  # Default: uniform in [-1, 1]

    # Whether to use base policy + noise (True) or pure uniform noise (False) during warmup
    # Note: Environment wrapper always applies base_action + residual_action
    # True: residual_action = noise (resulting in base_action + noise)
    # False: residual_action = pure_random - base_action (resulting in pure_random)
    use_base_policy_for_warmup: bool = True

    # ------------------------------------------------------------------
    # Standard deviation schedule -------------------------------------------
    # ------------------------------------------------------------------
    stddev_max: Any = 0.05
    stddev_min: Any = 0.05
    stddev_step: int = 300_000

    # Progressive clipping schedule for the residual actions
    # I.e., starts clipping linearly from 0 to action scale over progressive_clipping_steps steps
    progressive_clipping_steps: int = 0

    # ------------------------------------------------------------------
    # Offline RL (TD3-BC) Phase
    # ------------------------------------------------------------------
    train_offline_rl: bool = False
    offline_rl_steps: int = 50000
    offline_rl_bc_alpha: float = 0.2 # paper suggested value is 2.5 (but the bc loss is based on full action range an ndot residual.)
    offline_eval_interval_every_steps: int = 5000
    offline_save_freq: int = 5000
    offline_rl_load_ckpt: str = "latest"
    offline_rl_sampling_method: str = "fixed_ratio"  # "fixed_ratio" or "proportional"
    offline_rl_offline_ratio: float = 0.5            # fraction of batch from offline buffer (only for fixed_ratio)

    # ------------------------------------------------------------------
    # Offline-only pretraining mode (no live robot needed) --------------
    # ------------------------------------------------------------------
    # When true: skip get_envs()/env.reset() entirely (no follower/policy-server/
    # leader connection attempted at all - env/eval_env/obs become None) and exit
    # right after critic-warmup + (if train_offline_rl) the offline-RL phase
    # complete, BEFORE the main online training loop (which genuinely needs a live
    # env). Lets critic-warmup and/or the offline-RL phase run purely off the
    # offline/online replay-buffer CACHES on disk (see BUFFER_POPULATION_ARCHITECTURE.md) -
    # e.g. away from the lab, with no hardware connected at all. Combine with
    # train_offline_rl=false for "critic-warmup only", or train_offline_rl=true for
    # "critic-warmup then offline TD3-BC". Default false: every existing real-hardware
    # .sh is completely unaffected (this flag doesn't exist in any of them, so it's
    # absent -> false -> the exact same code path as before this flag was added).
    offline_pretrain_only: bool = False


# -----------------------------------------------------------------------------
# Top-level experiment config --------------------------------------------------
# -----------------------------------------------------------------------------
@dataclass
class ResidualTD3DexmgConfig(RLPDDexmgConfig):
    actor_name: str | None = None  # Inferred from base policy config

    # ------------------------------------------------------------------
    # Real-hardware training (Trossen, single station, no vectorization)
    # ------------------------------------------------------------------
    # When True, `get_envs()` connects to the REAL Trossen station instead of
    # building a MuJoCo/dexmg vectorized sim env - see
    # `docs/real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md` for the full
    # design. `num_envs`/`eval_num_envs` are forced to 1 in this mode
    # (asserted in `main()`); periodic evaluation is skipped entirely (see
    # `eval_interval_every_steps` handling in the training loop) - real
    # hardware relies purely on `save_freq` checkpointing, evaluated
    # separately/offline later, not on interleaved eval rollouts.
    real_hardware: bool = False
    # Station config name (see `trossen_real/config.py::load_station_config()`),
    # e.g. "trossen_station2_single". Required when real_hardware=True.
    station_config_name: str = ""
    # URL of the running `policy_server.py` serving the FROZEN base ACT policy
    # (same server/contract the inference web app already uses).
    policy_server_url: str = "http://127.0.0.1:5070"
    # "absolute" | "delta_joint" - matches the checkpoint's trained action space
    # (see `trossen_real/scripts/convert_to_delta_joint_dataset.py`).
    real_action_space: str = "delta_joint"
    # Per-episode step cap for the real env (no MuJoCo horizon to read from).
    real_max_steps: int | None = None
    # Seconds to sleep after the staged-position reset move (+ leader sync)
    # completes, before resetting the policy's action queue / taking the
    # first obs - mirrors infer_app's proven post-reset settle window
    # (lets the arm stop vibrating AND gives the operator time to release
    # any pedal still held from the previous episode).
    real_settle_time_s: float = 2.0
    # Opt-in human-intervention takeover during training (leader + foot pedal,
    # reusing `trossen_real/inference/intervention.py::InterventionManager` -
    # see `trossen_real/human_intervention/INTERVENTION_PLAN.md`). Default off;
    # when on, connecting the leader is REQUIRED (fails startup if it can't
    # connect - no silent degraded mode). Only meaningful when real_hardware=True.
    enable_intervention: bool = False
    # Optional path to a JSONL diagnostic log (one line per real-hardware env.step() -
    # reward/terminated/truncated/intervened/executed action/raw follower state/achieved Hz,
    # everything EXCEPT images). Purely for offline inspection - training never reads this
    # file, only the replay buffer. Empty/None = no logging (default).
    real_log_file: str | None = None

    # ------------------------------------------------------------------
    # How we talk to policy_server.py ----------------------------------
    # ------------------------------------------------------------------
    # These apply EVERYWHERE the base policy is queried - offline buffer
    # population, the online warm-up, and online RL - because they describe the
    # wire format, not a training choice. Both paths go through
    # `trossen_real/inference/policy_client.py`.

    # How camera / dataset frames are packed for the request body:
    #   "raw"  - uncompressed HWC bytes + base64. The original behaviour.
    #   "zlib" - the same bytes deflated. Lossless, ~3.3x smaller.
    #   "jpeg" - ~15x smaller at quality 95. Lossy, but measured on real
    #            wrist-cam frames it distorts LESS than the AV1 the LeRobot
    #            dataset is itself stored in (mean abs err 0.79 vs 1.39 on
    #            0-255), so it does not move inference further from the
    #            training distribution than training already was.
    # NOT part of either replay-buffer cache key: changing it will not
    # invalidate an existing buffer, even though the base actions it produces
    # differ slightly. Keep it fixed for a given buffer.
    policy_image_encoding: str = "raw"
    policy_jpeg_quality: int = 95

    # Fetch a whole action chunk per request instead of one action per request.
    #   0  - one POST /predict per query (the original behaviour)
    #   -1 - POST /predict_chunk, chunk length = the server's own n_action_steps
    #   N  - POST /predict_chunk, asking for N steps (server caps at chunk_size)
    # This changes the NETWORK pattern, not the actions: policy_server.py's
    # /predict already serves from an internal queue and only runs the model when
    # that queue is empty, so a client-side queue returns the identical sequence
    # (see ChunkedPolicyClient). The queue is dropped on every policy.reset(),
    # which happens at every episode boundary and on intervention release.
    policy_chunk_steps: int = 0

    # ------------------------------------------------------------------
    # Environment reward manipulation ----------------------------------
    # ------------------------------------------------------------------
    use_reward_manipulation_wrapper: bool = False
    terminate_on_success: bool = False
    dense_success_bonus_scale: float | None = None

    # ------------------------------------------------------------------
    # Algorithm & optimisation
    # ------------------------------------------------------------------
    algo: ResidualTD3AlgoConfig = field(default_factory=ResidualTD3AlgoConfig)

    # ------------------------------------------------------------------
    # Network architectures
    # ------------------------------------------------------------------
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-6,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            actor=ActorConfig(
                action_scale=0.1,
                actor_last_layer_init_scale=0.0,  # imp for residual
            ),
        )
    )

    # ------------------------------------------------------------------
    # Offline dataset
    # ------------------------------------------------------------------
    offline_data: OfflineDataConfig | None = field(default_factory=OfflineDataConfig)

    # ------------------------------------------------------------------
    # Base policy
    # ------------------------------------------------------------------
    base_policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)

    # ------------------------------------------------------------------
    # External stage-aware reward model (SARM / TCC) via HTTP
    # ------------------------------------------------------------------
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)

    # ------------------------------------------------------------------
    # Weights & Biases logging
    # ------------------------------------------------------------------
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------
    eval_interval_every_steps: int = 10_000

    # How often (in env steps) to save a checkpoint and push to WandB.
    # Set to -1 to disable periodic checkpointing (best-model saving still works).
    save_freq: int = 10_000

    # Whether to run an evaluation pass before training begins (at step 0)
    eval_first: bool = True

    # When True, skip cleanup of the run_cache_dir after training finishes.
    # Local checkpoints (latest/, best/, policy_step_*/) are preserved on disk.
    # Useful when network issues may prevent WandB artifact uploads from completing.
    no_cleanup: bool = False

    # ------------------------------------------------------------------
    # Training-rollout reward-model debugging
    # ------------------------------------------------------------------
    # When True, randomly record annotated TRAINING rollouts (not eval) to
    # inspect what the reward model is doing during learning: every step's frame
    # is overlaid with the raw (pre-hysteresis) predicted stage, the gated stage,
    # confidence, and the PBRS reward, and a per-frame JSON is written. Each
    # episode is sampled with probability ``training_rollout_prob``. Artifacts are
    # saved locally under ``<output_dir>/<run>/training_rollouts/`` and uploaded
    # to W&B under ``training/`` keyed by global step (non-overriding). Only
    # active for single-env training (num_envs == 1).
    save_training_rollouts: bool = False
    training_rollout_prob: float = 0.2

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    # Path to a local checkpoint file (checkpoint.pt) or directory containing
    # one.  When set, agent weights, optimizers, and global_step are restored
    # so training continues from where it left off.
    resume_ckpt: str | None = None

@dataclass
class ResidualTD3CanConfig(ResidualTD3DexmgConfig):
    task: str = "Can"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-can-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-can-bc/sdo8cku7",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-can-residual-td3"))


@dataclass
class ResidualTD3CubeLiftConfig(ResidualTD3DexmgConfig):
    task: str = "Lift"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-lift-image",
            num_episodes=50,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="resfit-robomimic-lift-bc/3pghetmx",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-lift-residual-td3"))


@dataclass
class ResidualTD3SquareConfig(ResidualTD3DexmgConfig):
    task: str = "Square"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-square-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-square-bc/dzbkdpwp",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-square-residual-td3"))


@dataclass
class ResidualTD3TrossenRealConfig(ResidualTD3DexmgConfig):
    """Real-hardware, single-station Trossen residual RL config. Points
    `get_envs()` at `TrossenResidualEnv` instead of a MuJoCo/dexmg vectorized
    sim env - see docs/real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md."""

    task: str = "trossen_real"
    real_hardware: bool = True
    num_envs: int = 1
    eval_num_envs: int = 1

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="trossen_pick_and_insert_cube_delta",  # placeholder - override via CLI with the real dataset repo_id
            # NOTE: main() asserts this is not None regardless of real_hardware - it CANNOT be
            # "use the whole dataset" via None. Override via CLI (offline_data.num_episodes=N)
            # to match your actual dataset's episode count.
            num_episodes=1,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="",  # unused for real hardware - base policy is served by policy_server.py, not loaded here
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="trossen-pick-and-insert-residual-td3"))


@dataclass
class ResidualTD3BoxCleanConfig(ResidualTD3DexmgConfig):
    task: str = "TwoArmBoxCleanup"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot1_eye_in_hand",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-box-clean-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-box-cleanup",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )


@dataclass
class ResidualTD3CoffeeConfig(ResidualTD3BoxCleanConfig):
    task: str = "TwoArmCoffee"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="dexmg-coffee-residual-td3", notes="all cameras")
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-coffee",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )

@dataclass
class ResidualTD3TwoArmCanSortConfig(ResidualTD3BoxCleanConfig):
    task: str = "TwoArmCanSortRandom"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.frontview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-cansort-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-can-sort-random",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )


# -----------------------------------------------------------------------------
# Register with Hydra
# -----------------------------------------------------------------------------
cs = ConfigStore.instance()
cs.store(name="residual_td3_dexmg_config", node=ResidualTD3DexmgConfig)
cs.store(name="residual_td3_can_config", node=ResidualTD3CanConfig)
cs.store(name="residual_td3_cube_lift_config", node=ResidualTD3CubeLiftConfig)
cs.store(name="residual_td3_square_config", node=ResidualTD3SquareConfig)
cs.store(name="residual_td3_box_clean_config", node=ResidualTD3BoxCleanConfig)
cs.store(name="residual_td3_coffee_config", node=ResidualTD3CoffeeConfig)
cs.store(name="residual_td3_two_arm_cansort_config", node=ResidualTD3TwoArmCanSortConfig)
cs.store(name="residual_td3_trossen_real_config", node=ResidualTD3TrossenRealConfig)
