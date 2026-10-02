# Residual RL on the real arm — checkpoints, logging, inference, UI

Status: **proposal. Nothing implemented.** Supersedes the first draft of
`RESIDUAL_RL_EVAL_PLAN.md`, which only covered eval.

Scope, after review:

1. Checkpoints become **hermetic** — inference needs the `.pt` and nothing else. **(done)**
2. One **run-record** system for per-tick data, shared by infer / eval / teleop /
   RL, replacing eight ad-hoc sinks.
3. Base-BC **provenance** recorded in the RL checkpoint and asserted at load. **(done)**
4. Residual inference as a **policy-client wrapper**, confirming your reading.
5. **Config schema + presets** instead of a growing checkbox wall.
6. **Rerun** for visualisation instead of more Flask UI.
7. Q plots as a **reusable util**, callable live or on logged data.
8. Eval protocol generalisation, and a note on resumability.

Everything under "Measured" was run in this repo. Numbers come from
`run_2026-10-02_12-37-47_..._off178ep_lr1e-06__seed42`.

---

## 0. Measured, not assumed

### Checkpoint

All 20 checkpoints (5k…100k) load into a `QAgent` built from `ck["config"]`
alone with **0 missing, 0 unexpected** keys. All `image_size=84`,
`pos_embed=(1,81,128)`, `actor_updates == step/2`.

Contents: `agent_state_dict` (118 tensors), `global_step`, `actor_updates`,
`optimizer_state_dict`, `config` (45 keys), `success_rate`. 432 MB on disk,
~115 MB of that is the fp32 model; the rest is optimizer state.

| module | params |
| --- | --- |
| `encoders` (2 cams) | 762,368 |
| `actor` | 3,861,895 |
| `critic` (10 vmapped Q heads) | 23,619,595 |

**Missing for deployment:** `ActionScaler` / `StateStandardizer`. Only
`actor.action_scale_tensor` is saved. §1 fixes this.

### Latency, laptop RTX 4090, batch 1

| stage | ms |
| --- | --- |
| encoder, 2 cams | 0.55 |
| + actor (`agent.act()`) | 0.74 |
| + all 10 critics, reusing `feat` | **1.38** |
| critics alone (marginal) | 0.64 |
| Rerun log, 2 images + 9 scalars | 0.79 |

At 20 Hz the budget is 50 ms/tick. Full RL stack **plus** logging ≈ **2.2 ms,
4.4%**. Critics are nearly free because the encoder forward is already done.

### Storage, real episode (T=290, 2 cams, exact 84×84 RL input)

| format | per episode | per 20-pose sweep |
| --- | --- | --- |
| `.npy` raw | 12.3 MB | 0.25 GB |
| `.npz` deflate | 3.2 MB | 0.06 GB |
| **zarr blosc/zstd-5** | **3.7 MB** (39 ms write) | **0.07 GB** |
| ffv1 lossless video | 2.0 MB (28 ms) | 0.04 GB |

**This is the finding that unblocks the design.** Storing the exact,
lossless model input for every tick of a whole sweep costs ~70 MB. There is no
need to trade fidelity for space, and no need to reconstruct inputs later.

### Already installed

`rerun-sdk 0.35.0`, `tensordict 0.9.1`, `pyarrow 24`, `zarr 2.18`, `h5py 3.16`,
`av 17.1`, `wandb 0.24`, `matplotlib 3.10`. Nothing new needs adding for any of
this.

---

## 1. Hermetic checkpoints — IMPLEMENTED

Status: **done and pushed.** Everything below is in the code, not proposed.
There is no backward compatibility and no backfill: checkpoints written from
now on carry everything, and ones written before do not. Retrain and move on.

### What every checkpoint now carries

`save_checkpoint()` gained a `deployment=` argument
(`resfit/rl_finetuning/utils/checkpoint.py`). It is built **once at startup** by
`build_deployment_meta()` (`resfit/rl_finetuning/utils/deployment.py`) and
passed to every save site — 7 in the distributed learner, 8 in the
single-process trainer, covering critic warm-up, offline RL, online RL and
resume alike.

```text
ck["deployment"] = {
  "schema_version": 1,
  "normalization": {
      "action_scaler":      {action_min, action_max, action_scale, min_range_per_dim},
      "state_standardizer": {state_mean, state_std, min_std},
  },
  "base_policy": {
      "kind": "act_http",
      "weights_sha256": "...",          # THE identity
      "weights_files": [...],
      "server_checkpoint_path": "...",  # descriptive only
      "image_keys", "non_image_keys",
      "likely_action_space", "action_space_source",
      "observed_n_action_steps", "observed_chunk_size",   # NOT identity
      "observed_at_utc",
  },
  "obs": {image_keys, rl_image_size, action_dim, prop_dim, real_action_space},
  "provenance": {dataset_name, dataset_root, dataset_num_episodes, task,
                 git: {commit, dirty}, created_utc},
}
```

### Normalizers reconstruct exactly

`ActionScaler` and `StateStandardizer` now keep the **raw dataset stats** they
were built from and expose `state_dict()` / `from_state_dict()`. The saved form
is the raw stats plus the scalars — deliberately *not* the derived
`_limits`/`_range` — so reconstruction re-runs the same `__init__` and cannot
drift from how training built it.

Verified bit-exact after a `torch.save`/`torch.load` round trip: `scale`,
`unscale` and `standardize` all return tensors `torch.equal` to the originals.

### Base-policy identity is hashed server-side

The hash lives in `policy_server.py::_hash_checkpoint_weights()` and is reported
as `weights_sha256` in `/health`. Server-side is the right place: that process
is the only one guaranteed to have the weights on local disk, so the same
trained model reports the same hash whether it is served from the GPU box, your
laptop, or someone else's machine.

It is a sha256 over the checkpoint's **weight files only** — `.safetensors`,
`.bin`, `.pt`, `.pth`, `.ckpt` — each contributing its relative name and its
bytes, names sorted.

Deliberately excluded:

- **`--n-action-steps` and every other inference-time knob.** They are applied
  to `policy.config` after loading and never touch the weight files, so a server
  started with `--n-action-steps 14` and one started with `20` report the same
  hash. That is what makes this a *checkpoint* identity rather than a
  *server-invocation* identity.
- **`config.json` / `train_config.json`.** They carry absolute paths and
  timestamps from whichever machine trained the model, which would make the hash
  non-portable.

Measured:

| case | result |
| --- | --- |
| same weights, different machine path, different `config.json`, `n_action_steps` 20 vs 14 | **identical hash** |
| different step of the same training run | different hash |
| sharded checkpoint, hashed twice | stable |
| one shard renamed | detected |
| path missing / no weight files | `(None, [])` → client refuses to run |

### Provenance is a hard requirement

For `real_hardware`, both trainers query `/health` **once at startup** and
refuse to start otherwise. One query covers the whole run — the server cannot
swap checkpoints without a restart — so online RL and resume pay nothing extra.

`BasePolicyProvenanceError` is raised, with an actionable message, when the
server is unreachable, has no policy loaded, or is an older build with no
`weights_sha256`.

**This changes how you launch mode 02.** Offline RL previously needed no policy
server, because the buffer comes from cache. It now does. What is being recorded
is the base policy's *identity*, not its output — a residual is a correction to
one specific ACT checkpoint and is meaningless against another, so a checkpoint
that cannot name its base is not reproducible. Start the policy server before
`02_offline_rl.sh`.

### Verifying at inference

`assert_base_policy_matches(ck["deployment"]["base_policy"], health, mode=...)`
with `strict` (default — hashes must match), `warn` (log and continue, for
deliberate cross-base probes) or `off`.

### Buffer caches get a sidecar, not a new metadata key

`user_metadata.json` is a verbatim dump of the dict whose sha1 **names the cache
directory**. Adding a key to it would change every cache hash and orphan every
buffer already on disk. So `write_buffer_sidecar()` writes a separate
`base_policy.json` next to it. Additive, invisible to the hash, and buffers
written before this simply have no sidecar.

Two related facts worth recording, both checked:

- The cache hash is computed over an explicit `offline_cache_meta` dict, not the
  whole config — so **adding config fields cannot obsolete buffers**.
- `user_metadata.json` is written but never read back for validation, so extra
  files alongside it are harmless.

Confirmed by diff that nothing inside either hashed metadata dict changed, and
`check_configs.sh` still reports all 8 configs composing with identical HASHED
values.

### Sim / robomimic

`base_policy` is `None` there and no `/health` query is attempted — there is no
policy server to ask. Normalization and provenance are still recorded.

### You must copy the server file

`policy_server.py` lives outside this repo, at
`.../GeneralistRewardModels/lerobot/custom_scripts/policy_server.py`. The hash
change cannot be pushed with this commit. Copy it to
`~/Research/LEROBOT/lerobot/custom_scripts/policy_server.py` on the server and
restart the policy server **before** starting training, or training will refuse
to start with the "does not report weights_sha256" message. A timestamped
`.bak-*` of the original is next to it.

## 2. The logging problem, stated concretely

Your instinct that it is "kind of messed up" is correct. There are **eight**
sinks across **four** roots with **no shared run identifier**:

| # | sink | root | written by |
| --- | --- | --- | --- |
| 1 | `app.log` | `logs/<app>/<date>/<time>/` | `log_setup.setup_logging` |
| 2 | `console_output.log` | same | `_Tee` stdout capture |
| 3 | `infer_<station>_<ts>.jsonl` | `logs/` | `TickLogger` |
| 4 | `*.mp4` | `infer_videos/` | `VideoRecorder` |
| 5 | LeRobot dataset | `infer_dataset/` | `DatasetRecorder` |
| 6 | `index.jsonl`, `summary.json`, `timeline.jsonl` | `eval_runs/` | `EvalWriter` |
| 7 | periodic latency dumps | into #1 | `call_stats` |
| 8 | RL training tick log | `RealTrainLogs/` | `REAL_LOG_FILE` |

Consequences you are already feeling:

- To reconstruct one episode you must join four artifacts by **wall-clock
  timestamp**, because nothing carries a run id.
- `TickLogger` is JSONL, flushed every line. Fine for 9 scalars; **unusable** for
  images or 10 Q heads × 350 ticks × 20 poses.
- Teleop shares `DatasetRecorder` but not `VideoRecorder`; infer has both. Adding
  RL fields means touching both paths independently.
- Nothing records *which checkpoint* produced a given tick, so a sweep's logs are
  only distinguishable by folder naming convention.

### Design principles

1. **One run id.** `run_<utc>_<shortuuid>` minted once, stamped into every
   artifact. Everything else joins on it.
2. **The archive is boring and columnar.** Parquet + zarr + mp4. No bespoke
   binary, no viewer-specific format as source of truth.
3. **Log the exact model input, not a reconstruction.** §0 shows it costs 70 MB
   per sweep. Any resize/codec applied after the fact is a confound.
4. **Logging never blocks control.** Writer owns a bounded queue and a background
   thread; the control loop does a non-blocking put. On overflow it drops and
   counts drops rather than stalling the arm.
5. **Viewers are regenerable.** Rerun `.rrd`, plots, HTML — all derived from the
   archive, never the only copy.
6. **One schema, declared once.** Adding a field is one edit, and every consumer
   sees it.

---

## 3. The run record

```text
runs/<run_id>/
  meta.json              run id, config snapshot, checkpoint provenance, git, hardware
  ticks.parquet          one row per tick, all scalars/vectors
  rl_obs.zarr            (T, n_cam, 3, 84, 84) uint8 — exact RL encoder input
  video/<cam>.mp4        native-res, human review (existing VideoRecorder)
  episodes.jsonl         one line per episode: pose, stages reached, stop reason
  app.log                python logging for this run (symlink or copy of sink #1)
  view.rrd               optional, generated — Rerun recording
```

One directory per run. `runs/index.parquet` at the root for one-pass
aggregation, replacing `eval_runs/index.jsonl`.

### `ticks.parquet` schema

Declared once as a dataclass + pyarrow schema in
`trossen_real/logging/schema.py`:

| column | type | notes |
| --- | --- | --- |
| `run_id`, `episode_id`, `step` | str, str, i32 | join keys |
| `t_wall`, `t_mono` | f64 | wall for humans, monotonic for timing |
| `dt_tick`, `achieved_hz` | f32 | |
| `dt_base_query`, `dt_rl_forward`, `dt_log` | f32 | **per-stage timing, always on** |
| `base_served_from_cache` | bool | chunk queue pop vs real round trip |
| `observation.state` | list<f32>[7] | raw joints |
| `observation.state_std` | list<f32>[7] | what the net saw |
| `base_action` | list<f32>[7] | real units from ACT |
| `base_naction` | list<f32>[7] | after `action_scaler.scale` |
| `residual_naction` | list<f32>[7] | actor output |
| `combined_naction` | list<f32>[7] | after clamp — **the `a` in Q(s,a)** |
| `executed_action` | list<f32>[7] | real units actually sent |
| `clipped_frac` | f32 | residual lost to the clamp |
| `intervened` | bool | leader took over → `a` is the human's |
| `q_values` | list<f32>[10] | null when critics disabled |
| `q_mean`, `q_min`, `q_max`, `q_std` | f32 | null when critics disabled |
| `rl_obs_index` | i32 | row into `rl_obs.zarr` |
| `checkpoint_label`, `base_checkpoint` | str | so a sweep is self-describing |

You asked specifically about timing: `dt_base_query` / `dt_rl_forward` /
`dt_log` as separate columns, from `time.perf_counter()`, are what let you answer
"why did this tick miss" without guessing. `call_stats` already does this for
HTTP; this extends it to the whole tick and puts it in the same table as
everything else.

### Why parquet and zarr rather than tensordict

`tensordict` is right for the replay buffer — fixed-shape batched tensors, GPU
transfer, `MultiStepTransform`. It is the wrong fit here: a run record is a
**ragged, mixed-type, append-only time series** that you want to open in pandas
and `groupby("checkpoint_label")`. Parquet does that natively; a saved
TensorDict does not.

The images are the one fixed-shape part, which is exactly where zarr fits:
chunked `(32, n_cam, 3, 84, 84)`, so §6's Q recompute reads batches of 32 ticks
straight off disk with no decode step. 39 ms to write a whole episode.

### Answering the "two video recordings" concern

There is only one *video*. The native-res mp4 stays as-is, for watching. The
84×84 record is not a second video — it is **the model's input tensor**,
losslessly stored, at 3.7 MB/episode.

Deriving it later from the mp4 instead would be false economy: the mp4 is lossy,
and the resize would have to be bit-identical to training forever. Storing it
makes the Q recompute exact and keeps the record valid if `image_size` ever
changes (§ the 112 experiment).

### Viewing the logs

- `ticks.parquet` opens in pandas directly (2.3.3 is installed) — one line, no
  custom reader: `pd.read_parquet("ticks.parquet")`. DuckDB/polars are nicer for
  ad-hoc SQL across a whole sweep but are **not** installed, so treat them as
  optional conveniences, not part of the plan.
- `scripts/run_record_to_rerun.py <run_dir>` produces `view.rrd` — images,
  Q bands, residual per joint, intervention spans, all on one scrubable
  timeline.
- `trossen_real/logging/inspect.py --run <id>` prints a text summary: Hz
  histogram, dropped-log count, intervention segments, stage transitions.

So the honest answer to "the logs won't be readable": parquet is more readable
than JSONL in every tool that matters, and the one thing parquet cannot show you
— the images against the signals — is exactly what Rerun does well.

---

## 4. Rerun instead of more web UI

This is the "does OSS already exist" answer. **Rerun is already installed
(0.35.0)** and is built by robotics people for precisely this: time-aligned
images, scalars, transforms, 3D, with a scrubbing viewer and no UI code.

Measured above: 0.79 ms/tick for 2 images + 9 scalars.

Two modes, same call sites:

| mode | call | use |
| --- | --- | --- |
| live | `rr.connect_grpc()` | watch an eval as it runs — images, Q, residual, Hz |
| offline | `rr.save(path)` | regenerate `view.rrd` from the archive afterwards |

What this buys, concretely: you stop building video display, time-series plots,
and scrubbing into Flask. The Flask app goes back to being **a robot control
panel** — start/stop, pose selection, stage annotation, pedal state — which is
the part Rerun does *not* do and which genuinely needs to be a web page on the
robot machine.

I would deliberately **not** try to embed Rerun inside the Flask page in phase
one. Run the viewer as its own window, launched from a button.

Worth knowing for later though: 0.35 ships `rr.serve_web_viewer()` and
`rr.serve_grpc()`, so Rerun can serve its own viewer into a browser tab. That is
a genuine path to "it's all in one web UI" without writing any of the
visualisation — but it is a phase-two nicety, and the separate window is what I
would ship first.

### What stays in Flask

Start / Stop / Reset, pose selection and progress, stage annotation buttons,
intervention and pedal status, a single low-rate camera thumbnail for framing,
live Hz, and preset selection (§5). That is a page that fits on one screen.

---

## 5. Config management and the checkbox problem

The UI is sprawling because every capability became an independent boolean, and
some of them are not independent — `collect_q` requires a residual checkpoint,
`policy_chunk_steps` requires server support, jpeg quality is meaningless when
encoding is raw.

The repo already solved this once, for RL: **dataclass schema + YAML + Hydra**.
Reuse the same pattern rather than inventing a UI-specific one.

### Schema

`trossen_real/infer_app/config.py` — one dataclass tree:

```python
@dataclass
class PolicyCfg:
    mode: str = "bc"            # "bc" | "residual"
    server_url: str = "http://127.0.0.1:5070"
    chunk_steps: int = 0
    image_encoding: str = "raw"
    jpeg_quality: int = 95
    residual_checkpoint: str | None = None
    base_policy_assert: str = "strict"      # strict | path | warn
    collect_q: bool = True
    device: str = "cuda"

@dataclass
class LoggingCfg:
    level: str = "full"        # "off" | "metrics" | "full"
    rl_obs: bool = True        # write rl_obs.zarr
    video_native: bool = True
    rerun_live: bool = False
    queue_capacity: int = 2048

@dataclass
class InferAppCfg:
    station: str = "trossen_station3_single"
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    eval: EvalRefCfg = ...
```

### Validation rules live with the schema, not the UI

One `validate()` returning a list of structured errors/warnings, so the API, the
CLI and the form all enforce the same rules:

```
policy.mode == "residual"  requires  policy.residual_checkpoint
policy.collect_q           requires  policy.mode == "residual"
policy.jpeg_quality        ignored unless image_encoding == "jpeg"   (warn)
policy.chunk_steps != 0    requires  health["supports_predict_chunk"]
logging.rl_obs             requires  policy.mode == "residual"       (warn)
logging.level == "off"     conflicts with eval.enabled               (error)
```

`GET /api/config/schema` returns fields, types, defaults, and these rules.
The form is **generated** from that, so a new field never needs hand-written
HTML, and the dependency arrows are declared in one place.

### Presets kill the checkbox wall

90% of runs are one of a handful of setups. Ship them as YAML in
`trossen_real/configs/infer/`:

| preset | what it is |
| --- | --- |
| `bc_baseline.yaml` | ACT only, chunked, jpeg, video, metrics logging |
| `residual_eval.yaml` | residual + critics + full logging + eval session |
| `residual_actor_only.yaml` | residual, `collect_q: false`, full logging |
| `debug_local.yaml` | mock follower, no video, rerun live on |

The UI becomes: **pick a preset → see a diff of what it changes → optionally
override a few fields → Start.** The checkboxes still exist for the long tail,
behind an "Advanced" disclosure, generated from the schema.

This also gives you reproducibility: the chosen preset plus the override diff is
exactly what lands in `meta.json`.

---

## 6. The residual inference path

Your reading is right. Confirming it precisely: the residual forward happens
**inside the policy-client object, before the control loop sees an action.**

`infer_loop.py:282` is the only place the policy is consulted:

```python
result = self.policy.predict_full(obs)      # {"action": real units, ...}
```

and `app.py:256-284` already builds then optionally wraps that object
(`PolicyClient` → `ChunkedPolicyClient`). A third wrap is the whole change.

```text
ResidualPolicyClient.predict_full(obs, raw_images)
  1. base   = inner.predict_full(obs)         # ChunkedPolicyClient: usually a queue pop
  2. base_n = action_scaler.scale(base["action"])
  3. rl_obs = resize(raw_images, 84) + state_std + base_n
  4. feat   = agent._encode(rl_obs)           # 0.55 ms
  5. resid  = agent.actor(...).mean           # 0.19 ms   (eval_mode -> deterministic)
  6. q      = agent.critic(feat, ...)         # 0.64 ms   only if collect_q
  7. comb   = clamp(base_n + resid, -1, 1)
  8. action = action_scaler.unscale(comb)
  9. tap.put(record)                          # non-blocking
 10. return {"action": action, ...}           # superset of PolicyClient's dict
```

Because the return dict is a **superset**, `infer_loop.py` keeps working
unchanged — it reads `result["action"]`. Video, pedal intervention, the eval
session, dataset recording and the writers all sit outside this call.

Composition verified against the training path
(`real_residual_env.py:266`, `residual_env_wrapper.py:135`):
`clamp(base_naction + residual, -1, 1)` then `unscale`. `agent.act()` returns
the **residual only**; `eval_mode=True` takes `dist.mean`, so eval is
deterministic.

### `collect_q: false`

You are right that the critics are not needed to run the actor. With
`collect_q: false` the critic is **never constructed** — skip it in the loader,
not just in the forward — so VRAM drops by 23.6 M params and the Q columns are
written as null. The schema rule in §5 makes the dependency explicit, and
§7's recompute can fill Q in afterwards from `rl_obs.zarr` for any checkpoint,
so turning critics off during the robot run costs you nothing analytically.

That is the real argument for the two-tier design: **the robot run does not need
to be the thing that computes Q.**

### Raw frames

The RL encoder needs pixels, but `build_observation` hands `predict_full`
*encoded* images. Add an optional `raw_images=None` kwarg to `predict_full` on
all three client classes; `infer_loop` passes the frames it already read. Three
one-line signature changes, backward compatible, no encode-then-decode per tick.
There is no client-side decoder today (only `encode_image`), so the alternative
would mean writing one and paying ~1-2 ms/tick for nothing.

---

## 7. Q plots as a reusable util

`scripts/TrossenStation1Real/evaluate_checkpoints.py` (605 lines) already
produces exactly the figures you mean — `plot_metric_vs_checkpoint`,
`create_q_trajectory_plot_like_training`, `plot_all_checkpoints_q_trajectories`,
`plot_trajectory_qvalues`. It reads from the **replay-buffer cache**, which is
why it needs no normalizers: buffer rows are already standardized.

Do not rewrite it. Factor it:

```text
resfit/rl_finetuning/eval/
  load_agent.py      checkpoint -> (agent, scaler, standardizer, meta)   [strict=True]
  q_eval.py          (agent, obs batch) -> per-tick Q stats
  sources.py         read from a replay-buffer cache  OR  from a run record
  plots.py           the four figures above, moved verbatim, source-agnostic
  recompute.py       CLI: run records x checkpoints -> q_table.parquet
```

`sources.py` is the only new idea: both the buffer cache and `rl_obs.zarr` can
produce `(images, state_std, base_naction, action)` batches, so `plots.py`
works unchanged on sim buffers, real buffers, and live eval records. That is the
reuse you asked for, and it keeps the sim-style figures identical.

**Recompute on fixed trajectories.** The reason this is worth the indirection:
every checkpoint gets scored on the *same* logged states. Comparing Q curves
where each checkpoint drove its own rollouts conflates "the critic changed" with
"the states visited changed". With `rl_obs.zarr` you hold the trajectories fixed
and vary only the weights.

Cost: 10 checkpoints × ~7,000 ticks, batched — seconds of GPU time, no robot.

Figures, deliberately plain (the ACT eval plots were rewritten once for being
over-built):

1. Success rate vs residual checkpoint, ACT baseline as a reference line.
2. Cumulative sub-stage reach rate vs checkpoint.
3. Q over time within an episode, min/max band across heads, one line per checkpoint.
4. Q at episode start vs checkpoint.
5. Ensemble spread `q_std` over time — the honest epistemic signal for offline RL.
6. Residual magnitude per joint, and `clipped_frac`, vs checkpoint.

---

## 8. Eval protocol and resumability

### Already more general than you think

`eval_config.py` is already a dataclass + YAML loader with configurable `stages`
and poses from a file or a seed
(`trossen_real/configs/eval/pick_cube_and_insert_station3.yaml`). The 20 poses
are **data, not code**. A different task is a new YAML: different stage list,
different pose set, different `max_steps_default`.

So "generalisable eval" mostly needs two additions, not a redesign:

- **A sweep dimension.** Today one eval session = one checkpoint. Add
  `checkpoint_label` as a first-class axis so a sweep is one logical campaign
  with N×M runs, and `runs/index.parquet` can pivot on it.
- **Task-agnostic success.** Success is currently "furthest stage reached ==
  last stage". That generalises fine as long as stage lists stay declarative.

### Resumability — analysed, as asked, not proposed

You flagged this as low priority. My read: it is **cheap if designed in now,
expensive to retrofit**, so the plan should not *implement* it but should not
*preclude* it.

What makes it cheap: a campaign manifest written up front listing every
(checkpoint, pose) cell with a status, and `runs/index.parquet` keyed on
`(campaign_id, checkpoint_label, pose_index)`. Resume = "find cells not marked
done". That is ~30 lines and no new concepts, because the run id already exists.

What makes it expensive later: if runs are only identifiable by folder
timestamps, reconstructing which cells completed means parsing paths.

Recommendation: write the manifest and the index from day one, implement the
resume *command* later. The 3–5 hour sweep in §10 is exactly the thing you will
want to interrupt.

---

## 9. What I would deliberately not do

Scope discipline, since you asked not to over-complicate:

- **No new web framework.** Flask stays. The UI gets a schema-driven form and
  presets, not a rewrite.
- **No embedded Rerun viewer** in phase one. Separate window, launched by a
  button.
- **No teleop migration** in phase one. The new writer is additive; teleop keeps
  `DatasetRecorder` until the infer path has proven the schema. Then it is a
  small swap.
- **No MCAP / ROS bag.** They are good formats, but they buy interop this repo
  does not need and cost a dependency plus a reader.
- **No replacement of `log_setup.py`.** It works. It gets a `run_id` so its
  output joins the rest.
- **No change to the replay buffer.** `tensordict` + the hashed caches stay
  exactly as they are.
- **No retraining.** Backfill the 20 existing checkpoints instead (§1).

---

## 10. Phasing

Phases 1–6 are verifiable with **no robot**, which is the point of the ordering.

| # | deliverable | robot |
| --- | --- | --- |
| 1 | ~~hermetic checkpoints~~ **DONE** — see §1. No backfill: old checkpoints stay as they are, the retrain produces hermetic ones | — |
| 2 | `trossen_real/logging/`: schema, `RunRecordWriter` (queue + background thread), `meta.json`, `index.parquet`. Unit-test drop behaviour under a deliberately stalled sink | no |
| 3 | `resfit/.../eval/load_agent.py` with `strict=True`; assert 0 missing/unexpected on all 20 | no |
| 4 | `ResidualPolicyClient` + `raw_images` kwarg. Replay test: feed logged ACT-eval frames through it, assert residual bounded by `action_scale`, combined action inside the box, and that `collect_q=false` constructs no critic | no |
| 5 | Factor `eval/{sources,plots,q_eval,recompute}.py` out of `evaluate_checkpoints.py`. Validate by reproducing today's figures from the buffer cache byte-for-byte | no |
| 6 | Config schema + `validate()` + presets + `GET /api/config/schema`; generated form; `run_record_to_rerun.py` | no |
| 7 | One pose, one checkpoint, pedal in hand: verify Hz, `dt_*` columns, residual magnitude, clipping, dropped-log count = 0 | **yes** |
| 8 | Screening sweep: 5 poses × 10 checkpoints (~1 h), `collect_q: false`, Q filled in afterwards by recompute | **yes** |
| 9 | Full 20-pose sweep on the shortlist; figures | **yes** |

### Robot-time cost, stated plainly

10 checkpoints × 20 poses = **200 episodes**. 350 steps at 20 Hz is 17.5 s of
motion, but with reset and annotation realistically 60–90 s/episode →
**3.5–5 hours attended.**

Phase 8's screening pass is why `collect_q: false` matters: run 50 episodes,
recompute Q for all 10 checkpoints offline on those same trajectories, then spend
the full 20-pose protocol only on the shortlist.

---

## 11. Decisions I need

**D1 — ~~backfill or retrain?~~ RESOLVED: retrain, no backward compatibility.**
Implemented in §1. One action left for you: copy the updated `policy_server.py`
to the server and restart it before training.

**D2 — `rl_obs` always on?** 3.7 MB/episode. Recommend always on for eval,
configurable off for long teleop sessions.

**D3 — Rerun live by default?** 0.79 ms/tick, but it is a second process and a
window. Recommend default **off**, on for debugging, and always generate
`view.rrd` from the archive afterwards.

**D4 — migrate `TickLogger` call sites now or keep both?** `live_infer_deploy.py`
and `infer_loop.py` share it. Recommend the writer subsumes it and `TickLogger`
is deleted in phase 2 — two call sites, and keeping both guarantees drift.

**D5 — does the RL training loop adopt the run record too?** It has its own
`REAL_LOG_FILE`. Recommend yes, but **after** phase 7, and as a separate change
so a logging bug cannot interact with a training run.

**D6 — one `runs/` root for everything, or keep `eval_runs/` separate?**
Recommend one root with a `kind` field in `meta.json` (`infer` | `eval` |
`teleop` | `rl_online`). Separate roots are how you end up joining on
timestamps again.
