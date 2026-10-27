# SARM stage and success detection in the inference app

The infer app can ask a trained SARM model which task stage the robot is in,
and show it live in the web UI, with a SUCCESS badge once the cube is
inserted. Off unless enabled in the station config.

Code: `trossen_real/inference/stage_detector.py` (client),
`infer_app/infer_loop.py` (per tick), `infer_app/app.py` (connect, health,
recording schema), `infer_app/static/` (stage panel). Server: opensarm
`reward_server.py` (multi-camera since 2026-10-22).

## How it works

- Every control tick, the loop hands the tick's frames (the cameras the SARM
  model uses: `cam_specialist`, `cam_left_wrist`, `cam_right_wrist`) and the
  joint state to the detector. That only appends to a list (~0.1 ms).
- Every `query_every` ticks (default: the ACT chunk size, `policy_chunk_steps`,
  or the policy server's `n_action_steps` if not chunked), a background thread
  sends the frames since the last query and asks the server to score the
  newest one (`num_anchors=1`). All frames are sent because the model's
  window looks back over the episode (8 frames, 8 ticks apart).
- The control loop never waits for the server. The UI and the recorded
  columns show the latest result, which lags by up to one query.
- A new episode (Start Inference) resets the server session.
- If the server errors or is unreachable, the error is shown in the stage
  panel and the stage stops updating; inference carries on.

**Detection rule:** a stage counts as detected when the last `consecutive`
queries (default 2) all predicted it with confidence above `min_conf`
(default 0.85). It is live: the next query that breaks the rule clears it.
**Success** = `success_stage` (4, "cube inserted") is detected.

Measured on the laptop (RTX 4090 Laptop): 37-53 ms per query with 3 cameras
and 15 frames sent, ~3 GB GPU memory for the server. A 20 Hz replay of
recorded episodes: success shown 0.75 s (one query) after "inserted" was
first predicted; the episode where the cube was left tipped never showed
success.

## Station config

```yaml
stage_detector:
  enabled: false
  server_url: "http://127.0.0.1:5080"  # 5060 = follower server, 5070 = policy server
  query_every: null        # ticks between queries; null = the ACT chunk size
  consecutive: 2
  min_conf: 0.85
  success_stage: 4         # "cube inserted"
  target_stage: null       # e.g. 2 ("cube above the hole"): shown in the UI only
  record_stage: true
  label_reward_on_success: false  # next.reward/next.done = 1 while success holds
  end_episode_on_success: false   # end the run on success (stop_reason "sarm_success")
```

When `enabled`, Connect fails if the reward server is not reachable, does not
report its cameras/stages, or needs a camera that is not a real camera of the
station.

## Running it

1. Reward server (opensarm, the config's `eval.ckpt_path` set to the
   checkpoint folder with plain spaces). Run it from the **opensarm** folder:
   `uv run` there uses opensarm's own venv. Never run `uv run`/`uv sync` in
   `residual-offpolicy-rl`: it re-syncs `.venv` and upgrades torch (fix:
   `docs/setup/UV_SERVER_SETUP.md` §4, "Repair the torch stack").
   ```bash
   cd <opensarm>
   CUDA_VISIBLE_DEVICES=0 uv run --with flask python reward_server.py \
       --config config/sarm_Station3_PickCubeAndInsert.yaml --host 127.0.0.1 --port 5080 --device cuda:0
   ```
2. Set `stage_detector.enabled: true` in the station config.
3. Start the infer app and connect as usual. The stage panel appears above
   the camera views.

## Recorded datasets

With dataset recording and `record_stage: true`, three columns are added to
every frame: `observation.sarm_stage` (int64, -1 before the first result),
`observation.sarm_stage_conf` (float32), `observation.sarm_success` (bool).
This changes the dataset schema: datasets recorded with them merge with older
ones only after dropping these columns (as with `cam_specialist`).

`label_reward_on_success` / `end_episode_on_success` are off: watch the
detector in the UI first. The reward and reset pedals keep working as before
in every case.

## Tests

`trossen_real/inference/tests/test_stage_detector.py` (fake server): query
cadence and payload, the consecutive/min_conf rule including clearing, stale
results dropped on a new episode, errors reported instead of raised.

## Residual RL from a SARM stage on

With a residual RL checkpoint loaded (connect panel), the residual can be
limited to the part of the task it was trained for:

```yaml
stage_detector:
  residual_from_stage: 2      # null (default) = residual for the whole episode
  residual_consecutive: 2
  residual_min_conf: 0.85
```

- Pure ACT until `residual_consecutive` SARM queries in a row predict a stage
  >= `residual_from_stage` with confidence > `residual_min_conf` (a stage-3
  answer counts for a stage-2 gate). The RL network is not run while off.
- Then the residual stays on until the episode ends, even if SARM later dips
  back (a stage-2 residual was trained on everything after the first stage-2
  detection, recoveries included). A new episode starts gated again.
- Connect fails if `residual_from_stage` is set with a residual checkpoint but
  the stage detector is disabled.
- The stage panel shows "residual RL: ON / OFF, pure ACT (from stage N)"; the
  residual tick log (`logs/residual/*.jsonl`) has `residual_active` per tick.
- The reward server pins the first 10 frames of an episode to stage 0, so with
  `query_every` < 10 the first queries say 0 and the residual can start no
  earlier than ~2 queries after tick 10.

Code: `StageDetector` (gate), `ResidualPolicyClient.residual_active`
(passthrough), `InferenceLoop._run` (sets it per tick). Tests:
`test_stage_detector.py`, `test_residual_gate_passthrough.py`.
