# Making remote inference run at 20 Hz

## The problem, measured

Run of 2026-09-30 12:25, `logs/infer_app/2026-09-30/12-24-56/app.log`:

```
policy:5070  /predict   calls=30  avg=475ms  p50=519ms  max=1306ms
follower     /getstate  calls=31  avg=3ms
follower     /move_...  calls=30  avg=3ms
achieved_hz  0.76 - 1.7
```

**30 predicts, 30 moves, 30 ticks: one network round trip per control tick.**
Everything local is 3 ms. The rate was exactly `1 / RTT`.

The GPU was never the bottleneck. `/predict` calls `ACTPolicy.select_action()`,
which runs the model only when its internal queue is empty and pops a cached
step otherwise (`modeling_act.py:127-133`). With `n_action_steps=15` the model
ran **twice in those 30 calls**; the other 28 uploaded 512 KB of images to
trigger a `deque.popleft()` on the far side.

## Three fixes

| # | what | measured effect |
| --- | --- | --- |
| 1 | HTTP keep-alive on both ends | one TCP connection instead of one per tick |
| 2 | `/predict_chunk` + client-side queue | round trips 45 -> 3 over 45 ticks (15x) |
| 3 | compressed image payload | 512 KB -> 34 KB per tick (15x) |

All three are **off by default**; nothing changes until you opt in.

### 1. Keep-alive (always on, no flag)

`policy_server.py`'s handler never set `protocol_version`, so it defaulted to
**HTTP/1.0** and closed the socket after every response; the client used bare
`requests.post()` rather than a `Session`. Both were needed - either one alone
forces a new connection, and through an SSH tunnel a new connection also means
a new tunnelled channel before any payload moves.

### 2. Chunked fetching (`INFER_POLICY_CHUNK_STEPS`)

The server already chunked; this moves the queue to the client. **The actions
are identical** - `select_action()`'s refill is literally
`predict_action_chunk(batch)[:, :n_action_steps]` then `popleft()`, which is
what `/predict_chunk` returns. It is *not* prefetching: the next chunk is
requested only once the current one runs out, from a fresh observation, so the
policy never sees a staler observation than before.

Refused (rather than silently returning different actions) for
`temporal_ensemble_coeff` checkpoints and `use_relative_actions=True`;
`/health` reports `supports_predict_chunk`.

Cost: the loop pauses one round trip per chunk. That is why fix 3 matters.

### 3. Image payload (`INFER_IMAGE_ENCODING`)

Measured on 16 real 256x256 wrist-cam frames, 2 images per observation:

| encoding | wire | encode | lossless | mean abs err |
| --- | --- | --- | --- | --- |
| `raw` (default, unchanged) | 512 KB | 0.33 ms | yes | 0 |
| `zlib` | 182 KB | 2.43 ms | **yes** | 0 |
| `jpeg` q95 | **34 KB** | 0.72 ms | no | 0.79 |
| _AV1 crf30 - what the dataset is stored in_ | - | - | no | _1.39_ |

**On JPEG being lossy:** the LeRobot dataset stores images as AV1 at crf=30, so
the policy was *trained* on lossily-compressed frames - it has never seen
pristine pixels. JPEG q95 distorts **less** than that AV1 pass (mean abs err
0.79 vs 1.39 on 0-255; PSNR 45.8 vs 43.1 dB), so it does not move inference
further from the training distribution than training already was. Below ~q85
that stops holding - re-measure before going there. Use `zlib` if you want
bit-identical pixels and will take 3.3x instead of 15x.

## `INFER_POLICY_CHUNK_STEPS` values

It is a plain integer, not a two-value switch:

| value | meaning |
| --- | --- |
| `0` | **default.** Per-tick `/predict`, exactly the old behaviour. |
| `-1` | Chunked. Length = the server's own `--n-action-steps`. **Use this.** |
| `15`, `10`, ... | Chunked, client overrides the length (server caps it at `chunk_size`). |

`-1` is preferred because then `--n-action-steps 15` on the server stays the
single source of truth - no second number to keep in sync.

With `--n-action-steps 15` and a checkpoint whose `chunk_size` is 20: the model
predicts 20 actions, the server slices `[:15]` and sends 15, the client executes
those 15 one per tick, then asks for the next 15 from a fresh observation. The
other 5 are discarded - exactly as `select_action()` already discarded them.

---

# Testing, step by step

Four runs. Each adds ONE change, so a regression has one suspect. Do a short
run each time (~100 ticks is plenty) with the same cube pose.

Baseline is already on disk - the slow run from 12:25 today:

```
infer_..._091700.jsonl  340 ticks  median 20.0 Hz   <- local policy server
infer_..._122133.jsonl   86 ticks  median  3.2 Hz   <- remote server, old code
infer_..._122516.jsonl   30 ticks  median  2.6 Hz   <- remote server, old code
```

## Step 0 - put the new server on the remote machine

The only file that changed there is `policy_server.py`. From this machine:

```sh
scp /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/custom_scripts/policy_server.py \
    <you>@<remote>:~/Research/LEROBOT/lerobot/custom_scripts/policy_server.py
```

Then on the remote machine, same command as always:

```sh
cd ~/Research/LEROBOT/lerobot
uv run python custom_scripts/policy_server.py \
    --checkpoint /home/qte9489/Research/LEROBOT/lerobot/outputs/train/ACT_BC_PickCubeAndInsert_Station3_Trial1_deltajoint/checkpoints/055000/pretrained_model/ \
    --n-action-steps 15
```

**Confirm the new code is actually running** before anything else - a stale copy
would make every later step look like a failure:

```sh
curl -s http://localhost:5070/health | python -m json.tool
```

Must contain both of these. If either is missing, the old file is still running:

```json
"supports_predict_chunk": true,
"supported_image_encodings": ["raw", "zlib", "jpeg"]
```

## Step 1 - keep-alive only

Local machine, terminal 1:

```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
python -m trossen_real.follower.follower_single_server --config trossen_station3_single --port 5060
```

Local machine, terminal 2 - **no env vars**:

```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
python -m trossen_real.infer_app.app --port 5080
```

Open the UI, enter the policy server URL, Connect, run ~100 ticks.

**What to look for** in `logs/infer_app/<date>/<time>/app.log`:

```
Policy fetching: per-tick (one /predict round trip per control step)
Camera frames -> policy server encoded as 'raw'
...
policy:5070  /predict  calls=100  avg=???ms
```

`calls` still equals the tick count (nothing changed there). `avg` should be
**below the 475 ms baseline** - that drop is the handshake you are no longer
paying. This is the only step where you cannot be sure how big the win is in
advance; it depends entirely on your link.

## Step 2 - add JPEG

Stop the infer app (Ctrl+C), restart it with:

```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
INFER_POLICY_CHUNK_STEPS=0 INFER_IMAGE_ENCODING=jpeg \
python -m trossen_real.infer_app.app --port 5080
```

**What to look for** in `app.log`:

```
Camera frames -> policy server encoded as 'jpeg' (quality 95)
policy:5070  /predict  calls=100  avg=???ms      <- should drop sharply
```

Payload goes 512 KB -> 34 KB per tick. **Also watch the robot**: same reaching,
same grasps. If behaviour looks different, rerun with
`INFER_IMAGE_ENCODING=zlib` - that is bit-identical to `raw` (proven by
`test_image_encoding.py`), so if the difference persists it is NOT the images.

## Step 3 - add chunking

```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
INFER_POLICY_CHUNK_STEPS=-1 INFER_IMAGE_ENCODING=jpeg \
python -m trossen_real.infer_app.app --port 5080
```

**What to look for** in `app.log`:

```
Policy fetching: CHUNKED, server default actions per round trip (server n_action_steps=15)
Camera frames -> policy server encoded as 'jpeg' (quality 95)
...
policy:5070  /predict_chunk  calls=7   avg=???ms
```

The line that proves it worked: **`/predict_chunk` replaces `/predict`, and
`calls` is about `ticks / 15`** - 100 ticks should be ~7 calls, not 100.
If you still see `/predict calls=100`, the env var did not reach the process.

## Reading the result

One command, all four runs, oldest first:

```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
.venv/bin/python -m trossen_real.scripts.infer_latency_report \
    logs/infer_trossen_station3_single_20260930_122516.jsonl \
    logs/infer_trossen_station3_single_<step1>.jsonl \
    logs/infer_trossen_station3_single_<step2>.jsonl \
    logs/infer_trossen_station3_single_<step3>.jsonl
```

```
log                          ticks  median   mean    p10    p90    min % of 20Hz
..._122516.jsonl (baseline)     30     2.6    2.6    1.5    4.4    1.2      13%
```

`median` is the headline. **Watch `p10` too**: with chunking the loop runs at
20 Hz and then pauses once per chunk for the round trip, so a high median with a
low `p10` means chunking is working but the round trip is still expensive - the
fix for that is Step 2, not more chunking.

Or just the newest log:

```sh
.venv/bin/python -m trossen_real.scripts.infer_latency_report "$(ls -t logs/infer_trossen_station3_single_*.jsonl | head -1)"
```

## Offline tests (no robot, no server)

```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
.venv/bin/python -m pytest trossen_real/inference/tests/ trossen_real/infer_app/tests/ -q
```

89 tests. `test_policy_chunking.py` proves chunked and per-tick produce the
**identical** action sequence against a stand-in server that mimics
`select_action()`'s queue. `test_image_encoding.py` round-trips every encoding
through the real `_decode_image()` from `policy_server.py` and checks the red
cube is still red - an RGB/BGR swap is the one failure that would look like
"the checkpoint got worse" and nothing else.

## If something breaks - rolling back

Each step is one env var, so drop them right to left:

| symptom | try |
| --- | --- |
| robot behaves differently in step 2 | `INFER_IMAGE_ENCODING=zlib` (lossless) |
| anything odd in step 3 | drop to `INFER_POLICY_CHUNK_STEPS=0` |
| still wrong with no env vars | the server file - `git`-less backups are `*.bak` next to each edited file |

## What was NOT done

**Prefetching the next chunk during execution.** It would hide the round trip
entirely, but the chunk would be computed from an observation taken ~1 RTT
before it starts executing - staleness the policy was not trained for. That is
what pi-0.5-style real-time chunking handles, and this checkpoint has no such
training. Left out deliberately.
