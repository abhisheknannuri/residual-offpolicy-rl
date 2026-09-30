# Can Task — Ablation Study Checklist

> **Methodology**: Change one variable at a time. Use 1-2 runs for **screening** (eliminate bad options), then 3-5 runs for **confirmation** (final config with statistical confidence).
>
> **Devices**: 🖥️ Server (critic warmup, no offline RL) | 💻 Laptop (offline RL configs)

---

## Round 0 — Critic Warmup & Robustness Baseline (🖥️ Server)

> **Goal**: (A) Does 10k critic warmup help? (B) How robust are we to a weaker base policy (AvgBC vs latest BC)?
> **How to read results**: Compare 1 vs 2 for warmup decision, compare 1 vs 3 for BC robustness.
> **Runs needed**: 2-3 per config is enough for screening.

| # | Config | Script | Run 1 | Run 2 | Run 3 |
|---|--------|--------|:-----:|:-----:|:-----:|
| 1 | No critic warmup (baseline) | `sparse_noCriticWarmup.sh` | [X] | [X] | [X] |
| 2 | 10k critic warmup | `sparse_criticWarmup.sh` | [X] | [X] | [X] |
| 3 | No critic warmup + AvgBC | `sparse_noCriticWarmup_withAvgBC.sh` | [X] | [X] | [X] |
| 4 | 10k critic warmup + AvgBC | `sparse_criticWarmup_withAvgBC.sh` | [X] | [X] | [X] |

**Decision (warmup)**: _______________________________________________
**Decision (robustness)**: _______________________________________________

---

## Round 1 — Offline RL Sampling Method (💻 Laptop, run FIRST)

> **Goal**: Find the best sampling config for the offline TD3-BC phase.
> **Baseline**: 25k steps, fixed_ratio, ratio=0.5, latest ckpt
> **Variable**: sampling method + ratio (one at a time from baseline)
> **Runs needed**: 2 per config for screening. Baseline already has Run 1.

| # | Config | Script | Run 1 | Run 2 | Run 3 |
|---|--------|--------|:-----:|:-----:|:-----:|
| 1 | fixed_ratio, 0.5 (baseline) | `sparse_with25kOffRL.sh` | [X] | [X] | [X] |
| 2 | proportional sampling | `sparse_with25kOffRL_proportional.sh` | [X] | [X] | [X-] |
| 3 | fixed_ratio, 0.25 | `sparse_with25kOffRL_ratio025.sh` | [X] | [X] | [X] |
| 4 | fixed_ratio, 0.75 | `sparse_with25kOffRL_ratio075.sh` | [X] | [X] | [X] |

**Decision**: Best sampling = _______________________________________________

---

## Round 2 — Checkpoint Loading + Step Sensitivity + Robustness (💻 Laptop)

> **Goal**: (A) Does `best` checkpoint fix overfitting compared to `latest`? (B) What is the optimal number of steps (25k vs 50k vs 75k)? (C) Do these findings hold for the weaker AvgBC model?
> **Baseline**: Best sampling from Round 1, latest ckpt, 25k steps
> **Variable**: load_ckpt (latest vs best) × steps (25k vs 50k vs 75k) × BC quality (latest vs AvgBC)
>
> ⚠️ **Before running**: Update `_best.sh` scripts with winning sampling from Round 1.

| # | Config | Script | Run 1 | Run 2 | Run 3 |
|---|--------|--------|:-----:|:-----:|:-----:|
| 1 | 25k, latest ckpt | `sparse_with25kOffRL.sh` | ← reuse Round 1 | | |
| 2 | 25k, best ckpt | `sparse_with25kOffRL_best.sh` | [ ] | [ ] | [ ] |
| 3 | 50k, latest ckpt | `sparse_with50kOffRL.sh` | [ ] | [ ] | [ ] |
| 4 | 50k, best ckpt | `sparse_with50kOffRL_best.sh` | [ ] | [ ] | [ ] |
| 5 | 75k, latest ckpt | `sparse_with75kOffRL.sh` | [--] | [ ] | [ ] |
| 6 | 75k, best ckpt | `sparse_with75kOffRL_best.sh` | [ ] | [ ] | [ ] |
| 7 | 25k, latest ckpt + AvgBC | `sparse_with25kOffRL_withAvgBC.sh` | [ ] | [ ] | [ ] |
| 8 | 25k, best ckpt + AvgBC | `sparse_with25kOffRL_withAvgBC_best.sh` | [ ] | [ ] | [ ] |
| 9 | 50k, latest ckpt + AvgBC | `sparse_with50kOffRL_withAvgBC.sh` | [ ] | [ ] | [ ] |
| 10| 50k, best ckpt + AvgBC | `sparse_with50kOffRL_withAvgBC_best.sh`| [ ] | [ ] | [ ] |
| 11| 75k, latest ckpt + AvgBC | `sparse_with75kOffRL_withAvgBC.sh` | [ ] | [ ] | [ ] |
| 12| 75k, best ckpt + AvgBC | `sparse_with75kOffRL_withAvgBC_best.sh`| [ ] | [ ] | [ ] |

**Decision**: Best ckpt strategy = ________________ Best steps = ________________

---

## Round 3 — Listed Res Scale (💻 Laptop)

> **Goal**: Does per-dimension action scale outperform scalar 0.2?
> **Baseline**: Best config from Rounds 0-2 (e.g. 50k best ckpt)
> **Variable**: scalar vs listed res scale
>
> ⚠️ **Before running**: Update these scripts with all winning params from previous rounds.

| # | Config | Script | Run 1 | Run 2 | Run 3 |
|---|--------|--------|:-----:|:-----:|:-----:|
| 1 | Best config (scalar 0.2) | (winner from Round 2) | ← reuse Round 2 | | |
| 2 | Best config + listedResScale | `sparse_with{25k/50k/75k}OffRL_listedResScale.sh` | [ ] | [ ] | [ ] |

**Decision**: _______________________________________________

---

## Round 4 — Final Confirmation (5 seeds)

> **Goal**: Get statistically robust results for the final best config.
> Run 5 seeds of the single best configuration from all rounds above.

| # | Config | Script | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|--------|--------|:-----:|:-----:|:-----:|:-----:|:-----:|
| 1 | Final best config | (TBD from above) | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## Final Best Config

| Parameter | Value |
|---|---|
| Critic warmup | |
| BC checkpoint | |
| Offline RL steps | |
| Sampling method | |
| Offline ratio | |
| Checkpoint loading | |
| Action scale | |

---

## Run Count Summary

| Round | Configs | Runs/config | Max total runs | Purpose |
|---|---|---|---|---|
| 0 (server) | 4 | 2-3 | ~10 | Screening: warmup + BC robustness |
| 1 (laptop) | 4 | 2 | ~8 | Screening: sampling method |
| 2 (laptop) | 12| 1-2 | ~20 | Screening: ckpt loading + steps + robustness |
| 3 (laptop) | 2 | 2-3 | ~6 | Screening: action scale |
| 4 (laptop) | 1 | 5 | 5 | Confirmation: final config |
| **Total** | | | **~49** | |

---

## Notes

- For each run, change `_Run1` → `_Run2` etc.: `WANDB_NAME="..._Run2" bash script.sh`
- Seed is fixed at 42 by default. For different seeds: `SEED=123 bash script.sh`
- All scripts log to WandB project `robomimic-can-residual-td3-ablation-studies`
- **[--]** = currently running
