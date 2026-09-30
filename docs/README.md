# Documentation index

| | |
| --- | --- |
| [overview/](overview/ResFi-ResidualFine-tuningWithOff-PolicyRL.md) | what ResFiT is and how the pieces fit together |
| [setup/](setup/) | environment setup: [uv](setup/UV_SERVER_SETUP.md), [docker](setup/DOCKER_SETUP.md), [known fixes](setup/SETUP_FIXES.md) |
| [policies/](policies/) | the base BC policy: [ACT architecture](policies/ACT_ARCHITECTURE.md), [its transformer](policies/ACT_TRANSFORMER.md), [vision backbone](policies/VISION_BACKBONE.md), [training it](policies/BC_POLICY_TRAINING.md) |
| [algorithms/](algorithms/) | the RL side: [TD3](algorithms/TD3_ALGORITHM.md), [residual learning](algorithms/RESIDUAL_LEARNING.md), [critic losses](algorithms/CRITIC_LOSSES_EXPLAINED.md), [n-step returns](algorithms/NSTEP_RETURNS.md), [value function and warm-up](algorithms/VALUE_FUNCTION_AND_WARMUP.md), [replay buffers](algorithms/REPLAY_BUFFERS.md), [action normalization](algorithms/ACTION_NORMALIZATION.md), [LayerNorm vs grad clip](algorithms/LAYERNORM_VS_GRADCLIP.md), [this QAgent vs IBRL's](algorithms/ORIGINAL_VS_IBRL_QAGENT.md) |
| [training/](training/) | running it: [residual RL training](training/RESIDUAL_RL_TRAINING.md), [TD3 end to end](training/TD3_TRAINING_END_TO_END.md), [checkpointing and resume](training/CHECKPOINTING_AND_RESUME.md), [experiment guide](training/EXPERIMENT_GUIDE.md), [W&B practices](training/WANDB_BEST_PRACTICES.md) |
| [rewards/](rewards/) | [reward and success](rewards/REWARD_AND_SUCCESS.md), [reward-model integration](rewards/REWARD_MODEL_INTEGRATION.md), [stage-aware PBRS pipeline](rewards/STAGE_AWARE_REWARD_PIPELINE.md) |
| [data/](data/DATASET_GUIDE.md) | dataset formats and conversion |
| [real/](real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md) | human-in-the-loop residual RL on the real robot |
| [archive/](archive/) | superseded notes, kept for provenance |

Docs that live beside the code they describe are **not** here on purpose - they
get updated, whereas docs in a distant folder rot:

* `trossen_real/` - the real-robot stack ([README](../trossen_real/README.md),
  [architecture](../trossen_real/REAL_TRAINING_ARCHITECTURE.md),
  [inference latency tuning](../trossen_real/INFER_LATENCY_TUNING.md),
  [eval mode](../trossen_real/infer_app/EVAL_MODE_PLAN.md),
  [pedal behaviour](../trossen_real/human_intervention/PEDAL_BEHAVIOR.md))
* `scripts/TrossenStation1Real/`, `scripts/TrossenStation3Real/` - per-station runbooks
* `../DEPENDENCIES.md` - the five vendored clones and their pins
* `../WORKSPACE_CLEANUP_PLAN.md` - repo/environment cleanup status
