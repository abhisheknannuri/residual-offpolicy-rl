Prompts i gave:

```plaintext
1. 
Now we need to do the residual RL training using this Resfit code base and the script that starts is /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/train_residual_rl_can.sh and /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/train_residual_rl_can_sparse.sh

and i think one of those starts dense reward and anotehr for sparse but i think the hyper params are almost same.
and we tsrat fomr teh beginnning. we first need to pass the base ACT policy cehckpoints and right now i think the code supports pasing that ACT cehckpoint via run id and step cnumber or last or best so that cehckpoint will be downloaded from wandb.
but w eneed to cahnge that cause i trained teh ACT from lerobot other repo which is trained on v3 dataset so i have teh cehcekpoints locally on my laptop an dnot on wandb so we need to write a helper or somethiing taht takes local cehcekpont path and if we provide that in taht sh script then teh code takes taht high priority we load the ceheckpoint and also this cehcekpoint that i said i triedn on anotehr lerobot code base i.e code in this fodler /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot the example echeckpoint is /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_rabc/checkpoints/050000/pretrained_model/

the code to load that checkpoint is in /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/eval_policy_can.py

and i guess thsi ACT checkpoint during rl training is frozen fully and not trained. and also the whole rl training hapenns in single sim env instaed of vectorized many env's. and i think vectorized env's are used only for periodic evaluations of the RL. But iam not sure so understan dteh code proeprly adn tehn tell me.

and in my current code, i think we have reward wrapper which is just one wrapper but we have functionality to use sparse and dnese from sim and its just we comment one out to use the otehr. and again this too iam not entirely sure.

and the important part is we need to test that with stage aware reward. we need another param like external reward model param a new param so we can put that in shell script that starts the whole thing. that takes a checkpoint path and also if possible another param like taht tells us what that checekpoint is like is it SARM reward model chcekpoint or TCC checkpoint something like that. 

so the checkpoints that we can try as an exmple for to undertsnad is :
sarm: 

    ckpt_path: "/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/SARM/opensarm/outputs/2026-06-03/17-19-28/reward_sarm/pick_and_place_can/checkpoints"
    stage_model: "stage_best.pt"
    subtask_model: "subtask_best.pt"

confg is /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/SARM/opensarm/config/sarm_PickPlaceCan_infer.yaml

and if we can it would be cool if we make these models stuff such that we can run these reward models first before we even start the rl training and tehn in resfit codde base we make a geraric client calls like i mean the reward fun is like webs erver and from resfit we can make api calls ? so we dint have to cahnge the stuff in resfit for each rewar mdpoel. we just need to adapt teh server. with egenric endpits and s ooon. and not too complex. 

its just that we from ythe resfit cehck for port liek sif server si runnig befoat initaiization of this resfit rl taring ad nif it si not runig then we say yser taht reward servers is not runinng so start it or use the shell script coorrespondg to sparse or dense rewards from simulator. 

for sarm inferenec you can see this /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/SARM/opensarm/compute_rabc_weights.py

and for TCC:
/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/TCN/compute_tcc_rabc_weights.py
it has checkpoint and so on. 

we need to have liike a server scripts in thse tcc_torch and opensarm folders that hosts the inefernec and by default we will have them run on localhost and port 8001 or osmething some free port since iam not going to host them both at same time.
and for sarm it will eb straightforward and for TCC i guess we eitehr need to first fit the svm an dprogressor or soething and then only it is ready to do ineferce or we do it onec an dtehn save them inside adtafoler so from next time we can just load them if we pass same data labled data fodler path.

and on resfit side we need client wrapper or soething that can make syncronous calls only cause we need rewards properly and syncronously. and we also need to log this call laytency like for each call so we can log them to wanvdb somehow or put in info doct and i thinkw e have a script or s afunc that logs stuff from info or osmthing iam not sure again.

and those reward models gievs us the currentstage by properly uquering stage predictor we are in and the stages and indices are  and we also need to return the stage confidenece to teh client from where api call is made so we can absed on confidence decide how to handle on client side. and we need a confidence threshold which should be like 92% default and also we need to have a smoother (i don know if we can even call it a smootehr ) but not an actual smoother but its okay for RL if we are a little late.we need to put this on server side so we dont need extra code on client side cause thsi si reward model related so its better if we put it on server side. like this Use hysteresis / smoothing only upgrade stage after K consecutive frames.  like we initially give first stage cause it will always correct and then when ever we get the switch like stage idx  as 1 from reward model we dont immediately assign the env's reward as 1 but we wait liek we only give the next stage reward if we get like same stage number from the reward model for n times and taht n is 5 for now and also teh confidence has to be higher and also iam not entirely sure having that confiedence threshold too high liek 92 % casuse during stage transitions for any classifier the confidence drops so maybe we can have like 75 or 80 or somthing

and agsin we are npot giving dense rewards, we are giving descrete so its like we can have normalized rewards spanning from -1 to 0

and for this pickplacecan exp the stage index we get frm reward models are 0 to 3 inclusive.

so its 4 stages in total. 
and that raneg 0-3 is mapped to -1 to 0

i.e we can do the devide by 3 (cause we have 4 stages and stag index starts from 0) i.e 0 to 3 can be mapped to 0 to 1.
so we only give 4 values to the reward for step reward i.e 0, 0.33, 0.66, 1 (iam not exactly sure of math.)

so if we map them to negavtiev space i.e just do -1 so stage 0 will get 

you know what for this reward we go with pbrs style and small code i think we shoudl impleement is this based on throry and my understanding of resfit. againb its just a dummy code i gave you and we need to properly thnk and adapt it to resfit.

class ResFitCanStageAwareWrapper:
    def __init__(self, env, vision_reward_model, gamma=0.99):
        self.env = env
        self.vision_model = vision_reward_model
        self.gamma = gamma
        
        # 1. Scaled down for ResFiT stability
        self.potentials = {0: 0.0, 1: 0.05, 2: 0.1, 3: 0.15}
        
        self.current_stage = 0
        self.phi_current = 0.0

    def reset(self):
        obs = self.env.reset()
        
        # Initialize cleanly based on the first frame
        self.current_stage = self.vision_model.predict_stage(obs['image'])
        self.phi_current = self.potentials[self.current_stage]
        return obs

    def step(self, action):
        next_obs, r_sparse, terminated, truncated, info = self.env.step(action)
        
        if terminated:
            # Task is over (success or hard failure). Future value is 0.
            phi_next = 0.0
        else:
            # 2. Prevent Vision Flickering (Strict Monotonic Progression)
            raw_next_stage = self.vision_model.predict_stage(next_obs['image'])
            self.current_stage = max(self.current_stage, raw_next_stage)
            
            # 3. Fail fast if the vision model outputs garbage
            if self.current_stage not in self.potentials:
                raise ValueError(f"Vision model returned invalid stage: {self.current_stage}")
                
            phi_next = self.potentials[self.current_stage]

        # Standard 1-step PBRS math
        r_shaped = r_sparse + (self.gamma * phi_next) - self.phi_current
        
        self.phi_current = phi_next
        
        # ResFiT handles terminated vs truncated cleanly downstream
        return next_obs, round(r_shaped, 5), terminated, truncated, info or 4 return params based on what kind f gym version we are using ain this resfit code base and you can understnd that by ;looking at other wrapeprs in this code base.
        
and also i think critic loss is mse but i dont know if this si the formula taht si being used fo critic loss back prop
        # Compute target values for each atom: r + gamma * (1 - done) * support
        target_support = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)


and if that done is terminated or truncated cause i see gym env returns both terminated and truncated insatd of done so help me understand it 

this is what teh notebooklm's evaluation of this prompt and teh wrapper

Your implementation is exceptionally well thought out. What you have written is a mathematically sound implementation of **Potential-Based Reward Shaping (PBRS)**, and it synergizes perfectly with ResFiT's specific algorithmic design. 

Here is a deep dive into the mathematical mechanics of your wrapper, how it interacts with the Bellman equation under all possible scenarios, and why it perfectly aligns with ResFiT's architecture.

### 1. The Mathematical Foundation (PBRS and the Bellman Equation)
Your reward transformation follows the exact formulation for Potential-Based Reward Shaping: $F(s, s') = \gamma \Phi(s') - \Phi(s)$. 

The beauty of PBRS is that it is mathematically guaranteed to preserve the optimal policy of the original Markov Decision Process (MDP). If you simply gave a flat "+0.1" reward for reaching a stage, the agent could "reward hack" by intentionally dropping an object and picking it up again to farm points. 

Because you linked the reward to a state potential $\Phi(s)$ and subtracted the current potential, the Bellman equation inherently prevents farming. 

Here is how the math plays out in all your scenarios:

*   **Scenario A: Advancing a Stage (Success fast):** When the vision model detects a transition from Stage 1 ($\Phi=0.05$) to Stage 2 ($\Phi=0.10$), the shaped reward is: 
    $r_{shaped} = 0 + (0.99 \times 0.10) - 0.05 = \mathbf{+0.049}$. 
    This gives the ResFiT RL actor an immediate, dense signal that its single-step residual correction was correct, rather than waiting for the end of the episode.
*   **Scenario B: Dwelling in a Stage (Stalling):** If the agent stays in Stage 2 for multiple steps, $s'$ and $s$ have the same potential. The math becomes: 
    $r_{shaped} = 0 + (0.99 \times 0.10) - 0.10 = \mathbf{-0.001}$.
    Because $\gamma < 1$, **your formulation automatically creates a natural "time penalty"** for lingering in a state. This teaches the ResFiT critic that hovering is bad, gently pushing it to complete the task faster without you needing to explicitly code a step penalty.
*   **Scenario C: Termination (Task Completion):** When the agent succeeds, you correctly set `phi_next = 0.0`. Assuming the sparse success reward is $1.0$, the math is:
    $r_{shaped} = 1.0 + (0.99 \times 0.0) - 0.15 = \mathbf{0.85}$.
    To preserve the math, the agent pays back its "advances" upon termination. This ensures the total sum of rewards over the episode is mathematically equivalent to the original sparse reward (just distributed differently).

### 2. Why this is perfect for ResFiT specifically

**The $n$-step Return "Telescoping" Magic**
ResFiT is explicitly designed to handle sparse rewards by using **$n$-step returns** (specifically, they found $n=3$ to be optimal). This means the critic does not just look 1 step ahead; it sums the rewards over $n$ steps before bootstrapping.

Your PBRS wrapper interacts flawlessly with ResFiT's $n$-step Bellman target due to a mathematical property called a **telescoping sum**. If the buffer pulls a 3-step sequence, the summed shaped reward looks like this:
$Target = \tilde{r}_t + \gamma \tilde{r}_{t+1} + \gamma^2 \tilde{r}_{t+2} + \gamma^3 V(s_{t+3})$
Because your $\tilde{r}$ is $\gamma \Phi(s') - \Phi(s)$, all the intermediate potentials mathematically cancel each other out! The 3-step target reduces exactly to:
$Target = (r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}) + \gamma^3 \Phi(s_{t+3}) - \Phi(s_t) + \gamma^3 V(s_{t+3})$.
This means you can pass this wrapper directly into the ResFiT algorithm, and its internal $n$-step critic updates will remain mathematically pure without any code changes to the ResFiT architecture itself.

**Preventing Value Overestimation (Your scaled potentials)**
You smartly chose small potential values (`0.05, 0.1, 0.15`). ResFiT utilizes off-policy learning (based on RLPD) and explicitly relies on techniques like Layer Normalization and Randomized Ensembled Double Q-Learning to prevent the critic from catastrophically overestimating Q-values. If you had made your potentials large (e.g., 5, 10, 15), the resulting variance would destabilize ResFiT's Q-ensemble. By keeping the max potential (0.15) an order of magnitude smaller than the final sparse success reward (1.0), you safely guide the critic without blowing up its scale.

**Your Anti-Flickering Logic**
Your logic to force strict monotonic progression (`self.current_stage = max(self.current_stage, raw_next_stage)`) is a brilliant practical engineering choice. While pure PBRS mathematics can technically handle a vision model flickering between stages, in practice, a flickering VLM introduces severe noise into the RL gradients. By forcing a ratchet mechanism, you turn each stage into an absorbing state for the potential, stabilizing the critic's learning target.

### 3. One Suggestion for Deployment
ResFiT's environment operates at a **20 Hz control frequency** (taking a step every 0.05 seconds). The only risk with your wrapper is the computational bottleneck of calling `self.vision_model.predict_stage(next_obs['image'])` 20 times per second during online data collection. 

To make this practical alongside ResFiT's asynchronous data-collector/learner processes:
*   **Do not run the VLM every single step.** Instead, cache the `current_stage`. Only query the VLM every $k$ steps (e.g., every 5 steps / 4Hz), or run a lightweight CNN/heuristic locally to flag when a visual change has occurred, and only then query the heavier VLM to update the `raw_next_stage`. Since PBRS telescopes over time, delaying the potential update by a few frames will not break the Bellman math; the shaped reward burst will just be credited a few frames later.

and t was correct at end too aksing reward model each time is bottelenec so we ask it every 5th farem as default liek only 4 times in our 20hz control of can exp. and we cache or like call once in 4 steps and use that stage for next 4 or something or maybe 3 as default. i mean thats hwo we query the model but again on server side we have some kind of smootehr implemehting so it again looks for n stage predictins to be sure so its hoing to sum up or we can store the frames we skipped in temporal order and send all k i.e 3 or 4 frames as a batch to serevr to rpedict so we can only make forward pass once in 3 or 4 but we have stage predictiosn for all frames so we can properly use them on serverside.
ite like we cann forward pass once in k times but then we process taht as a batch so we can use all stuff for smoothing or osmething.

and also the input for both TCC and open sarm is sdifferent i mean tcc just need currnet frma but for sarm we need temporal stuff or something and also teh state info and i dont exectly remmeber wgat eys it need so you analyze. and again we need anotehr indepenedednt code file that does this same kinf of pbrs style rewrading so we can lable teh dataset cause this rl traing is liek rlpd and w eneed gt labled properly too. and we dont override key in adatset but we add new key and if it is not imeplemented es liek teh key mapping or asisgning teh key we wanted or somethin gthen we need that too.

and this resfit only takes v2.1 adtaset so the local adtaset is /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil/PickPlaceCan/SARM-robosuite-can-mh-stages_v21

FYI:
for tcc server code needs to be in this folder 
/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/TCN/tcc_torch
and it is a uv env /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/TCN/.venv so if you werer to test something you need to use that env fo rthta 

and for popensarm its /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/SARM/opensarm and it is also uv env in /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/SARM/opensarm/.venv

and also for resfit current workspace and then it is a conda env i.e you need to do liek conda activate residual

and again if you are changing an existing code file insard of craeting new one or somthing tehn i sugges you to craetae a bak wuth time and s oon proepr name or sometihng before you make cahnges. and also proeprly docuemynt everystep you take before you take or modifuying yor step taht you alraedy took. lieke each decision so we can have a mapping on what we did properly. and that file liek hyou can craetea a .md fiel like i dont know rerwradmodel incorpoartion or somee h[roepr amd apt name 

and also for client side instaed of pasisng image maybe we can have liek npy blobs or somethng so it will be beterr or something if we are think of hostingteht rweardd servre s on remote machine or smthing i dont knwo you can seetaht in thsi robomemeter code base for inspiration
/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/ROBOMETER/robometer or if you want you can inspect teh roboreward and tip reward as wilell in taht geenralist models fodler i think i imeplemeted that in all thsoe liek serevr client approach.

and make teh plan as detailed as possible and nbe through as my life depends on it and i will ask oethr agent to review your plan as weell so be sure and through

2. 


```


# Stage-Aware Reward-Model Integration (SARM / TCC → Residual RL)

This document records the design + implementation of driving ResFiT residual RL
(TD3/RLPD on `Can`) with a **stage-aware** reward signal produced by an
**external reward model** (SARM first, TCC second) served over HTTP. Two reward
modes are supported (`reward_model.reward_mode`):

* **`pbrs`** (default) — Potential-Based Reward Shaping (policy-invariant,
  telescopes under n-step). Sections 2–9 below.
* **`milestone`** — a ratchet, non-potential-based stage reward used as an
  ablation (NOT policy-invariant). See **section 10**.

It also covers local ACT-checkpoint loading and the terminated-only bootstrap fix.

Author: automated implementation. (Early edits kept timestamped `.bak` siblings;
later milestone-mode edits were applied in-place with version control.)

---

## 1. What was built

| Area | File(s) |
|---|---|
| Local ACT checkpoint loading | `resfit/lerobot/utils/load_policy.py` (`resolve_local_policy_dir`), `resfit/rl_finetuning/config/residual_td3.py` (`BasePolicyConfig.local_path`), `resfit/rl_finetuning/scripts/train_residual_td3.py` |
| Terminated-only bootstrap (time-limit fix) | `resfit/rl_finetuning/config/rlpd.py` (`algo.terminated_only_bootstrap`), `resfit/rl_finetuning/utils/rb_transforms.py`, `resfit/rl_finetuning/scripts/train_residual_td3.py` |
| Reward-model config | `resfit/rl_finetuning/config/residual_td3.py` (`RewardModelConfig`, `cfg.reward_model`; `reward_mode`, `milestone_*`) |
| HTTP client | `resfit/rl_finetuning/reward_models/reward_client.py` |
| Stage-aware reward wrapper (PBRS **and** milestone) | `resfit/rl_finetuning/reward_models/pbrs_wrapper.py` |
| Env wiring | `resfit/dexmg/environments/dexmg.py`, `train_residual_td3.py` (`get_envs`) |
| SARM server | `SARM/opensarm/reward_server.py` |
| TCC server | `TCN/tcc_torch/reward_server.py` |
| Offline reward labeler (PBRS **and** milestone) | `scripts/label_offline_pbrs.py` |
| Shell scripts | `scripts/train_residual_rl_can_sarm_stage_pbrs.sh` (PBRS), `scripts/train_residual_rl_can_sarm_stage_milestone.sh` (milestone) |

---

## 2. Reward formulation (PBRS with sparse term)

Per env step (training env only):

```
r = r_sparse + gamma * phi(stage')  - phi(stage)
```

* `phi` = per-stage potential, default `{0:0.0, 1:0.05, 2:0.10, 3:0.15}` (4 stages).
* On a **true termination** (success), `phi(stage') = 0` (absorbing).
* Stage is **monotonic** and **server-gated** (never downgrades within an episode).
* PBRS is policy-invariant and telescopes cleanly under n-step returns.

Locked defaults: query every `k=5` steps (4 Hz), hysteresis `K=5` consecutive
high-confidence predictions to advance one stage, confidence threshold `0.80`.
(The current shell scripts pass `k=4` / `K=4`; the config dataclass default is
still `5`.)

> **Alternative reward mode:** setting `reward_model.reward_mode="milestone"`
> replaces this PBRS formula with a ratchet milestone reward — see **section 10**.

---

## 3. Client ↔ server contract (model-agnostic)

The RL client streams **only the new frames** collected since its last query.
The server owns all history, windowing, inference, and stage-transition gating —
so the RL side is identical for SARM and TCC.

```
GET  /health         -> {status, model_type, num_stages, ...}
POST /reset          form: session_id                     (clears session buffer + hysteresis)
POST /predict_stage  files: frames(.npy uint8 [W,H,W,3]),
                            states(.npy float32 [W,9])     (states ignored by TCC)
                     form:  session_id, task, num_anchors, reset,
                            hysteresis_k, conf_threshold, monotonic
                     -> {gated_stage, stage_conf, raw_stages, raw_confs,
                         taus, num_anchors, server_latency_s}
```

Payloads are `.npy` binary blobs (efficient, remote-host friendly). One query
batches the `k` skipped frames → a single forward pass → per-frame predictions →
server-side hysteresis.

### SARM windowing (server-side)
Causal window per anchor: sample `n_obs_steps=8` frames spaced `frame_gap=5`
ending at the anchor (`[t-35..t]`), adaptive-stride / edge-pad at episode start,
zero-pad the 4 rewind slots, `lengths = n_obs_steps+1 = 9`, evaluate at position
8 (the last real frame). State (9D) uses the same indices and is normalized with
`assets/pickplace_can.json`. CLIP image path matches training EXACTLY: images are
fed as **float32 CHW in `[0,1]`** (`[N,C,H,W]`) to `encode_image(do_rescale=False)`.
The client transports frames as uint8 HWC for bandwidth; the server restores
float `[0,1]` and permutes `NHWC->NCHW`. The CLIP **text** is conditioned on the
dataset's `task_name` (`"pick_and_place_can"`) which the server uses regardless of
the client's `task` field (SARM was trained on that exact string).

### TCC windowing (server-side)
Per anchor, `get_steps(anchor, num_steps, frame_stride)` context, clamped to
available/past frames for causality; one forward pass → embeddings → SVM
`predict` (stage) + `predict_proba().max` (confidence). SVM is fit once from the
labeled dataset and cached to `.pkl`.


## Mental model
**Client is dumb, server is smart.** The client (online wrapper OR offline labeler) just streams raw frames+states. The **server owns everything model-specific**: the rolling history buffer, the temporal windowing, the inference, and the hysteresis/gating. Client never builds a window.

Payload per call = only the **new** frames since the last call (not the whole history).

## What "anchor" means
An **anchor** = one frame you ask the server to output a stage for. For each anchor the server builds a **causal window ending at that anchor** and predicts one stage. So "score the last `num_anchors` frames" = "give me a per-frame stage prediction for each of those frames."

## Online RL timeline (k = `query_every_k`, default **5**, not 4)

```
reset (t=0): append frame0 -> POST predict_stage(frames=[f0], num_anchors=1, reset=1)
             server: clear buffer, store f0, predict stage(f0), hysteresis -> gated=0
             wrapper caches current_stage=0, phi=potentials[0]

t=1: append f1 (pending=[f1]).           step%5!=0 -> NO server call. reward uses cached stage.
t=2: append f2 (pending=[f1,f2]).        NO call.
t=3: append f3 (pending=[f1,f2,f3]).     NO call.
t=4: append f4 (pending=[f1..f4]).       NO call.
t=5: append f5 (pending=[f1..f5]).       step%5==0 -> POST predict_stage(frames=[f1,f2,f3,f4,f5], num_anchors=5)
     server: append all 5 to its buffer (now [f0..f5]),
             score the LAST 5 (f1..f5) each as an anchor -> 5 per-frame stages,
             feed those 5 through hysteresis (chronological) -> updated gated stage,
             return gated + per-frame raw_stages/confs.
     wrapper caches the new gated stage, clears pending.
t=6..9: NO calls (use cached stage). t=10: call with pending [f6..f10]. etc.
```

So: **query every 5 steps, batching the 5 skipped frames into ONE request → ONE forward pass, but 5 per-frame predictions come back.** Between queries the reward uses the last cached gated stage.

Truncation flushes early (query on the final frame) so the last non-terminal state gets scored.

## What the server does per `predict_stage` call
1. **Append** every sent frame to the session's rolling buffer (temporal order).
2. For each anchor (the last `num_anchors` buffer positions), **build its SARM window**: `get_frame_indices(anchor, n_obs_steps=8, frame_gap=5)` → 9 causal samples `[t-35 … t]`, zero-pad the 4 rewind slots, `lengths=9`, eval at position 8. All anchors stacked → **one batched forward** (CLIP → stage/subtask transformer).
3. **Hysteresis** over the anchors (below) → gated stage.
4. Return `{gated_stage, stage_conf, raw_stages[...], raw_confs[...], server_latency_s}`.

## Hysteresis — it's a running streak, NOT "last 5 must match"
Your guess ("take last 5 as anchor, if all 5 same then send") is close but not exact. The counter **persists in the session across queries**, per-frame:

```python
for (raw_stage, conf) in anchors:          # each scored frame, in order
    s = max(raw_stage, committed) if monotonic
    if s >= committed+1 and conf >= conf_threshold:   # a vote to advance
        count_toward_next += 1
        if count_toward_next >= hysteresis_k:   # 5 consecutive high-conf votes
            committed += 1                      # advance ONE stage
            count_toward_next = 0
    else:
        count_toward_next = 0                   # any bad/low-conf frame resets streak
```

Key points:
- To go **committed → committed+1**, need **K=5 consecutive** frames predicting ≥ committed+1 with conf ≥ 0.80. One low-confidence or lower-stage frame **resets the streak**.
- It advances **one stage at a time** (monotonic climb). Even if the model jumps to stage 3, it takes 5 confirming frames to reach 1, then 5 more for 2, etc.
- The streak spans queries — it's not reset per request. With k=5 and K=5, a "clean" transition advances one stage in roughly one query batch.
- Between queries the client just reuses the cached gated stage.

## The "40 / 36" — it's the window, NOT a skip
You remembered the 36–40. That's the **history the server needs to build one SARM window** for an anchor: `n_obs_steps=8` samples spaced `frame_gap=5` → `(8-1)*5+1 = 36` frames back (+ a few for batching ≈ 40). **We do NOT skip the first 40 frames.** We call from t=0. When the buffer has fewer than 36 frames, the server's `get_frame_indices` uses **adaptive stride + edge-padding** (repeat the earliest available frame) to still build a valid 9-sample window. Early frames just get a degraded window and almost always read as stage 0 — which is correct (task hasn't progressed). Everything is handled server-side.

## Offline labeling — same streaming, NOT whole-video-at-once
label_offline_pbrs.py uses the **exact same protocol and cadence** as online:
- Per episode: `reset` on frame 0, then append each frame and `_flush` (query) every `query_every_k` steps (default 5), batching pending frames.
- Same server, same hysteresis, same PBRS math.

**Why not batch the whole video in one shot?** Because hysteresis is **causal/streaming** — feeding all frames at once would produce different gating dynamics than the online run. To make the **offline labels match what the policy sees online**, we replay the episode the same way (stream + query every k). Within a query we still batch the k pending frames (one forward pass), and SARM inference itself is batched (B = num_anchors windows in one forward). So it's "streaming with k-sized mini-batches," not per-frame-serial and not whole-video-at-once.

So to your question **"is it the same for labeling and online?"** — **Yes, identical.** Same endpoint, same cadence (k=5), same hysteresis (K=5, conf 0.80), same windowing. The only difference is online is real-time on the live env; offline replays recorded episodes.

## The reward (both paths)
Between queries the cached gated `stage` feeds the PBRS transform each step:
```
r = r_sparse + gamma * phi(stage_next) - phi(stage_prev)     # phi = potentials[stage]
phi(terminal) = 0 on true success termination
```
Online: applied in the wrapper's `step()`. Offline: written to the sidecar parquet at the **source** frame (so it lines up with `next.reward` when the offline buffer is filled).

### Quick answers to your specific questions
- t=0: yes — send 1 image+state, `reset=1`, server stores + predicts + returns.
- Call every n steps? **Yes, every k=5** (default), both online AND offline.
- Send the k skipped frames as a batch at t=k? **Yes**, server puts them in the buffer in order and scores the newest `num_anchors` as anchors.
- "predict only latest frame OR last-5 hysteresis?" → server scores **all k newly sent frames** as anchors; those per-frame predictions feed the **persistent K=5 streak** hysteresis; between queries you reuse the last gated stage.
- "don't call for first 40 frames?" → **No**, we call from t=0; the 36–40 is the server-side window size, and short history is edge-padded.
- "offline pass all frames at once?" → **No**, streamed per episode exactly like online (to keep labels consistent).

---

## 4. Terminated-only bootstrap (time-limit fix)

Original code stored a single `next.done = terminated | truncated` and used it
as the C51/n-step bootstrap mask → the classic **time-limit bug** (truncation
wrongly zeros the future value, which would also corrupt PBRS telescoping).

Fix (config-gated by `algo.terminated_only_bootstrap`, default **False** →
byte-identical to the original sparse runs):

* Transitions now **always** store both `next.done` (= terminated | truncated,
  used for episode segmentation / n-step boundaries) and `next.terminated`
  (true absorbing terminals only).
* When the flag is **True**, `MultiStepTransform` computes the bootstrap mask
  `nonterminal = ~future_terminated` (gathered at the n-step-ahead / boundary
  index), so truncated transitions still bootstrap the (real) final observation.
* `info["final_obs"]` is already stored as `next_obs` on done, so bootstrapping
  on truncation uses the correct state.

Enabling the reward model in the shell scripts also sets this flag to `True`.

Buffer caches are bumped (`schema="terminated_v1"`) so stale caches regenerate.

---

## 5. Training env topology

* Training uses `num_envs=1`. When the reward model is enabled the training env
  is forced to `SyncVectorEnv` (in-process) so the synchronous per-step HTTP
  calls work simply and rewards are blocking/consistent.
* The PBRS wrapper is applied only to the **training** sub-env. **Eval envs are
  untouched** and continue to use the simulator's sparse/success reward.

---

## 6. How to run

### 6.1 Start the reward server first
SARM (from `SARM/opensarm`, its uv venv):
```bash
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/SARM/opensarm
uv run --with flask python reward_server.py --config config/sarm_PickPlaceCan_infer.yaml --host 127.0.0.1 --port 8001
```
TCC (from `TCN/tcc_torch`, the TCN uv venv):
```bash
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/TCN/tcc_torch
uv run --with flask python reward_server.py --checkpoint ../checkpoints_tcc/run_h6ui52ft/last.ckpt \
    --labeled_data_dir ../dataset/PickPlaceCan/labeled/ --host 127.0.0.1 --port 8001
```
Only one server runs at a time (both default to port 8001). `--with flask` injects
Flask ephemerally so the reward-model project deps stay untouched.

### 6.2 Run residual RL (resfit conda env `residual`)
Edit the shell script and set:
```bash
BASE_LOCAL_PATH=".../policy_rabc/checkpoints/050000/pretrained_model"   # local ACT ckpt
REWARD_MODEL_ENABLED="true"
REWARD_MODEL_BACKEND="sarm"          # or "tcc"
REWARD_SERVER_URL="http://127.0.0.1:8001"
# TERMINATED_ONLY_BOOTSTRAP="true" is applied automatically when enabled
```
Then:
```bash
conda activate residual
bash scripts/train_residual_rl_can_sparse.sh   # or train_residual_rl_can.sh (dense sim base)
```
Training aborts at startup with a clear message if the server is unreachable.

### 6.3 Offline PBRS labeling (optional, for consistent offline buffer)
```bash
conda run -n residual python scripts/label_offline_pbrs.py \
    --dataset_root .../SARM-robosuite-can-mh-stages_v21 \
    --server_url http://127.0.0.1:8001 \
    --output ./pbrs_sarm_progress.parquet
```
Writes a NEW sidecar parquet (`index, episode_index, frame_index, stage,
r_sparse, reward_pbrs`); the dataset's own rewards are never overwritten.
Rewards are emitted in **next.reward (source-frame) convention**: the value at
frame `i` is the PBRS reward of transition `i -> i+1`, so the offline buffer fill
assigns it to exactly that transition and the potentials telescope correctly.

To feed it into the offline replay buffer, set in the shell script:
```bash
OFFLINE_REWARD_PARQUET="./pbrs_sarm_progress.parquet"
OFFLINE_REWARD_COLUMN="reward_pbrs"
```
The offline fill reads this column keyed by the LeRobot global frame `index` and
treats it exactly like `next.reward` (priority over sparse/dense). This is
generic: any labeler that emits an `index` + reward column works today and in the
future — just point `OFFLINE_REWARD_*` at it. Changing the parquet/column
invalidates the offline buffer cache automatically.

---

## 7. Config reference (`cfg.reward_model`)

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Turn on PBRS reward-model shaping (training env only). |
| `backend` | `sarm` | Informational (`sarm`/`tcc`); the server decides the model. |
| `server_url` | `http://127.0.0.1:8001` | Reward server base URL. |
| `require_server` | `true` | Fail fast at startup if `/health` doesn't pass. |
| `task_prompt` | pick/place can | CLIP-conditioning task string. |
| `query_every_k` | `5` | Query cadence (env steps). |
| `num_stages` | `4` | Number of task stages. |
| `potentials` | `(0.0,0.05,0.1,0.15)` | Per-stage potentials. |
| `pbrs_gamma` | `null` | Shaping discount (falls back to `algo.gamma`). |
| `keep_sparse_term` | `true` | Keep sparse success reward as PBRS base term. |
| `reward_mode` | `pbrs` | `"pbrs"` (potential-based) or `"milestone"` (ratchet stage payouts; see §10). |
| `milestone_payouts` | `(0.0,0.1,0.1,0.1)` | Milestone mode: one-time payout on first entry into each stage (index = stage). |
| `milestone_success_bonus` | `0.7` | Milestone mode: terminal bonus on TRUE sim success (`r_sparse>0`). |
| `hysteresis_k` | `5` | Consecutive high-conf predictions to advance one stage. |
| `conf_threshold` | `0.80` | Min stage confidence toward an upgrade. |
| `monotonic` | `true` | Never downgrade the stage within an episode. |
| `image_key` | `observation.images.agentview` | Frame source key. |
| `image_vflip` | `false` | Vertically flip frames before sending (see risks). |
| `offline_key` | `null` | (Reserved.) Offline reward is wired via `offline_data.reward_parquet` / `reward_column` below. |

Offline reward source (generic, in `cfg.offline_data`):

| Field | Default | Meaning |
|---|---|---|
| `reward_parquet` | `null` | Sidecar parquet of per-transition rewards keyed by global frame `index`. When set, the offline buffer reward is read from it (treated like `next.reward`, source-frame convention), taking priority over sparse/dense. |
| `reward_column` | `reward_pbrs` | Column in `reward_parquet` to read. |

---

## 8. Verification done
* Unit test (conda `residual`): terminated-only n-step fix — truncation
  bootstraps under the fix, is cut under the original done-based mask. PASS.
* `reward_models` package imports; `ResidualTD3CanConfig` instantiates with the
  `reward_model` group; both shell scripts pass `bash -n`.
* **Live SARM server (RTX 4090) + real SARM v2.1 dataset**: the server's
  `predict()` path matches SARM's own reference inference **97.5% per-frame stage
  accuracy** vs the dataset's ground-truth `reward`. The full offline labeler ran
  end-to-end over HTTP on episode 0: stages progress `0→1→2→3`, `reward_pbrs` in
  `[-0.0045, 0.85]` (terminal success `1 − φ(0.15) = 0.85`, dwelling tiny negative)
  — PBRS math confirmed. npy transport, windowing, hysteresis all validated.

### Two SARM-server bugs found & fixed during real-data testing
(The synthetic-frame smoke test missed both — random pixels always read as stage 0.)
1. **Image layout**: LeRobot yields images as float32 **CHW `[0,1]`**; SARM
   `encode_image(do_rescale=False)` expects `[N,C,H,W]` floats. The server was
   feeding uint8 HWC → out-of-distribution → every frame stage 0. Fixed by
   `imgs.float()/255` + `permute(NHWC→NCHW)`.
2. **Task text**: SARM's CLIP text branch is conditioned on `task_name`
   (`"pick_and_place_can"`); the server was using the client's natural-language
   prompt → out-of-distribution → stage 0. Fixed: the server always uses
   `cfg.general.task_name` and ignores the client `task`.
* Dataset key auto-detection added to the labeler: the SARM v2.1 dataset uses
  keys `agentview-images-rgb` / `state` (not `observation.*`).

## 9. Open risks to verify with the live server
1. **Camera orientation (online only)**: robosuite *env* obs frames may be
   vertically flipped vs the reward model's training frames. Offline labeling on
   the SARM dataset is validated (no flip needed). For the live env, compare a
   rendered frame vs a dataset frame and toggle `reward_model.image_vflip`.
2. **Resolution**: SARM dataset frames are 256×256; the live env agentview is
   84×84. CLIP resizes both to 224, but the 84→224 upscale is a mild domain gap
   to watch during the online RL smoke run.
3. **Hysteresis latency**: `K=5` confirmations at `k=5` cadence advances one
   stage per query batch (offline ep0 showed the gated stage lags GT slightly,
   as designed); tune `hysteresis_k` / `conf_threshold` if too slow/fast.
4. **Dataset alignment for offline labeling**: label the SAME LeRobot dataset
   resfit loads as `offline_data` so the global `index` keys line up.

---

## 10. Stage-aware MILESTONE reward mode (non-PBRS ablation)

Selected by `reward_model.reward_mode` (`"pbrs"` default | `"milestone"`).
**Milestone** is a **ratchet, non-potential-based** stage reward. It reuses the
*entire* SARM/TCC stack unchanged — same HTTP client, same server, same
windowing, same monotonic **hysteresis gating** (§3, §Hysteresis). **Only the
per-step reward differs.** Unlike PBRS it does **not** telescope and is **not**
policy-invariant — that is the whole point of the ablation.

**Why it exists.** To test whether a *non-PBRS* stage-aware dense/discrete reward
changes sample efficiency vs. sparse / PBRS in the residual-RL setting. Working
hypothesis: because ResFiT starts from a strong BC base (already solves
exploration) and uses n-step returns, shaped rewards add little. Running
Sparse / PBRS / Milestone with everything else identical isolates the reward's
effect.

### 10.1 The milestone reward
```
# each env step, using the monotonic gated stage `s` and a ratchet `last_paid`:
r = 0
if s > last_paid:                         # entered one or more NEW stages
    r += sum(payouts[last_paid+1 .. s])   # pay each ONCE (skipped stages summed)
    last_paid = s
if terminated and r_sparse > 0.5:         # TRUE simulator success only
    r += success_bonus
```
* `payouts` = `milestone_payouts` (index = stage; index 0 = start, never paid).
* `success_bonus` = `milestone_success_bonus`, paid on the terminal success step.
* The ratchet (`last_paid`) pays each stage **once** — dropping / re-grasping
  cannot farm points (same spirit as PBRS monotonicity).
* A **failed** episode (truncation, no success) collects only the stage payouts
  it reached (e.g. `0.6` for reaching stage 2) and **no** bonus.

### 10.2 Reward scale / totals
| Design | payouts (stage 1/2/3) | bonus | max return (full success) |
|---|---|---|---|
| **Strong** (shell-script default) | `0.3, 0.3, 0.4` | `1.0` | **2.0** |
| **Safe** (config / labeler default) | `0.1, 0.1, 0.1` | `0.7` | **1.0** |

The critic is **MSE** (`CRITIC_LOSS_TYPE="mse"`, `clip_q_target_to_reward_range=False`)
which **ignores `v_min`/`v_max`** — so a max return of `2.0` is **not truncated**.
Only pick the `≤1.0` (safe) design if you switch to a *distributional* critic
(`c51`/`hl_gauss`) whose target is clamped to `[v_min, v_max]`.

### 10.3 Terminal / success handling — online vs offline
* **Online (live sim):** the success is the real sim signal (`terminated` +
  `r_sparse=1`). In milestone mode the wrapper **also queries the server on the
  terminal step** (PBRS skips it, since `phi'=0` there) and pays any pending
  stage payouts **plus** the bonus on that step — so a successful episode always
  totals `Σpayouts + bonus`.
* **Offline (demo dataset):** the LeRobot dataset has **no per-frame success
  flag**. The labeler *and* the offline buffer fill treat the **last frame of
  each episode** as the terminal (`next.done`/`terminated = frame_index ==
  length-1`) with `r_sparse = 1`. This is valid because the `mh` demos are **all
  successful** human episodes. **This is exactly how PBRS and the sparse baseline
  already worked — nothing new, no corruption.** (If the dataset ever contained
  *failed* episodes, a real success column would be required.)
* **Reward convention:** the labeler writes the reward of transition `i -> i+1`
  at source frame `i`; the offline fill stores it as that transition's
  `next.reward`. The terminal (success) reward therefore lands on the transition
  **into** the terminal frame — identical to PBRS.

### 10.4 Offline milestone labeling (MANDATORY for RLPD consistency)
ResFiT is RLPD (~50% offline demos per batch), so the offline buffer must carry
the SAME milestone reward the policy sees online, or the critic gets
contradictory targets. Relabel with the **same** payouts/bonus as the run:
```bash
python scripts/label_offline_pbrs.py --reward_mode milestone \
  --milestone_payouts 0.0 0.3 0.3 0.4 --milestone_success_bonus 1.0 \
  --dataset_root .../SARM-robosuite-can-mh-stages_v21 \
  --server_url http://127.0.0.1:8003 --hysteresis_k 4 --query_every_k 4 \
  --output ./milestone_sarm_progress.parquet
```
Writes a **separate** column `reward_milestone` (never reuses `reward_pbrs`) and
records `reward_mode` + `milestone_payouts` + `milestone_success_bonus` in the
parquet's `labeler_config`.

### 10.5 Config-guard (offline ↔ RL) — mode-aware
On load the RL run reads the parquet's `labeler_config` and **hard-errors** on a
mismatch:
* Always checks `reward_mode` (a milestone parquet cannot feed a PBRS run, and
  vice-versa).
* Milestone: checks `milestone_payouts` + `milestone_success_bonus` + `num_stages`.
* PBRS: checks `gamma` + `potentials` + `keep_sparse_term` + `num_stages`.
* Server-gating knobs (`hysteresis_k`/`query_every_k`/`conf_threshold`/`image_vflip`)
  are **warnings** only.

### 10.6 Run the milestone experiment
```bash
# 1) start the SARM (or TCC) reward server on :8003
# 2) relabel offline (§10.4) with matching payouts/bonus
# 3) train
bash scripts/train_residual_rl_can_sarm_stage_milestone.sh
```
Base ACT, hyper-params, offline dataset, MSE critic and terminated-only
bootstrap are **identical** to the PBRS script — only the reward differs, so
**Sparse / PBRS / Milestone** form a clean A/B/C.

### 10.7 Files touched
| Area | File |
|---|---|
| `reward_mode` / `milestone_payouts` / `milestone_success_bonus` | `resfit/rl_finetuning/config/residual_td3.py` |
| Reward-mode branch (PBRS unchanged; milestone added) | `resfit/rl_finetuning/reward_models/pbrs_wrapper.py` |
| Env wiring passes milestone params | `resfit/dexmg/environments/dexmg.py` |
| cfg dict + mode-aware config-guard | `resfit/rl_finetuning/scripts/train_residual_td3.py` |
| `--reward_mode` + `reward_milestone` column | `scripts/label_offline_pbrs.py` |
| Milestone training script | `scripts/train_residual_rl_can_sarm_stage_milestone.sh` |

### 10.8 Info keys logged (per step, training env)
`reward_model_reward` (the actual reward returned), `reward_model_r_milestone`,
`reward_model_last_paid_stage`, plus the shared `reward_model_stage`,
`reward_model_raw_stage`, `reward_model_stage_conf`, `reward_model_r_sparse`, and
latencies. `reward_model_r_shaped` mirrors the reward in both modes (PBRS shaped
value in PBRS mode, milestone reward in milestone mode).

### 10.9 Verification
* Config loads the new fields; milestone math checked (`.venv/bin/python`):
  normal / stage-skip / flicker successful episode → **2.0**; stage-2 failure →
  **0.6**; stage-3 landing on the terminal step → **2.0**; ratchet blocks farming.
* Wrapper exposes `reward_mode`/`milestone_payouts`; labeler exposes the args;
  `bash -n` on the milestone script passes.
* **Not yet smoke-tested end-to-end on a live server** (needs the SARM server up).
