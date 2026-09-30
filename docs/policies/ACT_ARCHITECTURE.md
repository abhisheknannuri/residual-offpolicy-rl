# ACT & Diffusion Policy — Full Architecture Deep Dive

This document provides a complete architectural breakdown of both policy architectures implemented in this codebase: **ACT** (Action Chunking Transformer) and **Diffusion Policy**. Every layer, every dimension, every mathematical operation is explained with concrete numbers from the TwoArmCoffee task.

---

## Table of Contents

### ACT (Action Chunking Transformer)
1. [ACT — The Big Picture](#1-act--the-big-picture)
2. [ACT — Input Processing](#2-act--input-processing)
3. [ACT — Vision Backbone (ResNet18)](#3-act--vision-backbone-resnet18)
4. [ACT — Transformer Encoder](#4-act--transformer-encoder)
5. [ACT — Transformer Decoder](#5-act--transformer-decoder)
6. [ACT — Action Head](#6-act--action-head)
7. [ACT — VAE (Optional)](#7-act--vae-optional)
8. [ACT — Positional Embeddings](#8-act--positional-embeddings)
9. [ACT — Loss Computation](#9-act--loss-computation)
10. [ACT — Inference (select_action)](#10-act--inference-select_action)
11. [ACT — Complete Dimension Trace](#11-act--complete-dimension-trace)

### Diffusion Policy
12. [Diffusion — The Big Picture](#12-diffusion--the-big-picture)
13. [Diffusion — Vision Encoder](#13-diffusion--vision-encoder)
14. [Diffusion — Conditional U-Net](#14-diffusion--conditional-u-net)
15. [Diffusion — Training (Noise Prediction)](#15-diffusion--training-noise-prediction)
16. [Diffusion — Inference (Iterative Denoising)](#16-diffusion--inference-iterative-denoising)
17. [Diffusion — Complete Dimension Trace](#17-diffusion--complete-dimension-trace)

### Comparison
18. [Side-by-Side Comparison](#18-side-by-side-comparison)

---

## 1. ACT — The Big Picture

**Paper:** [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (Zhao et al., 2023)](https://arxiv.org/abs/2304.13705)

**Core idea:** Given the current observation (images + robot state), predict a **chunk** of 20 future actions in a single forward pass using a transformer encoder-decoder architecture inspired by DETR (object detection).

### Architecture overview (text diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                        ACT Architecture                         │
│                                                                 │
│  INPUTS                                                         │
│  ──────                                                         │
│  Images (3 cameras):  (B, 3, 3, 84, 84)                       │
│  Robot state:         (B, 36)                                   │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │   ResNet18 ×3    │  One backbone shared across all cameras   │
│  │  (84×84 → 3×3    │  Feature map: (B, 512, 3, 3) per camera  │
│  │   feature map)   │                                           │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  1×1 Conv: (512 → 512)  +  2D sinusoidal pos embedding         │
│  Flatten: (B, 512, 3, 3) → 9 tokens of dim 512 per camera      │
│  3 cameras × 9 = 27 image tokens                                │
│                                                                 │
│  ┌───────────────────────────────────────────────┐              │
│  │  Transformer Encoder (4 layers)               │              │
│  │                                               │              │
│  │  Input tokens (sequence dim):                 │              │
│  │    [latent(1)] [state(1)] [img_tokens(27)]    │              │
│  │    = 29 tokens, each dim 512                  │              │
│  │                                               │              │
│  │  Self-attention across all tokens             │              │
│  │  8 heads, FFN dim 3200                        │              │
│  │                                               │              │
│  │  Output: 29 tokens × 512 dim                  │              │
│  └───────────────────┬───────────────────────────┘              │
│                      │                                          │
│                      ▼ (cross-attention)                        │
│  ┌───────────────────────────────────────────────┐              │
│  │  Transformer Decoder (1 layer)                │              │
│  │                                               │              │
│  │  Input: 20 learned query embeddings           │              │
│  │    (one per future timestep)                  │              │
│  │  Cross-attends to encoder output              │              │
│  │                                               │              │
│  │  Output: 20 tokens × 512 dim                  │              │
│  └───────────────────┬───────────────────────────┘              │
│                      │                                          │
│                      ▼                                          │
│  ┌───────────────────────────────────────────────┐              │
│  │  Action Head: Linear(512 → 24)                │              │
│  │                                               │              │
│  │  Output: (B, 20, 24) = 20 future actions      │              │
│  └───────────────────────────────────────────────┘              │
│                                                                 │
│  LOSS: L1(predicted_actions, ground_truth_actions)              │
│        masked by action_is_pad for episode boundaries           │
└─────────────────────────────────────────────────────────────────┘
```

### What is "action chunking"?

Without chunking, a policy predicts **one action** per observation and must be called 400 times per episode (at 20Hz, 20s episode). This creates temporal inconsistency — each prediction is independent and can jitter.

With chunking (`chunk_size=20`), the policy predicts **20 consecutive actions** in one shot. These 20 actions form a temporally coherent trajectory segment. The robot executes all 20 before querying the model again. This means only 20 model calls per 400-step episode, each producing a smooth sub-trajectory.

---

## 2. ACT — Input Processing

Before anything enters the transformer, inputs are normalized and projected.

### Normalization (happens first)

From [modeling_act.py line 234](resfit/lerobot/policies/act/modeling_act.py#L234):

**Images** (per-channel MEAN_STD):

$$I_{\text{norm}} = \frac{I - \mu_c}{\sigma_c + 10^{-8}}$$

where $\mu_c, \sigma_c \in \mathbb{R}^3$ are the per-channel RGB mean and std from the training dataset. Typical values: $\mu \approx [0.485, 0.456, 0.406]$, $\sigma \approx [0.229, 0.224, 0.225]$.

- Input: $(B, 3, 84, 84)$ pixels in $[0, 1]$
- Output: $(B, 3, 84, 84)$ centered values, roughly $[-2, +3]$

**Robot state** (per-dimension MEAN_STD):

$$s_{\text{norm}} = \frac{s - \mu_s}{\sigma_s + 10^{-8}}$$

where $\mu_s, \sigma_s \in \mathbb{R}^{36}$. Each dimension (ee position x/y/z, quaternion w/x/y/z, gripper angles) is independently normalized.

- Input: $(B, 36)$ raw sensor values
- Output: $(B, 36)$ centered, roughly $[-3, +3]$ per dimension

**Target actions** (per-dimension MEAN_STD, same formula):

- Input: $(B, 20, 24)$ raw action values
- Output: $(B, 20, 24)$ centered, roughly $[-3, +3]$ per dimension

### Input projections

After normalization, each modality is projected to the transformer hidden dimension $d = 512$:

| Input | Projection | Output |
|---|---|---|
| Robot state $(B, 36)$ | `Linear(36, 512)` | $(B, 512)$ → 1 token |
| Latent vector $(B, 32)$ | `Linear(32, 512)` | $(B, 512)$ → 1 token |
| Image features $(B, 512, 3, 3)$ | `Conv2d(512, 512, 1×1)` | $(B, 512, 3, 3)$ → 9 tokens per camera |

---

## 3. ACT — Vision Backbone (ResNet18)

### Architecture

ACT uses **one shared ResNet18** for all cameras. The backbone is from `torchvision.models.resnet18`, initialized with ImageNet pretrained weights (`ResNet18_Weights.IMAGENET1K_V1`), with the final FC layer and average pooling removed.

`FrozenBatchNorm2d` is used instead of standard `BatchNorm2d` — batch statistics from ImageNet pretraining are frozen (not updated during fine-tuning). This stabilizes training, especially with small batch sizes.

### Layer-by-layer for 84×84 input

Each `BasicBlock` is: Conv3×3 → BN → ReLU → Conv3×3 → BN → skip connection → ReLU.

| Layer | Operation | Output shape | Receptive field |
|---|---|---|---|
| Input | — | $(B, 3, 84, 84)$ | 1×1 |
| conv1 | Conv2d(3, 64, 7×7, stride=2, pad=3) | $(B, 64, 42, 42)$ | 7×7 |
| bn1 + relu | FrozenBatchNorm2d + ReLU | $(B, 64, 42, 42)$ | — |
| maxpool | MaxPool2d(3×3, stride=2, pad=1) | $(B, 64, 21, 21)$ | — |
| layer1 | 2× BasicBlock(64→64) | $(B, 64, 21, 21)$ | — |
| layer2 | 2× BasicBlock(64→128, stride=2) | $(B, 128, 11, 11)$ | — |
| layer3 | 2× BasicBlock(128→256, stride=2) | $(B, 256, 6, 6)$ | — |
| **layer4** | 2× BasicBlock(256→512, stride=2) | $(B, 512, 3, 3)$ | ~163×163 |

Only `layer4` output is extracted (via `IntermediateLayerGetter`). This is a $3 \times 3$ spatial grid of 512-dim feature vectors.

### From feature maps to transformer tokens

For each camera $c \in \{1, 2, 3\}$:

1. Feature map: $(B, 512, 3, 3)$
2. 1×1 conv projection: $(B, 512, 3, 3)$ → $(B, 512, 3, 3)$ (same size, but now in "transformer space")
3. Add 2D sinusoidal positional embedding (encodes spatial position within the 3×3 grid)
4. Reshape: $(B, 512, 3, 3)$ → $(9, B, 512)$ — 9 spatial tokens per camera

All cameras' tokens are concatenated: $3 \times 9 = 27$ image tokens.

### Why the ResNet backbone gets lower learning rate

The backbone is pretrained on ImageNet (1.2M images, 1000 classes). It already knows how to extract edges, textures, shapes. Fine-tuning too aggressively would overwrite these features. So:

- Backbone parameters: lr = $10^{-5}$ (slow adaptation)
- Everything else: lr = $10^{-4}$ (normal learning)

---

## 4. ACT — Transformer Encoder

> **Deep dive with full math and worked numerical examples**: See [ACT_TRANSFORMER.md](ACT_TRANSFORMER.md)

### Purpose

The encoder fuses **all input modalities** — the latent vector, robot state, and image features from all cameras — into a joint representation through self-attention. Every token can attend to every other token, allowing cross-modal reasoning (e.g., "the robot hand position correlates with what camera 2 sees").

### Input token sequence

| Position | Token(s) | Count | Source |
|---|---|---|---|
| 0 | Latent | 1 | Zero vector (no VAE) or sampled from VAE |
| 1 | Robot state | 1 | Linear(36 → 512) |
| 2-28 | Image features | 27 | 3 cameras × 9 spatial positions |
| **Total** | | **29** | |

Each token is a 512-dimensional vector.

### Encoder architecture

From [modeling_act.py `ACTEncoder` and `ACTEncoderLayer`](resfit/lerobot/policies/act/modeling_act.py#L591):

```
For each of 4 encoder layers:
    ┌────────────────────────────────────┐
    │  Input: x ∈ R^(29, B, 512)        │
    │                                    │
    │  q = k = x + pos_embed            │
    │  v = x                             │
    │                                    │
    │  x' = MultiheadAttention(q, k, v)  │
    │     8 heads, dropout=0.1           │
    │                                    │
    │  x = LayerNorm(x + Dropout(x'))    │  ← Post-norm residual
    │                                    │
    │  x' = Linear(512 → 3200)          │
    │  x' = ReLU(x')                     │
    │  x' = Dropout(x')                  │
    │  x' = Linear(3200 → 512)          │
    │                                    │
    │  x = LayerNorm(x + Dropout(x'))    │  ← Post-norm residual
    └────────────────────────────────────┘
```

### Multi-head attention math

For 8 heads with $d_{\text{model}} = 512$, each head has $d_k = d_v = 512 / 8 = 64$:

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

where $Q_i = x W_i^Q$, $K_i = x W_i^K$, $V_i = x W_i^V$, each $W_i \in \mathbb{R}^{512 \times 64}$.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_8) W^O$$

where $W^O \in \mathbb{R}^{512 \times 512}$.

Positional embeddings are **added to Q and K** (but not V). This encodes position information into the attention weights without modifying the values directly.

### Output

$(29, B, 512)$ — same shape as input. The 29 tokens are now fully contextualized — each token's representation incorporates information from all other tokens through 4 layers of self-attention.

---

## 5. ACT — Transformer Decoder

### Purpose

The decoder **queries** the encoder output to produce action predictions. It uses learned query embeddings — one per future timestep — that cross-attend to the encoder's fused representation.

### Why only 1 decoder layer?

The original ACT paper specifies 7 decoder layers, but due to a [bug in the original code](https://github.com/tonyzhaozh/act/issues/25), only the first layer was actually used. This implementation matches the bug: `n_decoder_layers=1`. Despite this, the model works well because most of the processing happens in the 4-layer encoder.

### Decoder input

The decoder input is a **zero tensor** of shape $(20, B, 512)$ — 20 queries, one per predicted timestep. Learned positional embeddings (`decoder_pos_embed`, shape $(20, 512)$) distinguish which timestep each query represents.

### Decoder architecture

From [modeling_act.py `ACTDecoderLayer`](resfit/lerobot/policies/act/modeling_act.py#L647):

```
┌────────────────────────────────────────────┐
│  Input: x ∈ R^(20, B, 512) (all zeros)    │
│                                            │
│  1. Self-attention among 20 queries        │
│     q = k = x + decoder_pos_embed          │
│     v = x                                   │
│     x = LayerNorm(x + Dropout(SelfAttn))   │
│                                            │
│  2. Cross-attention to encoder output      │
│     q = x + decoder_pos_embed              │
│     k = encoder_out + encoder_pos_embed    │
│     v = encoder_out                         │
│     x = LayerNorm(x + Dropout(CrossAttn))  │
│                                            │
│  3. Feed-forward network                   │
│     x = LayerNorm(x + Dropout(FFN(x)))     │
│     FFN: 512 → 3200 → ReLU → 512          │
└────────────────────────────────────────────┘
```

### Cross-attention: how actions are generated from observations

In the cross-attention step:
- **Queries** (20 of them): come from the decoder — "what action should I predict at timestep $t$?"
- **Keys/Values** (29 of them): come from the encoder — "here is the fused observation context"

Each query attends to the encoder tokens to extract the information relevant for its specific timestep. Earlier timesteps might attend more to current robot state; later timesteps might attend more to distant scene features.

### Output

$(20, B, 512)$ → transposed to $(B, 20, 512)$. Each of the 20 tokens is a 512-dim representation for one future action.

---

## 6. ACT — Action Head

A single linear layer maps each decoder output token to the action dimension:

$$a_t = W_{\text{head}} \cdot h_t + b_{\text{head}}$$

where $W_{\text{head}} \in \mathbb{R}^{24 \times 512}$, $b_{\text{head}} \in \mathbb{R}^{24}$.

- Input: $(B, 20, 512)$
- Output: $(B, 20, 24)$ — **20 predicted actions**, each 24-dimensional

The 24 action dimensions for TwoArmCoffee (GR1 humanoid):

| Index | Meaning |
|---|---|
| 0-2 | Right wrist Δposition (x, y, z) in meters |
| 3-5 | Right wrist Δrotation (axis-angle rx, ry, rz) |
| 6-11 | Right Inspire hand joints (thumb flex, thumb roll, index, middle, ring, pinky) |
| 12-14 | Left wrist Δposition |
| 15-17 | Left wrist Δrotation |
| 18-23 | Left Inspire hand joints |

---

## 7. ACT — VAE (Optional)

### What it is

The VAE (Variational Auto-Encoder) is an **optional** component (`use_vae=False` by default). When enabled, it adds a separate "VAE encoder" that encodes the ground-truth action sequence into a latent vector. This latent then conditions the main transformer, allowing it to capture **multi-modal** action distributions (multiple valid ways to do the same task).

### Why it exists

Without VAE, ACT regresses a single deterministic action chunk per observation. If the dataset contains diverse trajectories for the same situation (e.g., approaching an object from the left vs right), the model averages them, producing suboptimal actions.

With VAE, the model learns a latent space where different latent vectors produce different valid trajectories. During training, the latent is sampled from the data-conditioned distribution. During inference, it's set to zero (the prior mean).

### VAE architecture (when `use_vae=True`)

```
                                    ┌──────────────────────┐
Ground-truth actions (B, 20, 24)───►│                      │
                                    │   VAE Encoder        │
Robot state (B, 36) ───────────────►│   (4 transformer     │
                                    │    encoder layers)    │
[CLS] token (learned) ────────────►│                      │
                                    └──────────┬───────────┘
                                               │
                                         CLS output
                                               │
                                    ┌──────────▼───────────┐
                                    │  Linear(512 → 64)    │
                                    │  Split: μ(32), logσ²(32) │
                                    └──────────┬───────────┘
                                               │
                                    Reparameterization trick:
                                    z = μ + exp(logσ²/2) · ε
                                    where ε ~ N(0, I)
                                               │
                                    z ∈ R^(B, 32)
                                               │
                                    ┌──────────▼───────────┐
                                    │  Linear(32 → 512)    │
                                    │  → encoder token[0]  │
                                    └──────────────────────┘
```

**VAE Encoder input tokens:** `[CLS, robot_state, action_0, action_1, ..., action_19]`
- CLS: learned embedding $(1, 512)$
- Robot state: `Linear(36, 512)`
- Actions: `Linear(24, 512)` per timestep
- Total: 22 tokens (1 + 1 + 20)

**VAE Encoder:** 4 transformer encoder layers (same architecture as the main encoder)

**Latent projection:** `Linear(512, 64)` on the CLS token output, split into $\mu \in \mathbb{R}^{32}$ and $\log \sigma^2 \in \mathbb{R}^{32}$.

**Reparameterization:** $z = \mu + \exp(\log \sigma^2 / 2) \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.

**During inference:** VAE encoder is not used. $z = \mathbf{0} \in \mathbb{R}^{32}$ (zero vector — the prior mean).

---

## 8. ACT — Positional Embeddings

ACT uses three different types of positional embeddings:

### 1. Encoder 1D positional embeddings (learned)

`nn.Embedding(n_1d_tokens, 512)` where `n_1d_tokens` = 2 (latent + state) or 3 (+ env_state if present).

These are **learned** embeddings added to the latent and state tokens in the encoder input.

### 2. Encoder 2D sinusoidal positional embeddings (fixed)

For image feature map tokens. Uses `ACTSinusoidalPositionEmbedding2d(dim=256)`:

For a $H \times W$ feature map, the position of pixel $(y, x)$ is encoded using sinusoidal functions:

$$\text{PE}(y, x, 2i) = \sin\left(\frac{y / H \cdot 2\pi}{10000^{2i/d}}\right), \quad \text{PE}(y, x, 2i+1) = \cos\left(\frac{y / H \cdot 2\pi}{10000^{2i/d}}\right)$$

Separate embeddings for y-axis (first 256 dims) and x-axis (last 256 dims), concatenated to 512 dims. This encodes the spatial position within the 3×3 feature map grid.

### 3. Decoder positional embeddings (learned)

`nn.Embedding(20, 512)` — 20 learned embeddings, one per predicted timestep.

These distinguish which action timestep each decoder query represents. They are added to both the query and key in the decoder's self-attention, and to the query in cross-attention.

### 4. VAE encoder positional embeddings (fixed sinusoidal 1D)

Standard 1D sinusoidal embeddings (Attention Is All You Need style):

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Applied to the VAE encoder's input sequence (CLS + state + 20 actions = 22 positions).

---

## 9. ACT — Loss Computation

From [modeling_act.py `forward()` line 234](resfit/lerobot/policies/act/modeling_act.py#L234):

### L1 loss (always)

$$\mathcal{L}_{\text{L1}} = \frac{1}{|\mathcal{V}|} \sum_{b=1}^{B} \sum_{t=1}^{20} \sum_{d=1}^{24} \mathbb{1}[\text{not\_pad}_{b,t}] \cdot |a_{b,t,d} - \hat{a}_{b,t,d}|$$

where:
- $a$ = ground-truth actions (normalized), $\hat{a}$ = predicted actions (normalized)
- $\mathbb{1}[\text{not\_pad}]$ masks out padded actions at episode boundaries
- $|\mathcal{V}|$ = total number of valid (non-padded) elements

The loss is computed in **normalized space** (both prediction and target have been MEAN_STD normalized).

### KL divergence loss (only if `use_vae=True`)

$$\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{l=1}^{32} \left(1 + \log \sigma_l^2 - \mu_l^2 - \sigma_l^2\right)$$

This penalizes the latent distribution $q(z|a, s) = \mathcal{N}(\mu, \sigma^2)$ for deviating from the unit Gaussian prior $p(z) = \mathcal{N}(0, I)$. Summed over latent dimensions, averaged over batch.

### Total loss

$$\mathcal{L} = \mathcal{L}_{\text{L1}} + \lambda_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

where $\lambda_{\text{KL}} = 10.0$ (default `kl_weight`).

With `use_vae=False` (default): $\mathcal{L} = \mathcal{L}_{\text{L1}}$

---

## 10. ACT — Inference (select_action)

From [modeling_act.py `select_action()` line 155](resfit/lerobot/policies/act/modeling_act.py#L155):

### Action queue mechanism

ACT uses an **action queue** to avoid running the model at every timestep:

```
Step 0:   Queue empty → run model → get 20 actions → push all to queue
Step 0:   Pop action[0] from queue → execute
Step 1:   Queue has 19 actions → pop action[1] → execute
Step 2:   Queue has 18 actions → pop action[2] → execute
...
Step 19:  Queue has 1 action → pop action[19] → execute
Step 20:  Queue empty → run model again → get new 20 actions
...
```

With `n_action_steps=20` (equal to `chunk_size`), the model runs once every 20 environment steps. The queue is a `collections.deque` with `maxlen=20`.

### Per-environment queues

When running multiple parallel environments (eval), each environment has its own independent queue. If environment 3 finishes an episode and resets, only queue 3 is cleared — the other queues continue serving their remaining actions.

### Temporal ensembling (optional alternative)

If `temporal_ensemble_coeff` is set (e.g., 0.01), the action queue is replaced with a **temporal ensembler**. In this mode, `n_action_steps` must be 1, and the model runs at every step. Predictions from overlapping chunks are blended using exponential weights:

$$w_i = e^{-\alpha \cdot i}$$

where $\alpha = 0.01$ and $i$ is the age of the prediction (0 = latest). Older predictions get slightly higher weight (positive $\alpha$), preventing the newest prediction from dominating and causing jitter.

---

## 11. ACT — Complete Dimension Trace

Concrete dimensions for TwoArmCoffee with 3 cameras at 84×84:

```
INPUT:
  observation.images.agentview:              (B, 3, 84, 84)
  observation.images.robot0_eye_in_left_hand:  (B, 3, 84, 84)
  observation.images.robot0_eye_in_right_hand: (B, 3, 84, 84)
  observation.state:                         (B, 36)

NORMALIZATION:
  All images → MEAN_STD normalized:          (B, 3, 84, 84)
  State → MEAN_STD normalized:               (B, 36)

BACKBONE (shared ResNet18):
  Each camera → layer4 output:               (B, 512, 3, 3)
  
IMAGE PROJECTION:
  Conv2d(512, 512, 1×1):                     (B, 512, 3, 3)
  Reshape to tokens:                         (9, B, 512)  per camera
  2D sinusoidal pos embed:                   (9, B, 512)  per camera
  All 3 cameras concatenated:                (27, B, 512)

STATE PROJECTION:
  Linear(36, 512):                           (B, 512) → (1, B, 512)

LATENT PROJECTION (no VAE):
  zeros(B, 32) → Linear(32, 512):            (B, 512) → (1, B, 512)

ENCODER INPUT:
  Stack: [latent, state, img_cam1, img_cam2, img_cam3]
  Tokens:                                   (29, B, 512)
  Pos embeds:                                (29, B, 512)

ENCODER (4 layers × self-attention):
  Output:                                    (29, B, 512)

DECODER INPUT:
  Zero queries:                              (20, B, 512)
  Learned pos embed:                         (20, 1, 512)

DECODER (1 layer × self-attn + cross-attn):
  Output:                                    (20, B, 512)
  Transpose:                                 (B, 20, 512)

ACTION HEAD:
  Linear(512, 24):                           (B, 20, 24)

LOSS (training):
  L1( (B,20,24) predicted, (B,20,24) target ) → scalar (masked by pad)

OUTPUT (inference):
  Unnormalize:                               (B, 20, 24) → physical action units
  Action queue: pop 1 action per step:       (B, 24)
```

---

## 12. Diffusion — The Big Picture

**Paper:** [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Chi et al., 2023)](https://arxiv.org/abs/2303.11876)

**Core idea:** Frame action prediction as a **denoising diffusion process**. Instead of directly regressing actions (like ACT), train a neural network to iteratively remove noise from random Gaussian samples until clean action trajectories emerge.

### Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Diffusion Policy Architecture                │
│                                                                 │
│  INPUTS (2 timesteps of history: t-1 and t)                    │
│  ─────────────────────────────────────────                      │
│  Images:     (B, 2, n_cameras, C, H, W)                       │
│  Robot state: (B, 2, state_dim)                                 │
│                                                                 │
│  ┌──────────────────────────────┐                               │
│  │  DiffusionRgbEncoder ×n_cam │  One per camera                │
│  │  RandomCrop(240,240)        │  (center crop at eval)         │
│  │  ResNet18 → SpatialSoftmax  │                                │
│  │  → Linear(64) → ReLU        │                                │
│  │  Output: (B*2, 64) per cam  │                                │
│  └──────────────┬───────────────┘                               │
│                 │                                                │
│  Concatenate: [cam1_t-1, cam1_t, cam2_t-1, ..., state_t-1, state_t] │
│  global_cond ∈ R^(B, n_cameras*64*2 + state_dim*2)             │
│                                                                 │
│  DIFFUSION PROCESS:                                             │
│                                                                 │
│  Training:                                                      │
│    1. Sample noise ε ~ N(0, I)                                  │
│    2. Sample timestep t ~ Uniform{1..100}                       │
│    3. Noisy actions = √ᾱₜ · actions + √(1-ᾱₜ) · ε             │
│    4. U-Net predicts ε̂ = f(noisy_actions, t, global_cond)      │
│    5. Loss = MSE(ε̂, ε)                                         │
│                                                                 │
│  ┌──────────────────────────────────────────────┐               │
│  │  Conditional 1D U-Net                        │               │
│  │                                              │               │
│  │  Input: noisy action trajectory (B, H, D_a)  │               │
│  │  Conditioning: global_cond + timestep embed  │               │
│  │                                              │               │
│  │  Encoder:  D_a → 256 → 512 → 1024           │               │
│  │   (ResBlock + ResBlock + Downsample) × 3     │               │
│  │                                              │               │
│  │  Middle:   1024 → 1024                       │               │
│  │   ResBlock + ResBlock                        │               │
│  │                                              │               │
│  │  Decoder:  1024 → 512 → 256                  │               │
│  │   (skip + ResBlock + ResBlock + Upsample) × 2│               │
│  │                                              │               │
│  │  Final: Conv → D_a                           │               │
│  │                                              │               │
│  │  FiLM conditioning at every ResBlock:        │               │
│  │    h = γ · h + β  (scale + shift)            │               │
│  │    from (timestep_embed ⊕ global_cond)       │               │
│  │                                              │               │
│  │  Output: predicted noise (B, H, D_a)         │               │
│  └──────────────────────────────────────────────┘               │
│                                                                 │
│  Inference: start from pure noise, denoise for 100 steps        │
│  Output: (B, horizon=16, D_a=24) action trajectory              │
│  Execute first 8 actions, re-plan                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Diffusion — Vision Encoder

### DiffusionRgbEncoder

Unlike ACT which uses ResNet feature maps as transformer tokens, Diffusion Policy compresses each camera image into a **single 64-dimensional vector** using SpatialSoftmax.

Key difference from ACT's ResNet18: Diffusion Policy uses **GroupNorm** instead of FrozenBatchNorm2d, and has **no ImageNet pretrained weights** (`pretrained_backbone_weights=None`). The backbone is trained from scratch.

```
Input image: (B, 3, H, W)
    │
    ▼ RandomCrop(crop_shape) during training / CenterCrop at eval
    │  Default crop_shape=(240,240), but for 84×84 images this is
    │  skipped (crop_shape > image size → validation error if enabled,
    │  or config must set crop_shape=None or a smaller value)
    │
    ▼ ResNet18 backbone (GroupNorm instead of BatchNorm, trained from scratch)
    │  Remove avgpool, fc layers
    │  Output: (B, 512, H', W') feature map
    │
    ▼ SpatialSoftmax(num_keypoints=32)
    │  For each of 32 keypoints:
    │    1. Conv2d(512, 1) → heatmap (H', W')
    │    2. Softmax over spatial dims → probability distribution
    │    3. Expected position = Σ(grid_coords × probs) → (x, y)
    │  Output: (B, 32, 2) → flatten → (B, 64)
    │
    ▼ Linear(64, 64) + ReLU
    │
    Output: (B, 64) per image
```

### SpatialSoftmax — what it does

SpatialSoftmax converts a feature map into **keypoint coordinates**. Instead of flattening the entire feature map (which would be huge and lose spatial invariance), it learns 32 "keypoint detectors" that each attend to a specific spatial location:

For keypoint $k$:
1. Produce a heatmap: $h_k(i,j) = W_k^T \cdot f(i,j) + b_k$ where $f(i,j)$ is the 512-dim feature at position $(i,j)$
2. Softmax over space: $p_k(i,j) = \frac{e^{h_k(i,j)}}{\sum_{i',j'} e^{h_k(i',j')}}$
3. Expected coordinates: $(x_k, y_k) = \sum_{i,j} p_k(i,j) \cdot (i, j)$

Result: 32 keypoints × 2 coordinates = 64-dim vector per image. This is a compact, spatially-aware representation.

### Separate encoder per camera

With `use_separate_rgb_encoder_per_camera=True` (default), each camera has its **own** ResNet18 backbone. This allows each camera to specialize (e.g., the wrist camera learns different features from the overview camera). ACT, in contrast, uses one shared backbone.

---

## 14. Diffusion — Conditional U-Net

### Architecture

The U-Net operates on action trajectories as **1D sequences** (time axis only, not spatial). It's conditioned on the observation encoding and the diffusion timestep via FiLM (Feature-wise Linear Modulation).

### FiLM conditioning

At every ResBlock, the features are modulated:

$$h' = \gamma \cdot h + \beta$$

where $\gamma, \beta$ are predicted from the conditioning vector:

$$[\gamma, \beta] = \text{Linear}(\text{timestep\_embed} \oplus \text{global\_cond})$$

This allows the U-Net to adapt its behavior based on what the robot currently sees and what diffusion timestep it's at.

### U-Net dimension flow

Input: noisy action trajectory $(B, \text{horizon}=16, D_a=24)$ → permuted to $(B, 24, 16)$

| Stage | Operation | Shape |
|---|---|---|
| **Encoder stage 1** | 2× ResBlock1d(24→256, kernel=5) + Downsample | $(B, 256, 8)$ |
| **Encoder stage 2** | 2× ResBlock1d(256→512, kernel=5) + Downsample | $(B, 512, 4)$ |
| **Encoder stage 3** | 2× ResBlock1d(512→1024, kernel=5) + Downsample | $(B, 1024, 2)$ |
| **Mid** | 2× ResBlock1d(1024→1024, kernel=5) | $(B, 1024, 2)$ |
| **Decoder stage 1** | Skip-cat(enc3) → 2× ResBlock1d(2048→512) + Upsample | $(B, 512, 4)$ |
| **Decoder stage 2** | Skip-cat(enc2) → 2× ResBlock1d(1024→256) + Upsample | $(B, 256, 8)$ |
| **Final** | Skip-cat(enc1) → Conv1dBlock(512→256) → Conv1d(256→24) | $(B, 24, 16)$ |

Output: $(B, 24, 16)$ → permuted back to $(B, 16, 24)$ = predicted noise $\hat{\epsilon}$

### Timestep embedding

The diffusion timestep $t$ is encoded using sinusoidal embeddings then passed through an MLP:

$$e_t = \text{MLP}(\text{SinusoidalEmbed}(t)) \in \mathbb{R}^{128}$$

This is concatenated with `global_cond` to form the full conditioning signal for FiLM.

---

## 15. Diffusion — Training (Noise Prediction)

### The forward diffusion process

Given a clean action trajectory $a_0 \in \mathbb{R}^{16 \times 24}$, gradually add Gaussian noise over $T=100$ timesteps:

$$a_t = \sqrt{\bar{\alpha}_t} \cdot a_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$ and $\beta_s$ follows a cosine schedule (`squaredcos_cap_v2`).

At $t=0$: $a_0$ is the clean signal.
At $t=T$: $a_T \approx \mathcal{N}(0, I)$ — pure noise.

### Training step

1. Encode observations into `global_cond`
2. Get ground-truth action trajectory $a_0$ from batch
3. Sample random $\epsilon \sim \mathcal{N}(0, I)$
4. Sample random timestep $t \sim \text{Uniform}\{1, ..., 100\}$
5. Create noisy trajectory: $a_t = \sqrt{\bar{\alpha}_t} \cdot a_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$
6. U-Net predicts: $\hat{\epsilon} = f_\theta(a_t, t, \text{global\_cond})$
7. Loss: $\mathcal{L} = \text{MSE}(\hat{\epsilon}, \epsilon) = \frac{1}{16 \times 24} \sum_{i,j} (\hat{\epsilon}_{i,j} - \epsilon_{i,j})^2$

### Why predict noise instead of the clean signal?

Predicting $\epsilon$ (the noise) rather than $a_0$ (the clean trajectory) has been shown empirically to produce better results. The noise prediction parameterization has a natural connection to score matching — the model learns the gradient of the data log-probability.

---

## 16. Diffusion — Inference (Iterative Denoising)

### The reverse process

Starting from pure Gaussian noise, iteratively denoise using the trained U-Net:

$$a_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(a_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon}_\theta(a_t, t, \text{obs})\right) + \sigma_t z$$

where $z \sim \mathcal{N}(0, I)$ and $\sigma_t$ is the noise schedule variance.

### Inference loop

```python
# 1. Encode observations into global_cond
global_cond = encode_observations(obs)

# 2. Start from pure noise
a_T = torch.randn(B, horizon, action_dim)  # (B, 16, 24)

# 3. Iteratively denoise (100 steps for DDPM)
for t in reversed(range(100)):
    predicted_noise = unet(a_t, t, global_cond)
    a_{t-1} = scheduler.step(predicted_noise, t, a_t)

# 4. Extract executable actions from clean trajectory
# Skip first (n_obs_steps-1)=1 actions, take next n_action_steps=8
actions = a_0[:, 1:9, :]  # (B, 8, 24)
```

### DDIM acceleration (optional)

DDPM requires 100 denoising steps at inference. DDIM (Denoising Diffusion Implicit Models) can skip steps, reducing to e.g. 10-20 steps with minimal quality loss. Set via `noise_scheduler_type="DDIM"` and `num_inference_steps`.

### Action queue for Diffusion

Like ACT, Diffusion Policy uses an action queue. But it only executes `n_action_steps=8` out of `horizon=16` predicted actions before re-planning. This overlap means predictions from consecutive chunks are averaged, providing smoother transitions.

---

## 17. Diffusion — Complete Dimension Trace

For TwoArmCoffee with 3 cameras at 84×84, n_obs_steps=2:

```
INPUT (2 timesteps):
  Each camera: (B, 2, 3, 84, 84)  per timestep
  State: (B, 2, 36)

RGB ENCODING (per camera, per timestep):
  Reshape: (B*2, 3, 84, 84)
  Crop: (B*2, 3, 84, 84) → (B*2, 3, crop_H, crop_W)  [adjusted for 84×84]
  ResNet18 backbone: (B*2, 512, H', W')
  SpatialSoftmax(32): (B*2, 64)
  Linear + ReLU: (B*2, 64)
  Reshape: (B, 2, 64) per camera

GLOBAL CONDITIONING:
  Flatten all features across time and cameras:
  [cam1_t-1, cam1_t, cam2_t-1, cam2_t, cam3_t-1, cam3_t, state_t-1, state_t]
  = (B, 3*64*2 + 36*2) = (B, 456)

DIFFUSION TRAINING:
  Ground-truth actions: (B, 16, 24)     [horizon × action_dim]
  Normalize to [-1, 1] via MIN_MAX

  Sample ε ~ N(0,I): (B, 16, 24)
  Sample t ~ Uniform{1..100}
  Noisy actions: (B, 16, 24)

  U-NET:
    Input: (B, 24, 16)  [permuted: channels × time]
    + timestep embed: (B, 128)
    + global_cond: (B, 456)
    FiLM cond: (B, 128+456) = (B, 584)

    Encoder: 24→256→512→1024, time: 16→8→4→2
    Mid: 1024, time: 2
    Decoder: 1024→512→256, time: 2→4→8
    Final: 256→24, time: 8→16

    Output: (B, 24, 16) → (B, 16, 24)  predicted noise

  Loss: MSE((B,16,24) pred_noise, (B,16,24) true_noise) → scalar

DIFFUSION INFERENCE:
  Sample a_T ~ N(0,I): (B, 16, 24)
  For t = 99, 98, ..., 0:
    ε̂ = UNet(a_t, t, global_cond)
    a_{t-1} = denoise_step(a_t, ε̂, t)
  
  Unnormalize a_0: (B, 16, 24)
  Extract actions[1:9]: (B, 8, 24)
  Queue: pop 1 per step, re-plan after 8 steps
```

---

## 18. Side-by-Side Comparison

| Property | ACT | Diffusion Policy |
|---|---|---|
| **Paper** | Zhao et al., 2023 | Chi et al., 2023 |
| **Generation paradigm** | Direct regression | Iterative denoising |
| **Core network** | Transformer encoder-decoder | 1D U-Net with FiLM |
| **Vision encoder** | ResNet18 (shared) → 2D feature map → 27 tokens | ResNet18 (per camera) → SpatialSoftmax → 64-dim vector |
| **Observation history** | 1 frame | 2 frames |
| **Chunk size / horizon** | 20 | 16 |
| **Actions executed** | 20 (all) | 8 (first half) |
| **Loss function** | L1 on actions (±KL) | MSE on noise |
| **State normalization** | MEAN_STD | MIN_MAX $[-1,1]$ |
| **Action normalization** | MEAN_STD | MIN_MAX $[-1,1]$ |
| **Multimodality handling** | Optional VAE | Inherent in diffusion |
| **Temporal ensembling** | Optional (exponential weights) | Overlapping chunks (predict 16, execute 8) |
| **Inference cost** | 1 forward pass per 20 steps | 100 forward passes per 8 steps |
| **Training loss** | $\|a - \hat{a}\|_1$ | $\|\epsilon - \hat{\epsilon}\|_2^2$ |
| **Model parameters** | ~34M | ~75M |
| **Best for** | Speed, simplicity | Multi-modal tasks, contact-rich manipulation |

### When to use which?

- **ACT**: Faster inference (20× fewer model calls), simpler to tune, works well for tasks with relatively deterministic solutions. Good default choice.
- **Diffusion**: Better handles multi-modal action distributions (multiple valid solutions), produces smoother trajectories for contact-rich tasks, but slower at inference and harder to tune (noise schedule, number of steps, etc.).

Both can serve as the frozen BC base for residual RL fine-tuning in the ResFiT pipeline. The choice depends on task complexity and compute budget.
