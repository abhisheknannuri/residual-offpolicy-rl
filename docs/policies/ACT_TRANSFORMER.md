# The Transformer Inside ACT — Deep Dive with Math & Worked Examples

A complete explanation of **how the transformer encoder-decoder works** inside ACT (Action Chunking with Transformers). Every operation is traced with concrete numbers — from raw tokens all the way to predicted actions.

Back to: [ACT_ARCHITECTURE.md](ACT_ARCHITECTURE.md) | [BC_POLICY_TRAINING.md](BC_POLICY_TRAINING.md) | [VISION_BACKBONE.md](VISION_BACKBONE.md)

---

## Table of Contents

1. [What Is a Transformer? (30-Second Version)](#1-what-is-a-transformer-30-second-version)
2. [Self-Attention from Scratch](#2-self-attention-from-scratch)
3. [Multi-Head Attention](#3-multi-head-attention)
4. [The Encoder Layer — Full Math](#4-the-encoder-layer--full-math)
5. [The Decoder Layer — Full Math](#5-the-decoder-layer--full-math)
6. [Positional Embeddings — Why & How](#6-positional-embeddings--why--how)
7. [Putting It All Together — ACT's Token Flow](#7-putting-it-all-together--acts-token-flow)
8. [Full Numerical Example — 4 Tokens Through 1 Encoder Layer](#8-full-numerical-example--4-tokens-through-1-encoder-layer)
9. [Cross-Attention — How Actions Are Born](#9-cross-attention--how-actions-are-born)
10. [Why These Hyperparameters?](#10-why-these-hyperparameters)
11. [DETR Connection — Why ACT Looks Like Object Detection](#11-detr-connection--why-act-looks-like-object-detection)
12. [Code ↔ Math Mapping](#12-code--math-mapping)

---

## 1. What Is a Transformer? (30-Second Version)

A transformer is a neural network that processes a **set of tokens** (vectors) and lets each token look at every other token to decide what's important. It has two core operations:

1. **Self-attention**: Each token asks "which other tokens should I pay attention to?" and updates itself based on the answer.
2. **Feed-forward network (FFN)**: Each token is independently transformed through two linear layers with a non-linearity.

These two operations, wrapped in residual connections and layer normalization, form one **transformer layer**. Stack multiple layers and you get a deep transformer.

### In ACT specifically:

```
                    29 input tokens (1 latent + 1 state + 27 image)
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Encoder (4 layers)  │  ← Self-attention: tokens attend to each other
                    │  Self-Attn + FFN     │
                    └─────────┬───────────┘
                              │ 29 contextualized tokens
                              │
                    ┌─────────▼───────────┐
                    │  Decoder (1 layer)   │  ← Cross-attention: queries attend to encoder
                    │  Self-Attn           │
                    │  Cross-Attn + FFN    │
                    └─────────┬───────────┘
                              │ 20 action tokens
                              ▼
                    Linear(512 → 24) per token
                              │
                              ▼
                    20 predicted actions
```

---

## 2. Self-Attention from Scratch

### The intuition

Imagine 3 tokens: `[robot_state]`, `[image_center]`, `[image_top_left]`.

The robot state token wants to know: "Where is the gripper relative to the object?" This information is in the image tokens. Self-attention lets `[robot_state]` **query** all tokens, find that `[image_center]` has the relevant object features, and pull that information into its own representation.

### The math

Given $n$ input tokens, each of dimension $d$, stacked as $X \in \mathbb{R}^{n \times d}$:

**Step 1: Project to Q, K, V**

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$.

- $Q$ (Query): "What am I looking for?"
- $K$ (Key): "What do I contain?"
- $V$ (Value): "What information do I provide if selected?"

**Step 2: Compute attention scores**

$$S = \frac{Q K^T}{\sqrt{d_k}}$$

$S \in \mathbb{R}^{n \times n}$. Entry $S_{ij}$ measures how much token $i$'s query matches token $j$'s key.

The $\sqrt{d_k}$ scaling prevents the dot products from becoming too large (which would push softmax into saturation where gradients vanish).

**Step 3: Softmax to get attention weights**

$$A = \text{softmax}(S) \quad \text{(row-wise)}$$

$A_{ij} \in [0, 1]$ and $\sum_j A_{ij} = 1$ for each row $i$. These are the attention weights — how much token $i$ attends to token $j$.

**Step 4: Weighted sum of values**

$$\text{Output} = A V$$

Each output token is a weighted mixture of all value vectors, where the weights come from attention.

### Complete single-head formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

---

## 3. Multi-Head Attention

### Why multiple heads?

A single attention head can only focus on one type of relationship at a time. With 8 heads running in parallel, different heads can learn different things:

- Head 1 might attend to spatial neighbors ("what's next to me?")
- Head 2 might attend to the robot state ("where is the gripper?")
- Head 3 might attend to same-camera tokens ("what else is in this view?")
- Head 4 might look for color features
- etc.

### The math

In ACT: $d_{\text{model}} = 512$, $n_{\text{heads}} = 8$, so $d_k = d_v = 512 / 8 = 64$ per head.

**For each head $i \in \{1, ..., 8\}$:**

$$W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{512 \times 64}$$

$$\text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V) \in \mathbb{R}^{n \times 64}$$

**Concatenate all heads:**

$$\text{MultiHead}(X) = [\text{head}_1 \| \text{head}_2 \| ... \| \text{head}_8] W^O$$

where $[\cdot \| \cdot]$ is concatenation along the feature dimension, giving $\mathbb{R}^{n \times 512}$, and $W^O \in \mathbb{R}^{512 \times 512}$ is the output projection.

### Parameter count for one multi-head attention

| Parameter | Shape | Count |
|---|---|---|
| $W^Q$ (all heads packed) | $512 \times 512$ | 262,144 |
| $W^K$ | $512 \times 512$ | 262,144 |
| $W^V$ | $512 \times 512$ | 262,144 |
| $W^O$ | $512 \times 512$ | 262,144 |
| Biases ($b^Q, b^K, b^V, b^O$) | $4 \times 512$ | 2,048 |
| **Total** | | **1,050,624** |

PyTorch's `nn.MultiheadAttention` packs all head projections into single large matrices for efficiency.

---

## 4. The Encoder Layer — Full Math

**File**: `resfit/lerobot/policies/act/modeling_act.py`, class `ACTEncoderLayer`

ACT uses **post-norm** (default `pre_norm=False`). Each encoder layer has two sub-layers:

### Sub-layer 1: Self-Attention + Residual + LayerNorm

$$\begin{aligned}
q &= x + \text{pos\_embed} \\
k &= x + \text{pos\_embed} \\
v &= x \\
x' &= \text{MultiHeadAttn}(q, k, v) \\
x &= \text{LayerNorm}(x + \text{Dropout}(x'))
\end{aligned}$$

**Key detail**: Positional embeddings are added to Q and K **but NOT to V**. This means position information affects *which tokens attend to each other* (via Q·K compatibility) but doesn't directly modify *what information is passed* (V stays position-free).

Why? If you added pos_embed to V, then every output would have positional information mixed into the content. By keeping V clean, the network can choose to use or ignore position through the attention weights.

### Sub-layer 2: Feed-Forward Network + Residual + LayerNorm

$$\begin{aligned}
x' &= W_2 \cdot \text{Dropout}(\text{ReLU}(W_1 \cdot x + b_1)) + b_2 \\
x &= \text{LayerNorm}(x + \text{Dropout}(x'))
\end{aligned}$$

where $W_1 \in \mathbb{R}^{3200 \times 512}$, $W_2 \in \mathbb{R}^{512 \times 3200}$.

The FFN is applied **independently to each token** — no interaction between tokens. It's a per-token nonlinear transformation that lets each token process the information gathered from attention.

### Why 3200 for FFN dimension?

The original "Attention Is All You Need" paper uses $d_{\text{ff}} = 4 \times d_{\text{model}}$. For $d_{\text{model}} = 512$, that would be 2048. ACT uses 3200, which is $6.25 \times d_{\text{model}}$ — a slightly larger FFN for more capacity. This is inherited from the DETR object detection model that ACT is based on.

### FFN Parameter Count

| Parameter | Shape | Count |
|---|---|---|
| $W_1$ | $512 \times 3200$ | 1,638,400 |
| $b_1$ | $3200$ | 3,200 |
| $W_2$ | $3200 \times 512$ | 1,638,400 |
| $b_2$ | $512$ | 512 |
| **Total** | | **3,280,512** |

### Post-Norm vs Pre-Norm

**Post-norm** (ACT default): Apply LayerNorm **after** the residual addition.
```
x = LayerNorm(x + sublayer(x))
```

**Pre-norm**: Apply LayerNorm **before** the sublayer.
```
x = x + sublayer(LayerNorm(x))
```

Post-norm can be harder to train for very deep transformers (gradient issues), but for ACT's 4-layer encoder, it works fine. Pre-norm is available via `pre_norm=True` in the config.

---

## 5. The Decoder Layer — Full Math

**File**: `resfit/lerobot/policies/act/modeling_act.py`, class `ACTDecoderLayer`

The decoder layer has **three** sub-layers instead of two:

### Sub-layer 1: Self-Attention (among decoder queries)

$$\begin{aligned}
q &= x + \text{decoder\_pos\_embed} \\
k &= x + \text{decoder\_pos\_embed} \\
v &= x \\
x &= \text{LayerNorm}(x + \text{Dropout}(\text{SelfAttn}(q, k, v)))
\end{aligned}$$

The 20 query tokens attend to each other. This lets action predictions be temporally coherent — action at $t=5$ can look at what $t=4$ and $t=6$ are predicting. The `decoder_pos_embed` encodes which timestep each query represents.

### Sub-layer 2: Cross-Attention (decoder queries attend to encoder output)

$$\begin{aligned}
q &= x + \text{decoder\_pos\_embed} \\
k &= \text{encoder\_out} + \text{encoder\_pos\_embed} \\
v &= \text{encoder\_out} \\
x &= \text{LayerNorm}(x + \text{Dropout}(\text{CrossAttn}(q, k, v)))
\end{aligned}$$

This is the crucial step where **observation information flows into action predictions**:
- **Queries** (20): "I'm producing action for timestep $t$, what do I need to know?"
- **Keys** (29): "I'm the [latent/state/image_pixel] token, here's my identifier"
- **Values** (29): "Here's the actual information about the observation"

The cross-attention matrix is $(20 \times 29)$ — each action query computes a weighted sum over all 29 encoder tokens.

### Sub-layer 3: Feed-Forward Network

$$x = \text{LayerNorm}(x + \text{Dropout}(\text{FFN}(x)))$$

Same structure as encoder FFN: $512 \to 3200 \to 512$.

### Why Only 1 Decoder Layer?

The original ACT paper specifies `n_decoder_layers=7`, but a [bug in their code](https://github.com/tonyzhaozh/act/issues/25) meant only the first layer's output was used. This codebase faithfully matches the bug: `n_decoder_layers=1`. The model works well because:

1. The 4-layer encoder does most of the heavy lifting (fusing all modalities)
2. The single cross-attention step is sufficient to extract task-relevant features
3. The action head is just a linear layer — no complex post-processing needed

---

## 6. Positional Embeddings — Why & How

Transformers process tokens as a **set** — without positional information, they can't distinguish `[state, image]` from `[image, state]`. Positional embeddings inject position information.

### The 4 types of positional embeddings in ACT

#### Type 1: Encoder 1D Positional Embeddings (Learned)

```python
self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, 512)
```

- Applies to: latent token (position 0) and state token (position 1)
- **Learned** parameters — the network decides what position encoding works best
- Shape: $(2, 512)$

#### Type 2: Encoder 2D Sinusoidal Positional Embeddings (Fixed)

```python
self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(dim=256)
```

- Applies to: image feature map tokens (9 per camera)
- **Fixed** (not learned) — computed from the (row, col) position in the 3×3 feature map
- Encodes x-position in the first 256 dims, y-position in the last 256 dims

For a 3×3 grid, positions are normalized to $[0, 2\pi]$:

| Position | $(y, x)$ | $y_{\text{norm}}$ | $x_{\text{norm}}$ |
|---|---|---|---|
| top-left | (1, 1) | $\frac{1}{3} \times 2\pi = 2.09$ | $\frac{1}{3} \times 2\pi = 2.09$ |
| top-center | (1, 2) | $\frac{1}{3} \times 2\pi = 2.09$ | $\frac{2}{3} \times 2\pi = 4.19$ |
| top-right | (1, 3) | $\frac{1}{3} \times 2\pi = 2.09$ | $\frac{3}{3} \times 2\pi = 6.28$ |
| center | (2, 2) | $\frac{2}{3} \times 2\pi = 4.19$ | $\frac{2}{3} \times 2\pi = 4.19$ |
| bottom-right | (3, 3) | $\frac{3}{3} \times 2\pi = 6.28$ | $\frac{3}{3} \times 2\pi = 6.28$ |

Then for each dimension $j$:

$$\text{inv\_freq}(j) = 10000^{2\lfloor j/2 \rfloor / d}$$

$$\text{embed}(x, j) = \begin{cases} \sin(x_{\text{norm}} / \text{inv\_freq}(j)) & \text{if } j \text{ even} \\ \cos(x_{\text{norm}} / \text{inv\_freq}(j)) & \text{if } j \text{ odd} \end{cases}$$

Lower-frequency dimensions change slowly across positions (encoding coarse location), while higher-frequency dimensions change rapidly (encoding fine location).

#### Type 3: Decoder Positional Embeddings (Learned)

```python
self.decoder_pos_embed = nn.Embedding(20, 512)
```

- Applies to: 20 decoder query tokens
- **Learned** — each timestep gets a unique 512-dim embedding
- Tells the decoder "this query is responsible for predicting action at timestep $t$"

#### Type 4: VAE Encoder Positional Embeddings (Fixed Sinusoidal 1D)

Standard "Attention Is All You Need" sinusoidal embeddings:

$$\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Applied to the 22 tokens of the VAE encoder input (CLS + state + 20 actions).

### Why some are learned and some are fixed?

| Type | Why |
|---|---|
| Encoder 1D (learned) | Only 2-3 tokens — easy to learn, and the meaning of "latent" vs "state" has no geometric structure |
| Encoder 2D (fixed sinusoidal) | Image positions have spatial structure. Sinusoidal embeddings capture this and generalize to unseen positions. Shared across cameras — same grid encoding for all |
| Decoder (learned) | Temporal structure of action chunks could have complex non-linear dependencies. Learned embeddings give maximum flexibility |

---

## 7. Putting It All Together — ACT's Token Flow

### Step-by-step for TwoArmCoffee (3 cameras, 84×84)

**Input preparation:**

| Token | Source | Projection | Dim |
|---|---|---|---|
| `token[0]`: latent | $z \in \mathbb{R}^{32}$ (zeros or VAE sample) | `Linear(32, 512)` | 512 |
| `token[1]`: state | robot state $\in \mathbb{R}^{36}$ | `Linear(36, 512)` | 512 |
| `token[2..10]`: cam0 | ResNet18 feature map $(512, 3, 3)$ | `Conv2d(512, 512, 1)` + reshape | 512 each |
| `token[11..19]`: cam1 | same | same | 512 each |
| `token[20..28]`: cam2 | same | same | 512 each |

Total: **29 tokens**, each $\mathbb{R}^{512}$.

**Encoder (4 layers):**

```
Layer 1: tokens attend to each other → latent learns about state and images
Layer 2: deeper fusion → spatial relationships between camera views emerge
Layer 3: higher-level reasoning → "object is at position X, gripper is at position Y"
Layer 4: final contextualization → all tokens carry globally fused information
```

Each layer: self-attention (29×29 matrix, 8 heads) + FFN (512→3200→512).

**Decoder (1 layer):**

Input: 20 zero-init queries (one per future timestep)

```
Step 1: Self-attention (20×20) — queries coordinate with each other
Step 2: Cross-attention (20×29) — each query extracts relevant observation info
Step 3: FFN — per-query nonlinear transformation
```

**Action head:**

$$\text{action}_t = W_{\text{head}} \cdot \text{decoder\_out}_t + b_{\text{head}}, \quad W_{\text{head}} \in \mathbb{R}^{24 \times 512}$$

Output: $(B, 20, 24)$ — 20 actions, each 24-dimensional.

---

## 8. Full Numerical Example — 4 Tokens Through 1 Encoder Layer

Let's trace 4 tokens through one encoder layer with **2 heads** and $d_{\text{model}} = 4$ (simplified from ACT's 29 tokens, 8 heads, 512 dims — same math, just smaller numbers).

### Setup

4 tokens, each 4-dimensional:

$$X = \begin{bmatrix} 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 1.0 \\ 1.0 & 1.0 & 0.0 & 0.0 \\ 0.0 & 0.0 & 1.0 & 1.0 \end{bmatrix}$$

Rows: token 0 (latent), token 1 (state), token 2 (image pixel 1), token 3 (image pixel 2).

### Step 1: Compute Q, K, V for Head 1

With 2 heads and $d_{\text{model}} = 4$: $d_k = 4/2 = 2$.

Head 1 projection matrices (simplified — in reality these are learned):

$$W_1^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad W_1^K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad W_1^V = \begin{bmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$$

(In ACT, positional embeddings are added to Q and K first: $q = (x + \text{pos}) W^Q$, $k = (x + \text{pos}) W^K$. For simplicity, we skip pos_embed here.)

$$Q_1 = X W_1^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}, \quad K_1 = X W_1^K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}, \quad V_1 = X W_1^V = \begin{bmatrix} 1 & 1 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$$

### Step 2: Attention Scores

$$S_1 = \frac{Q_1 K_1^T}{\sqrt{d_k}} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1\cdot0+0\cdot1 & 1\cdot1+0\cdot0 & 1\cdot1+0\cdot1 & 1\cdot0+0\cdot0 \\ 0\cdot0+1\cdot1 & 0\cdot1+1\cdot0 & 0\cdot1+1\cdot1 & 0\cdot0+1\cdot0 \\ 1\cdot0+1\cdot1 & 1\cdot1+1\cdot0 & 1\cdot1+1\cdot1 & 1\cdot0+1\cdot0 \\ 0\cdot0+0\cdot1 & 0\cdot1+0\cdot0 & 0\cdot1+0\cdot1 & 0\cdot0+0\cdot0 \end{bmatrix}$$

$$= \frac{1}{1.414} \begin{bmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 \\ 1 & 1 & 2 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0.71 & 0.71 & 0 \\ 0.71 & 0 & 0.71 & 0 \\ 0.71 & 0.71 & 1.41 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

### Step 3: Softmax (row-wise)

$$A_1 = \text{softmax}(S_1) \approx \begin{bmatrix} 0.17 & 0.35 & 0.35 & 0.17 \\ 0.35 & 0.17 & 0.35 & 0.17 \\ 0.15 & 0.15 & 0.43 & 0.15 \\ 0.25 & 0.25 & 0.25 & 0.25 \end{bmatrix}$$

**Reading row 0** (latent token): It attends 35% to token 1 (state), 35% to token 2 (image), 17% to itself, 17% to token 3. The latent is gathering information from state and image — exactly what we want!

**Reading row 3** (image pixel 2): All scores were 0, so softmax gives uniform attention (25% each). This token has no strong preference — it will average all information equally.

### Step 4: Weighted Sum of Values

$$\text{head}_1 = A_1 V_1 \approx \begin{bmatrix} 0.17\cdot1+0.35\cdot0+0.35\cdot1+0.17\cdot0 & 0.17\cdot1+0.35\cdot0+0.35\cdot0+0.17\cdot1 \\ 0.35\cdot1+0.17\cdot0+0.35\cdot1+0.17\cdot0 & 0.35\cdot1+0.17\cdot0+0.35\cdot0+0.17\cdot1 \\ 0.15\cdot1+0.15\cdot0+0.43\cdot1+0.15\cdot0 & 0.15\cdot1+0.15\cdot0+0.43\cdot0+0.15\cdot1 \\ 0.25\cdot1+0.25\cdot0+0.25\cdot1+0.25\cdot0 & 0.25\cdot1+0.25\cdot0+0.25\cdot0+0.25\cdot1 \end{bmatrix}$$

$$= \begin{bmatrix} 0.52 & 0.34 \\ 0.70 & 0.52 \\ 0.58 & 0.30 \\ 0.50 & 0.50 \end{bmatrix}$$

### Step 5: Concatenate Heads + Output Projection

Head 2 would produce another $(4 \times 2)$ matrix. Concatenate both:

$$\text{concat} = [\text{head}_1 \| \text{head}_2] \in \mathbb{R}^{4 \times 4}$$

Then multiply by $W^O \in \mathbb{R}^{4 \times 4}$:

$$\text{attn\_out} = \text{concat} \cdot W^O \in \mathbb{R}^{4 \times 4}$$

### Step 6: Residual + LayerNorm

$$x' = \text{LayerNorm}(X + \text{Dropout}(\text{attn\_out}))$$

The original input $X$ is added back (residual connection), then LayerNorm normalizes each token independently.

### Step 7: FFN + Residual + LayerNorm

$$\text{ffn\_out} = W_2 \cdot \text{ReLU}(W_1 \cdot x' + b_1) + b_2$$

$$\text{output} = \text{LayerNorm}(x' + \text{Dropout}(\text{ffn\_out}))$$

### Key Takeaway

After one layer, every token has been updated with information from every other token. The latent and state tokens now carry image information; the image tokens now carry state information. After 4 such layers, all tokens are deeply fused.

---

## 9. Cross-Attention — How Actions Are Born

The decoder's cross-attention is where observation information becomes action predictions. Let's trace this carefully.

### Setup

- **Encoder output**: 29 tokens $\in \mathbb{R}^{29 \times 512}$ — fused observation context
- **Decoder queries**: 20 tokens $\in \mathbb{R}^{20 \times 512}$ — initialized as zeros, then go through self-attention first

### Cross-Attention Matrix

$$Q_{\text{dec}} = (\text{decoder\_tokens} + \text{decoder\_pos\_embed}) \cdot W^Q \in \mathbb{R}^{20 \times 64} \text{ per head}$$

$$K_{\text{enc}} = (\text{encoder\_out} + \text{encoder\_pos\_embed}) \cdot W^K \in \mathbb{R}^{29 \times 64} \text{ per head}$$

$$V_{\text{enc}} = \text{encoder\_out} \cdot W^V \in \mathbb{R}^{29 \times 64} \text{ per head}$$

Attention weights:

$$A = \text{softmax}\!\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{64}}\right) \in \mathbb{R}^{20 \times 29}$$

### What the 20×29 Attention Matrix Looks Like (Conceptual)

```
                    Encoder tokens (Keys)
                    ┌─────┬───────┬──────────────────────────────┐
                    │ lat │ state │ cam0(9) │ cam1(9) │ cam2(9)  │
    ┌───────────────┼─────┼───────┼─────────┼─────────┼──────────┤
    │ action_t=0    │ 0.0 │  0.3  │  0.05.. │  0.02.. │  0.01.. │
    │ action_t=1    │ 0.0 │  0.25 │  0.05.. │  0.02.. │  0.01.. │
D   │ action_t=2    │ 0.0 │  0.2  │  0.06.. │  0.03.. │  0.02.. │
e   │    ...        │     │       │         │         │          │
c   │ action_t=10   │ 0.0 │  0.1  │  0.04.. │  0.04.. │  0.04.. │
    │    ...        │     │       │         │         │          │
    │ action_t=19   │ 0.0 │  0.05 │  0.03.. │  0.03.. │  0.05.. │
    └───────────────┴─────┴───────┴─────────┴─────────┴──────────┘

    (Values are illustrative — actual weights are learned)
```

Possible patterns the network might learn:
- **Early timesteps** (t=0..5): Attend heavily to **state** (current gripper position matters most for immediate next actions) and nearby image pixels
- **Later timesteps** (t=15..19): Attend more to **image features** (need to plan toward the goal object, which is in the scene)
- **Latent token**: Rarely attended to when VAE is off (it's all zeros)

### Output

Each action query becomes a 512-dim vector — a weighted combination of encoder features tailored for that specific timestep:

$$\text{decoder\_out}_t = \sum_{j=0}^{28} A_{t,j} \cdot V_{\text{enc},j}$$

This goes through the FFN and then the action head `Linear(512, 24)` to produce one 24-dim action.

---

## 10. Why These Hyperparameters?

### ACT's transformer configuration

| Parameter | Value | Why |
|---|---|---|
| `dim_model` | 512 | Same as original DETR. Large enough for rich representations, small enough for 84×84 images |
| `n_heads` | 8 | Standard choice. $d_k = 512/8 = 64$ per head — well-studied sweet spot |
| `dim_feedforward` | 3200 | From DETR. ~6× model dim (vs standard 4×). Extra capacity for this task |
| `n_encoder_layers` | 4 | Sufficient for fusing 3 cameras + state. More layers = diminishing returns + slower training |
| `n_decoder_layers` | 1 | Bug-compatible with original ACT (see section 5). Works well in practice |
| `dropout` | 0.1 | Standard regularization. Prevents overfitting on limited robot data |
| `feedforward_activation` | ReLU | Simple, works well. GeLU also available but not default |
| `pre_norm` | False | Post-norm is DETR's original design. Pre-norm is optional |

### Compute cost

For one forward pass with $B=32$, the attention computation is:

**Encoder**: Per layer: $(29 \times 29) \times 8 \times 2 \times 64 = 860K$ multiply-adds for attention. 4 layers = 3.4M.

**Decoder**: $(20 \times 20) + (20 \times 29) = 980$ attention entries per head. 8 heads × 64 dim = 500K multiply-adds.

**FFN** dominates: $5 \times (29 \times 512 \times 3200 \times 2) = 476M$ multiply-adds.

Total per forward: ~500M multiply-adds — fast enough for 20Hz control on a GPU.

---

## 11. DETR Connection — Why ACT Looks Like Object Detection

ACT's architecture is directly adapted from **DETR** (DEtection TRansformer, Carion et al. 2020), a model for object detection. The mapping is:

| DETR | ACT |
|---|---|
| Input image | Input camera images (84×84 × 3 cameras) |
| ResNet backbone → feature map | Same: ResNet18 → (512, 3, 3) per camera |
| Feature map → tokens + 2D pos embed | Same: 9 tokens per camera + sinusoidal 2D pos embed |
| Transformer encoder (6 layers) | Transformer encoder (4 layers) |
| 100 object queries (learned) | 20 action queries (learned, one per timestep) |
| Transformer decoder (6 layers) | Transformer decoder (1 layer) |
| Object class + bounding box heads | Action head: Linear(512, 24) |
| Predicts: "what objects are where" | Predicts: "what actions to take when" |

The key insight: **predicting a sequence of future actions is structurally similar to predicting a set of object bounding boxes.** Both require:
1. Processing visual input through a backbone + transformer encoder
2. Using learned queries that cross-attend to visual features
3. Each query independently predicting its output (object/action)

The difference: DETR's queries are permutation-invariant (objects have no ordering), while ACT's queries have a temporal ordering (action at $t=3$ must come before $t=4$). This ordering is provided by the decoder positional embeddings.

---

## 12. Code ↔ Math Mapping

### Encoder Layer Code

```python
# File: resfit/lerobot/policies/act/modeling_act.py, class ACTEncoderLayer

def forward(self, x, pos_embed=None, key_padding_mask=None):
    skip = x                                    # Save for residual
    # Post-norm: no LayerNorm here (pre_norm=False)
    q = k = x if pos_embed is None else x + pos_embed  # Add pos to Q and K only
    x = self.self_attn(q, k, value=x, ...)[0]          # Multi-head self-attention
    x = skip + self.dropout1(x)                 # Residual connection
    x = self.norm1(x)                           # LayerNorm (post-norm)
    skip = x                                    # Save for second residual
    x = self.linear2(self.dropout(self.activation(self.linear1(x))))  # FFN: 512→3200→ReLU→512
    x = skip + self.dropout2(x)                 # Residual connection
    x = self.norm2(x)                           # LayerNorm (post-norm)
    return x
```

Maps to:

$$x \leftarrow \text{LN}(x + \text{drop}(\text{MHA}(x+p, x+p, x)))$$
$$x \leftarrow \text{LN}(x + \text{drop}(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2))$$

### Decoder Layer Code

```python
# File: resfit/lerobot/policies/act/modeling_act.py, class ACTDecoderLayer

def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
    skip = x
    q = k = x + decoder_pos_embed           # Self-attention among queries
    x = self.self_attn(q, k, value=x)[0]
    x = self.norm1(skip + self.dropout1(x))  # Residual + LayerNorm
    skip = x
    x = self.multihead_attn(
        query=x + decoder_pos_embed,         # Queries from decoder + pos
        key=encoder_out + encoder_pos_embed,  # Keys from encoder + pos
        value=encoder_out,                    # Values from encoder (no pos!)
    )[0]
    x = self.norm2(skip + self.dropout2(x))  # Residual + LayerNorm
    skip = x
    x = self.linear2(self.dropout(self.activation(self.linear1(x))))  # FFN
    x = self.norm3(skip + self.dropout3(x))  # Residual + LayerNorm
    return x
```

Maps to:

$$x \leftarrow \text{LN}(x + \text{drop}(\text{SelfAttn}(x+p_d, x+p_d, x)))$$
$$x \leftarrow \text{LN}(x + \text{drop}(\text{CrossAttn}(x+p_d, \text{enc}+p_e, \text{enc})))$$
$$x \leftarrow \text{LN}(x + \text{drop}(\text{FFN}(x)))$$

### Parameter count summary

| Component | Params |
|---|---|
| 1 encoder layer (self-attn + FFN + norms) | ~4.3M |
| 4 encoder layers | ~17.3M |
| 1 decoder layer (self-attn + cross-attn + FFN + norms) | ~5.4M |
| Encoder 1D pos embed | 1,024 |
| Decoder pos embed | 10,240 |
| Action head (512→24) | 12,312 |
| **Transformer total** | **~22.8M** |
| ResNet18 backbone | ~11.2M |
| Input projections | ~0.3M |
| **Full ACT model** | **~34.3M** |

---

Back to: [ACT_ARCHITECTURE.md](ACT_ARCHITECTURE.md) | [BC_POLICY_TRAINING.md](BC_POLICY_TRAINING.md) | [VISION_BACKBONE.md](VISION_BACKBONE.md)
