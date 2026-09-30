# Vision Backbone: ResNet18 — Deep Dive with Worked Examples

How camera images become transformer tokens in ACT (and feature vectors in Diffusion Policy). Covers the ResNet18 architecture, image normalization, multi-camera handling, and what `(B, 512, 3, 3)` actually means.

Back to: [BC_POLICY_TRAINING.md](BC_POLICY_TRAINING.md) | [ACT_ARCHITECTURE.md](ACT_ARCHITECTURE.md)

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Image Normalization — Where Do Those Numbers Come From?](#2-image-normalization--where-do-those-numbers-come-from)
3. [ResNet18 Architecture Layer by Layer](#3-resnet18-architecture-layer-by-layer)
4. [What Is (B, 512, 3, 3)?](#4-what-is-b-512-3-3)
5. [From Feature Map to Transformer Tokens](#5-from-feature-map-to-transformer-tokens)
6. [Multi-Camera Handling — Shared vs Separate](#6-multi-camera-handling--shared-vs-separate)
7. [Is the Backbone Trained or Frozen?](#7-is-the-backbone-trained-or-frozen)
8. [What Changes With Different Input Sizes?](#8-what-changes-with-different-input-sizes)
9. [ACT vs Diffusion Policy: Side-by-Side](#9-act-vs-diffusion-policy-side-by-side)
10. [During Residual RL — What Happens to the Encoder?](#10-during-residual-rl--what-happens-to-the-encoder)
11. [Complete Numerical Walkthrough](#11-complete-numerical-walkthrough)

---

## 1. The Big Picture

```
Camera Image (84×84×3)
    │
    ▼
Normalize (per-channel mean/std from YOUR dataset)
    │
    ▼
ResNet18 (conv1 → layer1 → layer2 → layer3 → layer4)
    │
    ▼ Feature Map: (B, 512, 3, 3) ← 512 features at 3×3 spatial grid
    │
    ▼
1×1 Conv Projection (512 → 512)
    │
    ▼ Projected: (B, 512, 3, 3)
    │
    ▼
Reshape to tokens: (9, B, 512)  ← 9 tokens = 3×3 spatial positions
    │
    ▼
+ 2D Sinusoidal Positional Embedding
    │
    ▼
Fed into ACT Transformer Encoder alongside [latent], [state] tokens
```

Each **camera** produces **9 spatial tokens** (for 84×84 input). With 2 cameras, you get 18 image tokens. These are concatenated with the latent token and state token(s), giving the transformer a rich visual context.

---

## 2. Image Normalization — Where Do Those Numbers Come From?

### Short Answer

The mean and std values are **computed from YOUR dataset**, NOT the fixed ImageNet constants. They are dataset-specific.

### How Stats Are Computed

**File**: `deps/lerobot/lerobot/common/datasets/compute_stats.py`

When you create a LeRobot dataset, it computes statistics per feature:

```python
def compute_episode_stats(episode_data, features):
    for key, data in episode_data.items():
        if features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)        # load sampled images as uint8
            axes_to_reduce = (0, 2, 3)                # reduce over (batch, H, W)
            keepdims = True                            # keep channel dim → shape (C, 1, 1)
        # ...
        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
        
        if features[key]["dtype"] in ["image", "video"]:
            # Normalize from [0, 255] range to [0, 1] range
            ep_stats[key] = {k: v / 255.0 for k, v in ep_stats[key].items()}
```

Key points:
- Images loaded as **uint8** [0, 255]
- Stats computed per-channel (reduce over batch, H, W) → shape **(C, 1, 1)** = **(3, 1, 1)**
- Stats then **divided by 255** to match the [0, 1] range that the model expects
- Stats aggregated across episodes using parallel variance algorithm

### What the Stats Look Like

For a robosuite simulation dataset (not natural photos), the values will **NOT** be ImageNet-like. They depend on the scene:

| Stat | Typical ImageNet | Typical Robosuite |
|---|---|---|
| Mean R | 0.485 | ~0.3–0.5 (depends on table/objects) |
| Mean G | 0.456 | ~0.2–0.4 |
| Mean B | 0.406 | ~0.2–0.4 |
| Std R | 0.229 | ~0.1–0.3 |
| Std G | 0.224 | ~0.1–0.2 |
| Std B | 0.225 | ~0.1–0.2 |

The values I listed earlier (`0.485, 0.456, 0.406` / `0.229, 0.224, 0.225`) were **illustrative approximations** — a habit from ImageNet-pretrained models. In practice, each dataset will have its own stats computed and stored.

### The Normalization Formula

Applied by `Normalize` module in `deps/lerobot/lerobot/common/policies/normalize.py`:

For `MEAN_STD` mode:

$$x_{\text{norm}} = \frac{x - \mu}{\sigma + 10^{-8}}$$

where $\mu$ and $\sigma$ are per-channel, shape $(3, 1, 1)$, broadcast across H and W.

### What Range Do Normalized Values End Up In?

| Feature Type | Normalization Mode | Output Range |
|---|---|---|
| **Images** | `MEAN_STD` per-channel | **Unbounded** — roughly $[-3, +4]$ depending on data distribution. NOT clipped to [-1, 1]. |
| **Robot State** | `MEAN_STD` per-dimension | **Unbounded** — roughly $[-3, +3]$ (most values within 3 standard deviations). NOT clipped. |
| **Actions** | `MEAN_STD` per-dimension | **Unbounded** — same as state. The model predicts in this z-scored space, then `Unnormalize` converts back. |

**Important**: MEAN_STD normalization produces z-scores. Unlike MIN_MAX (which maps to exactly [-1, 1]), z-scores are unbounded. Outlier values can exceed ±3.

### Are Stats the Same Every Time?

**No.** Stats are computed fresh for each dataset. Different datasets (different tasks, cameras, lighting) will have different per-channel means and stds. The stats are saved with the dataset and loaded into the model's `Normalize` buffer during training.

---

## 3. ResNet18 Architecture Layer by Layer

ResNet18 is a **18-layer deep** convolutional neural network from Microsoft Research (2015). The "18" counts convolutional layers (not including pooling or FC).

### Full Architecture

```
Input: (B, 3, 84, 84)         ← 3 color channels, 84×84 pixels
│
├─ conv1: Conv2d(3→64, 7×7, stride=2, pad=3)
│  Output: (B, 64, 42, 42)    ← spatial: 84/2 = 42
│
├─ bn1: FrozenBatchNorm2d(64)  ← BatchNorm with frozen running stats
├─ relu
│
├─ maxpool: MaxPool2d(3×3, stride=2, pad=1)
│  Output: (B, 64, 21, 21)    ← spatial: 42/2 = 21
│
├─ layer1: 2× BasicBlock(64→64)
│  │  ┌─ conv(64→64, 3×3) → BN → ReLU → conv(64→64, 3×3) → BN ─┐
│  │  │                                                           │
│  │  └──── + (identity shortcut) ────────────────────────────────┘
│  │  → ReLU
│  Output: (B, 64, 21, 21)    ← NO spatial change (stride=1)
│
├─ layer2: 2× BasicBlock(64→128, stride=2 on first block)
│  │  First block: conv(64→128, 3×3, stride=2) + downsample shortcut
│  │  Second block: conv(128→128, 3×3) identity shortcut
│  Output: (B, 128, 11, 11)   ← spatial: 21/2 = 11 (with ceil)
│
├─ layer3: 2× BasicBlock(128→256, stride=2 on first block)
│  Output: (B, 256, 6, 6)     ← spatial: 11/2 = 6 (with ceil)
│
├─ layer4: 2× BasicBlock(256→512, stride=2 on first block)
│  Output: (B, 512, 3, 3)     ← spatial: 6/2 = 3
│                                  ↑↑↑ THIS is what (B, 512, 3, 3) means! ↑↑↑
│
├─ avgpool: AdaptiveAvgPool2d(1,1)  ← NOT USED in ACT (stripped by IntermediateLayerGetter)
├─ fc: Linear(512, 1000)           ← NOT USED in ACT (stripped)
```

### BasicBlock (Residual Block)

Each BasicBlock has two 3×3 convolutions with a skip connection:

```
Input x
  │
  ├─ conv1(3×3) → BN → ReLU
  ├─ conv2(3×3) → BN
  │
  + ←── shortcut(x)  [identity if channels match, else 1×1 conv to match]
  │
  ReLU
  │
Output
```

The skip connection is what makes it a "residual" network — each block learns a residual correction on top of the identity mapping.

### Parameter Count

| Layer | Params | Cumulative |
|---|---|---|
| conv1 (7×7, 3→64) | 9,408 | 9,408 |
| layer1 (64→64, ×2 blocks) | 147,456 | 156,864 |
| layer2 (64→128, ×2 blocks) | 525,568 | 682,432 |
| layer3 (128→256, ×2 blocks) | 2,099,712 | 2,782,144 |
| layer4 (256→512, ×2 blocks) | 8,393,728 | 11,175,872 |
| **Total backbone** (no FC) | **~11.2M** | |
| 1×1 projection conv | 262,656 | 11,438,528 |

---

## 4. What Is (B, 512, 3, 3)?

This is the **feature map** output from `layer4` of ResNet18:

- **B** = batch size (e.g., 32)
- **512** = number of feature channels (each channel detects a different pattern)
- **3, 3** = spatial dimensions (height × width of the feature map)

### Why 3×3?

The spatial dimensions shrink as the image passes through stride-2 layers:

$$84 \xrightarrow{\text{conv1, s=2}} 42 \xrightarrow{\text{maxpool, s=2}} 21 \xrightarrow{\text{layer2, s=2}} 11 \xrightarrow{\text{layer3, s=2}} 6 \xrightarrow{\text{layer4, s=2}} 3$$

Five halving operations: $84 \div 2^5 = 84 \div 32 = 2.625 \rightarrow 3$ (due to padding/ceiling).

### What Does Each Position Represent?

Each of the 3×3 = 9 positions in the feature map corresponds to a **receptive field** covering a region of the original image:

```
Original 84×84 image:
┌─────────────────────────────────────────┐
│ region(0,0) │ region(0,1) │ region(0,2) │
│   ~28×28    │   ~28×28    │   ~28×28    │
├─────────────┼─────────────┼─────────────┤
│ region(1,0) │ region(1,1) │ region(1,2) │
│   ~28×28    │   ~28×28    │   ~28×28    │
├─────────────┼─────────────┼─────────────┤
│ region(2,0) │ region(2,1) │ region(2,2) │
│   ~28×28    │   ~28×28    │   ~28×28    │
└─────────────┴─────────────┴─────────────┘

Feature map 3×3:
┌─────┬─────┬─────┐
│ f₀₀ │ f₀₁ │ f₀₂ │  ← 512-dim vector at each position
├─────┼─────┼─────┤     summarizes what's in that image region
│ f₁₀ │ f₁₁ │ f₁₂ │
├─────┼─────┼─────┤
│ f₂₀ │ f₂₁ │ f₂₂ │
└─────┴─────┴─────┘
```

Each position has a **512-dimensional** feature vector capturing what patterns were detected in that region — edges, textures, object parts, etc.

The theoretical receptive field of each position is actually larger than 28×28 (due to padding), but the **effective** receptive field is roughly this size.

---

## 5. From Feature Map to Transformer Tokens

**File**: `resfit/lerobot/policies/act/modeling_act.py`, lines 557–576

### Step 1: 1×1 Convolution Projection

```python
self.encoder_img_feat_input_proj = nn.Conv2d(
    backbone_model.fc.in_features,  # 512 for ResNet18
    config.dim_model,               # 512 (transformer dimension)
    kernel_size=1
)
```

Maps each 512-dim feature to 512-dim transformer space. For ResNet18 with `dim_model=512`, this is 512→512 (same dimensionality, but learned linear projection per spatial position).

For ResNet50,`fc.in_features` = 2048, so this would be a 2048→512 compression.

### Step 2: Reshape to Sequence

```python
cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
# (B, 512, 3, 3) → (9, B, 512)
```

The 3×3 spatial grid is flattened into **9 tokens**, each of dimension 512. This is exactly the format the transformer expects: `(sequence_length, batch, features)`.

### Step 3: Add 2D Sinusoidal Positional Embedding

```python
cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)
```

Uses `ACTSinusoidalPositionEmbedding2d` with `dim_model // 2 = 256` — generates sin/cos embeddings for each (x, y) position in the feature map. This tells the transformer **where** each token came from spatially.

### The Token Sequence

For a setup with 2 cameras (e.g., `shouldercamera0`, `shouldercamera1`):

```
Token Index:  [0]    [1]      [2]         [3..11]           [12..20]
Content:     latent  state   (env_state)  cam0 (9 tokens)   cam1 (9 tokens)
              ↑       ↑                    ↑                  ↑
           learned  projected           ResNet18           ResNet18
           z~N(0,1) from MLP          feature map        feature map
```

Total: 2 + 9×2 = **20 tokens** fed to the transformer encoder (or 21 with env_state).

---

## 6. Multi-Camera Handling — Shared vs Separate

### ACT: One Shared Backbone for All Cameras

```python
# In ACT — ONE self.backbone used in a loop
for img in batch["observation.images"]:
    cam_features = self.backbone(img)["feature_map"]   # same backbone!
    cam_features = self.encoder_img_feat_input_proj(cam_features)  # same projection!
    all_cam_features.append(cam_features)
```

- **1 ResNet18 instance** processes all cameras sequentially
- **1 projection layer** shared across cameras
- The transformer uses positional embeddings to distinguish cameras (tokens from different cameras are concatenated along the sequence dimension)
- **Why?** Parameter efficiency + the transformer can learn cross-camera attention

### Diffusion Policy: Separate Encoder Per Camera (by default)

```python
# In Diffusion Policy — separate encoders
if self.config.use_separate_rgb_encoder_per_camera:  # default: True
    encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
    self.rgb_encoder = nn.ModuleList(encoders)
else:
    self.rgb_encoder = DiffusionRgbEncoder(config)
```

- **N ResNet18 instances** (one per camera) by default
- Each camera gets its own weights
- **Why?** No transformer to distinguish cameras — features are concatenated directly into a flat conditioning vector

### Summary

| | ACT | Diffusion Policy |
|---|---|---|
| **Number of backbone instances** | 1 (shared) | N (one per camera, by default) |
| **How cameras are distinguished** | Positional encoding in transformer | Separate weights per camera |
| **Output per camera** | 9 tokens of dim 512 | 1 vector of dim 64 |
| **Config to change** | N/A (always shared) | `use_separate_rgb_encoder_per_camera` |

### Same During Training and Evaluation

Yes — the encoder architecture is identical during training and evaluation. During eval, the same checkpoint is loaded and the same forward pass runs.

---

## 7. Is the Backbone Trained or Frozen?

### Short Answer

**Trained (fine-tuned) with a 10× lower learning rate.** BatchNorm statistics are frozen, but convolutional weights are trainable.

### Details

**File**: `resfit/lerobot/policies/act/configuration_act.py`

```python
pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
optimizer_lr: float = 1e-4           # main network learning rate
optimizer_lr_backbone: float = 1e-5  # backbone learning rate (10× lower)
```

**File**: `resfit/lerobot/policies/act/modeling_act.py`

```python
# Backbone uses FrozenBatchNorm2d (stats frozen, affine params trainable)
backbone_model = getattr(torchvision.models, config.vision_backbone)(
    weights=config.pretrained_backbone_weights,
    norm_layer=FrozenBatchNorm2d,  # ← running mean/var frozen from ImageNet
)

# Separate optimizer param groups
def get_optim_params(self):
    return [
        {"params": [p for n, p in self.named_parameters() 
                     if not n.startswith("model.backbone") and p.requires_grad]},
        {"params": [p for n, p in self.named_parameters() 
                     if n.startswith("model.backbone") and p.requires_grad],
         "lr": self.config.optimizer_lr_backbone},  # 1e-5
    ]
```

### What's Frozen vs Trainable?

| Component | Status | Why |
|---|---|---|
| Conv weights (all layers) | **Trainable** at LR 1e-5 | Fine-tune features for robot domain |
| BatchNorm running mean/var | **Frozen** | Preserve ImageNet statistics for stable training |
| BatchNorm γ, β (affine) | **Trainable** at LR 1e-5 | Allow per-channel scale/shift adaptation |
| 1×1 projection conv | **Trainable** at LR 1e-4 | New layer, needs full LR |

### Why 10× Lower LR?

The backbone starts from ImageNet weights that already know useful low-level (edges, textures) and mid-level (shapes, parts) features. Aggressive fine-tuning would destroy these features. The lower LR allows gentle adaptation to the robot vision domain while preserving general visual knowledge.

### `FrozenBatchNorm2d` Explained

Standard BatchNorm tracks running statistics (mean, variance) during training and uses them during inference. `FrozenBatchNorm2d` instead:
- Keeps the running mean and variance **fixed** at their ImageNet-pretrained values
- Still has **learnable** scale (γ) and shift (β) parameters
- This prevents the BatchNorm statistics from changing due to the very different distribution of robot images vs ImageNet images, which would destabilize early training

---

## 8. What Changes With Different Input Sizes?

The ResNet has no fully-connected layers (those are stripped), so it accepts **any input size**. The spatial dimensions of the output change:

### Spatial Dimension Formula

$$H_{\text{out}} = \left\lceil \frac{H_{\text{in}}}{32} \right\rceil, \quad W_{\text{out}} = \left\lceil \frac{W_{\text{in}}}{32} \right\rceil$$

(approximately — actual values depend on padding behavior)

### Common Input Sizes

| Input Size | layer4 Output | Tokens Per Camera | Total Tokens (2 cams) |
|---|---|---|---|
| **84 × 84** | (B, 512, **3, 3**) | **9** | 18 + 2 = 20 |
| **96 × 96** | (B, 512, **3, 3**) | **9** | 18 + 2 = 20 |
| **112 × 112** | (B, 512, **4, 4**) | **16** | 32 + 2 = 34 |
| **224 × 224** | (B, 512, **7, 7**) | **49** | 98 + 2 = 100 |
| **480 × 640** | (B, 512, **15, 20**) | **300** | 600 + 2 = 602 |

### Impact on Transformer

More tokens = more computation in self-attention ($O(n^2)$ complexity):
- 84×84: 20 tokens → 400 attention pairs ✓ fast
- 224×224: 100 tokens → 10,000 attention pairs — 25× more expensive
- 480×640: 602 tokens → 362,404 attention pairs — impractical

This is why the codebase uses **84×84** images — it balances visual detail with transformer efficiency.

### `replace_final_stride_with_dilation`

Setting this to `True` replaces layer4's stride=2 with dilated convolutions:
- layer4 maintains spatial resolution instead of halving it
- Output becomes (B, 512, **6, 6**) instead of (B, 512, 3, 3)
- Gives 36 tokens per camera instead of 9
- More spatial detail but more computation
- Default: `False`

---

## 9. ACT vs Diffusion Policy: Side-by-Side

### Architecture Comparison

| Aspect | ACT | Diffusion Policy |
|---|---|---|
| **Backbone** | ResNet18 | ResNet18 |
| **Pretrained?** | Yes (ImageNet) | **No** (from scratch) |
| **Norm layer** | FrozenBatchNorm2d | **GroupNorm** (BatchNorm replaced) |
| **Backbone LR** | 1e-5 (10× lower) | Same as network (no separate group) |
| **Feature extraction** | `IntermediateLayerGetter(layer4)` | `nn.Sequential(children[:-2])` |
| **Pooling** | None (keep spatial) | **SpatialSoftmax** → 32 keypoints |
| **Output per camera** | 2D feature map → **9 tokens** | **64-dim vector** (32 keypoints × 2 coords) |
| **Multi-camera** | 1 shared backbone | N separate backbones (default) |
| **Image augmentation** | None | Random crop (train) / center crop (eval) |

### Why the Differences?

**ACT** uses a transformer encoder that can naturally handle variable-length token sequences with positional encodings. Spatial features are preserved as tokens — the transformer decides what to attend to.

**Diffusion Policy** uses the image features as a **flat conditioning vector** for the U-Net denoiser. It needs a fixed-size representation, hence SpatialSoftmax which produces exactly 64 values regardless of spatial dimensions.

**ACT** can use pretrained ImageNet weights because `FrozenBatchNorm2d` preserves the BatchNorm statistics. **Diffusion Policy** replaces BatchNorm with GroupNorm (which has no running statistics), so pretrained weights would be corrupted — hence training from scratch.

### SpatialSoftmax (Diffusion Policy only)

Instead of flattening the feature map into tokens, SpatialSoftmax treats each channel as a heatmap and extracts the expected (x, y) position of the maximum:

$$\text{softmax}_c(i, j) = \frac{e^{f_c(i,j)}}{\sum_{i',j'} e^{f_c(i',j')}}$$

$$(x_c, y_c) = \sum_{i,j} \text{softmax}_c(i,j) \cdot (i, j)$$

With 32 keypoints, this gives a 64-dimensional vector (32×2) — a very compact representation that captures **where** important features are, not what they are.

---

## 10. During Residual RL — What Happens to the Encoder?

During residual RL fine-tuning, the **ACT base policy's ResNet18 is completely frozen** and used only for inference:

```python
# In BasePolicyVecEnvWrapper
self.base_policy.eval()  # frozen, no grad
for param in self.base_policy.parameters():
    param.requires_grad_(False)
```

The **RL agent has its own vision encoder** — a separate, smaller ViT (MinViT):

| | BC (ACT) Encoder | RL (Residual) Encoder |
|---|---|---|
| **Architecture** | ResNet18 (~11.2M params) | MinViT (~1.5M params) |
| **Input** | 84×84 images | 84×84 images |
| **Output** | (B, 512, 3, 3) → 9 tokens | (B, 81, 128) → spatial embedding |
| **Pretrained?** | ImageNet | No (from scratch, trained during RL) |
| **During RL** | Frozen inside base policy | Trained with critic/actor |

The RL encoder is smaller and trained from scratch on the RL objective. The base policy's encoder is frozen — it only runs during the base policy forward pass to produce base actions.

---

## 11. Complete Numerical Walkthrough

Let's trace a single 84×84 image through the full pipeline.

### Input

Camera image: shape `(3, 84, 84)`, pixel values in [0, 255] (uint8)

After conversion to float: `(3, 84, 84)`, values in [0.0, 1.0]

### Step 1: Normalize

Dataset stats (computed for this specific dataset):
- mean = [0.42, 0.35, 0.31] (shape: (3, 1, 1))
- std = [0.18, 0.15, 0.14] (shape: (3, 1, 1))

For a pixel with RGB = [0.7, 0.5, 0.3]:

$$R_{\text{norm}} = \frac{0.7 - 0.42}{0.18} = \frac{0.28}{0.18} = 1.56$$

$$G_{\text{norm}} = \frac{0.5 - 0.35}{0.15} = \frac{0.15}{0.15} = 1.00$$

$$B_{\text{norm}} = \frac{0.3 - 0.31}{0.14} = \frac{-0.01}{0.14} = -0.07$$

Normalized pixel: [1.56, 1.00, -0.07] — **unbounded**, not [-1, 1]

### Step 2: conv1 (7×7, stride 2)

Input: (1, 3, 84, 84)

$$H_{\text{out}} = \lfloor(84 - 7 + 2 \times 3) / 2\rfloor + 1 = \lfloor 83/2\rfloor + 1 = 42$$

Output: (1, 64, 42, 42)

Each of the 64 filters detects a different low-level pattern (edges, colors, textures).

### Step 3: FrozenBatchNorm2d + ReLU

$$x = \text{ReLU}\left(\gamma \cdot \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}} + \beta\right)$$

μ_running and σ²_running are frozen at ImageNet values. γ, β are trainable.

### Step 4: MaxPool (3×3, stride 2)

$$H_{\text{out}} = \lfloor(42 - 3 + 2 \times 1) / 2\rfloor + 1 = \lfloor 41/2\rfloor + 1 = 21$$

Output: (1, 64, 21, 21)

### Step 5: layer1 (no spatial change)

Two BasicBlocks, 64→64, stride 1.

Output: (1, 64, 21, 21)

### Step 6: layer2 (halve spatial)

Two BasicBlocks, 64→128, first has stride 2.

$$H_{\text{out}} = \lfloor(21 - 3 + 2 \times 1) / 2\rfloor + 1 = \lfloor 20/2\rfloor + 1 = 11$$

Output: (1, 128, 11, 11)

### Step 7: layer3

Two BasicBlocks, 128→256, first has stride 2.

$$H_{\text{out}} = \lfloor(11 - 3 + 2 \times 1) / 2\rfloor + 1 = \lfloor 10/2\rfloor + 1 = 6$$

Output: (1, 256, 6, 6)

### Step 8: layer4

Two BasicBlocks, 256→512, first has stride 2.

$$H_{\text{out}} = \lfloor(6 - 3 + 2 \times 1) / 2\rfloor + 1 = \lfloor 5/2\rfloor + 1 = 3$$

Output: **(1, 512, 3, 3)** ← the feature map!

### Step 9: 1×1 Projection

Conv2d(512→512, kernel=1) — learned linear projection per position.

Output: (1, 512, 3, 3)

### Step 10: Reshape to Tokens

```python
einops.rearrange("b c h w -> (h w) b c")
# (1, 512, 3, 3) → (9, 1, 512)
```

9 tokens, each 512-dimensional.

### Step 11: Add Positional Embedding

2D sinusoidal embedding encodes (row, col) position:
- Token 0 (pos 0,0): gets sin/cos encoding for top-left
- Token 4 (pos 1,1): gets sin/cos encoding for center
- Token 8 (pos 2,2): gets sin/cos encoding for bottom-right

### Step 12: Enter Transformer

These 9 tokens (per camera) are concatenated with:
- 1 latent token (from VAE or zeros)
- 1 state token (from robot state projection)

Total: 11 tokens (1 camera) or 20 tokens (2 cameras) → fed into ACT transformer encoder.

---

Back to: [BC_POLICY_TRAINING.md](BC_POLICY_TRAINING.md) | [ACT_ARCHITECTURE.md](ACT_ARCHITECTURE.md)
