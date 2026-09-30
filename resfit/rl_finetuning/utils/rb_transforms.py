# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import NestedKey, TensorDictBase
from tensordict.utils import expand_right
from torchrl.envs.transforms.transforms import Transform


class MultiStepTransform(Transform):
    """A MultiStep transformation for ReplayBuffers.

    This transform keeps the previous ``n_steps`` observations in a local buffer.
    The inverse transform (called during :meth:`~torchrl.data.ReplayBuffer.extend`)
    outputs the transformed previous ``n_steps`` with the ``T-n_steps`` current
    frames.

    All entries in the ``"next"`` tensordict that are not part of the ``done_keys``
    or ``reward_keys`` will be mapped to their respective ``t + n_steps - 1``
    correspondent.

    This transform is a more hyperparameter resistant version of
    :class:`~torchrl.data.postprocs.postprocs.MultiStep`:
    the replay buffer transform will make the multi-step transform insensitive
    to the collectors hyperparameters, whereas the post-process
    version will output results that are sensitive to these
    (because collectors have no memory of previous output).

    Args:
        n_steps (int): Number of steps in multi-step. The number of steps can be
            dynamically changed by changing the ``n_steps`` attribute of this
            transform.
        gamma (:obj:`float`): Discount factor.

    Keyword Args:
        reward_keys (list of NestedKey, optional): the reward keys in the input tensordict.
            The reward entries indicated by these keys will be accumulated and discounted
            across ``n_steps`` steps in the future. A corresponding ``<reward_key>_orig``
            entry will be written in the ``"next"`` entry of the output tensordict
            to keep track of the original value of the reward.
            Defaults to ``["reward"]``.
        done_key (NestedKey, optional): the done key in the input tensordict, used to indicate
            an end of trajectory.
            Defaults to ``"done"``.
        done_keys (list of NestedKey, optional): the list of end keys in the input tensordict.
            All the entries indicated by these keys will be left untouched by the transform.
            Defaults to ``["done", "truncated", "terminated"]``.
        mask_key (NestedKey, optional): the mask key in the input tensordict.
            The mask represents the valid frames in the input tensordict and
            should have a shape that allows the input tensordict to be masked
            with.
            Defaults to ``"mask"``.

    Examples:
        >>> from torchrl.envs import GymEnv, TransformedEnv, StepCounter, MultiStepTransform, SerialEnv
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(100, ndim=2),
        ...     transform=MultiStepTransform(n_steps=3, gamma=0.95)
        ... )
        >>> base_env = SerialEnv(2, lambda: GymEnv("CartPole"))
        >>> env = TransformedEnv(base_env, StepCounter())
        >>> _ = env.set_seed(0)
        >>> _ = torch.manual_seed(0)
        >>> tdreset = env.reset()
        >>> for _ in range(100):
        ...     rollout = env.rollout(max_steps=50, break_when_any_done=False,
        ...         tensordict=tdreset, auto_reset=False)
        ...     indices = rb.extend(rollout)
        ...     tdreset = rollout[..., -1]["next"]
        >>> print("step_count", rb[:]["step_count"][:, :5])
        step_count tensor([[[ 9],
                 [10],
                 [11],
                 [12],
                 [13]],
        <BLANKLINE>
                [[12],
                 [13],
                 [14],
                 [15],
                 [16]]])
        >>> # The next step_count is 3 steps in the future
        >>> print("next step_count", rb[:]["next", "step_count"][:, :5])
        next step_count tensor([[[13],
                 [14],
                 [15],
                 [16],
                 [17]],
        <BLANKLINE>
                [[16],
                 [17],
                 [18],
                 [19],
                 [20]]])

    """

    ENV_ERR = "The MultiStepTransform is only an inverse transform and can be applied exclusively to replay buffers."

    def __init__(
        self,
        n_steps,
        gamma,
        *,
        reward_keys: list[NestedKey] | None = None,
        done_key: NestedKey | None = None,
        done_keys: list[NestedKey] | None = None,
        mask_key: NestedKey | None = None,
        use_terminated_for_bootstrap: bool = False,
        terminated_key: NestedKey = "terminated",
    ):
        super().__init__()
        self.n_steps = n_steps
        self.reward_keys = reward_keys
        self.done_key = done_key
        self.done_keys = done_keys
        self.mask_key = mask_key
        self.gamma = gamma
        # When True, the n-step bootstrap mask (``nonterminal``) is derived from
        # ``terminated`` instead of ``done`` (= terminated | truncated). This fixes the
        # time-limit bug so truncated transitions still bootstrap. Trajectory segmentation
        # and reward accumulation continue to use ``done``.
        self.use_terminated_for_bootstrap = use_terminated_for_bootstrap
        self.terminated_key = terminated_key
        self._buffer = None
        self._validated = False

    @property
    def n_steps(self):
        """The look ahead window of the transform.

        This value can be dynamically edited during training.
        """
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value):
        if not isinstance(value, int) or not (value >= 1):
            raise ValueError("The value of n_steps must be a strictly positive integer.")
        self._n_steps = value

    @property
    def done_key(self):
        return self._done_key

    @done_key.setter
    def done_key(self, value):
        if value is None:
            value = "done"
        self._done_key = value

    @property
    def done_keys(self):
        return self._done_keys

    @done_keys.setter
    def done_keys(self, value):
        if value is None:
            value = ["done", "terminated", "truncated"]
        self._done_keys = value

    @property
    def reward_keys(self):
        return self._reward_keys

    @reward_keys.setter
    def reward_keys(self, value):
        if value is None:
            value = [
                "reward",
            ]
        self._reward_keys = value

    @property
    def mask_key(self):
        return self._mask_key

    @mask_key.setter
    def mask_key(self, value):
        if value is None:
            value = "mask"
        self._mask_key = value

    def _validate(self):
        if self.parent is not None:
            raise ValueError(self.ENV_ERR)
        self._validated = True

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase | None:
        """
        Intercepts the transition right before it gets added to the buffer.
        `tensordict` is the 1-step transition we just passed to `rb.add()`.
        """
        if not self._validated:
            self._validate()

        # Append the new 1-step transition to our internal FIFO queue (cache)
        # total_cat is a TensorDict containing the last `n_steps` transitions stacked together.
        total_cat = self._append_tensordict(tensordict)
        
        # If we have accumulated at least `n_steps` transitions in the queue...
        if total_cat.shape[-1] >= self.n_steps:
            # Calculate the N-step return across the entire queue!
            # It modifies the OLDEST transition in the queue to look `n_steps` ahead.
            out = _multi_step_func(
                total_cat,
                done_key=self.done_key,  # Defaults to "done", meaning it looks for ("next", "done")
                done_keys=self.done_keys,
                reward_keys=self.reward_keys,
                mask_key=self.mask_key,
                n_steps=self.n_steps,
                gamma=self.gamma,
                use_terminated_for_bootstrap=self.use_terminated_for_bootstrap,
                terminated_key=self.terminated_key,
            )
            # Return ONLY the oldest transition (now heavily modified with N-step math).
            # This returned transition is what actually gets written to the ReplayBuffer memory.
            return out[..., -self.n_steps]

        # If the queue isn't full yet (e.g. step 0 or 1), return None. 
        # The ReplayBuffer ignores None and writes nothing to memory yet.
        return None

    def _append_tensordict(self, data):
        """
        Maintains a FIFO queue (self._buffer) of exactly `n_steps` transitions.
        """
        if self._buffer is None:
            total_cat = data
            self._buffer = data[..., -self.n_steps :].copy()
        else:
            # Concatenate the existing cache with the new transition along the time dimension
            total_cat = torch.cat([self._buffer, data], -1)
            # Keep only the most recent `n_steps` transitions for the next round
            self._buffer = total_cat[..., -self.n_steps :].copy()
        return total_cat


def _multi_step_func(
    tensordict: TensorDictBase,
    *,
    done_key,
    done_keys,
    reward_keys,
    mask_key,
    n_steps,
    gamma,
    use_terminated_for_bootstrap: bool = False,
    terminated_key: NestedKey = "terminated",
):
    # in accordance with common understanding of what n_steps should be
    # E.g. to get a 3-step return, we look ahead 2 steps past the current state.
    n_steps = n_steps - 1
    
    # We clone the TensorDict structure (but NOT the underlying data tensors) so we can 
    # freely rename keys or add new keys without modifying the original input object.
    tensordict = tensordict.clone(False)
    # breakpoint()
    # done_key defaults to "done", so this gets tensordict["next"]["done"]
    done = tensordict.get(("next", done_key)) # eg: tensor([False, False, False])

    # We will use the 'done' boolean mask to figure out where episodes end early.
    # To do this safely, the 'done' tensor must have the exact same shape as the TensorDict's
    # logical "batch_size" (usually [BatchSize, TimeDimension]).
    ndim = tensordict.ndim
    if done.shape != tensordict.shape: # we dont enter here for sparse run.
        # If 'done' has an extra trailing dimension (e.g. shape is [Batch, Time, 1] 
        # while tensordict shape is [Batch, Time]), we just squeeze it out.
        if done.shape[-1] == 1 and done.shape[:-1] == tensordict.shape:
            done = done.squeeze(-1)
        else:
            try:
                # If the dimensions are totally misaligned (e.g. batch dimension is at the end),
                # we attempt to transpose the TensorDict so it matches the 'done' mask's orientation.
                tensordict.batch_size = done.shape
                tensordict = tensordict.transpose(ndim - 1, tensordict.ndim - 1)
                done = tensordict.get(("next", done_key))
            except Exception as err:
                raise RuntimeError(
                    "tensordict shape must be compatible with the done's shape (trailing singleton dimension excluded)."
                ) from err

    # Optional terminated signal for time-limit-correct bootstrapping. When enabled,
    # the bootstrap mask uses ``terminated`` (true absorbing terminals only) instead of
    # ``done`` (which also fires on truncation). Fetched from the (possibly transposed)
    # tensordict so its orientation matches the finalized ``done`` above.
    terminated = None
    if use_terminated_for_bootstrap:
        terminated = tensordict.get(("next", terminated_key), None)
        if terminated is None:
            raise KeyError(
                f"MultiStepTransform.use_terminated_for_bootstrap=True but key "
                f"('next', '{terminated_key}') is missing from the replay-buffer transitions."
            )
        if terminated.shape != done.shape:
            if terminated.shape[-1] == 1 and terminated.shape[:-1] == done.shape:
                terminated = terminated.squeeze(-1)
            else:
                raise RuntimeError(
                    "terminated shape must be compatible with done's shape "
                    "(trailing singleton dimension excluded)."
                )

    # The user can pass a 'mask_key' to explicitly ignore certain padded transitions.
    # By default, mask_key="mask". If it's not present in the transition, mask is None.
    if mask_key is not None:
        mask = tensordict.get(mask_key, None)
    else:
        mask = None

    # Unpack the shape of the TensorDict. 
    # Because we concatenated 3 transitions along the time dimension, 
    # tensordict.batch_size is usually something like [1, 3] or just [3].
    # *batch captures any leading dimensions (like [1]), and T captures the time dimension (3).
    *batch, T = tensordict.batch_size # batch is [] and T is 3

    summed_rewards = []
    # reward_keys defaults to ["reward"]. It loops through in case you have multiple reward signals.
    for reward_key in reward_keys:
        # reward is NOT a single number! It is a tensor of shape [BatchSize, 3]. It literally contains [r_0, r_1, r_2]!
        reward = tensordict.get(("next", reward_key)) # eg: tensor([0., 0., 0.])

        # sum rewards: This calculates the N-step discounted sum (r_t + gamma * r_{t+1} + ...)
        # time_to_obs tells us how far we actually looked ahead. Normally this is n_steps (e.g. 2).
        # But if the episode hit a 'done' flag early, time_to_obs will be less than n_steps.
        # _get_reward(gamma, reward, done, n_steps) = _get_reward(0.995, tensor([0., 0., 0.]), tensor([False, False, False]), 2)
        summed_reward, time_to_obs = _get_reward(gamma, reward, done, n_steps)
        # summed_reward is tensor([0., 0., 0.]) and time_to_obs is tensor([2, 1, 0])
        summed_rewards.append(summed_reward)
    # summed_rewards = [tensor([0., 0., 0.])]
    # Create an index tensor to select the "future" observation.
    # If T=3, torch.arange is [0, 1, 2]. 
    # For index 0, time_to_obs is usually 2 (looking 2 steps ahead).
    # So idx_to_gather[0] = 0 + 2 = 2. This means "to get the future obs for index 0, pull from index 2".
    idx_to_gather = torch.arange(T, device=time_to_obs.device, dtype=time_to_obs.dtype).expand(*batch, T)
    # idx_to_gather is tensor([[[0, 1, 2]]]) and time_to_obs is tensor([[[2, 1, 0]]]) 
    idx_to_gather = idx_to_gather + time_to_obs
    # idx_to_gather is tensor([[[2, 2, 2]]])

    # What `.gather` does: It uses the index array to copy data from the future.
    # Because your `idx_to_gather` was `tensor([2, 2, 2])`:
    # - For transition 0: it grabs `next.obs` from index 2 (which is 2 steps ahead!)
    # - For transition 1: it also grabs `next.obs` from index 2 (because 2 is the max index in a shape of 3)
    # - For transition 2: it also grabs `next.obs` from index 2
    # This `tensordict_gather` now holds the future observations that we will use to overwrite the old 1-step observations.
    # We exclude reward/done keys because we are manually calculating the N-step versions of those below.
    tensordict_gather = tensordict.get("next").exclude(*reward_keys, *done_keys).gather(-1, idx_to_gather)

    # Save metadata about how far we looked ahead (usually n_steps + 1)
    tensordict.set("steps_to_next_obs", time_to_obs + 1)
    for reward_key, summed_reward in zip(reward_keys, summed_rewards):
        # Backup the original 1-step reward under a new name
        tensordict.rename_key_(("next", reward_key), ("next", "original_reward"))
        # Overwrite the 'reward' key with the accumulated N-step sum!
        tensordict.set(("next", reward_key), summed_reward)

    # Overwrite the 'next' dictionary with the future observation we gathered earlier
    tensordict.get("next").update(tensordict_gather)
    # Store the correct discount factor to apply to Q(s_{t+N}) (e.g. gamma^3)
    tensordict.set("gamma", gamma ** (time_to_obs + 1))

    # Fetch the 'done' flag from the future state we landed on
    future_done = done.gather(-1, idx_to_gather).squeeze(-1)
    if use_terminated_for_bootstrap:
        # Time-limit-correct mask: bootstrap unless the state we land on is a TRUE terminal.
        # idx_to_gather already points at the trajectory boundary (or n-step-ahead state).
        # On a truncation boundary terminated is False -> nonterminal True -> the critic
        # bootstraps the (real final) obs with discount gamma**(steps_to_next_obs). On a real
        # termination boundary terminated is True -> nonterminal False -> no bootstrap.
        future_terminated = terminated.gather(-1, idx_to_gather).squeeze(-1)
        nonterminal = ~future_terminated.bool()
    else:
        # nonterminal is the crucial boolean mask for the Bellman target: target = reward + (gamma * nonterminal * Q)
        # It is True only if we successfully looked ahead `n_steps` AND the future state is NOT done.
        nonterminal = (time_to_obs == n_steps) & (~future_done) # tensor([ True, False, False])

    if mask is not None:
        mask = mask.view(*batch, T)
        nonterminal[~mask] = False
    
    # Save the nonterminal mask to the TensorDict so the loss function can use it
    tensordict.set("nonterminal", nonterminal)
    if tensordict.ndim != ndim:
        tensordict = tensordict.apply(
            lambda x: x.transpose(ndim - 1, tensordict.ndim - 1),
            batch_size=done.transpose(ndim - 1, tensordict.ndim - 1).shape,
        )
        tensordict.batch_size = tensordict.batch_size[:ndim]
    return tensordict


def _get_reward(
    gamma: float,
    reward: torch.Tensor,
    done: torch.Tensor,
    max_steps: int,
):
    """
    Calculates the N-step discounted sum of rewards (e.g., r_t + gamma*r_{t+1} + gamma^2*r_{t+2}).
    
    IMPORTANT: The `reward` tensor here is NOT just a single number! Because `_multi_step_func` 
    was called with `total_cat`, `reward` has a time dimension of length `max_steps + 1` (e.g. 3).
    It contains the entire queue of recent rewards!

    This function uses a 1D Convolution (conv1d) to efficiently calculate the sliding window 
    sum over the time dimension for the entire batch simultaneously.
    """
    # Create the discount filter: [gamma^0, gamma^1, gamma^2, ...]
    filt = torch.tensor(
        [gamma**i for i in range(max_steps + 1)],
        device=reward.device,
        dtype=reward.dtype,
    ).view(1, 1, -1)
    
    # We must not sum rewards across episode boundaries. 
    # done_cumsum creates a unique ID for each trajectory. If 'done' is True, 
    # the cumsum increments, meaning the next step belongs to a new trajectory.
    done_cumsum = done.cumsum(-1)
    done_cumsum = torch.cat([torch.zeros_like(done_cumsum[..., :1]), done_cumsum[..., :-1]], -1)
    num_traj = done_cumsum.max().item() + 1
    done_cumsum = done_cumsum.expand(num_traj, *done.shape)
    
    # Create a boolean mask for each individual trajectory
    traj_ids = done_cumsum == torch.arange(num_traj, device=done.device, dtype=done_cumsum.dtype).view(
        num_traj, *[1 for _ in range(done_cumsum.ndim - 1)]
    )
    
    if reward.shape != traj_ids.shape[1:]:
        traj_ids_expand = expand_right(traj_ids, (num_traj, *reward.shape))
        reward_traj = traj_ids_expand * reward
        reward_traj = reward_traj.transpose(-1, traj_ids.ndim - 1)
    else:
        # Zero out rewards that don't belong to the current trajectory
        reward_traj = traj_ids * reward

    # Pad the end so the convolution doesn't shrink the tensor length
    reward_traj = torch.nn.functional.pad(reward_traj, [0, max_steps], value=0.0)
    shape = reward_traj.shape[:-1]
    if len(shape) > 1:
        reward_traj = reward_traj.flatten(0, reward_traj.ndim - 2)
    reward_traj = reward_traj.unsqueeze(-2)
    
    # Perform the sliding window N-step sum using 1D convolution!
    summed_rewards = torch.conv1d(reward_traj, filt)
    summed_rewards = summed_rewards.squeeze(-2)
    if len(shape) > 1:
        summed_rewards = summed_rewards.unflatten(0, shape)
    # let's check that our summed rewards have the right size
    if reward.shape != traj_ids.shape[1:]:
        summed_rewards = summed_rewards.transpose(-1, traj_ids.ndim - 1)
        summed_rewards = (summed_rewards * traj_ids_expand).sum(0)
    else:
        summed_rewards = (summed_rewards * traj_ids).sum(0)

    # time_to_obs is the tensor of the time delta to the next obs
    # 0 = take the next obs (ie do nothing)
    # 1 = take the obs after the next
    time_to_obs = traj_ids.flip(-1).cumsum(-1).clamp_max(max_steps + 1).flip(-1) * traj_ids
    time_to_obs = time_to_obs.sum(0)
    time_to_obs = time_to_obs - 1
    return summed_rewards, time_to_obs

"""
================================================================================
SUMMARY: How MultiStepTransform Works Under the Hood
================================================================================

When you use `MultiStepTransform(n_steps=3)` in a PyTorch RL ReplayBuffer, 
transitions are NOT written directly to memory when you call `rb.add(transition)`.

1. THE WAITING ROOM (Queueing)
   - `rb.add()` passes the transition to `_inv_call`.
   - The transform keeps a sliding window cache (`self._buffer`) of the last 3 transitions.
   - It concatenates the new transition to the cache to create `total_cat` (size 4).

2. THE MATH (_multi_step_func)
   - The function calculates the N-step math for the OLDEST transition in the queue (index 1).
   - It uses `_get_reward(conv1d)` to sum the rewards: r_t + gamma*r_{t+1} + gamma^2*r_{t+2}.
   - It creates `idx_to_gather` to figure out where the future state is.
   - It overwrites `next.obs` with the observation from 3 steps in the future!
   - It backs up the 1-step reward to `next.original_reward`.
   - It overwrites the standard `next.reward` with the summed N-step reward.
   - It calculates a `nonterminal` boolean mask. If we hit a `done=True` flag before 
     reaching 3 steps, `nonterminal` becomes False so the Bellman equation knows NOT to bootstrap.

3. WRITING TO BUFFER
   - Finally, it returns `out[..., -self.n_steps]` (which is the oldest transition, now 
     fully mutated with future data).
   - THIS mutated transition is what actually gets written to the memmap storage!

This allows the training loop to do a simple 1-step Bellman backup:
    target = reward + (gamma_tensor * nonterminal * Q(next_obs))
And secretly be performing highly efficient N-step RL!
================================================================================

Viewed rb_transforms.py:392-440

I completely understand why this is confusing. Let's break down EXACTLY what happened in your debugger traces. It is incredibly elegant once you see the pattern.

When we process the offline dataset, `MultiStepTransform` works on a sliding window of 4 transitions (`total_cat`). But remember, we only care about the result it computes for **Index 1** of that window (which is the oldest transition waiting to be finalized).

Let's look at the frames leading up to the end of your episode (where `done=True` starts padding at index 113 and the reward is `1.0`). 

---

### Step A: Processing Frame 113
**Queue contains frames:** `[110, 111, 112, 113]`
**Computing N-step return for:** Frame 111
```python
(Pdb) p done
tensor([False, False, False,  True])  # <--- Frame 113 is done!
(Pdb) p time_to_obs
tensor([2, 2, 1, 0])
(Pdb) p summed_reward
tensor([0.0000, 0.9900, 0.9950, 1.0000])
```
- **What happens to Frame 111 (Index 1)?**
  - The code wants to look 2 steps ahead. Are there any `done=True` flags blocking it? No. (Frame 112 and 113 are safely within the window).
  - So `time_to_obs` is **2**.
  - It successfully grabs `next.obs` from Frame 113.
  - The reward calculation is: $r_{111} + (\gamma \cdot r_{112}) + (\gamma^2 \cdot r_{113})$. Since the sparse reward is `1.0` at frame 113, the math is: $0 + 0 + (0.995^2 \cdot 1.0) = \mathbf{0.9900}$.

---

### Step B: Processing Frame 114
**Queue contains frames:** `[111, 112, 113, 114]`
**Computing N-step return for:** Frame 112
```python
(Pdb) p done
tensor([False, False,  True,  True])
(Pdb) p time_to_obs
tensor([2, 1, 0, 0])
(Pdb) p summed_reward
tensor([0.9900, 0.9950, 1.0000, 1.0000])
```
- **What happens to Frame 112 (Index 1)?**
  - The code wants to look 2 steps ahead (to frame 114). But wait! Frame 113 has `done=True`. It cannot look past a `done` flag!
  - The trajectory isolation logic (`done_cumsum`) automatically catches this and forces the lookahead to shrink.
  - So `time_to_obs` drops to **1**. It only looks 1 step ahead to Frame 113.
  - The reward calculation shrinks to: $r_{112} + (\gamma \cdot r_{113})$. Which is $0 + (0.995 \cdot 1.0) = \mathbf{0.9950}$.

---

### Step C: Processing Frame 115
**Queue contains frames:** `[112, 113, 114, 115]`
**Computing N-step return for:** Frame 113
```python
(Pdb) p done
tensor([False,  True,  True,  True])
(Pdb) p time_to_obs
tensor([1, 0, 0, 0])
(Pdb) p summed_reward
tensor([0.9950, 1.0000, 1.0000, 1.0000])
```
- **What happens to Frame 113 (Index 1)?**
  - Frame 113 literally IS a `done=True` frame. It cannot look ahead AT ALL.
  - `time_to_obs` drops to **0**. It grabs the immediate next observation (which is technically just the first padding frame).
  - The reward calculation shrinks to just 0 steps: $r_{113}$. Which is exactly $\mathbf{1.0000}$.
  - Crucially, `nonterminal` becomes `False`. This means the RL Bellman equation won't even use the `next.obs` anyway. It will just do `target = reward + 0`.

---

### Summary of exactly what got saved:
Because of how the trajectory logic gracefully "shrinks" the window as it hits the episode boundary, here is the exact data that was written to your buffer:

| Transition | Lookahead (`time_to_obs`) | Future Obs It Saved | Discounted N-Step Reward | `nonterminal` |
| :--- | :--- | :--- | :--- | :--- |
| **Frame 110** | 2 steps | `next.obs` of 112 | $0.0000$ | True |
| **Frame 111** | 2 steps | `next.obs` of 113 | $0.9900$ | False *(because 113 is done)* |
| **Frame 112** | 1 step | `next.obs` of 113 | $0.9950$ | False *(because 113 is done)* |
| **Frame 113** | 0 steps | `next.obs` of 113 | $1.0000$ | False *(because 113 is done)* |
| **Frame 114** | 0 steps | `next.obs` of 114 | $1.0000$ | False |
| **Frame 115** | 0 steps | `next.obs` of 115 | $1.0000$ | False |

As you can see, the consecutive padding frames (`114, 115, ...`) are perfectly isolated. They just turn into 0-step dummy transitions with `nonterminal=False` that will never bootstrap or corrupt your Bellman math!

"""
