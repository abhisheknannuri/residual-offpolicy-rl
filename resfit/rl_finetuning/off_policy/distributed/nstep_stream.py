"""Actor-side n-step return computation.

Why this exists
---------------
``resfit/rl_finetuning/utils/rb_transforms.py::MultiStepTransform`` is a
*modified* copy of torchrl's. Where upstream returns a slice of every
newly-resolved transition::

    if total_cat.shape[-1] > self.n_steps:
        return out[..., : -self.n_steps]        # upstream: a SLICE

ours returns exactly one::

    if total_cat.shape[-1] >= self.n_steps:
        return out[..., -self.n_steps]          # ResFiT: a single INDEX

That specialises it to ``rb.add()`` one transition at a time, which is how
``train_residual_td3.py::_add_transitions_to_buffer`` uses it. Calling
``rb.extend()`` on a buffer carrying this transform does not silently drop
data - it raises::

    RuntimeError: expand_as_right requires the destination tensor to have less
    dimensions than the input tensor, got tensor.ndimension()=1 and
    dest.ndimension()=0

...because the 0-dim result cannot take broadcast PER priorities.

In the distributed setup the learner receives transitions in bursts, so the
natural insert is a bulk ``extend()``. We therefore run the n-step transform on
the **actor**, where transitions are produced strictly in temporal order by
construction, and leave the learner's ``online_rb`` with no transform at all.

Verified equivalent to the current in-buffer path, including across episode
boundaries - see ``tests/test_nstep_stream.py``.
"""

from __future__ import annotations

from tensordict import TensorDictBase

from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform


class NStepStream:
    """Turns a stream of 1-step transitions into n-step transitions.

    Drop-in for the ``transform=MultiStepTransform(...)`` that
    ``train_residual_td3.py`` attaches to ``online_rb``, but usable without a
    replay buffer.

    Usage on the actor::

        nstep = NStepStream(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma,
                            use_terminated_for_bootstrap=cfg.algo.terminated_only_bootstrap)
        ...
        out = nstep.push(td)          # td as built by _add_transitions_to_buffer
        if out is not None:
            outbox.insert(out)        # ready for the wire

    The first ``n_steps - 1`` pushes return ``None`` while the internal window
    fills. This matches today's behaviour exactly: with ``n_step=3``, adding 10
    transitions yields 8 stored ones. The last ``n_steps - 1`` transitions of a
    run stay in the window and are never emitted - also true today.

    Episode boundaries need no special handling: the window is never cleared,
    and ``_multi_step_func`` truncates the lookahead using ``next.done`` /
    ``next.terminated`` so a transition never bootstraps across a boundary.
    """

    def __init__(self, n_steps: int, gamma: float, *, use_terminated_for_bootstrap: bool = False):
        self._tf = MultiStepTransform(
            n_steps=n_steps,
            gamma=gamma,
            use_terminated_for_bootstrap=use_terminated_for_bootstrap,
        )
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_pushed = 0
        self.n_emitted = 0

    def push(self, td: TensorDictBase) -> TensorDictBase | None:
        """Feed one 1-step transition; get back an n-step one, or ``None``.

        The returned TensorDict is moved to CPU and given a leading batch
        dimension of 1, so a burst can be stacked with ``torch.cat`` and handed
        straight to ``rb.extend()`` on the learner. ``_inv_call`` returns a
        0-dim TensorDict, which ``torch.cat`` cannot stack as-is.

        CPU matters: transitions are built on ``device`` in the training loop,
        and a CUDA tensor carries its device through pickle.
        """
        self.n_pushed += 1
        out = self._tf._inv_call(td)
        if out is None:
            return None
        self.n_emitted += 1
        if out.ndim == 0:
            out = out.unsqueeze(0)
        return out.cpu()

    @property
    def pending(self) -> int:
        """Transitions held in the lookahead window, not yet emitted."""
        return self.n_pushed - self.n_emitted

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"NStepStream(n_steps={self.n_steps}, gamma={self.gamma}, "
            f"pushed={self.n_pushed}, emitted={self.n_emitted})"
        )
