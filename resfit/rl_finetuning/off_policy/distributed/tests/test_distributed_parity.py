"""Parity tests: the distributed path must reproduce today's buffer contents.

Run:  <python> -m pytest resfit/rl_finetuning/off_policy/distributed/tests/ -v
(or execute this file directly - it has a __main__ fallback with no pytest dep)
"""
from __future__ import annotations

import pickle
import sys
import threading

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

from resfit.rl_finetuning.off_policy.distributed import (
    NStepStream,
    OnlineTransitionInbox,
    TransitionOutbox,
)
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform

N_STEP, GAMMA = 3, 0.9
NSTEP_KEYS = ["gamma", "nonterminal", "steps_to_next_obs"]


def _mk(i: int, done: bool = False, rew: float = 1.0) -> TensorDict:
    """Mimics the TensorDict built by _add_transitions_to_buffer()."""
    return TensorDict(
        {
            "obs": TensorDict({"s": torch.tensor([float(i)])}, batch_size=[]),
            "next": TensorDict(
                {
                    "obs": TensorDict({"s": torch.tensor([float(i + 1)])}, batch_size=[]),
                    "done": torch.tensor(done),
                    "terminated": torch.tensor(done),
                    "reward": torch.tensor(rew),
                },
                batch_size=[],
            ),
            "action": torch.zeros(2),
            "intervened": torch.tensor(False),
            "_priority": torch.tensor(10.0),
        },
        batch_size=[],
    ).unsqueeze(0)


def _rb(with_transform: bool, max_size: int = 200):
    return TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=max_size, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority",
        transform=MultiStepTransform(
            n_steps=N_STEP, gamma=GAMMA, use_terminated_for_bootstrap=True
        ) if with_transform else None,
        batch_size=2,
    )


# episode terminates at i == 4; a second episode follows
EPISODE = [(i, i == 4) for i in range(10)]


def _today(episode):
    """Current behaviour: transform on the buffer, add() one at a time."""
    rb = _rb(with_transform=True)
    for i, d in episode:
        rb.add(_mk(i, d))
    return rb


def _distributed(episode):
    """Proposed: n-step on the actor, bulk extend() on a transform-free buffer."""
    nstep = NStepStream(N_STEP, GAMMA, use_terminated_for_bootstrap=True)
    outbox = TransitionOutbox(capacity=1000)
    for i, d in episode:
        out = nstep.push(_mk(i, d))
        if out is not None:
            outbox.insert(out)
    rb = _rb(with_transform=False)
    inbox = OnlineTransitionInbox(rb)
    # everything the actor would have shipped, as one burst
    inbox.batch_insert(outbox.get_latest_data(-1))
    inbox.drain_into_buffer()
    return rb


def test_parity_including_episode_boundary():
    a, b = _today(EPISODE)[:], _distributed(EPISODE)[:]
    assert len(a) == len(b) == 8, (len(a), len(b))
    for k in NSTEP_KEYS:
        assert torch.equal(a[k], b[k]), f"{k} differs"
    assert torch.allclose(a["next", "reward"], b["next", "reward"])
    assert torch.equal(a["obs", "s"], b["obs", "s"])
    assert torch.equal(a["next", "obs", "s"], b["next", "obs", "s"])


def test_boundary_does_not_bootstrap_across_episodes():
    g = _today(EPISODE)[:]
    idx = {int(g[j]["obs", "s"].item()): j for j in range(len(g))}
    # transition at s=3: window 3,4 truncated by the terminal at 4
    j = idx[3]
    assert int(g[j]["steps_to_next_obs"]) == 2
    assert abs(float(g[j]["next", "reward"]) - (1 + GAMMA)) < 1e-5
    assert not bool(g[j]["nonterminal"])
    # the terminal transition itself
    j = idx[4]
    assert int(g[j]["steps_to_next_obs"]) == 1
    assert not bool(g[j]["nonterminal"])
    # first transition of the NEXT episode gets a full n-step window again
    j = idx[5]
    assert int(g[j]["steps_to_next_obs"]) == N_STEP
    assert bool(g[j]["nonterminal"])
    assert abs(float(g[j]["next", "reward"]) - (1 + GAMMA + GAMMA**2)) < 1e-5


def test_bursty_arrival_matches_contiguous():
    """Transform state must survive being fed in network-sized chunks."""
    nstep = NStepStream(N_STEP, GAMMA, use_terminated_for_bootstrap=True)
    rb = _rb(with_transform=False)
    inbox = OnlineTransitionInbox(rb)
    for lo, hi in ((0, 3), (3, 4), (4, 10)):        # deliberately uneven bursts
        burst = [o for i, d in EPISODE[lo:hi] if (o := nstep.push(_mk(i, d))) is not None]
        inbox.batch_insert(burst)
        inbox.drain_into_buffer()
    a = _today(EPISODE)[:]
    b = rb[:]
    assert len(a) == len(b)
    for k in NSTEP_KEYS:
        assert torch.equal(a[k], b[k]), f"{k} differs under bursty arrival"


def test_capacity_is_bounded_by_max_size():
    """extend() must wrap, not grow the storage."""
    rb = _rb(with_transform=False, max_size=5)
    for chunk in range(4):
        rb.extend(torch.cat([_mk(i) for i in range(chunk * 10, chunk * 10 + 10)], dim=0))
        assert len(rb) == 5, f"storage grew to {len(rb)}"


def test_transitions_survive_pickle_and_are_cpu():
    """The wire format is pickle; CUDA tensors would carry their device across."""
    nstep = NStepStream(N_STEP, GAMMA, use_terminated_for_bootstrap=True)
    staged = [o for i, d in EPISODE if (o := nstep.push(_mk(i, d))) is not None]
    for td in staged:
        for t in td.values(include_nested=True, leaves_only=True):
            assert t.device.type == "cpu", f"non-CPU tensor would break pickling: {t.device}"
    back = pickle.loads(pickle.dumps(staged))
    rb = _rb(with_transform=False)
    OnlineTransitionInbox(rb).__class__  # noqa: B018
    inbox = OnlineTransitionInbox(rb)
    inbox.batch_insert(back)
    assert inbox.drain_into_buffer() == len(staged)
    assert len(rb) == len(staged)


def test_inbox_len_reports_buffer_not_queue():
    """The startup gate reads len(inbox); it must mean 'in the buffer'."""
    rb = _rb(with_transform=False)
    inbox = OnlineTransitionInbox(rb)
    inbox.batch_insert([_mk(i) for i in range(4)])
    assert len(inbox) == 0 and inbox.pending == 4     # queued, not yet inserted
    inbox.drain_into_buffer()
    assert len(inbox) == 4 and inbox.pending == 0


def test_concurrent_insert_while_draining():
    """Net thread appends while the main loop drains; nothing is lost."""
    rb = _rb(with_transform=False, max_size=5000)
    inbox = OnlineTransitionInbox(rb)
    total = 2000
    stop = threading.Event()

    def producer():                     # stands in for the agentlace net thread
        for i in range(total):
            inbox.insert(_mk(i))
        stop.set()

    t = threading.Thread(target=producer)
    t.start()
    drained = 0
    while not stop.is_set() or inbox.pending:
        drained += inbox.drain_into_buffer()
    t.join()
    drained += inbox.drain_into_buffer()
    assert drained == total, f"lost transitions: {drained} != {total}"
    assert len(rb) == total


def test_outbox_cursor_semantics():
    """get_latest_data(from_id) is the resume cursor; must not re-send or skip."""
    ob = TransitionOutbox(capacity=100)
    for i in range(10):
        ob.insert(_mk(i))
    first = ob.get_latest_data(-1)
    assert len(first) == 10
    # server acked up to id 9 -> nothing new
    assert ob.get_latest_data(ob.latest_data_id()) == []
    for i in range(10, 15):
        ob.insert(_mk(i))
    assert len(ob.get_latest_data(9)) == 5




# ---------------------------------------------------------------------------
# Added with the entrypoint: chunked push + weight sync
# ---------------------------------------------------------------------------
def test_outbox_get_range_chunks_are_contiguous_and_lossless():
    ob = TransitionOutbox(capacity=1000)
    for i in range(130):
        ob.insert(i)
    got, cursor = [], -1
    while True:
        items, last = ob.get_range(cursor, 64)
        if not items:
            break
        assert last == cursor + len(items), (cursor, last, len(items))
        got.extend(items)
        cursor = last
    assert got == list(range(130))


def test_outbox_get_range_after_eviction_resumes_at_oldest():
    ob = TransitionOutbox(capacity=10)
    for i in range(25):  # ids 0..24, only 15..24 retained
        ob.insert(i)
    items, last = ob.get_range(3, 100)  # cursor points at evicted data
    assert items == list(range(15, 25)) and last == 24


def _tiny_agent(scale=0.0):
    import torch.nn as nn

    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Linear(4, 2)
            self.encoders = nn.ModuleList([nn.Linear(3, 3)])
            self.critic = nn.Linear(9, 1)
            with torch.no_grad():
                for p in self.parameters():
                    p.fill_(scale)

    return A()


def test_weight_payload_excludes_critic_and_follows_freeze_flag():
    from resfit.rl_finetuning.off_policy.distributed import extract_sync_weights

    ag = _tiny_agent(1.0)
    p = extract_sync_weights(ag, freeze_encoder=False, grad_step=7)
    assert set(p) == {"actor", "encoders", "grad_step"}
    p2 = extract_sync_weights(ag, freeze_encoder=True, grad_step=7)
    assert set(p2) == {"actor", "grad_step"}
    # snapshot must not alias live params
    with torch.no_grad():
        ag.actor.weight.fill_(5.0)
    assert float(p["actor"]["weight"].flatten()[0]) == 1.0


def test_receiver_stages_only_and_applies_latest():
    from resfit.rl_finetuning.off_policy.distributed import WeightReceiver, extract_sync_weights

    live = _tiny_agent(0.0)
    r = WeightReceiver(freeze_encoder=False)
    r.stage(extract_sync_weights(_tiny_agent(1.0), freeze_encoder=False, grad_step=1))
    r.stage(extract_sync_weights(_tiny_agent(2.0), freeze_encoder=False, grad_step=2))
    assert float(live.actor.weight.flatten()[0]) == 0.0, "stage() must never touch live weights"
    assert r.apply_if_new(live) == 2
    assert float(live.actor.weight.flatten()[0]) == 2.0
    assert float(live.encoders[0].weight.flatten()[0]) == 2.0
    assert float(live.critic.weight.flatten()[0]) == 0.0, "critic is never synced"
    assert r.n_dropped == 1 and r.apply_if_new(live) is None


def test_receiver_refuses_actor_without_encoder_when_encoder_trains():
    from resfit.rl_finetuning.off_policy.distributed import WeightReceiver, extract_sync_weights

    r = WeightReceiver(freeze_encoder=False)
    r.stage(extract_sync_weights(_tiny_agent(1.0), freeze_encoder=True, grad_step=3))  # learner thinks frozen
    try:
        r.apply_if_new(_tiny_agent(0.0))
    except RuntimeError as e:
        assert "freeze_encoder" in str(e)
    else:
        raise AssertionError("mismatched freeze_encoder must raise, not silently pair a stale encoder")

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
