"""Transition transport between the actor and the learner's online buffer.

Threading model
---------------
agentlace runs its ReqRep server on a daemon thread
(``TrainerServer.start(threaded=True)``), so received transitions arrive on a
*different* thread than the learner's update loop.

torchrl's ReplayBuffer is internally locked (``_replay_lock`` / ``_write_lock``
around ``add`` / ``extend`` / ``_sample``), but ``update_tensordict_priority()``
-> the PrioritizedSampler sum-tree is not obviously covered by that same lock,
and the learner's main loop calls it on every update. Rather than rely on an
audit of torchrl's sum-tree, we keep *all* buffer mutation on one thread:

    net thread  ->  collections.deque  ->  learner main loop  ->  online_rb

``deque.append`` / ``popleft`` are atomic under the GIL, so the handoff needs no
lock of its own. This is simpler than HIL-SERL, which inserts directly on the
network thread and bolts a ``threading.Lock`` onto its datastore
(``serl_launcher/data/data_store.py:57``).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from itertools import islice
from typing import Any

import torch

logger = logging.getLogger(__name__)


class TransitionOutbox:
    """Actor side: an ordered outbox of n-step transitions awaiting transmission.

    Implements agentlace's ``DataStoreBase`` protocol structurally (we do not
    import it here so this module stays usable without agentlace installed).

    Deliberately the same shape as agentlace's ``QueuedDataStore``: a deque plus
    a monotonically increasing sequence id, where ``get_latest_data(from_id)``
    returns everything after ``from_id``. That id is the resume cursor - it is
    what lets a reconnecting actor avoid both re-sending and losing data.
    """

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self._seq_ids: deque[int] = deque(maxlen=capacity)
        self._data: deque[Any] = deque(maxlen=capacity)
        self.latest_seq_id = -1
        # The control loop inserts while agentlace's update thread
        # (TrainerClient.start_async_update) reads - iterating a deque that is
        # being appended to raises "deque mutated during iteration".
        self._lock = threading.Lock()

    def insert(self, data: Any) -> None:
        with self._lock:
            self.latest_seq_id += 1
            self._seq_ids.append(self.latest_seq_id)
            self._data.append(data)

    def batch_insert(self, batch_data: list[Any]) -> None:
        for d in batch_data:
            self.insert(d)

    def latest_data_id(self) -> int:
        with self._lock:
            return self.latest_seq_id

    def get_latest_data(self, from_id: int) -> list[Any]:
        with self._lock:
            return self._get_latest_data_locked(from_id)

    def _get_latest_data_locked(self, from_id: int) -> list[Any]:
        if not self._seq_ids or from_id >= self.latest_seq_id:
            return []
        if from_id < self._seq_ids[0]:
            # Either the very first fetch (from_id == -1, nothing dropped), or
            # the consumer fell behind far enough that the deque evicted data.
            # Send everything we still hold; warn only in the latter case,
            # because silently losing transitions is much worse than a log line.
            n_dropped = self._seq_ids[0] - from_id - 1
            if n_dropped > 0:
                logger.warning(
                    "Outbox overflow: requested from_id=%d but oldest held is %d. "
                    "%d transitions were dropped. Raise capacity or push more often.",
                    from_id, self._seq_ids[0], n_dropped,
                )
            return list(self._data)
        start = from_id - self._seq_ids[0] + 1
        return list(self._data)[start:]

    def get_range(self, from_id: int, max_items: int) -> tuple[list[Any], int]:
        """Up to ``max_items`` transitions after ``from_id``, plus the seq id of the last one.

        Used instead of agentlace's ``TrainerClient.update()``, which ships *everything*
        since the server's cursor in ONE message - after a long disconnect that is up to
        ``capacity`` x ~85 KB of camera frames in a single pickle. Chunking keeps every
        message bounded, and the returned id is the exact cursor to advance to.
        """
        with self._lock:
            if not self._seq_ids or from_id >= self.latest_seq_id:
                return [], from_id
            oldest = self._seq_ids[0]
            if from_id < oldest:
                n_dropped = oldest - from_id - 1
                if n_dropped > 0:
                    logger.warning(
                        "Outbox overflow: %d transitions were evicted before they could be sent "
                        "(learner unreachable for too long). Raise dist.outbox_capacity.", n_dropped,
                    )
                start = 0
            else:
                start = from_id - oldest + 1
            items = list(islice(self._data, start, start + max_items))
            return items, oldest + start + len(items) - 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._seq_ids)


class OnlineTransitionInbox:
    """Learner side: agentlace datastore adapter in front of ``online_rb``.

    ``batch_insert`` runs on agentlace's network thread and only appends to a
    deque. ``drain_into_buffer`` runs on the learner's main loop and performs
    the single bulk ``extend()``.

    ``__len__`` reports the *replay buffer's* length, not the pending queue's,
    so the HIL-SERL-style startup gate reads naturally::

        while len(inbox) < cfg.algo.learning_starts:
            inbox.drain_into_buffer()
            time.sleep(1)
    """

    def __init__(self, online_rb, capacity: int = 200_000):
        self.online_rb = online_rb
        self.capacity = capacity
        self._pending: deque[Any] = deque()
        self.total_received = 0
        self.total_inserted = 0

    # ---- called on the agentlace network thread -------------------------
    def insert(self, data: Any) -> None:
        self._pending.append(data)
        self.total_received += 1

    def batch_insert(self, batch_data: list[Any]) -> None:
        self._pending.extend(batch_data)
        self.total_received += len(batch_data)

    def latest_data_id(self) -> int:
        # The learner never sends data back to the actor, so this direction is
        # unused. HIL-SERL raises here too (serl_launcher/data/data_store.py:41).
        raise NotImplementedError("learner-side store is write-only")

    def get_latest_data(self, from_id: int) -> list[Any]:
        raise NotImplementedError("learner-side store is write-only")

    # ---- called on the learner's main loop -------------------------------
    def drain_into_buffer(self) -> int:
        """Move every pending transition into ``online_rb``. Returns the count.

        One bulk ``extend()`` per call. Safe because the transitions arriving
        here already carry their n-step fields (computed on the actor), so
        ``online_rb`` carries no transform - see ``nstep_stream.py`` for why
        ``extend()`` and ``MultiStepTransform`` cannot coexist.
        """
        if not self._pending:
            return 0
        batch = []
        while True:
            try:
                batch.append(self._pending.popleft())
            except IndexError:
                break
        if not batch:
            return 0
        stacked = torch.cat([b if b.ndim else b.unsqueeze(0) for b in batch], dim=0)
        self.online_rb.extend(stacked)
        self.total_inserted += len(batch)
        return len(batch)

    def __len__(self) -> int:
        return len(self.online_rb)

    @property
    def pending(self) -> int:
        return len(self._pending)
