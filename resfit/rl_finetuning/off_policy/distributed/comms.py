"""Actor and learner networking runtimes built on agentlace.

Threading contract (the whole point of this module)
---------------------------------------------------
**Actor.** The 20 Hz control loop never does network I/O. It only:
  - appends n-step transitions to the ``TransitionOutbox``,
  - drops episode stats / timing into ``ActorComms`` (thread-safe setters),
  - calls ``WeightReceiver.apply_if_new(agent)`` between env steps.
A background thread pushes transitions in bounded chunks and sends a heartbeat
once per interval; agentlace's own broadcast thread stages incoming weights. A
slow or dead network therefore degrades to stale weights, never to a stalled
robot.

**Learner.** agentlace's REP thread only appends to deques and reads/writes a
few scalar status fields. Every replay-buffer mutation, every ``wandb.log`` and
every weight publish happens on the learner main loop.

Session handling
----------------
agentlace's server keeps a per-store cursor (``last_update_id_map``). A
restarted actor process starts its outbox at seq id -1, so without intervention
the server would silently skip its first N transitions (N = old cursor).
``actor-register`` therefore sets the cursor to the registering actor's current
latest id: a fresh actor gets -1 (send everything); an actor re-registering after
a *learner* restart does not replay transitions the old learner already had.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from typing import Any

from resfit.rl_finetuning.off_policy.distributed.data_store import (
    OnlineTransitionInbox,
    TransitionOutbox,
)
from resfit.rl_finetuning.off_policy.distributed.transport import (
    REQ_GET_INIT,
    REQ_HEARTBEAT,
    REQ_REGISTER,
    STORE_ONLINE,
    import_agentlace,
    make_trainer_config,
)
from resfit.rl_finetuning.off_policy.distributed.weight_sync import (
    WeightPublisher,
    WeightReceiver,
)

logger = logging.getLogger(__name__)

# Learner phases, in order. The actor's behaviour is a pure function of these.
PHASE_INIT = "init"                # loading dataset / offline buffer - actor idles
PHASE_WARMUP = "warmup"            # online_rb < learning_starts - actor collects base+noise
PHASE_PRETRAINING = "pretraining"  # offline RL / critic warmup - actor idles
PHASE_TRAINING = "training"        # actor runs the residual policy
PHASE_DONE = "done"                # actor exits
ALL_PHASES = (PHASE_INIT, PHASE_WARMUP, PHASE_PRETRAINING, PHASE_TRAINING, PHASE_DONE)


# =============================================================================
# Learner
# =============================================================================
class LearnerComms:
    def __init__(self, dist_cfg, *, freeze_encoder: bool, learning_starts: int):
        TrainerServer, _, TrainerConfig = import_agentlace(dist_cfg.codec)
        self.dist_cfg = dist_cfg
        self.session_id = uuid.uuid4().hex[:12]
        self.learning_starts = int(learning_starts)

        # --- scalar status (written by main loop, read by REP thread) ---------
        self.phase = PHASE_INIT
        self.grad_step = 0
        self.online_size = 0
        self.init_info: dict[str, Any] | None = None

        # --- written by REP thread, read by main loop --------------------------
        self.actor_spec: dict[str, Any] | None = None
        self.actor_session: str | None = None
        self.env_step_reported = -1
        self.actor_done = False
        self.done_acked = False  # actor heartbeat received while phase == DONE
        self.last_heartbeat_t = 0.0
        self.n_heartbeats = 0
        self._stats: deque[dict[str, Any]] = deque()

        self.inbox: OnlineTransitionInbox | None = None
        self.server = TrainerServer(
            make_trainer_config(dist_cfg, TrainerConfig),
            request_callback=self._on_request,
        )
        self.publisher = WeightPublisher(
            self.server, steps_per_update=int(dist_cfg.steps_per_update), freeze_encoder=freeze_encoder
        )
        self._last_publish_t = 0.0
        self._last_published_step: int | None = None
        self._started = False
        self._pending_cursor: int | None = None
        # Extra scalar fields merged into every status reply (e.g. start_env_step, env_step).
        self.extra: dict[str, Any] = {}

    # ---- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        """Start serving immediately so the actor can connect while we load data."""
        self.server.start(threaded=True)
        self._started = True
        print(
            f"[learner] serving on :{self.dist_cfg.port} (req/rep) and :{self.dist_cfg.broadcast_port} "
            f"(weights), codec={self.dist_cfg.codec}, session={self.session_id}"
        )

    def attach_online_buffer(self, online_rb) -> OnlineTransitionInbox:
        self.inbox = OnlineTransitionInbox(online_rb)
        self.server.register_data_store(STORE_ONLINE, self.inbox)
        # The actor may have registered during PHASE_INIT, before this store existed.
        # register_data_store() resets the cursor to -1, so re-apply the actor's.
        if self._pending_cursor is not None:
            self.server.last_update_id_map[STORE_ONLINE] = self._pending_cursor
        return self.inbox

    def set_phase(self, phase: str) -> None:
        assert phase in ALL_PHASES, phase
        if phase != self.phase:
            print(f"[learner] phase: {self.phase} -> {phase}")
        self.phase = phase

    def stop(self) -> None:
        if self._started:
            try:
                self.server.stop()
            except Exception:  # pragma: no cover - best effort at shutdown
                logger.exception("error stopping TrainerServer")

    # ---- main-loop helpers ------------------------------------------------------
    def drain(self) -> int:
        """Move received transitions into online_rb. Main loop only."""
        if self.inbox is None:
            return 0
        n = self.inbox.drain_into_buffer()
        self.online_size = len(self.inbox)
        return n

    def pop_stats(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self._stats.popleft())
            except IndexError:
                return out

    def publish(self, agent, grad_step: int) -> None:
        if self._last_published_step is None:
            n_bytes = sum(
                t.numel() * t.element_size()
                for m in ([agent.actor] + ([] if self.publisher.freeze_encoder else [agent.encoders]))
                for t in m.state_dict().values()
            )
            print(f"[learner] weight payload per publish: {n_bytes / 1e6:.1f} MB (pre-codec)")
        self.publisher.publish(agent, grad_step)
        self._last_publish_t = time.monotonic()
        self._last_published_step = grad_step

    def maybe_publish(self, agent, grad_step: int, *, republish_every_s: float = 10.0) -> bool:
        """Publish every ``steps_per_update`` grad steps, and at least every ``republish_every_s``.

        The time-based republish matters: ZMQ PUB drops messages when no subscriber is
        connected, and with ``dist.target_utd`` set the learner can legitimately stop
        stepping while it waits for env data - without a timer, an actor that missed the
        first publish would wait for weights forever while the learner waits for data.
        """
        # Delta, not modulo: grad_step advances by num_updates_per_iteration (e.g. 4) per
        # iteration, so `grad_step % 50 == 0` would only ever fire at multiples of 100.
        last = self._last_published_step if self._last_published_step is not None else 0
        due_steps = grad_step - last >= self.publisher.steps_per_update
        due_time = (time.monotonic() - self._last_publish_t) >= republish_every_s
        if due_steps or due_time:
            self.publish(agent, grad_step)
            return True
        return False

    def wait_for_actor_ack_done(self, timeout_s: float) -> bool:
        """After set_phase(DONE): wait until the actor has observed it (or never connected)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.done_acked or self.actor_done or self.actor_session is None:
                return True
            time.sleep(0.1)
        return False

    def wait_for_actor_spec(self, poll_s: float = 1.0) -> dict[str, Any]:
        last_msg = 0.0
        while self.actor_spec is None:
            if time.monotonic() - last_msg > 15:
                print("[learner] waiting for an actor to register (needed for obs/action dims)...")
                last_msg = time.monotonic()
            time.sleep(poll_s)
        return self.actor_spec

    # ---- REP thread ---------------------------------------------------------------
    def _status(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "learner_session": self.session_id,
            "grad_step": int(self.grad_step),
            "published": int(self.publisher.n_published),
            "online_size": int(self.online_size),
            "learning_starts": self.learning_starts,
            **self.extra,
        }

    def _on_request(self, req_type: str, payload: dict) -> dict:
        try:
            if req_type == REQ_GET_INIT:
                if self.init_info is None:
                    return {"ready": False, **self._status()}
                return {"ready": True, "init": self.init_info, **self._status()}

            if req_type == REQ_REGISTER:
                self.actor_spec = payload.get("spec")
                self.actor_session = payload.get("actor_session")
                cursor = int(payload.get("outbox_latest_id", -1))
                self._pending_cursor = cursor
                if STORE_ONLINE in self.server.last_update_id_map:
                    self.server.last_update_id_map[STORE_ONLINE] = cursor
                print(f"[learner] actor registered: session={self.actor_session} cursor={cursor}")
                return {"ok": True, **self._status()}

            if req_type == REQ_HEARTBEAT:
                self.last_heartbeat_t = time.monotonic()
                self.n_heartbeats += 1
                env_step = int(payload.get("env_step", -1))
                if env_step > self.env_step_reported:
                    self.env_step_reported = env_step
                if payload.get("done"):
                    self.actor_done = True
                if self.phase == PHASE_DONE:
                    self.done_acked = True
                for item in payload.get("stats", ()):
                    self._stats.append(item)
                return {"ok": True, "registered_session": self.actor_session, **self._status()}
        except Exception as exc:  # never let a bad message kill the REP thread
            logger.exception("learner request handler failed for %s", req_type)
            return {"ok": False, "error": repr(exc), **self._status()}
        return {"ok": False, "error": f"unknown request {req_type}"}


# =============================================================================
# Actor
# =============================================================================
def _say(msg: str, color: str | None = None, *, level: int = logging.INFO) -> None:
    """Print AND log.

    The actor writes `app.log` through the logging module, but every
    distributed diagnostic in this file and in the trainer was a bare
    `print()` - 84 of them against 11 logger calls. So when an actor sat idle
    for 68 s and was killed, its log contained nothing but call_stats: the
    "[actor] idle - learner phase: init" line that would have explained it went
    to a terminal nobody had kept.
    """
    try:
        from termcolor import colored

        print(colored(msg, color) if color else msg)
    except ImportError:
        print(msg)
    logger.log(level, msg)


class ActorComms:
    def __init__(self, dist_cfg, *, freeze_encoder: bool):
        _, TrainerClient, TrainerConfig = import_agentlace(dist_cfg.codec)
        self.dist_cfg = dist_cfg
        self.actor_session = uuid.uuid4().hex[:12]
        self.outbox = TransitionOutbox(capacity=int(dist_cfg.outbox_capacity))
        self.receiver = WeightReceiver(freeze_encoder=freeze_encoder)

        print(f"[actor] connecting to learner at {dist_cfg.ip}:{dist_cfg.port} (waits until reachable)...")
        self.client = TrainerClient(
            "actor_env",
            dist_cfg.ip,
            make_trainer_config(dist_cfg, TrainerConfig),
            data_stores={STORE_ONLINE: self.outbox},
            wait_for_server=True,
            timeout_ms=int(dist_cfg.timeout_ms),
        )
        self.client.recv_network_callback(self.receiver.stage)
        print(f"[actor] connected. session={self.actor_session}")

        # --- shared with the control loop ---------------------------------------
        self._lock = threading.Lock()
        self._stats: deque[dict[str, Any]] = deque()
        self._env_step = 0
        self._done = False
        self.status: dict[str, Any] = {}
        self._registered_learner: str | None = None
        self._spec: dict[str, Any] | None = None

        self.n_pushed = 0
        self.n_push_failures = 0
        self.n_heartbeat_failures = 0
        self._last_status_t = time.monotonic()
        self._last_stale_warn = 0.0
        self._last_push_warn = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- handshake (blocking, before the control loop) ----------------------------
    def get_init(self, poll_s: float = 2.0) -> dict[str, Any]:
        last_msg = 0.0
        while True:
            res = self.client.request(REQ_GET_INIT, {})
            if res is not None:
                self.status = {k: v for k, v in res.items() if k not in ("init", "ready")}
                if res.get("ready"):
                    return res["init"]
            if time.monotonic() - last_msg > 15:
                phase = res.get("phase") if res else "unreachable"
                print(f"[actor] waiting for learner init (learner phase: {phase})...")
                last_msg = time.monotonic()
            time.sleep(poll_s)

    def register(self, spec: dict[str, Any]) -> dict[str, Any]:
        self._spec = spec
        while True:
            res = self.client.request(
                REQ_REGISTER,
                {"spec": spec, "actor_session": self.actor_session, "outbox_latest_id": self.outbox.latest_data_id()},
            )
            if res is not None and res.get("ok"):
                self._registered_learner = res.get("learner_session")
                self.status = res
                return res
            time.sleep(1.0)

    # ---- called from the control loop (cheap, non-blocking) ------------------------
    def set_env_step(self, env_step: int) -> None:
        self._env_step = int(env_step)

    def log(self, env_step: int, data: dict[str, Any]) -> None:
        """Queue a wandb-style dict for the learner to log at ``env_step``."""
        self._stats.append({"env_step": int(env_step), "data": data})

    @property
    def phase(self) -> str:
        return self.status.get("phase", PHASE_INIT)

    # ---- background thread ----------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="actor-comms", daemon=True)
        self._thread.start()

    def _push_transitions(self) -> None:
        from_id = self.client.get_server_last_update_id(STORE_ONLINE)
        if from_id is None:
            # This is the call behind "Failed to get last update id" in the
            # agentlace logs. It used to bump a counter nobody printed, so a
            # actor that could not push a single transition looked identical to
            # one that had nothing to push.
            self.n_push_failures += 1
            if time.monotonic() - self._last_push_warn > 10:
                self._last_push_warn = time.monotonic()
                _say(
                    f"[actor] cannot read the learner's store cursor "
                    f"({self.n_push_failures} failures) - NO transitions are "
                    f"reaching the learner. {self.outbox.size() if hasattr(self.outbox, 'size') else '?'} "
                    f"queued locally.",
                    "red", level=logging.WARNING,
                )
            return
        max_items = int(self.dist_cfg.push_chunk)
        # Bounded per call: a backlog must never starve the heartbeat, which is the ONLY
        # way the actor learns the learner's phase. (Found by e2e_localhost.py: an
        # unbounded drain kept the actor in warm-up after the learner had finished.)
        for _ in range(int(self.dist_cfg.push_max_chunks_per_tick)):
            items, last_id = self.outbox.get_range(from_id, max_items)
            if not items:
                return
            msg = {"type": "datastore", "store_name": STORE_ONLINE, "payload": {"data": items, "last_id": last_id}}
            res = self.client.req_rep_client.send_msg(msg)
            if res is None or not res.get("success"):
                # Server cursor did not advance; the next tick re-reads it and resends.
                self.n_push_failures += 1
                return
            self.n_pushed += len(items)
            from_id = last_id

    def _heartbeat(self, *, done: bool = False) -> None:
        stats = []
        while True:
            try:
                stats.append(self._stats.popleft())
            except IndexError:
                break
        res = self.client.request(
            REQ_HEARTBEAT,
            {
                "env_step": self._env_step,
                "done": done or self._done,
                "stats": stats,
                "sync": self.receiver.staleness_info,
                "actor_session": self.actor_session,
            },
        )
        if res is None:
            # Undelivered stats go back to the front of the queue for the next tick.
            for s in reversed(stats):
                self._stats.appendleft(s)
            # The heartbeat is the ONLY way the actor learns the learner's
            # phase, so a failed one means `self.status` - and therefore
            # `self.phase` - is now STALE. Silence here is what let an actor sit
            # in PHASE_INIT for 68 s while the learner trained and exited: the
            # actor had no idea the learner had ever moved on.
            self.n_heartbeat_failures += 1
            stale_s = time.monotonic() - self._last_status_t
            if time.monotonic() - self._last_stale_warn > 10:
                self._last_stale_warn = time.monotonic()
                _say(
                    f"[actor] HEARTBEAT FAILED ({self.n_heartbeat_failures} in a row). "
                    f"Learner status is {stale_s:.0f}s stale, so phase is still "
                    f"{self.phase!r} and the actor will NOT act on anything newer. "
                    f"Check the REQ/REP channel to {self.dist_cfg.ip}:{self.dist_cfg.port}.",
                    "red", level=logging.WARNING,
                )
            return

        if self.n_heartbeat_failures:
            _say(f"[actor] heartbeat recovered after {self.n_heartbeat_failures} failure(s)",
                 "green")
            self.n_heartbeat_failures = 0
        prev_phase = self.status.get("phase")
        self.status = res
        self._last_status_t = time.monotonic()
        if res.get("phase") != prev_phase:
            _say(f"[actor] learner phase: {prev_phase} -> {res.get('phase')}", "cyan")
        learner_session = res.get("learner_session")
        if (
            self._spec is not None
            and (learner_session != self._registered_learner or res.get("registered_session") != self.actor_session)
        ):
            print(f"[actor] learner session changed ({self._registered_learner} -> {learner_session}); re-registering")
            self.register(self._spec)

    def _run(self) -> None:
        interval = float(self.dist_cfg.push_interval_s)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._heartbeat()  # status first: phase changes must never wait on a backlog
                self._push_transitions()
            except Exception:  # keep the comms thread alive no matter what
                logger.exception("actor comms tick failed")
            self._stop.wait(max(0.0, interval - (time.monotonic() - t0)))

    def flush_and_stop(self, *, done: bool = True, timeout_s: float = 30.0) -> None:
        """Stop the thread, then synchronously push everything left and send a final heartbeat."""
        self._done = done
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            before = self.n_pushed
            try:
                self._push_transitions()
            except Exception:
                logger.exception("final push failed")
                break
            if self.n_pushed == before:
                break
        try:
            self._heartbeat(done=done)
        except Exception:
            logger.exception("final heartbeat failed")
        try:
            self.client.stop()
        except Exception:
            logger.exception("error stopping TrainerClient")
