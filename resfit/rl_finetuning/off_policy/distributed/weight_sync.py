"""Learner -> actor weight synchronisation.

Two ResFiT-specific concerns, neither of which HIL-SERL has to deal with.

1. The ViT encoder is TRAINED, so it must be synced with the actor head.
   ``FREEZE_ENCODER="false"`` in all five real-station scripts; ``q_agent.py:105``
   only freezes params when the flag is *true*, and ``encoder_opt.step()`` runs
   at ``q_agent.py:419`` inside ``update_critic()``. So the encoder changes on
   every critic update, and ``agent.act()`` calls ``self._encode()`` with those
   same encoders. Publishing the actor head against a different encoder snapshot
   means the head consumes features from a representation it was never trained
   against. Both must come from ONE gradient step, in ONE message.

   When ``freeze_encoder=True`` the encoder is static and is sent once at
   handshake instead of on every publish.

2. PyTorch ``load_state_dict()`` mutates in place.
   HIL-SERL's receive callback rebinds ``agent`` straight from the broadcast
   thread (``train_rlpd.py:133-135``). That is safe *only* because JAX params are
   an immutable pytree and ``agent.replace(...)`` is a pointer swap. Doing the
   same with ``load_state_dict`` lets the control loop run a forward pass over
   half-old, half-new weights - no exception, just a bad action sent to a real
   arm. So the receiver only *stages* the payload; the actor's main loop applies
   it between env steps.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import torch

logger = logging.getLogger(__name__)


def extract_sync_weights(agent, *, freeze_encoder: bool, grad_step: int) -> dict[str, Any]:
    """Snapshot exactly what the actor needs, as CPU tensors.

    Deliberately excludes ``critic``, ``critic_target``, ``actor_target`` and all
    optimizer state: the actor never uses them and they are the bulk of the bytes.
    """
    def cpu_sd(module) -> dict[str, torch.Tensor]:
        # .cpu() already copies off-GPU; clone only tensors that were on CPU to begin with,
        # so the snapshot can never alias a live parameter that keeps training.
        return {
            k: (v.detach().clone() if v.device.type == "cpu" else v.detach().cpu())
            for k, v in module.state_dict().items()
        }

    payload: dict[str, Any] = {"actor": cpu_sd(agent.actor), "grad_step": grad_step}
    if not freeze_encoder:
        payload["encoders"] = cpu_sd(agent.encoders)
    return payload


class WeightPublisher:
    """Learner side: publish a consistent snapshot every N gradient steps."""

    def __init__(self, server, *, steps_per_update: int, freeze_encoder: bool):
        self.server = server
        self.steps_per_update = steps_per_update
        self.freeze_encoder = freeze_encoder
        self.n_published = 0

    def maybe_publish(self, agent, grad_step: int, *, force: bool = False) -> bool:
        if not force and (grad_step <= 0 or grad_step % self.steps_per_update != 0):
            return False
        self.publish(agent, grad_step)
        return True

    def publish(self, agent, grad_step: int) -> None:
        payload = extract_sync_weights(
            agent, freeze_encoder=self.freeze_encoder, grad_step=grad_step
        )
        self.server.publish_network(payload)
        self.n_published += 1


class WeightReceiver:
    """Actor side: stage on the broadcast thread, apply on the main loop.

    ``stage`` is the agentlace ``recv_network_callback``. It must do nothing but
    store a reference - see the module docstring.
    """

    def __init__(self, *, freeze_encoder: bool):
        self.freeze_encoder = freeze_encoder
        self._staged: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self.last_applied_grad_step: int = -1
        self.n_applied = 0
        self.n_dropped = 0

    # ---- broadcast thread -------------------------------------------------
    def stage(self, payload: dict[str, Any]) -> None:
        """Callback for agentlace. Never touches the live networks."""
        with self._lock:
            if self._staged is not None:
                # Superseded before the main loop got to it. Expected whenever
                # the learner outruns the 20 Hz control loop; zmq.CONFLATE drops
                # older messages at the socket, this drops them one layer up.
                self.n_dropped += 1
            self._staged = payload

    # ---- actor main loop --------------------------------------------------
    def apply_if_new(self, agent) -> int | None:
        """Install staged weights if any. Call BETWEEN env steps, never during act().

        Returns the applied grad_step, or ``None`` if nothing was pending.
        """
        with self._lock:
            payload, self._staged = self._staged, None
        if payload is None:
            return None

        agent.actor.load_state_dict(payload["actor"])
        if "encoders" in payload:
            agent.encoders.load_state_dict(payload["encoders"])
        elif not self.freeze_encoder:
            # The learner is training the encoder but did not send it. Silently
            # continuing would pair a fresh actor head with a stale encoder.
            raise RuntimeError(
                "Weight payload has no 'encoders' but freeze_encoder=False. "
                "Actor and learner disagree about cfg.agent.freeze_encoder."
            )

        self.last_applied_grad_step = int(payload.get("grad_step", -1))
        self.n_applied += 1
        return self.last_applied_grad_step

    @property
    def staleness_info(self) -> dict[str, int]:
        return {
            "sync/last_applied_grad_step": self.last_applied_grad_step,
            "sync/n_applied": self.n_applied,
            "sync/n_superseded": self.n_dropped,
        }
