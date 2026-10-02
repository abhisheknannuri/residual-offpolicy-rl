# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Everything a checkpoint needs in order to be deployable on its own.

A residual RL checkpoint is only meaningful next to two things that do not live
in its weights:

  * the normalization the observations and actions were expressed in, which
    came from the offline dataset;
  * the EXACT base BC policy the residual was learned on top of, since the
    residual is a correction to one specific ACT checkpoint and is arbitrary
    against any other.

Previously neither was recorded, so inference had to re-open the dataset and
trust an unwritten convention about which ACT server to point at. This module
produces one `deployment` block that `save_checkpoint()` embeds, after which
loading a checkpoint needs the `.pt` and nothing else.

The base-policy identity is the server's `weights_sha256` from `/health` - a
sha256 over the ACT checkpoint's WEIGHT FILES ONLY. It is computed server-side
deliberately: the policy server is the only process guaranteed to have the
weights on local disk, and doing it there means the same trained model reports
the same hash whether it is served from this machine, a laptop, or a remote
box. It excludes `--n-action-steps` and every other inference-time knob, so it
identifies the checkpoint rather than the server invocation.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any


class BasePolicyProvenanceError(RuntimeError):
    """Raised when the base BC policy cannot be identified.

    Deliberately fatal. A residual trained against an unknown base is not
    reproducible and cannot be safely deployed later, and the failure is silent
    at every other layer - the arm simply moves slightly wrong.
    """


def base_policy_identity(policy_server_url: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    """Query `/health` ONCE and return the base policy's identity block.

    Call at startup, not per save: the server cannot swap checkpoints without
    being restarted, so one query covers the whole run.

    Raises `BasePolicyProvenanceError` if the server is unreachable, has no
    policy loaded, or is an older build that does not report `weights_sha256`.
    """
    from trossen_real.inference.policy_client import PolicyClient

    client = PolicyClient(policy_server_url)
    try:
        health = client.health()
    except Exception as exc:
        raise BasePolicyProvenanceError(
            f"cannot reach policy_server.py at {policy_server_url} to record which base BC "
            f"checkpoint this run is training against ({exc!r}).\n"
            "Every residual checkpoint records this, so training refuses to start without it.\n"
            "Start the policy server first (it is needed even when the offline buffer is "
            "already cached - the identity is recorded, not the actions)."
        ) from exc
    finally:
        try:
            client.close()
        except Exception:
            pass

    if not health.get("loaded"):
        raise BasePolicyProvenanceError(
            f"policy_server.py at {policy_server_url} is up but has no policy loaded - {health}"
        )

    sha = health.get("weights_sha256")
    if not sha:
        raise BasePolicyProvenanceError(
            f"policy_server.py at {policy_server_url} does not report 'weights_sha256' in /health.\n"
            "That field is what pins a residual checkpoint to one exact base BC checkpoint, "
            "independently of where the server runs or which --n-action-steps it was started with.\n"
            "Update policy_server.py (it needs _hash_checkpoint_weights) and restart it.\n"
            f"/health returned keys: {sorted(health)}"
        )

    return {
        "kind": "act_http",
        # THE identity. Portable across machines; changes with the weights,
        # including between steps of the same training run.
        "weights_sha256": sha,
        "weights_files": health.get("weights_files", []),
        # Everything below is descriptive only - useful for debugging a
        # mismatch, never part of the identity.
        "server_checkpoint_path": health.get("checkpoint"),
        "image_keys": health.get("image_keys", []),
        "non_image_keys": health.get("non_image_keys", []),
        "likely_action_space": health.get("likely_action_space"),
        "action_space_source": health.get("action_space_source"),
        # NOT part of the identity on purpose: these are inference-time knobs,
        # recorded so you can see how the server happened to be running.
        "observed_n_action_steps": health.get("n_action_steps"),
        "observed_chunk_size": health.get("chunk_size"),
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except Exception:
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def build_deployment_meta(
    cfg: Any,
    action_scaler: Any,
    state_standardizer: Any,
    base_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the `deployment` block embedded in every checkpoint.

    `base_policy` is the result of one `base_policy_identity()` call made at
    startup, or None when no base BC policy applies (sim/robomimic runs, which
    have no policy server to ask).
    """
    return {
        "schema_version": 1,
        "normalization": {
            "action_scaler": action_scaler.state_dict(),
            "state_standardizer": state_standardizer.state_dict(),
        },
        "base_policy": base_policy,
        "obs": {
            # What the RL encoder consumes, so inference resizes identically.
            "image_keys": [cfg.rl_camera] if isinstance(cfg.rl_camera, str) else list(cfg.rl_camera),
            "rl_image_size": cfg.offline_data.image_size,
            "action_dim": 7,
            "prop_dim": 7,
            "real_action_space": cfg.get("real_action_space", None),
        },
        "provenance": {
            "dataset_name": cfg.offline_data.name,
            "dataset_root": cfg.offline_data.get("root", None),
            "dataset_num_episodes": cfg.offline_data.num_episodes,
            "task": cfg.task,
            "git": _git_state(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def assert_base_policy_matches(
    ckpt_base_policy: dict[str, Any] | None,
    live_health: dict[str, Any],
    *,
    mode: str = "strict",
) -> None:
    """Check a loaded checkpoint against the policy server it is about to run on.

    `mode`:
      strict  weights_sha256 must match (the default, and the only one that
              actually guarantees the residual is valid)
      warn    log the mismatch and continue - for deliberate cross-base probes
      off     skip entirely
    """
    if mode == "off":
        return
    if not ckpt_base_policy:
        raise BasePolicyProvenanceError(
            "this checkpoint carries no base_policy block, so it cannot be verified against "
            "the running policy server. It predates deployment metadata - retrain, or pass "
            "mode='off' if you accept running an unverified residual."
        )
    want = ckpt_base_policy.get("weights_sha256")
    got = live_health.get("weights_sha256")
    if want and got and want == got:
        return
    msg = (
        "base BC policy MISMATCH - this residual was trained against a different ACT "
        "checkpoint than the one this server is serving. The residual is a correction to "
        "one specific base policy and is meaningless against another.\n"
        f"  checkpoint expects weights_sha256 = {want}\n"
        f"    (trained against: {ckpt_base_policy.get('server_checkpoint_path')})\n"
        f"  server reports     weights_sha256 = {got}\n"
        f"    (now serving:     {live_health.get('checkpoint')})"
    )
    if mode == "warn":
        import logging

        logging.getLogger(__name__).warning(msg)
        return
    raise BasePolicyProvenanceError(msg)


def write_buffer_sidecar(cache_dir, deployment_meta: dict[str, Any] | None) -> None:
    """Record the base BC policy next to a replay-buffer cache.

    Written as its OWN file, never merged into `user_metadata.json`: that file
    is a verbatim dump of the dict whose sha1 names the cache directory, so
    adding a key to it would change every cache hash and orphan every buffer
    already on disk.

    A sidecar is additive and invisible to the hash. Buffers written before
    this existed simply have no sidecar, and nothing reads `user_metadata.json`
    back for validation anyway.
    """
    import json
    from pathlib import Path

    if not deployment_meta or not deployment_meta.get("base_policy"):
        return
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / "base_policy.json").write_text(
            json.dumps(deployment_meta["base_policy"], indent=2, default=str)
        )
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "could not write base_policy.json next to %s - the buffer itself is fine, "
            "only its base-policy record is missing.", cache_dir, exc_info=True,
        )
