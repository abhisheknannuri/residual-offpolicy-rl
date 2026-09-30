"""agentlace bootstrap: import path, wire codec, and the ResFiT TrainerConfig.

Why this module exists
----------------------
1. **No install needed.** agentlace is used straight from the clone at
   ``<repo>/agentlace`` (added to ``sys.path`` if ``import agentlace`` fails).
   That clone's ``ReqRepClient`` is also more robust than the 0.1.3 wheel
   HIL-SERL pins (it survives socket errors instead of wedging the REQ socket).

2. **Deterministic wire codec without lz4.** agentlace hard-codes
   ``compression='lz4'`` and imports ``lz4.frame`` at module import time
   (``agentlace/internal/utils.py``). ``lz4`` is not installed in the training
   ``.venv`` (which has no pip), so rather than touching the environment we:

   - install a stub ``lz4.frame`` in ``sys.modules`` *only* if real lz4 is
     missing, purely so ``agentlace.internal.utils`` can be imported, and
   - replace ``agentlace.internal.utils.make_compression_method`` with ours
     **before** ``agentlace.zmq_wrapper.{req_rep,broadcast}`` are imported
     (both bind that name at import time).

   Our codec ignores the ``'lz4'`` name agentlace passes and always uses
   ``dist.codec``, so both machines speak the same format whether or not either
   has lz4. The codec name is folded into ``TrainerConfig.version``, which
   agentlace hashes at connect time - a codec mismatch refuses to connect
   instead of producing garbage.
"""

from __future__ import annotations

import pickle
import sys
import types
import zlib
from pathlib import Path
from typing import Any, Callable

# Custom request types (actor -> learner, answered on the learner's REP thread).
REQ_GET_INIT = "get-init"          # normalization stats, config fingerprint, start env step
REQ_REGISTER = "actor-register"    # actor spec + resets the transition cursor for this session
REQ_HEARTBEAT = "actor-heartbeat"  # env_step, episode stats, timing; reply carries learner status
REQUEST_TYPES = [REQ_GET_INIT, REQ_REGISTER, REQ_HEARTBEAT]

# Datastore name for the single transition channel (interventions stay in the
# online buffer, flagged via the `intervened` key - see migration doc §10).
STORE_ONLINE = "actor_env"

PROTOCOL_VERSION = "resfit-dist-1"
SUPPORTED_CODECS = ("zlib", "none", "lz4")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_state: dict[str, Any] = {"codec": None, "mods": None}


def _make_codec(codec: str) -> tuple[Callable[[Any], bytes], Callable[[bytes], Any]]:
    proto = pickle.HIGHEST_PROTOCOL
    if codec == "none":
        return (lambda o: pickle.dumps(o, protocol=proto)), pickle.loads
    if codec == "zlib":
        # level 1: uint8 camera frames still compress well; float weights barely
        # compress at any level, so paying for higher levels buys nothing.
        return (lambda o: zlib.compress(pickle.dumps(o, protocol=proto), 1)), (
            lambda b: pickle.loads(zlib.decompress(b))
        )
    if codec == "lz4":
        import lz4.frame as _lz4  # must be REAL lz4 on both machines

        if getattr(_lz4, "__resfit_stub__", False):
            raise RuntimeError("dist.codec=lz4 but the lz4 package is not installed on this machine.")
        return (lambda o: _lz4.compress(pickle.dumps(o, protocol=proto))), (
            lambda b: pickle.loads(_lz4.decompress(b))
        )
    raise ValueError(f"dist.codec must be one of {SUPPORTED_CODECS}, got {codec!r}")


def _ensure_agentlace_on_path() -> None:
    """Make the REAL agentlace package importable.

    Trap this guards against: the clone lives at ``<repo>/agentlace/`` and the package
    is one level down at ``<repo>/agentlace/agentlace/``. With the repo root (or cwd)
    on ``sys.path``, ``import agentlace`` "succeeds" by importing the clone *folder* as
    an empty namespace package - and every submodule import then fails. A real package
    has a ``__file__``; a namespace package does not.
    """
    try:
        import agentlace

        if getattr(agentlace, "__file__", None):
            return
    except ImportError:
        pass
    for name in [m for m in sys.modules if m == "agentlace" or m.startswith("agentlace.")]:
        del sys.modules[name]
    clone = _REPO_ROOT / "agentlace"
    if not (clone / "agentlace" / "trainer.py").exists():
        raise ImportError(
            f"agentlace is not installed and no clone was found at {clone}. "
            "Clone https://github.com/youliangtan/agentlace there."
        )
    sys.path.insert(0, str(clone))
    import agentlace

    if not getattr(agentlace, "__file__", None):
        raise ImportError(f"could not import the agentlace package from {clone}")


def import_agentlace(codec: str = "zlib"):
    """Import agentlace with ``codec`` pinned on every socket.

    Returns ``(TrainerServer, TrainerClient, TrainerConfig)``. Idempotent for the
    same codec; raises if called again with a different one (the zmq wrappers
    have already bound the first codec).
    """
    if _state["mods"] is not None:
        if _state["codec"] != codec:
            raise RuntimeError(f"agentlace already imported with codec={_state['codec']!r}, not {codec!r}")
        return _state["mods"]

    if "agentlace.zmq_wrapper.req_rep" in sys.modules or "agentlace.trainer" in sys.modules:
        raise RuntimeError(
            "agentlace.trainer/zmq_wrapper was imported before import_agentlace(); the wire codec "
            "can no longer be pinned. Import agentlace only through this function."
        )

    _ensure_agentlace_on_path()

    try:
        import lz4.frame  # noqa: F401
    except ImportError:
        # Stub ONLY so agentlace.internal.utils imports. It is never used to encode:
        # make_compression_method is replaced right below.
        stub_pkg = types.ModuleType("lz4")
        stub_frame = types.ModuleType("lz4.frame")

        def _missing(*_a, **_k):
            raise RuntimeError("lz4 is not installed; this stub must never be called")

        stub_frame.compress = stub_frame.decompress = _missing
        stub_frame.__resfit_stub__ = True
        stub_pkg.frame = stub_frame
        sys.modules["lz4"] = stub_pkg
        sys.modules["lz4.frame"] = stub_frame

    compress, decompress = _make_codec(codec)  # validate before patching

    import agentlace.internal.utils as _al_utils

    def make_compression_method(_ignored_name: str):
        return compress, decompress

    _al_utils.make_compression_method = make_compression_method

    from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer

    _state["codec"] = codec
    _state["mods"] = (TrainerServer, TrainerClient, TrainerConfig)
    return _state["mods"]


def make_trainer_config(dist_cfg, TrainerConfig):
    """Both nodes must build this identically - agentlace hashes it on connect."""
    return TrainerConfig(
        port_number=int(dist_cfg.port),
        broadcast_port=int(dist_cfg.broadcast_port),
        request_types=list(REQUEST_TYPES),
        version=f"{PROTOCOL_VERSION}+{dist_cfg.codec}",
    )
