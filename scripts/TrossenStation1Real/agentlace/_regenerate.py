from pathlib import Path

SRC = Path("/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/TrossenStation1Real")
DST = SRC / "agentlace"

PY_OLD = '    python resfit/rl_finetuning/scripts/train_residual_td3.py'
PY_NEW = '    "${PYTHON_BIN}" resfit/rl_finetuning/scripts/train_residual_td3_distributed.py'
CN_OLD = 'CONFIG_NAME="residual_td3_trossen_real_config"'
CN_NEW = 'CONFIG_NAME="residual_td3_trossen_real_dist_config"'
ANCHOR = '    --config-name="${CONFIG_NAME}"'

BANNER = '''# =============================================================================
# GENERATED COPY - distributed (actor/learner) variant
# =============================================================================
# Source of truth for ALL hyper-parameters: ../@@NAME@@
# This file is a byte-for-byte copy of that script except for:
#   1. sources ./common_dist.sh  (role parsing, IP/ports, PYTHON_BIN)
#   2. entrypoint  -> train_residual_td3_distributed.py
#   3. CONFIG_NAME -> ..._dist_config
#   4. CMD gains   -> role= / dist.* arguments
#
# Run BOTH roles from this same file so the two nodes cannot disagree on any
# hyper-parameter (a silent actor/learner config mismatch is the single most
# dangerous failure mode of this architecture):
#
#     ./@@NAME@@ learner     # on the GPU server
#     ./@@NAME@@ actor       # on the laptop  (LEARNER_IP=<server> ./@@NAME@@ actor)
#
# Regenerate after editing the original:  ./_regenerate.sh
# =============================================================================
'''

SOURCE_BLOCK = '''# ── Distributed role / transport (see common_dist.sh) ────────────────────────
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_dist.sh
source "${_HERE}/common_dist.sh"
@@FORCE@@
'''

DIST_ARGS = '''
    # ── Distributed actor/learner (agentlace) ──
    role="${ROLE}"
    dist.ip="${LEARNER_IP}"
    dist.port="${LEARNER_PORT}"
    dist.broadcast_port="${LEARNER_BROADCAST_PORT}"
    dist.steps_per_update="${STEPS_PER_UPDATE}"
    dist.target_utd="${TARGET_UTD}"
'''

FILES = [
    ("train_residual_rl_real.sh", None),
    ("train_residual_rl_real_withOffRL.sh", None),
    ("train_residual_rl_real_continue_from_criticwarmup.sh", None),
    ("train_residual_rl_real_continue_from_offlineRL.sh", None),
    # offline pretrain touches no environment -> learner only (see migration doc §7)
    ("train_residual_rl_offline_pretrain.sh", "learner"),
]

for name, force_role in FILES:
    txt = (SRC / name).read_text()
    assert txt.count(PY_OLD) == 1 and txt.count(CN_OLD) == 1 and txt.count(ANCHOR) == 1, name

    txt = txt.replace(PY_OLD, PY_NEW)
    txt = txt.replace(CN_OLD, CN_NEW)
    txt = txt.replace(ANCHOR, ANCHOR + "\n" + DIST_ARGS.rstrip("\n"))

    force = ""
    if force_role:
        force = ('\n# This phase touches no environment at all - it is learner-only.\n'
                 'if [[ "${ROLE}" == "actor" ]]; then\n'
                 '    echo "ERROR: this script runs offline_pretrain_only (no env, no robot) - run it with: learner" >&2\n'
                 '    exit 1\n'
                 'fi\n'
                 f'ROLE="{force_role}"\n')
    inject = BANNER.replace("@@NAME@@", name) + "\n" + SOURCE_BLOCK.replace("@@FORCE@@", force)

    # insert right after the shebang line
    lines = txt.split("\n")
    assert lines[0].startswith("#!"), name
    out = lines[0] + "\n" + inject + "\n".join(lines[1:])

    p = DST / name
    p.write_text(out)
    p.chmod(0o755)
    print(f"  wrote {p.relative_to(SRC.parent.parent)}  ({len(out.splitlines())} lines)")
