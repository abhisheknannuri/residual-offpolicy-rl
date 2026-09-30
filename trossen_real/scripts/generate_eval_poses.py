import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# CONFIGURATION CONSTANTS (All dimensions in centimeters)
# ==============================================================================
WORKSPACE_WIDTH = 22.5       # X dimension (width)
WORKSPACE_HEIGHT = 14.5       # Y dimension (depth/height)
CUBE_SIZE = 3.0               # Cube width & height (X and Y)

N_POSES = 20                  # Total evaluation poses to generate
RANDOM_SEED = 42              # Fixed seed for reproducibility

# Allowed overhang: 1/3 of cube size on -X, +X, -Y; 0 on +Y
MAX_OVERHANG = CUBE_SIZE / 3.0

# Discrete rotation angles (in degrees).
# For a square cube, rotations wrap at 90 deg, so [-45, 45] covers all physical appearances.
# DISCRETE_YAW_DEG = [0.0, 20.0, -20.0, 45.0, -45.0]
DISCRETE_YAW_DEG = [0.0, 45.0, -45.0]

# ==============================================================================
# POSE GENERATOR
# ==============================================================================
def generate_eval_poses(n_poses, seed=42):
    """Generates deterministic (x, y, yaw_deg) poses allowing partial overhang."""
    rng = np.random.default_rng(seed)
    half_c = CUBE_SIZE / 2.0

    # Overhang boundaries for cube center
    # Standard inside bounds: [half_c, DIM - half_c]
    # Allowed overhang expands -X, +X, -Y by MAX_OVERHANG
    x_min = half_c - MAX_OVERHANG
    x_max = (WORKSPACE_WIDTH - half_c) + MAX_OVERHANG
    
    y_min = half_c - MAX_OVERHANG
    y_max = WORKSPACE_HEIGHT - half_c  # +Y stays strictly within workspace

    poses = []
    for _ in range(n_poses):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        yaw = float(rng.choice(DISCRETE_YAW_DEG))
        poses.append((x, y, yaw))

    return poses

def generate_eval_poses_stratified(n_poses, seed=42, jitter=0.75):
    """Stratified (jittered-grid) poses - one cube per cell of a grid covering
    the workspace, jittered inside its cell.

    Why not plain `rng.uniform()` (see `generate_eval_poses`): i.i.d. uniform
    sampling gives uniform *density*, not uniform *coverage*. At n=20 in 2D you
    reliably get clumps and voids - the seed-42 set has a 5.3 cm (23% of width)
    band through the middle of X with no pose in it, 8 of 20 grid cells empty,
    and two poses 0.65 cm apart (indistinguishable for a 3 cm cube). Re-rolling
    the seed just produces a different bad layout.

    Stratifying bounds the worst-case gap by the cell size and makes duplicates
    impossible, while staying fully deterministic. `jitter` is the fraction of
    each cell the sample may move within (1.0 = anywhere in the cell, which lets
    neighbours touch at a shared edge; 0.75 keeps a margin so poses stay
    physically distinguishable).

    Yaw is assigned round-robin over DISCRETE_YAW_DEG and then shuffled, so the
    three orientations are as evenly represented as n allows - rather than the
    9/5/6 split random choice produced.
    """
    rng = np.random.default_rng(seed)
    half_c = CUBE_SIZE / 2.0
    x_min = half_c - MAX_OVERHANG
    x_max = (WORKSPACE_WIDTH - half_c) + MAX_OVERHANG
    y_min = half_c - MAX_OVERHANG
    y_max = WORKSPACE_HEIGHT - half_c

    # Grid shaped to the usable area's aspect ratio, so cells stay near-square.
    span_x, span_y = x_max - x_min, y_max - y_min
    cols = max(1, int(round(math.sqrt(n_poses * span_x / span_y))))
    rows = int(math.ceil(n_poses / cols))
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)                      # decorrelate pose index from position
    cells = cells[:n_poses]

    cw, ch = span_x / cols, span_y / rows
    # Balanced yaws: round-robin, then shuffled.
    yaws = [DISCRETE_YAW_DEG[i % len(DISCRETE_YAW_DEG)] for i in range(n_poses)]
    rng.shuffle(yaws)

    poses = []
    for (c, r), yaw in zip(cells, yaws):
        cx0, cy0 = x_min + c * cw, y_min + r * ch
        pad_x, pad_y = cw * (1 - jitter) / 2, ch * (1 - jitter) / 2
        x = rng.uniform(cx0 + pad_x, cx0 + cw - pad_x)
        y = rng.uniform(cy0 + pad_y, cy0 + ch - pad_y)
        poses.append((x, y, float(yaw)))
    return poses


GENERATORS = {"uniform": generate_eval_poses, "stratified": generate_eval_poses_stratified}


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def get_rotated_corners(cx, cy, size, yaw_deg):
    """Computes the 4 corners of a rotated square around center (cx, cy)."""
    half = size / 2.0
    corners = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half]
    ])
    rad = math.radians(yaw_deg)
    rot_mat = np.array([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad),  math.cos(rad)]
    ])
    rotated = np.dot(corners, rot_mat.T)
    return rotated + np.array([cx, cy])

def visualize_poses(poses):
    n = len(poses)
    cols = 3 if n <= 9 else 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.2), squeeze=False)
    fig.suptitle(f"Evaluation Cube Poses (Seed: {RANDOM_SEED})", fontsize=14, fontweight="bold")

    for i, pose in enumerate(poses):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        cx, cy, yaw = pose

        # 1. Draw workspace boundary (0 to WORKSPACE_WIDTH, 0 to WORKSPACE_HEIGHT)
        table_rect = patches.Rectangle(
            (0, 0), WORKSPACE_WIDTH, WORKSPACE_HEIGHT,
            linewidth=1.2, edgecolor="#555555", facecolor="#F8F9FA"
        )
        ax.add_patch(table_rect)

        # 2. Draw Dotted Red Cube
        corners = get_rotated_corners(cx, cy, CUBE_SIZE, yaw)
        cube_poly = patches.Polygon(
            corners, closed=True,
            linewidth=2.0, linestyle=":", edgecolor="#D32F2F", facecolor="#FFCDD2", alpha=0.8
        )
        ax.add_patch(cube_poly)

        # 3. Heading arrow (shows cube orientation)
        arrow_len = CUBE_SIZE * 0.7
        rad = math.radians(yaw)
        ax.arrow(
            cx, cy, arrow_len * math.cos(rad), arrow_len * math.sin(rad),
            head_width=1.5, head_length=1.5, fc="#B71C1C", ec="#B71C1C", zorder=5
        )

        # 4. Viewport framing (with padding around workspace)
        pad_x = 4.0
        pad_y = 2.0
        ax.set_xlim(-pad_x, WORKSPACE_WIDTH + pad_x)
        ax.set_ylim(-pad_y, WORKSPACE_HEIGHT + pad_y)
        ax.set_aspect("equal")

        # Subplot Title with exact manual coordinates for physical placement
        ax.set_title(f"Pose #{i+1}: X={cx:.1f} Y={cy:.1f} cm | {yaw:+.0f}°", fontsize=9, fontweight="semibold")
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide unused grid subplots if N is not a multiple of cols
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()

def order_poses_for_printing(poses, per_page=5):
    """Reorder AND RENUMBER poses so the printed mat runs in index order.

    Two things the operator wants at once, which plain generation cannot give:
      * page 1 holds poses 1..5, page 2 holds 6..10 - so you can work straight
        down the stack instead of hunting for pose 7;
      * no two cubes on the same page overlap, or you cannot tell which outline
        is which (20 x 3 cm outlines cover ~55% of a 22.5 x 14.5 cm workspace).

    Achieved by choosing WHICH poses share a page first (each pose goes to the
    page where it sits furthest from what is already there), then relabelling
    sequentially. Within a page, poses are ordered top-row-first, left to right,
    so the numbers scan like reading.

    NOTE: this CHANGES pose indices. A results file recorded against the old
    numbering refers to different physical placements afterwards - so only apply
    it before starting a campaign, or remap existing records.
    """
    n_pages = max(1, math.ceil(len(poses) / per_page))
    pages = [[] for _ in range(n_pages)]
    for pose in poses:
        best, best_score = None, None
        for pg in pages:
            if len(pg) >= per_page:
                continue
            score = min((math.dist(pose[:2], q[:2]) for q in pg), default=float("inf"))
            if best_score is None or score > best_score:
                best, best_score = pg, score
        (best if best is not None else pages[0]).append(pose)

    ordered = []
    for pg in pages:
        # reading order within the sheet: top band first, then left -> right.
        pg.sort(key=lambda q: (-round(q[1] / max(CUBE_SIZE, 1e-6)), q[0]))
        ordered.extend(pg)
    return ordered


# Standard paper, PORTRAIT, in cm. The mat MUST be a real paper size: a
# custom-sized page makes print dialogs default to "shrink oversized pages",
# which silently rescales the sheet and destroys the 1:1 scale it depends on.
PAPER_CM = {"a4": (21.0, 29.7), "letter": (21.59, 27.94), "a3": (29.7, 42.0)}


def _rotate_pose_90(x, y, yaw):
    """Rotate a pose 90 deg CCW about the workspace origin, for printing the
    22.5 cm-wide workspace onto a 21.0 cm-wide portrait sheet.

    A proper rotation, NOT a swap of x and y - swapping would MIRROR the layout
    and every cube would be placed in the wrong spot (and the yaw handedness
    would flip), which is exactly the kind of error a printed reference must not
    be able to make. det([[0,-1],[1,0]]) = +1.

        u = H - y      (page horizontal, spans the workspace's 14.5 cm depth)
        v = x          (page vertical,   spans the workspace's 22.5 cm width)
        yaw -> yaw + 90
    """
    return WORKSPACE_HEIGHT - y, x, yaw + 90.0


def make_placement_mat(poses, path, per_page=5, seed=RANDOM_SEED, method="uniform",
                       regroup=True, paper="a4", rotate="auto"):
    """Print-at-1:1 placement mat: put the cube inside outline N, no ruler needed.

    The whole point is to remove measuring from the loop. Placing a cube by
    reading "X=17.14, Y=5.99, yaw +45" off a table and measuring it out takes
    the best part of a minute; dropping it into a printed outline takes a few
    seconds. Over a 300-run campaign that is hours.

    Drawn at TRUE physical scale, so it MUST be printed at 100% / "actual size"
    (never "fit to page"). Each page carries a 100 mm check bar - measure it
    before trusting the sheet.

    Poses are split across pages (`per_page`) because 20 x 3 cm outlines on one
    22.5 x 14.5 cm workspace overlap badly - about 55% area coverage - which is
    worse than useless when you are trying to find outline 13 in a hurry.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    # Sized to sit comfortably inside A4 LANDSCAPE's printable area (~28.7 x 20 cm
    # after typical unprintable edges). If the sheet is any closer to the paper
    # edge the printer quietly rescales it to fit, which destroys the 1:1 scale
    # the whole mat depends on.
    if paper not in PAPER_CM:
        raise ValueError(f"paper must be one of {sorted(PAPER_CM)}, got {paper!r}")
    page_w, page_h = PAPER_CM[paper]          # PORTRAIT - many printers refuse landscape

    fits_upright = WORKSPACE_WIDTH <= page_w - 2.0 and WORKSPACE_HEIGHT <= page_h - 5.0
    if rotate == "auto":
        do_rotate = not fits_upright
    else:
        do_rotate = bool(rotate)
    draw_w = WORKSPACE_HEIGHT if do_rotate else WORKSPACE_WIDTH
    draw_h = WORKSPACE_WIDTH if do_rotate else WORKSPACE_HEIGHT
    if draw_w > page_w - 2.0 or draw_h > page_h - 5.0:
        raise ValueError(
            f"workspace {WORKSPACE_WIDTH} x {WORKSPACE_HEIGHT} cm does not fit on {paper} "
            f"portrait ({page_w} x {page_h} cm) at 1:1, rotated or not - try --paper a3"
        )
    fig_w, fig_h = page_w / 2.54, page_h / 2.54
    strip_cm = 1.8                     # bottom band for the print-scale check bar
    title_cm = 1.8                     # top band for the heading
    # Workspace centred horizontally, and vertically between the two bands.
    left_cm = (page_w - draw_w) / 2
    bottom_cm = strip_cm + (page_h - strip_cm - title_cm - draw_h) / 2

    # Group poses onto pages by SPATIAL SEPARATION, not index order. Chunking
    # 1-5, 6-10, ... routinely puts neighbouring cubes on the same sheet, where
    # their 3 cm outlines overlap and you cannot tell which is which. Greedy
    # farthest-first: each pose goes on the page where it sits furthest from
    # what is already there.
    numbered = list(enumerate(poses, 1))
    if regroup:
        n_pages = max(1, math.ceil(len(poses) / per_page))
        pages = [[] for _ in range(n_pages)]
        for idx, pose in numbered:
            best, best_score = None, None
            for pg in pages:
                if len(pg) >= per_page:
                    continue
                score = min((math.dist(pose[:2], q[:2]) for _, q in pg), default=float("inf"))
                if best_score is None or score > best_score:
                    best, best_score = pg, score
            (best if best is not None else pages[0]).append((idx, pose))
    else:
        # Already ordered by order_poses_for_printing(): plain sequential pages
        # are both in index order AND overlap-free.
        pages = [numbered[i:i + per_page] for i in range(0, len(numbered), per_page)]

    with PdfPages(path) as pdf:
        for chunk in pages:
            chunk = sorted(chunk)
            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = fig.add_axes([
                left_cm / page_w, bottom_cm / page_h,
                draw_w / page_w, draw_h / page_h,
            ])
            ax.set_xlim(0, draw_w)
            ax.set_ylim(0, draw_h)
            ax.set_aspect("equal")
            if do_rotate:
                # Horizontal axis is the workspace's Y, running right-to-left
                # (u = H - y); vertical axis is the workspace's X.
                yt = list(range(0, int(WORKSPACE_HEIGHT) + 1, 5))
                ax.set_xticks([WORKSPACE_HEIGHT - v for v in yt])
                ax.set_xticklabels([str(v) for v in yt])
                ax.set_yticks(range(0, int(WORKSPACE_WIDTH) + 1, 5))
                ax.set_xlabel("workspace Y (cm)", fontsize=8)
                ax.set_ylabel("workspace X (cm)", fontsize=8)
            else:
                ax.set_xticks(range(0, int(WORKSPACE_WIDTH) + 1, 5))
                ax.set_yticks(range(0, int(WORKSPACE_HEIGHT) + 1, 5))
                ax.set_xlabel("workspace X (cm)", fontsize=8)
                ax.set_ylabel("workspace Y (cm)", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            for idx, (cx, cy, yaw) in chunk:
                if do_rotate:
                    cx, cy, yaw = _rotate_pose_90(cx, cy, yaw)
                corners = get_rotated_corners(cx, cy, CUBE_SIZE, yaw)
                ax.add_patch(patches.Polygon(corners, closed=True, linewidth=2.0,
                                             linestyle="--", edgecolor="#D32F2F",
                                             facecolor="none"))
                ax.plot([cx], [cy], "+", color="#D32F2F", markersize=8, markeredgewidth=1.4)
                ax.text(cx, cy, f"{idx}", ha="center", va="center", fontsize=16,
                        fontweight="bold", color="#D32F2F",
                        bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="none", alpha=0.85))
                rad = math.radians(yaw)
                ax.arrow(cx, cy, CUBE_SIZE * 0.62 * math.cos(rad), CUBE_SIZE * 0.62 * math.sin(rad),
                         head_width=0.45, head_length=0.45, fc="#B71C1C", ec="#B71C1C", lw=1.2)

            nums = ", ".join(str(i) for i, _ in chunk)
            if do_rotate:
                # Without this the sheet is ambiguous: a 90 deg-rotated mat laid
                # down the wrong way puts every cube in the wrong place.
                ax.annotate("", xy=(draw_w * 0.5, draw_h + 0.9), xytext=(draw_w * 0.5, draw_h + 0.1),
                            annotation_clip=False,
                            arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=1.8))
                ax.text(draw_w * 0.5 + 0.35, draw_h + 0.45, "workspace +X (away from robot base)",
                        fontsize=8, color="#1565C0", va="center", clip_on=False)
                ax.text(-0.5, draw_h * 0.5, "SHEET ROTATED 90deg", fontsize=8, color="#1565C0",
                        rotation=90, ha="center", va="center", fontweight="bold", clip_on=False)
            ax.set_title(f"Eval placement mat - poses {nums}\n"
                         f"{method}, seed {seed}, cube {CUBE_SIZE:g} cm, "
                         f"workspace {WORKSPACE_WIDTH:g} x {WORKSPACE_HEIGHT:g} cm",
                         fontsize=10, fontweight="bold")

            # Print-scale check bar, in figure coords at a known physical length.
            bar_cm = 10.0
            x0 = left_cm / page_w
            y0 = 0.9 / page_h
            fig.add_artist(plt.Line2D([x0, x0 + bar_cm / page_w], [y0, y0],
                                      color="black", linewidth=2.5))
            for t in (0.0, bar_cm):
                xt = x0 + t / page_w
                fig.add_artist(plt.Line2D([xt, xt], [y0 - 0.008, y0 + 0.008],
                                          color="black", linewidth=2.5))
            fig.text(x0, y0 + 0.020,
                     f"PRINT AT 100% / 'actual size' on {paper.upper()} PORTRAIT (never "
                     f"'fit to page'). This bar must measure exactly 100 mm.",
                     fontsize=8.5, fontweight="bold")
            pdf.savefig(fig)
            plt.close(fig)
    return Path(path)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def export_poses(poses, path, seed=RANDOM_SEED, method="uniform", ordered=False, per_page=5):
    """Write the pose list as JSON for the infer app's eval mode to load
    (`trossen_real/infer_app/eval_config.py`), so the UI can tell the operator
    where to place the cube and every checkpoint is scored on the identical set.

    The seed and workspace geometry travel with the poses: a results file is
    only comparable to another if these match.
    """
    cmd = (f"generate_eval_poses.py --method {method} --seed {seed} "
           f"--n-poses {len(poses)}" + (f" --order-for-printing --mat-per-page {per_page}" if ordered else ""))
    payload = {
        # Everything needed to regenerate this file byte-for-byte. Nothing here
        # was hand-picked: the poses are a pure function of these parameters.
        "generator": {
            "method": method,
            "seed": seed,
            "n_poses": len(poses),
            "ordered_for_printing": ordered,
            "mat_per_page": per_page if ordered else None,
            "regenerate_with": cmd,
        },
        "seed": seed,
        "workspace_cm": [WORKSPACE_WIDTH, WORKSPACE_HEIGHT],
        "cube_cm": CUBE_SIZE,
        "yaw_choices_deg": list(DISCRETE_YAW_DEG),
        "max_overhang_cm": MAX_OVERHANG,
        "poses": [
            {"index": i, "x_cm": round(x, 4), "y_cm": round(y, 4), "yaw_deg": round(yaw, 4)}
            for i, (x, y, yaw) in enumerate(poses, 1)
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


if __name__ == "__main__":
    # NOTE: running with NO arguments behaves exactly as it always has (print the
    # table, then open the matplotlib window). Every flag below is additive.
    parser = argparse.ArgumentParser(description="Generate the deterministic evaluation cube poses.")
    parser.add_argument("--out", default=None,
                        help="also write the poses to this JSON file (for the infer app's eval mode). "
                             "Implies --no-show unless --show is passed, so it works headless.")
    parser.add_argument("--mat", default=None,
                        help="also write a 1:1 printable placement mat PDF (put the cube in the "
                             "numbered outline - no measuring). Implies --no-show like --out.")
    parser.add_argument("--order-for-printing", action="store_true",
                        help="reorder and RENUMBER poses so mat page 1 = poses 1-5, page 2 = 6-10, "
                             "... with no overlaps on any page. Changes pose indices, so use it "
                             "before starting a campaign (or remap existing results).")
    parser.add_argument("--paper", choices=sorted(PAPER_CM), default="a4",
                        help="mat paper size, PORTRAIT (default a4). A real paper size, so "
                             "printing at 100%% needs no scaling and no landscape setting.")
    parser.add_argument("--no-rotate", action="store_true",
                        help="do not rotate the workspace to fit portrait (will fail if it "
                             "does not fit at 1:1)")
    parser.add_argument("--mat-per-page", type=int, default=5,
                        help="poses per mat page (default 5; all 20 on one page overlap badly)")
    parser.add_argument("--show", action="store_true", help="force the plot window even with --out")
    parser.add_argument("--no-show", action="store_true", help="never open the plot window")
    parser.add_argument("--n-poses", type=int, default=N_POSES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--method", choices=sorted(GENERATORS), default="uniform",
                        help="uniform (default, original behaviour: i.i.d. random) or "
                             "stratified (jittered grid - even coverage and balanced yaw). "
                             "Changing this changes the poses, and therefore the eval config "
                             "hash: results are NOT comparable across methods.")
    args = parser.parse_args()

    eval_poses = GENERATORS[args.method](n_poses=args.n_poses, seed=args.seed)
    if args.order_for_printing:
        eval_poses = order_poses_for_printing(eval_poses, per_page=args.mat_per_page)
        print(f"(reordered + renumbered for printing: pages of {args.mat_per_page}, in index order)")

    print(f"Generated {args.n_poses} Evaluation Poses (Workspace: {WORKSPACE_WIDTH}x{WORKSPACE_HEIGHT} cm):")
    print("-" * 55)
    for idx, (x, y, yaw) in enumerate(eval_poses, 1):
        print(f"Pose #{idx:02d} | Center: (X={x:6.2f}, Y={y:5.2f}) cm | Yaw: {yaw:+5.1f}°")
    print("-" * 55)

    if args.out:
        print(f"Wrote {export_poses(eval_poses, args.out, seed=args.seed, method=args.method, ordered=args.order_for_printing, per_page=args.mat_per_page)}")

    if args.mat:
        print(f"Wrote {make_placement_mat(eval_poses, args.mat, per_page=args.mat_per_page, seed=args.seed, method=args.method, regroup=not args.order_for_printing, paper=args.paper, rotate=False if args.no_rotate else 'auto')}")
        print("  -> print at 100% scale and check the 100 mm bar before using it.")

    show = args.show or (not args.no_show and not args.out and not args.mat)
    if show:
        visualize_poses(eval_poses)