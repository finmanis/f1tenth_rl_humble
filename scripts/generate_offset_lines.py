#!/usr/bin/env python3
"""
Generate left/right offset race lines from a map's centerline.

Each waypoint is shifted perpendicular to the track direction by the given
offset distance, producing parallel lines the opponent pure-pursuit can follow.

Usage:
    python3 scripts/generate_offset_lines.py --map maps/levine_race/levine_race --offset 0.8
    python3 scripts/generate_offset_lines.py --map maps/spielberg/Spielberg --offset 0.6 --validate
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _seg_intersect_params(p1, p2, p3, p4):
    """Return (t, u) for p1+t*(p2-p1) ∩ p3+u*(p4-p3), or (None, None) if parallel."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-10:
        return None, None
    diff = p3 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / denom
    return t, u


def _remove_loops(xy: np.ndarray) -> np.ndarray:
    """
    Remove self-intersections (loops) from a closed offset polyline.

    At inside corners the offset curve folds back on itself.  We find each
    segment crossing, replace the loop with the intersection point, and repeat
    until no crossings remain.
    """
    pts = [np.array(p, dtype=float) for p in xy]
    for _ in range(len(pts)):          # at most N passes
        n = len(pts)
        found = False
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:  # don't test segments that share endpoint at wrap
                    continue
                p3, p4 = pts[j], pts[(j + 1) % n]
                t, u = _seg_intersect_params(p1, p2, p3, p4)
                if t is not None and 0.0 < t < 1.0 and 0.0 < u < 1.0:
                    cross_pt = p1 + t * (p2 - p1)
                    pts = pts[: i + 1] + [cross_pt] + pts[j + 1 :]
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return np.array(pts)


def compute_offset_line(waypoints: np.ndarray, offset: float, side: str) -> np.ndarray:
    """
    Shift each waypoint perpendicular to the track by `offset` metres,
    then remove any loops that form on the inside of tight corners.

    Parameters
    ----------
    waypoints : (N, 3) array  [x, y, vx]
    offset    : perpendicular distance in metres
    side      : "left" or "right" (relative to forward travel direction)

    Returns
    -------
    (M, 3) array with shifted x, y and nearest-centeline vx  (M ≤ N after loop removal)
    """
    xy = waypoints[:, :2]
    n = len(xy)
    normals = np.zeros_like(xy)

    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        dx = xy[next_i, 0] - xy[prev_i, 0]
        dy = xy[next_i, 1] - xy[prev_i, 1]
        length = np.hypot(dx, dy)
        if length < 1e-9:
            normals[i] = normals[i - 1]
            continue
        tx, ty = dx / length, dy / length
        # Left normal: 90° CCW from tangent
        normals[i] = [-ty, tx]

    if side == "right":
        normals = -normals

    shifted_xy = _remove_loops(xy + offset * normals)

    # Re-interpolate velocity from nearest centerline point (point count may have changed)
    vx_orig = waypoints[:, 2] if waypoints.shape[1] >= 3 else np.ones(n)
    vx = np.array([
        vx_orig[np.argmin(np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1]))]
        for p in shifted_xy
    ])
    return np.column_stack([shifted_xy, vx])


def validate_against_map(offset_line: np.ndarray, map_path: str, min_wall_dist: float = 0.2) -> int:
    """
    Check each offset waypoint is in free space on the occupancy map.
    Returns number of waypoints that are too close to walls (warns, does not clamp).
    """
    try:
        import yaml
        from PIL import Image
    except ImportError:
        print("  [validate] PIL not available, skipping wall check.")
        return 0

    yaml_path = map_path + ".yaml"
    if not os.path.exists(yaml_path):
        print(f"  [validate] No map YAML found at {yaml_path}, skipping.")
        return 0

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = meta["resolution"]
    origin = meta["origin"]
    negate = meta.get("negate", 0)
    occupied_thresh = meta.get("occupied_thresh", 0.65)

    img_name = meta.get("image", os.path.basename(map_path) + ".png")
    img_path = os.path.join(os.path.dirname(map_path), img_name)
    if not os.path.exists(img_path):
        print(f"  [validate] Map image not found at {img_path}, skipping.")
        return 0

    img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
    # Convert to occupancy: with negate=0, dark pixels are walls (occ = 1 - brightness).
    occ = (1.0 - img) if not negate else img
    h, w = occ.shape

    n_bad = 0
    min_px = max(1, int(min_wall_dist / resolution))

    for pt in offset_line[:, :2]:
        # World -> pixel (row 0 = top = max y in world)
        px = int((pt[0] - origin[0]) / resolution)
        py = h - int((pt[1] - origin[1]) / resolution) - 1
        x0, x1 = max(0, px - min_px), min(w, px + min_px + 1)
        y0, y1 = max(0, py - min_px), min(h, py + min_px + 1)
        patch = occ[y0:y1, x0:x1]
        if patch.size == 0 or np.any(patch > occupied_thresh):
            n_bad += 1

    return n_bad


def plot_offset_lines_check(map_path, centerline, left_line, right_line, output_path, offset):
    """
    Save a PNG showing left / center / right lines overlaid on the occupancy map.
    Mirrors the style of generate_centerline.py's visualize().
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    from PIL import Image

    # Load map image and metadata
    img = None
    resolution, origin = 0.05, [0.0, 0.0, 0.0]
    for yaml_candidate in (map_path + "_map.yaml", map_path + ".yaml"):
        if os.path.exists(yaml_candidate):
            with open(yaml_candidate) as f:
                meta = yaml.safe_load(f)
            resolution = meta.get("resolution", 0.05)
            origin = meta.get("origin", [0.0, 0.0, 0.0])
            img_name = meta.get("image", os.path.basename(map_path) + ".png")
            img_path = os.path.join(os.path.dirname(map_path), img_name)
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path).convert("L"))
            break

    def to_px(line):
        """Convert world-coord waypoints to pixel coords."""
        px = (line[:, 0] - origin[0]) / resolution
        py = (img.shape[0] if img is not None else 0) - (line[:, 1] - origin[1]) / resolution
        return px, py

    fig, ax = plt.subplots(figsize=(10, 10))

    if img is not None:
        ax.imshow(img, cmap="gray", origin="upper")
        ax.set_title(f"Opponent Lines — offset ±{offset} m", fontsize=14)
    else:
        ax.set_title(f"Opponent Lines — offset ±{offset} m (no map image)", fontsize=14)
        ax.invert_yaxis()

    line_styles = [
        (left_line,   "#2196F3", "Left line"),
        (centerline,  "#F44336", "Centerline"),
        (right_line,  "#4CAF50", "Right line"),
    ]

    all_px, all_py = [], []
    for line, color, label in line_styles:
        px, py = to_px(line)
        ax.plot(px, py, "-", color=color, linewidth=2, alpha=0.85, label=label)
        ax.plot(px[0], py[0], "o", color=color, markersize=8, zorder=5)
        all_px.extend(px)
        all_py.extend(py)

    if all_px:
        margin = 60
        ax.set_xlim(min(all_px) - margin, max(all_px) + margin)
        ax.set_ylim(max(all_py) + margin, min(all_py) - margin)

    ax.legend(fontsize=11, loc="best")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved check plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate left/right offset lines from centerline")
    parser.add_argument("--map", required=True,
                        help="Map path prefix, e.g. maps/levine_race/levine_race")
    parser.add_argument("--offset", type=float, default=0.4,
                        help="Perpendicular offset in metres (default: 0.4)")
    parser.add_argument("--validate", action="store_true",
                        help="Check offset lines against occupancy map for wall proximity")
    args = parser.parse_args()

    map_path = str(project_root / args.map) if not os.path.isabs(args.map) else args.map
    centerline_path = map_path + "_centerline.csv"

    if not os.path.exists(centerline_path):
        print(f"ERROR: Centerline not found: {centerline_path}")
        print("  Run scripts/generate_centerline.py first.")
        sys.exit(1)

    waypoints = np.loadtxt(centerline_path, delimiter=",", skiprows=1)
    if waypoints.ndim == 1:
        waypoints = waypoints.reshape(1, -1)
    if waypoints.shape[1] < 3:
        waypoints = np.column_stack([waypoints, np.ones(len(waypoints))])

    print(f"Loaded centerline: {centerline_path} ({len(waypoints)} waypoints)")
    print(f"Offset: ±{args.offset} m")

    left_line = compute_offset_line(waypoints, args.offset, "left")
    right_line = compute_offset_line(waypoints, args.offset, "right")

    for side, line in (("left", left_line), ("right", right_line)):
        out_path = map_path + f"_{side}_line.csv"

        if args.validate:
            n_bad = validate_against_map(line, map_path)
            if n_bad > 0:
                print(f"  [WARNING] {side} line: {n_bad}/{len(line)} waypoints near walls — "
                      f"consider reducing --offset")
            else:
                print(f"  [validate] {side} line: all waypoints clear of walls")

        header = "x_m,y_m,vx_mps"
        np.savetxt(out_path, line, delimiter=",", header=header, comments="")
        print(f"  Saved {side} line: {out_path}")

    # Always generate the check plot
    check_path = map_path + "_offset_lines_check.png"
    plot_offset_lines_check(map_path, waypoints, left_line, right_line, check_path, args.offset)

    print("Done. Run training — wrapper will auto-detect the new lines.")
    print(f"  >>> CHECK: {check_path} <<<")


if __name__ == "__main__":
    main()
