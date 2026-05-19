"""Test the 4-orientation corner-ordering sweep in run_calibration_core.

We can't easily exercise run_calibration_core end-to-end without
hardware, but we can exercise _grid_with_ordering + a manually mis-
ordered image_pts to prove the sweep recovers the right orientation.
"""
from __future__ import annotations
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from calibrate_chessboard import _grid_with_ordering, _INNER_COLS, _INNER_ROWS, _SQUARE_MM
from calibrate_extrinsics import solve_survey1_extrinsics


def _project(pts3d_base, R_cam_in_base, C_cam_in_base, K):
    R_base_to_cam = R_cam_in_base.T
    pts_cam = (R_base_to_cam @ (pts3d_base.T - C_cam_in_base.reshape(3, 1))).T
    u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)


def test_grid_ordering_shapes():
    """All four orderings must produce a (35, 3) float32 grid with the
    same set of points, just in different orderings."""
    origin = np.array([0.50, 0.10, 0.02])
    grids = {
        (a, b): _grid_with_ordering(origin, a, b)
        for a in (False, True) for b in (False, True)
    }
    shape = (_INNER_ROWS * _INNER_COLS, 3)
    for g in grids.values():
        assert g.shape == shape
        assert g.dtype == np.float32
    # Same set of XY positions across all orderings.
    base = {tuple(p.tolist()) for p in grids[(False, False)]}
    for g in grids.values():
        assert {tuple(p.tolist()) for p in g} == base


def test_sweep_recovers_above_plane_solution_under_mirror_ordering():
    """Simulate the lab case: OpenCV returned image_pts in 'flip-cols'
    order (physical top-right is index 0). The original-ordering
    solvePnP call produces a below-plane solution; only the sweep
    discovers the correct 'flip-cols' ordering and returns an
    above-plane camera.

    Key logic:
      - We project _grid_with_ordering(flip_i=True, flip_j=False) through
        the true camera to get image_pts in 'flip-cols' order.
      - The sweep orientation "flip-cols (i reversed)" generates the same
        grid → same correspondence → correctly recovers the true pose.
    """
    K = np.array([
        [912.0,   0.0, 635.0],
        [  0.0, 911.0, 343.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    origin = np.array([0.50, 0.10, 0.02])
    R_cam_in_base = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(np.pi + np.deg2rad(30)), -np.sin(np.pi + np.deg2rad(30))],
        [0.0, np.sin(np.pi + np.deg2rad(30)),  np.cos(np.pi + np.deg2rad(30))],
    ])
    C_cam_in_base = np.array([0.60, 0.30, 0.30])

    # Build image_pts by projecting the 'flip-cols' grid through the true pose.
    # This simulates OpenCV returning corners in flip-cols order.
    flipped_grid = _grid_with_ordering(origin, flip_i=True, flip_j=False)
    image_pts = _project(flipped_grid, R_cam_in_base, C_cam_in_base, K)

    # Run the same 4-orientation sweep as run_calibration_core:
    orientations = [
        ("original",                False, False),
        ("flip-cols (i reversed)",  True,  False),
        ("flip-rows (j reversed)",  False, True),
        ("180° rotation",           True,  True),
    ]
    best = None
    for name, fi, fj in orientations:
        grid = _grid_with_ordering(origin, flip_i=fi, flip_j=fj)
        try:
            R_c, C_c, rms_c = solve_survey1_extrinsics(
                corners_2d=image_pts,
                corners_3d_base=grid,
                K=K, dist_coeffs=dist,
                chess_origin_z_in_base=float(origin[2]),
            )
        except RuntimeError:
            continue
        if best is None or rms_c < best[2]:
            best = (R_c, C_c, rms_c, name)

    assert best is not None, "no orientation succeeded"
    R_rec, C_rec, rms, chosen = best

    # The correct match: image_pts was projected from the flipped grid, so
    # the sweep orientation that regenerates that same flipped grid wins.
    assert chosen == "flip-cols (i reversed)", (
        f"expected 'flip-cols (i reversed)' to win; got '{chosen}' "
        f"(rms={rms:.4f})")

    # Camera must be above chessboard plane.
    assert C_rec[2] > origin[2], f"chosen C is below plane: {C_rec}"

    # Recovered position must be close to the true camera centre.
    np.testing.assert_allclose(C_rec, C_cam_in_base, atol=2e-3)
