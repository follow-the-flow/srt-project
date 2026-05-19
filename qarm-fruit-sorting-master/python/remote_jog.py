"""Helpers shared by the remote-mode HMI (Simulink-side MATLAB Function
blocks call into these via py.remote_jog.*). No Simulink dependency here
so the logic stays unit-testable in plain Python.

Workspace box and joint limits match the spec (Section 2.1 safety).
"""
import numpy as np
from qarm_driver import QArmDriver

WORKSPACE_BOX = {
    'x': (-0.10, 0.55),
    'y': (-0.55, 0.55),
    'z': (0.02,  0.50),
}


def _inside_box(xyz):
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return (WORKSPACE_BOX['x'][0] <= x <= WORKSPACE_BOX['x'][1] and
            WORKSPACE_BOX['y'][0] <= y <= WORKSPACE_BOX['y'][1] and
            WORKSPACE_BOX['z'][0] <= z <= WORKSPACE_BOX['z'][1])


def cartesian_nudge(current_xyz, axis, step):
    """Return a new xyz (3-vector) with axis shifted by step if the result
    is inside the workspace box; otherwise None (caller should log a
    warning and leave the arm at current_xyz)."""
    i = {'x': 0, 'y': 1, 'z': 2}[axis]
    out = np.array(current_xyz, dtype=float).copy()
    out[i] += step
    if not _inside_box(out):
        return None
    return out


def joint_nudge(current_joints, joint, step):
    """Return a new (4,) joint vector with joint shifted by step if the
    result is inside the QArm joint limits; otherwise None."""
    out = np.array(current_joints, dtype=float).copy()
    out[joint] += step
    lo, hi = QArmDriver.JOINT_LIMITS[joint]
    if out[joint] < lo or out[joint] > hi:
        return None
    return out
