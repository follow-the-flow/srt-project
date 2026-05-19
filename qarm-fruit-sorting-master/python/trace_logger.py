"""Append-only event logger shared by teach_points.py, sorting_controller.py,
and anything else that wants a timestamped trace of runtime events.

One line per event: [t=seconds since session start]  TAG=name  key=val ...

Kept simple on purpose: plain text, line-buffered, crash-safe (each write
is flushed). Parse with regex from generate_report_plots.py.
"""
import os
import sys
import time
import numpy as np
from datetime import datetime


class TraceLogger:
    IMPORTANT_TAGS = {
        "SESSION_START", "SESSION_END", "HIL_ERROR", "IK_FAIL",
        "JOINT_LIMIT", "EXCEPT", "ROUTINE_START", "ROUTINE_DONE",
        "GOTO", "GRIPPER_CMD", "PICK_ATTEMPT", "GRIPPER_READBACK",
        "SORT_COMPLETE", "DETECTION_FRAME", "PRE_FLIGHT_SKIP",
    }

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.f = open(path, "a", buffering=1)
        self.t0 = time.time()
        self.log("SESSION_START",
                  ts=datetime.now().isoformat(timespec="seconds"))

    @staticmethod
    def _fmt_val(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=float)
            return "[" + ",".join(f"{x:.4f}" for x in arr.flatten()) + "]"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def log(self, tag, **kv):
        t = time.time() - self.t0
        parts = [f"t={t:.3f}", f"TAG={tag}"]
        for k, v in kv.items():
            parts.append(f"{k}={self._fmt_val(v)}")
        line = "  ".join(parts)
        try:
            self.f.write(line + "\n")
        except Exception:
            pass
        if tag in self.IMPORTANT_TAGS:
            sys.stderr.write("[trace] " + line + "\n")
            sys.stderr.flush()

    def close(self):
        try:
            self.log("SESSION_END")
            self.f.close()
        except Exception:
            pass
