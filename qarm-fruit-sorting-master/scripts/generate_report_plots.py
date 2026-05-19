"""Generate the four report figures from a single run's log artefacts.

Inputs
------
  logs/robot_trace_*.log   (latest)
  logs/autonomous_run.mat  (optional; produced by the Simulink run)

Outputs
-------
  figures/plot_joints.png
  figures/plot_gripper_readback.png
  figures/plot_fsm_timeline.png
  figures/plot_detections.png
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
LOG_DIR = os.path.join(REPO, "logs")
FIG_DIR = os.path.join(REPO, "figures")


def latest_trace_log():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "robot_trace_*.log")))
    if not files:
        return None
    return files[-1]


def parse_events(path):
    events = []
    rx = re.compile(r"t=([0-9.]+)\s+TAG=(\w+)\s*(.*)")
    kv_rx = re.compile(r"(\w+)=(\S+)")
    with open(path) as f:
        for line in f:
            m = rx.match(line.strip())
            if not m:
                continue
            t = float(m.group(1))
            tag = m.group(2)
            kv = {}
            for km in kv_rx.finditer(m.group(3)):
                kv[km.group(1)] = km.group(2)
            events.append((t, tag, kv))
    return events


def plot_gripper_readback(events, out):
    tgts, acts, stalls = [], [], []
    for t, tag, kv in events:
        if tag != "GRIPPER_READBACK":
            continue
        tgts.append(float(kv.get("target", "0")))
        acts.append(float(kv.get("actual", "0")))
        stalls.append(float(kv.get("stall", "0")))
    if not tgts:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.arange(len(tgts))
    ax.plot(idx, tgts, "o-", label="commanded")
    ax.plot(idx, acts, "s-", label="actual")
    ax.bar(idx, stalls, alpha=0.3, label="stall delta", color="red")
    ax.set_xlabel("pick #")
    ax.set_ylabel("gripper command")
    ax.set_title("Gripper commanded vs settled (readback)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return True


def plot_fsm_timeline(events, out):
    state_events = [(t, kv.get("state", "")) for t, tag, kv in events
                     if tag == "STATE_ENTER"]
    if not state_events:
        markers = [(t, tag) for t, tag, _ in events
                   if tag in ("PICK_ATTEMPT", "SORT_COMPLETE")]
        if not markers:
            return False
        fig, ax = plt.subplots(figsize=(8, 2.5))
        for t, tag in markers:
            ax.axvline(t, color="C0" if tag == "PICK_ATTEMPT" else "C1",
                        label=tag)
        ax.set_xlabel("time (s)")
        ax.set_yticks([])
        ax.set_title("FSM event markers")
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, lab in zip(handles, labels):
            seen.setdefault(lab, h)
        ax.legend(seen.values(), seen.keys())
        fig.tight_layout(); fig.savefig(out); plt.close(fig)
        return True
    states = sorted(set(s for _, s in state_events))
    y = {s: i for i, s in enumerate(states)}
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(states))))
    for (t0, s0), (t1, _) in zip(state_events[:-1], state_events[1:]):
        ax.hlines(y[s0], t0, t1, lw=6)
    ax.set_yticks(list(y.values()))
    ax.set_yticklabels(list(y.keys()))
    ax.set_xlabel("time (s)")
    ax.set_title("FSM state timeline")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return True


def plot_detections(events, out):
    rows = []
    for t, tag, kv in events:
        if tag != "DETECTION_FRAME":
            continue
        rows.append((t, int(kv.get("n", "0")),
                     float(kv.get("mean_conf", "0"))))
    if not rows:
        return False
    t = np.array([r[0] for r in rows])
    n = np.array([r[1] for r in rows])
    c = np.array([r[2] for r in rows])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t, n, "o-"); ax1.set_ylabel("detections")
    ax1.set_title("Detection quality")
    ax2.plot(t, c, "s-", color="C1"); ax2.set_ylabel("mean conf")
    ax2.set_xlabel("time (s)")
    for a in (ax1, ax2): a.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return True


def plot_joints_from_mat(out):
    mat_path = os.path.join(LOG_DIR, "autonomous_run.mat")
    if not os.path.exists(mat_path):
        return False
    try:
        from scipy.io import loadmat
    except ImportError:
        return False
    data = loadmat(mat_path)
    if "joint_cmd" not in data or "joint_actual" not in data:
        return False
    t = data.get("time", np.arange(data["joint_cmd"].shape[0])).flatten()
    cmd = data["joint_cmd"]
    act = data["joint_actual"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for j in range(4):
        axes[j].plot(t, np.rad2deg(cmd[:, j]), label="cmd")
        axes[j].plot(t, np.rad2deg(act[:, j]), label="actual",
                       linestyle="--")
        axes[j].set_ylabel(f"joint {j+1} (deg)")
        axes[j].grid(True, alpha=0.3); axes[j].legend()
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Joint trajectories over autonomous run")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return True


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    log = latest_trace_log()
    if log is None:
        print("no robot_trace_*.log — run the autonomous pipeline first")
        return 1
    events = parse_events(log)
    print(f"parsed {len(events)} events from {log}")

    results = {
        "gripper_readback":
            plot_gripper_readback(events, os.path.join(FIG_DIR,
                "plot_gripper_readback.png")),
        "fsm_timeline":
            plot_fsm_timeline(events, os.path.join(FIG_DIR,
                "plot_fsm_timeline.png")),
        "detections":
            plot_detections(events, os.path.join(FIG_DIR,
                "plot_detections.png")),
        "joints":
            plot_joints_from_mat(os.path.join(FIG_DIR,
                "plot_joints.png")),
    }
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'no data'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
