"""
Probe available USB webcams via cv2.VideoCapture and save one frame from
each. Used to locate the UGreen ground-level camera. The D415 is routed
via Quanser SDK, not cv2, so any cv2-accessible index should be the
UGreen (or another USB webcam).
"""
import os
import sys
import cv2

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs")
)
os.makedirs(OUT_DIR, exist_ok=True)

# Try DirectShow (Windows) and default backends for each index
backends = [("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]
found_any = False
for idx in range(6):
    for name, backend in backends:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # Drop first few frames, webcams often return junk at startup
        frame = None
        for _ in range(5):
            ok, f = cap.read()
            if ok and f is not None and f.size > 0:
                frame = f
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if frame is None:
            continue
        path = os.path.join(OUT_DIR, f"probe_cam{idx}_{name}.png")
        cv2.imwrite(path, frame)
        print(f"  idx={idx:d} backend={name:5s} "
              f"size={w}x{h}  mean={frame.mean():.1f}  -> {path}")
        found_any = True
        break  # no need to try second backend on this index

if not found_any:
    print("No cv2-accessible cameras found. If the UGreen is plugged in:")
    print("  - verify Windows Device Manager sees it as a camera")
    print("  - close any app holding it (Skype, Zoom, etc.)")
    sys.exit(1)
