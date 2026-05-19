"""
Live preview of the RealSense D415 (color + depth side-by-side).

Keys in the OpenCV window:
  q or ESC : quit
  s        : save current frame to figures/preview_color.png + preview_depth.png
  d        : toggle depth view on/off
"""
import os
import time
import numpy as np
import cv2

from camera import QArmCamera

FIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "figures")
)


def colorize_depth(depth_u16):
    d = depth_u16.astype(np.float32)
    if d.max() <= 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    d = np.clip(d / 3000.0, 0, 1)  # 0..3 m range
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


def main():
    cam = QArmCamera()
    cam.open()
    show_depth = True
    t_last, frames = time.time(), 0
    fps = 0.0
    try:
        while True:
            try:
                color, depth = cam.read()
            except Exception:
                continue

            frames += 1
            if frames % 15 == 0:
                now = time.time()
                fps = 15.0 / (now - t_last)
                t_last = now

            disp = color.copy()
            if show_depth:
                depth_vis = colorize_depth(depth)
                disp = np.hstack([disp, depth_vis])

            cv2.putText(disp, f"{fps:.1f} fps  (s: save  d: toggle depth  q: quit)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)
            cv2.imshow("QArm camera preview", disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            if k == ord('d'):
                show_depth = not show_depth
            if k == ord('s'):
                os.makedirs(FIG_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(FIG_DIR, "preview_color.png"), color)
                cv2.imwrite(os.path.join(FIG_DIR, "preview_depth.png"),
                            colorize_depth(depth))
                print(f"saved to {FIG_DIR}")
    finally:
        cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    main()
