"""
Intel RealSense D415 camera interface via Quanser SDK Video3D.
Provides RGB and Depth image capture for fruit detection.
"""

import numpy as np
from quanser.multimedia import (Video3D, Video3DStreamType,
                                 ImageFormat, ImageDataType,
                                 Video3DProperty)


# D415 approximate intrinsics for common modes (kept in one place)
_D415_INTRINSICS = {
    (640, 480):   {'fx': 615.0, 'fy': 615.0, 'cx': 320.0, 'cy': 240.0},
    (1280, 720):  {'fx': 920.0, 'fy': 920.0, 'cx': 640.0, 'cy': 360.0},
    (1920, 1080): {'fx': 1380.0, 'fy': 1380.0, 'cx': 960.0, 'cy': 540.0},
}


class QArmCamera:
    """Interface to the QArm's Intel RealSense D415 camera."""

    def __init__(self, device_id="0", color_res=(1280, 720),
                 depth_res=(1280, 720), fps=30.0):
        """
        Initialize camera interface.

        Parameters
        ----------
        device_id : str
            Camera device ID (usually "0")
        color_res : tuple
            (width, height) for color stream
        depth_res : tuple
            (width, height) for depth stream
        fps : float
            Frame rate in Hz
        """
        self.device_id = device_id
        self.color_w, self.color_h = color_res
        self.depth_w, self.depth_h = depth_res
        self.fps = fps

        self.video3d = None
        self.color_stream = None
        self.depth_stream = None
        self._streaming = False

        # Buffers
        self._color_buf = np.zeros(
            (self.color_h, self.color_w, 3), dtype=np.uint8
        )
        self._depth_buf = np.zeros(
            (self.depth_h, self.depth_w), dtype=np.uint16
        )

        # Default intrinsics for D415 at this resolution
        self.intrinsics = _D415_INTRINSICS.get(
            color_res, _D415_INTRINSICS[(1280, 720)]
        ).copy()

    def open(self):
        """Open camera and start streaming."""
        self.video3d = Video3D()
        self.video3d.open(self.device_id)

        # Open color stream (BGR for OpenCV compatibility)
        self.color_stream = self.video3d.stream_open(
            Video3DStreamType.COLOR, 0,
            self.fps, self.color_w, self.color_h,
            ImageFormat.ROW_MAJOR_INTERLEAVED_BGR,
            ImageDataType.UINT8
        )

        # Open depth stream
        self.depth_stream = self.video3d.stream_open(
            Video3DStreamType.DEPTH, 0,
            self.fps, self.depth_w, self.depth_h,
            ImageFormat.ROW_MAJOR_GREYSCALE,
            ImageDataType.UINT16
        )

        self.video3d.start_streaming()
        self._streaming = True

        # Enable auto-exposure + auto white balance on the color stream.
        # Enable the IR emitter on the depth stream.
        try:
            self._set_stream_props(self.color_stream, {
                Video3DProperty.ENABLE_AUTO_EXPOSURE: 1.0,
                Video3DProperty.ENABLE_AUTO_WHITE_BALANCE: 1.0,
            })
        except Exception as e:
            print(f"  [warn] color auto-exposure setup failed: {e}")
        try:
            self._set_stream_props(self.depth_stream, {
                Video3DProperty.ENABLE_EMITTER: 1.0,
                Video3DProperty.ENABLE_AUTO_EXPOSURE: 1.0,
            })
        except Exception as e:
            print(f"  [warn] depth emitter/ae setup failed: {e}")

        # Read real intrinsics from the sensor (replaces hardcoded defaults)
        self._read_real_intrinsics()

        print(f"Camera opened: color {self.color_w}x{self.color_h}, "
              f"depth {self.depth_w}x{self.depth_h} @ {self.fps}fps")
        print(f"  intrinsics: fx={self.intrinsics['fx']:.1f} "
              f"fy={self.intrinsics['fy']:.1f} "
              f"cx={self.intrinsics['cx']:.1f} "
              f"cy={self.intrinsics['cy']:.1f}")

    def _read_real_intrinsics(self):
        """Read real camera intrinsics from the D415 sensor via the SDK.

        The SDK returns K in transposed layout:
            K_c[0][0]=fx  K_c[1][1]=fy  K_c[2][0]=cx  K_c[2][1]=cy
        """
        try:
            from quanser.multimedia.video import media_lib, ffi
            K_c = ffi.new('float[3][3]')
            coeffs_c = ffi.new('float[5]')
            model_c = ffi.new('t_video3d_distortion_model *')

            result = media_lib.video3d_stream_get_camera_intrinsics(
                self.color_stream._stream, K_c, model_c, coeffs_c)
            if result < 0:
                print(f"  [warn] get_camera_intrinsics returned {result} — using defaults")
                return

            fx = float(K_c[0][0])
            fy = float(K_c[1][1])
            cx = float(K_c[2][0])  # transposed layout
            cy = float(K_c[2][1])

            if fx > 0 and fy > 0:
                self.intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
                self.distortion_coeffs = np.array(
                    [float(coeffs_c[i]) for i in range(5)], dtype=np.float32)
                print(f"  [ok] read real intrinsics from sensor")
                return
            print(f"  [warn] sensor returned zero intrinsics — using defaults")
        except Exception as e:
            print(f"  [warn] could not read intrinsics: {e} — using defaults")

    @staticmethod
    def _set_stream_props(stream, prop_value_map):
        props = np.array(list(prop_value_map.keys()), dtype=np.int32)
        vals = np.array(list(prop_value_map.values()), dtype=np.float64)
        stream.set_properties(props, len(props), vals)

    def read(self):
        """
        Read one frame (color + depth).

        Returns
        -------
        color : np.ndarray
            HxWx3 BGR image (uint8)
        depth : np.ndarray
            HxW depth map (uint16, in mm)
        """
        if not self._streaming:
            raise RuntimeError("Camera not streaming. Call open() first.")

        color_frame = self.color_stream.get_frame()
        if color_frame is not None:
            color_frame.get_data(self._color_buf)
            color_frame.release()

        depth_frame = self.depth_stream.get_frame()
        if depth_frame is not None:
            depth_frame.get_data(self._depth_buf)
            depth_frame.release()

        return self._color_buf.copy(), self._depth_buf.copy()

    def pixel_to_world(self, row, col, depth_mm, T_cam_to_base):
        """
        Convert pixel coordinates + depth to world (robot base) coordinates.

        Parameters
        ----------
        row, col : int
            Pixel coordinates
        depth_mm : float
            Depth in mm
        T_cam_to_base : np.ndarray
            4x4 camera-to-base transformation matrix

        Returns
        -------
        p_world : np.ndarray
            (3,) position in robot base frame (metres)
        """
        Z = depth_mm / 1000.0  # Convert to metres
        X = (col - self.intrinsics['cx']) * Z / self.intrinsics['fx']
        Y = (row - self.intrinsics['cy']) * Z / self.intrinsics['fy']

        p_cam = np.array([X, Y, Z, 1.0])
        p_base = T_cam_to_base @ p_cam
        return p_base[:3]

    def close(self):
        """Stop streaming and close camera."""
        if self._streaming:
            self.video3d.stop_streaming()
            self._streaming = False
        if self.color_stream:
            self.color_stream.close()
        if self.depth_stream:
            self.depth_stream.close()
        if self.video3d:
            self.video3d.close()
        print("Camera closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
