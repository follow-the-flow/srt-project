"""
QArm Hardware Driver using the Quanser SDK for Windows.
Provides a high-level Python interface to the QArm robot.

Board type: "qarm_usb"
Channel mapping (Other I/O):
  Read  1000-1003: Joint positions [yaw, shoulder, elbow, wrist] (rad)
  Read  1004:      Gripper position
  Read  3000-3004: Angular velocities
  Read  10000-10004: Motor temperatures
  Write 1000-1003: Joint position commands (rad)
  Write 1004:      Gripper command (0.0=open, 1.0=closed)
  Write 11005-11008: LED brightness (R, G, B, W)

Card-specific options:
  j0_mode=0 (position mode), j0_mode=1 (PWM mode)
"""

import numpy as np
import time
from quanser.hardware import HIL, MAX_STRING_LENGTH


class QArmDriver:
    """High-level driver for the Quanser QArm robot."""

    # Channel definitions
    JOINT_READ_CHANNELS = np.array([1000, 1001, 1002, 1003], dtype=np.uint32)
    GRIPPER_READ_CHANNEL = np.array([1004], dtype=np.uint32)
    VELOCITY_READ_CHANNELS = np.array([3000, 3001, 3002, 3003], dtype=np.uint32)
    TEMP_READ_CHANNELS = np.array([10000, 10001, 10002, 10003, 10004], dtype=np.uint32)

    JOINT_WRITE_CHANNELS = np.array([1000, 1001, 1002, 1003], dtype=np.uint32)
    GRIPPER_WRITE_CHANNEL = np.array([1004], dtype=np.uint32)
    LED_WRITE_CHANNELS = np.array([11005, 11006, 11007, 11008], dtype=np.uint32)

    ALL_READ_CHANNELS = np.array([1000, 1001, 1002, 1003, 1004], dtype=np.uint32)
    ALL_WRITE_CHANNELS = np.array([1000, 1001, 1002, 1003, 1004], dtype=np.uint32)

    # Joint limits (radians)
    JOINT_LIMITS = np.array([
        [-2.967, 2.967],   # Yaw: +/-170 deg
        [-1.484, 1.484],   # Shoulder: +/-85 deg
        [-1.658, 1.309],   # Elbow: -95/+75 deg
        [-2.793, 2.793],   # Wrist: +/-160 deg
    ])

    def __init__(self, board_id="0", position_mode=True):
        """
        Initialize the QArm driver.

        Parameters
        ----------
        board_id : str
            Board identifier (default "0" for first connected QArm)
        position_mode : bool
            If True, use position control mode. If False, use PWM mode.
        """
        self.card = HIL()
        self.board_id = board_id
        self.position_mode = position_mode
        self._connected = False

        # State buffers
        self._joint_positions = np.zeros(4, dtype=np.float64)
        self._gripper_position = np.zeros(1, dtype=np.float64)
        self._joint_velocities = np.zeros(4, dtype=np.float64)
        self._temperatures = np.zeros(5, dtype=np.float64)

    def connect(self):
        """Open connection to the QArm hardware."""
        try:
            self.card.open("qarm_usb", self.board_id)

            # Set position mode for all joints
            if self.position_mode:
                options = ("j0_mode=0,j1_mode=0,j2_mode=0,j3_mode=0,"
                           "gripper_mode=0")
            else:
                options = ("j0_mode=1,j1_mode=1,j2_mode=1,j3_mode=1,"
                           "gripper_mode=1")

            self.card.set_card_specific_options(options, MAX_STRING_LENGTH)
            self._connected = True
            print(f"QArm connected (board {self.board_id}, "
                  f"{'position' if self.position_mode else 'PWM'} mode)")
        except Exception as e:
            self._connected = False
            raise RuntimeError(f"Failed to connect to QArm: {e}")

    def disconnect(self):
        """Safely disconnect from the QArm."""
        if self._connected:
            try:
                # Move to home position first
                self.set_joint_positions(np.zeros(4))
                self.set_gripper(0.0)
                time.sleep(1.0)
            except Exception:
                pass
            finally:
                self.card.close()
                self._connected = False
                print("QArm disconnected")

    def read_joint_positions(self):
        """
        Read current joint positions.

        Returns
        -------
        np.ndarray
            4-element array of joint angles [yaw, shoulder, elbow, wrist] in radians
        """
        self.card.read_other(
            self.JOINT_READ_CHANNELS, 4, self._joint_positions
        )
        return self._joint_positions.copy()

    def read_gripper_position(self):
        """Read current gripper position (0.0=open, 1.0=closed)."""
        self.card.read_other(
            self.GRIPPER_READ_CHANNEL, 1, self._gripper_position
        )
        return float(self._gripper_position[0])

    def read_joint_velocities(self):
        """Read current joint angular velocities (rad/s)."""
        self.card.read_other(
            self.VELOCITY_READ_CHANNELS, 4, self._joint_velocities
        )
        return self._joint_velocities.copy()

    def read_temperatures(self):
        """Read motor temperatures for all 5 motors."""
        self.card.read_other(
            self.TEMP_READ_CHANNELS, 5, self._temperatures
        )
        return self._temperatures.copy()

    def read_all(self):
        """
        Read all joint positions and gripper in a single call.

        Returns
        -------
        tuple
            (joint_positions [4], gripper_position [scalar])
        """
        buf = np.zeros(5, dtype=np.float64)
        self.card.read_other(self.ALL_READ_CHANNELS, 5, buf)
        self._joint_positions[:] = buf[:4]
        self._gripper_position[0] = buf[4]
        return buf[:4].copy(), float(buf[4])

    def set_joint_positions(self, positions):
        """
        Command joint positions.

        Parameters
        ----------
        positions : array_like
            4-element array of target joint angles in radians.
            Will be clamped to joint limits.
        """
        pos = np.array(positions, dtype=np.float64).flatten()[:4]
        # Clamp to limits
        for i in range(4):
            pos[i] = np.clip(pos[i], self.JOINT_LIMITS[i, 0], self.JOINT_LIMITS[i, 1])
        self.card.write_other(self.JOINT_WRITE_CHANNELS, 4, pos)

    def set_gripper(self, value):
        """
        Command gripper position.

        Parameters
        ----------
        value : float
            0.0 = fully open, 1.0 = fully closed
        """
        val = np.array([np.clip(value, 0.0, 1.0)], dtype=np.float64)
        self.card.write_other(self.GRIPPER_WRITE_CHANNEL, 1, val)

    def set_joints_and_gripper(self, positions, gripper):
        """
        Command joints and gripper in a single write.

        Parameters
        ----------
        positions : array_like
            4-element array of joint angles (rad)
        gripper : float
            Gripper command (0=open, 1=closed)
        """
        pos = np.array(positions, dtype=np.float64).flatten()[:4]
        for i in range(4):
            pos[i] = np.clip(pos[i], self.JOINT_LIMITS[i, 0], self.JOINT_LIMITS[i, 1])
        buf = np.array([pos[0], pos[1], pos[2], pos[3], np.clip(gripper, 0, 1)],
                        dtype=np.float64)
        self.card.write_other(self.ALL_WRITE_CHANNELS, 5, buf)

    def set_led(self, r=0.0, g=0.0, b=0.0, w=0.0):
        """Set LED brightness (0.0 to 1.0 for each channel)."""
        buf = np.array([r, g, b, w], dtype=np.float64)
        self.card.write_other(self.LED_WRITE_CHANNELS, 4, buf)

    def home(self, duration=3.0, steps=100):
        """
        Smoothly move to home position [0, 0, 0, 0] with gripper open.

        Parameters
        ----------
        duration : float
            Time to reach home position (seconds)
        steps : int
            Number of interpolation steps
        """
        current, cur_grip = self.read_all()
        target = np.zeros(4)
        # Keep the gripper where it physically is — commanding 0.0 (fully
        # open) can drive the servo past its physical endstop and trigger
        # QERR_QARM_OVERLOAD_GRIPPER. We use the current measured grip.
        safe_grip = float(max(cur_grip, 0.10))
        dt = duration / steps

        for i in range(steps + 1):
            t = i / steps
            # Smooth interpolation (cubic)
            s = 3 * t**2 - 2 * t**3
            pos = current + s * (target - current)
            self.set_joints_and_gripper(pos, safe_grip)
            time.sleep(dt)

        # Hold final target so the joint PID can settle out of steady-state error.
        for _ in range(int(2.0 / 0.05)):
            self.set_joints_and_gripper(target, safe_grip)
            time.sleep(0.05)

        print("QArm at home position")

    def is_connected(self):
        """Check if the QArm is connected."""
        return self._connected

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
