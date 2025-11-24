"""
Complementary Filter for IMU Orientation Estimation

The complementary filter combines accelerometer and gyroscope data to estimate
orientation. It uses a high-pass filter on gyroscope data and a low-pass filter
on accelerometer data, then combines them with a tuning parameter α.

Mathematical Foundation:
-------------------------
For roll (φ) and pitch (θ):

1. Accelerometer-based angles (low-frequency):
   φ_acc = atan2(ay, az)
   θ_acc = atan2(-ax, sqrt(ay² + az²))

2. Gyroscope integration (high-frequency):
   φ_gyro = φ_prev + gx * dt
   θ_gyro = θ_prev + gy * dt

3. Complementary fusion:
   φ = α * φ_gyro + (1 - α) * φ_acc
   θ = α * θ_gyro + (1 - α) * θ_acc

4. Yaw (ψ) from gyroscope only (no magnetometer):
   ψ = ψ_prev + gz * dt

Where:
- α: tuning parameter (0-1), typically 0.95-0.98
- Higher α: more trust in gyroscope (better dynamic response)
- Lower α: more trust in accelerometer (better long-term stability)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Orientation:
    """Orientation representation in Euler angles (roll, pitch, yaw)."""
    roll: float  # φ (phi) - rotation around X-axis
    pitch: float  # θ (theta) - rotation around Y-axis
    yaw: float  # ψ (psi) - rotation around Z-axis

    def to_quaternion(self) -> np.ndarray:
        """Convert Euler angles to quaternion [w, x, y, z]."""
        cr = np.cos(self.roll / 2)
        sr = np.sin(self.roll / 2)
        cp = np.cos(self.pitch / 2)
        sp = np.sin(self.pitch / 2)
        cy = np.cos(self.yaw / 2)
        sy = np.sin(self.yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert Euler angles to rotation matrix (ZYX convention)."""
        R = np.zeros((3, 3))
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)

        R[0, 0] = cp * cy
        R[0, 1] = cp * sy
        R[0, 2] = -sp
        R[1, 0] = sr * sp * cy - cr * sy
        R[1, 1] = sr * sp * sy + cr * cy
        R[1, 2] = sr * cp
        R[2, 0] = cr * sp * cy + sr * sy
        R[2, 1] = cr * sp * sy - sr * cy
        R[2, 2] = cr * cp

        return R


class ComplementaryFilter:
    """
    Complementary filter for IMU orientation estimation.

    Combines accelerometer and gyroscope measurements using a tunable
    blending parameter α.

    Parameters:
    -----------
    alpha : float, default=0.96
        Filter blending parameter (0-1). Higher values trust gyroscope more.
    dt : float, default=0.01
        Sampling period in seconds.
    """

    def __init__(self, alpha: float = 0.96, dt: float = 0.01):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.alpha = alpha
        self.dt = dt
        self.orientation = Orientation(0.0, 0.0, 0.0)
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.bias_estimation_samples = 100
        self.bias_samples_collected = 0

    def reset(self) -> None:
        """Reset filter state."""
        self.orientation = Orientation(0.0, 0.0, 0.0)
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.bias_samples_collected = 0

    def estimate_bias(self, gyro: np.ndarray) -> None:
        """
        Estimate gyroscope bias from initial samples.

        Parameters:
        -----------
        gyro : np.ndarray, shape=(3,)
            Gyroscope reading [gx, gy, gz] in rad/s.
        """
        if self.bias_samples_collected < self.bias_estimation_samples:
            self.gyro_bias = (
                self.gyro_bias * self.bias_samples_collected + gyro
            ) / (self.bias_samples_collected + 1)
            self.bias_samples_collected += 1

    def update(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: Optional[float] = None
    ) -> Orientation:
        """
        Update orientation estimate from IMU measurements.

        Parameters:
        -----------
        accel : np.ndarray, shape=(3,)
            Accelerometer reading [ax, ay, az] in m/s².
        gyro : np.ndarray, shape=(3,)
            Gyroscope reading [gx, gy, gz] in rad/s.
        dt : float, optional
            Time step in seconds. Uses self.dt if not provided.

        Returns:
        --------
        Orientation
            Updated orientation estimate.
        """
        if dt is None:
            dt = self.dt

        # Estimate bias during initial samples
        if self.bias_samples_collected < self.bias_estimation_samples:
            self.estimate_bias(gyro)
            return self.orientation

        # Remove bias from gyroscope
        gyro_corrected = gyro - self.gyro_bias

        # Normalize accelerometer
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 1e-6:
            # No valid accelerometer reading, use gyroscope only
            self.orientation.roll += gyro_corrected[0] * dt
            self.orientation.pitch += gyro_corrected[1] * dt
            self.orientation.yaw += gyro_corrected[2] * dt
            return self.orientation

        accel_normalized = accel / accel_norm

        # Accelerometer-based angles (low-frequency estimate)
        roll_acc = np.arctan2(accel_normalized[1], accel_normalized[2])
        pitch_acc = np.arctan2(
            -accel_normalized[0],
            np.sqrt(accel_normalized[1]**2 + accel_normalized[2]**2)
        )

        # Gyroscope integration (high-frequency estimate)
        roll_gyro = self.orientation.roll + gyro_corrected[0] * dt
        pitch_gyro = self.orientation.pitch + gyro_corrected[1] * dt
        yaw_gyro = self.orientation.yaw + gyro_corrected[2] * dt

        # Complementary fusion
        self.orientation.roll = (
            self.alpha * roll_gyro + (1 - self.alpha) * roll_acc
        )
        self.orientation.pitch = (
            self.alpha * pitch_gyro + (1 - self.alpha) * pitch_acc
        )
        self.orientation.yaw = yaw_gyro  # Yaw from gyro only

        return self.orientation

    def process_batch(
        self,
        accel_data: np.ndarray,
        gyro_data: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of IMU measurements.

        Parameters:
        -----------
        accel_data : np.ndarray, shape=(N, 3)
            Accelerometer readings.
        gyro_data : np.ndarray, shape=(N, 3)
            Gyroscope readings.
        timestamps : np.ndarray, shape=(N,), optional
            Timestamps in seconds. If None, uses uniform dt.

        Returns:
        --------
        orientations : np.ndarray, shape=(N, 3)
            Estimated orientations [roll, pitch, yaw] for each sample.
        quaternions : np.ndarray, shape=(N, 4)
            Estimated quaternions [w, x, y, z] for each sample.
        """
        n_samples = len(accel_data)
        orientations = np.zeros((n_samples, 3))
        quaternions = np.zeros((n_samples, 4))

        if timestamps is None:
            dts = np.full(n_samples, self.dt)
        else:
            dts = np.diff(np.concatenate([[timestamps[0]], timestamps]))

        for i in range(n_samples):
            self.update(accel_data[i], gyro_data[i], dt=dts[i])
            orientations[i] = [
                self.orientation.roll,
                self.orientation.pitch,
                self.orientation.yaw
            ]
            quaternions[i] = self.orientation.to_quaternion()

        return orientations, quaternions

