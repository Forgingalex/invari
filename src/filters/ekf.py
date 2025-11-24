"""
Extended Kalman Filter (EKF) for IMU Orientation Estimation

The EKF provides optimal state estimation by modeling the system dynamics
and measurement uncertainties. It maintains a state vector and covariance
matrix, updating them through prediction and correction steps.

Mathematical Foundation:
------------------------
State Vector (quaternion representation):
x = [q0, q1, q2, q3]ᵀ  (quaternion [w, x, y, z])

Process Model (gyroscope integration):
x_k = f(x_{k-1}, u_k, w_k)
where u_k is gyroscope input and w_k is process noise.

Measurement Model (accelerometer):
z_k = h(x_k) + v_k
where v_k is measurement noise.

Prediction Step:
P_k|k-1 = F_k P_{k-1|k-1} F_kᵀ + Q_k
x_k|k-1 = f(x_{k-1|k-1}, u_k)

Update Step:
K_k = P_k|k-1 H_kᵀ (H_k P_k|k-1 H_kᵀ + R_k)⁻¹
x_k|k = x_k|k-1 + K_k (z_k - h(x_k|k-1))
P_k|k = (I - K_k H_k) P_k|k-1

Where:
- F_k: Jacobian of process model
- H_k: Jacobian of measurement model
- Q_k: Process noise covariance
- R_k: Measurement noise covariance
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.linalg import block_diag


@dataclass
class EKFState:
    """EKF state representation."""
    quaternion: np.ndarray  # [w, x, y, z]
    covariance: np.ndarray  # 4x4 covariance matrix
    roll: float
    pitch: float
    yaw: float

    def to_euler(self) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles."""
        qw, qx, qy, qz = self.quaternion

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for IMU orientation estimation.

    Uses quaternion representation for state to avoid gimbal lock and
    provide smooth orientation tracking.

    Parameters:
    -----------
    dt : float, default=0.01
        Sampling period in seconds.
    Q : np.ndarray, optional
        Process noise covariance (4x4). Defaults to identity scaled by 0.01.
    R : np.ndarray, optional
        Measurement noise covariance (3x3). Defaults to identity scaled by 0.1.
    """

    def __init__(
        self,
        dt: float = 0.01,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None
    ):
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.dt = dt

        # Process noise covariance (quaternion uncertainty)
        self.Q = Q if Q is not None else np.eye(4) * 0.01

        # Measurement noise covariance (accelerometer uncertainty)
        self.R = R if R is not None else np.eye(3) * 0.1

        # Initialize state: identity quaternion [1, 0, 0, 0]
        self.state = EKFState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            covariance=np.eye(4) * 0.1,
            roll=0.0,
            pitch=0.0,
            yaw=0.0
        )

        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.bias_estimation_samples = 100
        self.bias_samples_collected = 0

    def reset(self) -> None:
        """Reset filter state."""
        self.state = EKFState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            covariance=np.eye(4) * 0.1,
            roll=0.0,
            pitch=0.0,
            yaw=0.0
        )
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.bias_samples_collected = 0

    def estimate_bias(self, gyro: np.ndarray) -> None:
        """Estimate gyroscope bias from initial samples."""
        if self.bias_samples_collected < self.bias_estimation_samples:
            self.gyro_bias = (
                self.gyro_bias * self.bias_samples_collected + gyro
            ) / (self.bias_samples_collected + 1)
            self.bias_samples_collected += 1

    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm

    def _process_model(
        self,
        q: np.ndarray,
        gyro: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Process model: quaternion integration from gyroscope.

        q_k = q_{k-1} ⊗ [1, 0.5*ωx*dt, 0.5*ωy*dt, 0.5*ωz*dt]
        """
        # Angular velocity quaternion
        omega = gyro * dt
        omega_norm = np.linalg.norm(omega)

        if omega_norm < 1e-10:
            return q

        # Quaternion derivative
        omega_quat = np.array([
            np.cos(omega_norm / 2),
            omega[0] / omega_norm * np.sin(omega_norm / 2),
            omega[1] / omega_norm * np.sin(omega_norm / 2),
            omega[2] / omega_norm * np.sin(omega_norm / 2)
        ])

        # Quaternion multiplication
        q_new = np.array([
            q[0] * omega_quat[0] - q[1] * omega_quat[1] -
            q[2] * omega_quat[2] - q[3] * omega_quat[3],
            q[0] * omega_quat[1] + q[1] * omega_quat[0] +
            q[2] * omega_quat[3] - q[3] * omega_quat[2],
            q[0] * omega_quat[2] - q[1] * omega_quat[3] +
            q[2] * omega_quat[0] + q[3] * omega_quat[1],
            q[0] * omega_quat[3] + q[1] * omega_quat[2] -
            q[2] * omega_quat[1] + q[3] * omega_quat[0]
        ])

        return self._normalize_quaternion(q_new)

    def _process_jacobian(
        self,
        q: np.ndarray,
        gyro: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute Jacobian of process model with respect to state.

        F = ∂f/∂x
        """
        omega = gyro * dt
        omega_norm = np.linalg.norm(omega)

        if omega_norm < 1e-10:
            return np.eye(4)

        # Approximate Jacobian (linearized around current state)
        F = np.eye(4)
        F[0, 1] = -omega[0] * dt / 2
        F[0, 2] = -omega[1] * dt / 2
        F[0, 3] = -omega[2] * dt / 2
        F[1, 0] = omega[0] * dt / 2
        F[1, 2] = omega[2] * dt / 2
        F[1, 3] = -omega[1] * dt / 2
        F[2, 0] = omega[1] * dt / 2
        F[2, 1] = -omega[2] * dt / 2
        F[2, 3] = omega[0] * dt / 2
        F[3, 0] = omega[2] * dt / 2
        F[3, 1] = omega[1] * dt / 2
        F[3, 2] = -omega[0] * dt / 2

        return F

    def _measurement_model(self, q: np.ndarray) -> np.ndarray:
        """
        Measurement model: expected accelerometer reading in body frame.

        Assuming gravity vector in world frame is [0, 0, -g], we rotate it
        to body frame using the quaternion rotation.
        """
        qw, qx, qy, qz = q

        # Gravity vector in world frame (normalized)
        g_world = np.array([0.0, 0.0, -1.0])

        # Rotate to body frame: R^T * g_world
        # Using quaternion rotation matrix
        accel_body = np.array([
            2 * (qx * qz - qw * qy),
            2 * (qy * qz + qw * qx),
            qw * qw - qx * qx - qy * qy + qz * qz
        ]) * (-g_world[2])  # Scale by gravity magnitude

        return accel_body

    def _measurement_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of measurement model with respect to state.

        H = ∂h/∂x
        """
        qw, qx, qy, qz = q

        H = np.zeros((3, 4))
        H[0, 0] = 2 * qy
        H[0, 1] = 2 * qz
        H[0, 2] = -2 * qw
        H[0, 3] = -2 * qx

        H[1, 0] = -2 * qx
        H[1, 1] = 2 * qw
        H[1, 2] = 2 * qz
        H[1, 3] = -2 * qy

        H[2, 0] = 2 * qw
        H[2, 1] = -2 * qx
        H[2, 2] = -2 * qy
        H[2, 3] = 2 * qz

        return H

    def predict(self, gyro: np.ndarray, dt: Optional[float] = None) -> None:
        """
        Prediction step: propagate state using process model.

        Parameters:
        -----------
        gyro : np.ndarray, shape=(3,)
            Gyroscope reading [gx, gy, gz] in rad/s.
        dt : float, optional
            Time step in seconds. Uses self.dt if not provided.
        """
        if dt is None:
            dt = self.dt

        # Remove bias
        gyro_corrected = gyro - self.gyro_bias

        # Process model
        q_pred = self._process_model(self.state.quaternion, gyro_corrected, dt)

        # Process Jacobian
        F = self._process_jacobian(self.state.quaternion, gyro_corrected, dt)

        # Covariance prediction
        P_pred = F @ self.state.covariance @ F.T + self.Q

        # Update state
        self.state.quaternion = q_pred
        self.state.covariance = P_pred

    def update(self, accel: np.ndarray) -> None:
        """
        Update step: correct state using measurement.

        Parameters:
        -----------
        accel : np.ndarray, shape=(3,)
            Accelerometer reading [ax, ay, az] in m/s².
        """
        # Normalize accelerometer
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 1e-6:
            return  # Invalid measurement

        accel_normalized = accel / accel_norm

        # Expected measurement
        h = self._measurement_model(self.state.quaternion)
        h_normalized = h / (np.linalg.norm(h) + 1e-10)

        # Measurement residual
        y = accel_normalized - h_normalized

        # Measurement Jacobian
        H = self._measurement_jacobian(self.state.quaternion)

        # Innovation covariance
        S = H @ self.state.covariance @ H.T + self.R

        # Kalman gain
        K = self.state.covariance @ H.T @ np.linalg.inv(S)

        # State update
        dq = K @ y
        self.state.quaternion = self._normalize_quaternion(
            self.state.quaternion + dq
        )

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ H
        self.state.covariance = (
            I_KH @ self.state.covariance @ I_KH.T +
            K @ self.R @ K.T
        )

        # Update Euler angles
        roll, pitch, yaw = self.state.to_euler()
        self.state.roll = roll
        self.state.pitch = pitch
        self.state.yaw = yaw

    def step(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: Optional[float] = None
    ) -> EKFState:
        """
        Perform one complete EKF step (predict + update).

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
        EKFState
            Updated state estimate.
        """
        if dt is None:
            dt = self.dt

        # Estimate bias during initial samples
        if self.bias_samples_collected < self.bias_estimation_samples:
            self.estimate_bias(gyro)
            return self.state

        # Prediction
        self.predict(gyro, dt)

        # Update
        self.update(accel)

        return self.state

    def process_batch(
        self,
        accel_data: np.ndarray,
        gyro_data: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        covariances : np.ndarray, shape=(N, 4, 4)
            Covariance matrices for each sample.
        """
        n_samples = len(accel_data)
        orientations = np.zeros((n_samples, 3))
        quaternions = np.zeros((n_samples, 4))
        covariances = np.zeros((n_samples, 4, 4))

        if timestamps is None:
            dts = np.full(n_samples, self.dt)
        else:
            dts = np.diff(np.concatenate([[timestamps[0]], timestamps]))

        for i in range(n_samples):
            self.step(accel_data[i], gyro_data[i], dt=dts[i])
            orientations[i] = [
                self.state.roll,
                self.state.pitch,
                self.state.yaw
            ]
            quaternions[i] = self.state.quaternion
            covariances[i] = self.state.covariance

        return orientations, quaternions, covariances

