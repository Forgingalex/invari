"""
Evaluation metrics for orientation estimation.

Provides RMSE, MAE, and orientation error calculations for comparing
filter performance.
"""

import numpy as np
from typing import Optional


def compute_rmse(
    true: np.ndarray,
    estimated: np.ndarray,
    axis: Optional[int] = None
) -> float:
    """
    Compute Root Mean Square Error (RMSE).

    Parameters:
    -----------
    true : np.ndarray
        True values.
    estimated : np.ndarray
        Estimated values.
    axis : int, optional
        Axis along which to compute RMSE. If None, computes over all elements.

    Returns:
    --------
    float
        RMSE value.
    """
    mse = np.mean((true - estimated) ** 2, axis=axis)
    return np.sqrt(mse)


def compute_mae(
    true: np.ndarray,
    estimated: np.ndarray,
    axis: Optional[int] = None
) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Parameters:
    -----------
    true : np.ndarray
        True values.
    estimated : np.ndarray
        Estimated values.
    axis : int, optional
        Axis along which to compute MAE. If None, computes over all elements.

    Returns:
    --------
    float
        MAE value.
    """
    return np.mean(np.abs(true - estimated), axis=axis)


def compute_orientation_error(
    true_quat: np.ndarray,
    estimated_quat: np.ndarray
) -> np.ndarray:
    """
    Compute orientation error between quaternions.

    The error is computed as the angular difference between two quaternions.

    Parameters:
    -----------
    true_quat : np.ndarray, shape=(N, 4) or (4,)
        True quaternions [w, x, y, z].
    estimated_quat : np.ndarray, shape=(N, 4) or (4,)
        Estimated quaternions [w, x, y, z].

    Returns:
    --------
    np.ndarray
        Angular error in radians. Shape: (N,) or scalar.
    """
    true_quat = np.atleast_2d(true_quat)
    estimated_quat = np.atleast_2d(estimated_quat)

    # Normalize quaternions
    true_quat = true_quat / np.linalg.norm(true_quat, axis=1, keepdims=True)
    estimated_quat = estimated_quat / np.linalg.norm(
        estimated_quat, axis=1, keepdims=True
    )

    # Compute relative rotation quaternion: q_error = q_true * q_est^-1
    # For unit quaternions: q^-1 = [w, -x, -y, -z]
    q_est_inv = estimated_quat.copy()
    q_est_inv[:, 1:] *= -1

    # Quaternion multiplication
    q_error = np.zeros_like(true_quat)
    for i in range(len(true_quat)):
        q_error[i, 0] = (
            true_quat[i, 0] * q_est_inv[i, 0] -
            true_quat[i, 1] * q_est_inv[i, 1] -
            true_quat[i, 2] * q_est_inv[i, 2] -
            true_quat[i, 3] * q_est_inv[i, 3]
        )
        q_error[i, 1] = (
            true_quat[i, 0] * q_est_inv[i, 1] +
            true_quat[i, 1] * q_est_inv[i, 0] +
            true_quat[i, 2] * q_est_inv[i, 3] -
            true_quat[i, 3] * q_est_inv[i, 2]
        )
        q_error[i, 2] = (
            true_quat[i, 0] * q_est_inv[i, 2] -
            true_quat[i, 1] * q_est_inv[i, 3] +
            true_quat[i, 2] * q_est_inv[i, 0] +
            true_quat[i, 3] * q_est_inv[i, 1]
        )
        q_error[i, 3] = (
            true_quat[i, 0] * q_est_inv[i, 3] +
            true_quat[i, 1] * q_est_inv[i, 2] -
            true_quat[i, 2] * q_est_inv[i, 1] +
            true_quat[i, 3] * q_est_inv[i, 0]
        )

    # Extract rotation angle: θ = 2 * arccos(|w|)
    # Clamp to avoid numerical issues
    w = np.clip(np.abs(q_error[:, 0]), 0, 1)
    angle_error = 2 * np.arccos(w)

    return angle_error.squeeze()

