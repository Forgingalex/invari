"""
Visualization utilities for IMU data and orientation estimates.

Provides plotting functions for signals, orientations, and residuals.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path


def plot_signals(
    timestamps: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    title: str = "IMU Raw Signals",
    save_path: Optional[str] = None
) -> None:
    """
    Plot raw accelerometer and gyroscope signals.

    Parameters:
    -----------
    timestamps : np.ndarray, shape=(N,)
        Time in seconds.
    accel : np.ndarray, shape=(N, 3)
        Accelerometer data [ax, ay, az].
    gyro : np.ndarray, shape=(N, 3)
        Gyroscope data [gx, gy, gz].
    title : str, default="IMU Raw Signals"
        Plot title.
    save_path : str, optional
        Path to save figure. If None, displays plot.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Accelerometer
    axes[0].plot(timestamps, accel[:, 0], label="ax", alpha=0.7)
    axes[0].plot(timestamps, accel[:, 1], label="ay", alpha=0.7)
    axes[0].plot(timestamps, accel[:, 2], label="az", alpha=0.7)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title("Accelerometer")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gyroscope
    axes[1].plot(timestamps, gyro[:, 0], label="gx", alpha=0.7)
    axes[1].plot(timestamps, gyro[:, 1], label="gy", alpha=0.7)
    axes[1].plot(timestamps, gyro[:, 2], label="gz", alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular Velocity (rad/s)")
    axes[1].set_title("Gyroscope")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_orientation(
    timestamps: np.ndarray,
    orientations: np.ndarray,
    labels: Optional[Tuple[str, str, str]] = None,
    title: str = "Orientation Estimate",
    save_path: Optional[str] = None
) -> None:
    """
    Plot orientation angles (roll, pitch, yaw).

    Parameters:
    -----------
    timestamps : np.ndarray, shape=(N,)
        Time in seconds.
    orientations : np.ndarray, shape=(N, 3)
        Orientation angles [roll, pitch, yaw] in radians.
    labels : tuple, optional
        Labels for each orientation estimate. If None, uses default.
    title : str, default="Orientation Estimate"
        Plot title.
    save_path : str, optional
        Path to save figure. If None, displays plot.
    """
    if labels is None:
        labels = ("Roll", "Pitch", "Yaw")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    angles = ["Roll (φ)", "Pitch (θ)", "Yaw (ψ)"]
    colors = ["r", "g", "b"]

    for i in range(3):
        axes[i].plot(
            timestamps,
            np.degrees(orientations[:, i]),
            label=labels[i],
            color=colors[i],
            alpha=0.7
        )
        axes[i].set_ylabel(f"{angles[i]} (deg)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_residuals(
    timestamps: np.ndarray,
    residuals: np.ndarray,
    title: str = "Filter Residuals",
    save_path: Optional[str] = None
) -> None:
    """
    Plot filter residuals (measurement - predicted).

    Parameters:
    -----------
    timestamps : np.ndarray, shape=(N,)
        Time in seconds.
    residuals : np.ndarray, shape=(N, 3)
        Residual values for each axis.
    title : str, default="Filter Residuals"
        Plot title.
    save_path : str, optional
        Path to save figure. If None, displays plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ["X", "Y", "Z"]
    colors = ["r", "g", "b"]

    for i in range(3):
        axes[i].plot(
            timestamps,
            residuals[:, i],
            label=f"{labels[i]}-axis",
            color=colors[i],
            alpha=0.7
        )
        axes[i].axhline(y=0, color="k", linestyle="--", alpha=0.3)
        axes[i].set_ylabel(f"Residual ({labels[i]})")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_comparison(
    timestamps: np.ndarray,
    true: Optional[np.ndarray],
    complementary: np.ndarray,
    ekf: np.ndarray,
    title: str = "Filter Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Compare multiple orientation estimates.

    Parameters:
    -----------
    timestamps : np.ndarray, shape=(N,)
        Time in seconds.
    true : np.ndarray, shape=(N, 3), optional
        Ground truth orientations.
    complementary : np.ndarray, shape=(N, 3)
        Complementary filter estimates.
    ekf : np.ndarray, shape=(N, 3)
        EKF estimates.
    title : str, default="Filter Comparison"
        Plot title.
    save_path : str, optional
        Path to save figure. If None, displays plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    angles = ["Roll (φ)", "Pitch (θ)", "Yaw (ψ)"]
    colors = ["r", "g", "b"]

    for i in range(3):
        if true is not None:
            axes[i].plot(
                timestamps,
                np.degrees(true[:, i]),
                label="Ground Truth",
                color="k",
                linestyle="--",
                alpha=0.7
            )
        axes[i].plot(
            timestamps,
            np.degrees(complementary[:, i]),
            label="Complementary",
            color=colors[i],
            linestyle="-",
            alpha=0.7
        )
        axes[i].plot(
            timestamps,
            np.degrees(ekf[:, i]),
            label="EKF",
            color=colors[i],
            linestyle=":",
            alpha=0.7
        )
        axes[i].set_ylabel(f"{angles[i]} (deg)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

