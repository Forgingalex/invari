"""
IMU Data Loader

Provides utilities for loading IMU data from CSV files and converting
between different formats.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IMUData:
    """Container for IMU measurement data."""
    timestamps: np.ndarray  # Time in seconds
    accelerometer: np.ndarray  # Shape: (N, 3) - [ax, ay, az] in m/s²
    gyroscope: np.ndarray  # Shape: (N, 3) - [gx, gy, gz] in rad/s
    sample_rate: float  # Hz

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.timestamps)

    def get_duration(self) -> float:
        """Get total duration in seconds."""
        return self.timestamps[-1] - self.timestamps[0]

    def get_time_range(self, start: float, end: float) -> "IMUData":
        """
        Extract a time range from the data.

        Parameters:
        -----------
        start : float
            Start time in seconds.
        end : float
            End time in seconds.

        Returns:
        --------
        IMUData
            Subset of data in the specified time range.
        """
        mask = (self.timestamps >= start) & (self.timestamps <= end)
        return IMUData(
            timestamps=self.timestamps[mask],
            accelerometer=self.accelerometer[mask],
            gyroscope=self.gyroscope[mask],
            sample_rate=self.sample_rate
        )


class IMULoader:
    """
    Loader for IMU data from CSV files.

    Expected CSV format:
    - Columns: timestamp, ax, ay, az, gx, gy, gz
    - Timestamp can be absolute (seconds) or relative (will be normalized)
    - Accelerometer units: m/s²
    - Gyroscope units: rad/s
    """

    @staticmethod
    def load_csv(
        filepath: str,
        timestamp_col: str = "timestamp",
        accel_cols: Tuple[str, str, str] = ("ax", "ay", "az"),
        gyro_cols: Tuple[str, str, str] = ("gx", "gy", "gz"),
        sample_rate: Optional[float] = None
    ) -> IMUData:
        """
        Load IMU data from CSV file.

        Parameters:
        -----------
        filepath : str
            Path to CSV file.
        timestamp_col : str, default="timestamp"
            Name of timestamp column.
        accel_cols : tuple, default=("ax", "ay", "az")
            Names of accelerometer columns.
        gyro_cols : tuple, default=("gx", "gy", "gz")
            Names of gyroscope columns.
        sample_rate : float, optional
            Sample rate in Hz. If None, estimated from timestamps.

        Returns:
        --------
        IMUData
            Loaded IMU data.
        """
        df = pd.read_csv(filepath)

        # Extract timestamps
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")

        timestamps = df[timestamp_col].values.astype(float)

        # Normalize timestamps to start at 0
        if timestamps[0] != 0:
            timestamps = timestamps - timestamps[0]

        # Extract accelerometer data
        accel_cols_list = list(accel_cols)
        missing_accel = [col for col in accel_cols_list if col not in df.columns]
        if missing_accel:
            raise ValueError(f"Accelerometer columns not found: {missing_accel}")

        accelerometer = df[accel_cols_list].values.astype(float)

        # Extract gyroscope data
        gyro_cols_list = list(gyro_cols)
        missing_gyro = [col for col in gyro_cols_list if col not in df.columns]
        if missing_gyro:
            raise ValueError(f"Gyroscope columns not found: {missing_gyro}")

        gyroscope = df[gyro_cols_list].values.astype(float)

        # Estimate sample rate if not provided
        if sample_rate is None:
            if len(timestamps) > 1:
                dt = np.mean(np.diff(timestamps))
                sample_rate = 1.0 / dt if dt > 0 else 100.0
            else:
                sample_rate = 100.0

        return IMUData(
            timestamps=timestamps,
            accelerometer=accelerometer,
            gyroscope=gyroscope,
            sample_rate=sample_rate
        )

    @staticmethod
    def save_csv(
        data: IMUData,
        filepath: str,
        timestamp_col: str = "timestamp",
        accel_cols: Tuple[str, str, str] = ("ax", "ay", "az"),
        gyro_cols: Tuple[str, str, str] = ("gx", "gy", "gz")
    ) -> None:
        """
        Save IMU data to CSV file.

        Parameters:
        -----------
        data : IMUData
            Data to save.
        filepath : str
            Output file path.
        timestamp_col : str, default="timestamp"
            Name of timestamp column.
        accel_cols : tuple, default=("ax", "ay", "az")
            Names of accelerometer columns.
        gyro_cols : tuple, default=("gx", "gy", "gz")
            Names of gyroscope columns.
        """
        df = pd.DataFrame({
            timestamp_col: data.timestamps,
            accel_cols[0]: data.accelerometer[:, 0],
            accel_cols[1]: data.accelerometer[:, 1],
            accel_cols[2]: data.accelerometer[:, 2],
            gyro_cols[0]: data.gyroscope[:, 0],
            gyro_cols[1]: data.gyroscope[:, 1],
            gyro_cols[2]: data.gyroscope[:, 2],
        })
        df.to_csv(filepath, index=False)

