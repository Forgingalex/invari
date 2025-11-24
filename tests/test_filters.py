"""
Unit tests for filter implementations.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters import ComplementaryFilter, ExtendedKalmanFilter
from src.filters.complementary import Orientation


class TestComplementaryFilter(unittest.TestCase):
    """Test cases for Complementary Filter."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = ComplementaryFilter(alpha=0.96, dt=0.01)

    def test_initialization(self):
        """Test filter initialization."""
        self.assertEqual(self.filter.alpha, 0.96)
        self.assertEqual(self.filter.dt, 0.01)
        self.assertIsNotNone(self.filter.orientation)

    def test_reset(self):
        """Test filter reset."""
        self.filter.orientation.roll = 1.0
        self.filter.reset()
        self.assertEqual(self.filter.orientation.roll, 0.0)
        self.assertEqual(self.filter.orientation.pitch, 0.0)
        self.assertEqual(self.filter.orientation.yaw, 0.0)

    def test_update_static(self):
        """Test filter update with static IMU (gravity only)."""
        # Gravity vector pointing down
        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.array([0.0, 0.0, 0.0])

        # Process bias estimation samples
        for _ in range(101):
            self.filter.update(accel, gyro)

        # Should converge to near-zero orientation
        self.assertAlmostEqual(self.filter.orientation.roll, 0.0, places=1)
        self.assertAlmostEqual(self.filter.orientation.pitch, 0.0, places=1)

    def test_batch_processing(self):
        """Test batch processing."""
        n_samples = 100
        accel_data = np.tile([0.0, 0.0, -9.81], (n_samples, 1))
        gyro_data = np.zeros((n_samples, 3))
        timestamps = np.linspace(0, 1, n_samples)

        orientations, quaternions = self.filter.process_batch(
            accel_data, gyro_data, timestamps
        )

        self.assertEqual(orientations.shape, (n_samples, 3))
        self.assertEqual(quaternions.shape, (n_samples, 4))

    def test_orientation_conversion(self):
        """Test orientation to quaternion conversion."""
        orient = Orientation(0.1, 0.2, 0.3)
        quat = orient.to_quaternion()
        
        self.assertEqual(len(quat), 4)
        # Check quaternion normalization
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0, places=5)


class TestExtendedKalmanFilter(unittest.TestCase):
    """Test cases for Extended Kalman Filter."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = ExtendedKalmanFilter(dt=0.01)

    def test_initialization(self):
        """Test filter initialization."""
        self.assertEqual(self.filter.dt, 0.01)
        self.assertIsNotNone(self.filter.state)
        self.assertEqual(self.filter.state.quaternion[0], 1.0)  # Identity quaternion

    def test_reset(self):
        """Test filter reset."""
        self.filter.state.quaternion[0] = 0.5
        self.filter.reset()
        self.assertEqual(self.filter.state.quaternion[0], 1.0)

    def test_quaternion_normalization(self):
        """Test quaternion normalization."""
        q = np.array([2.0, 0.0, 0.0, 0.0])
        q_norm = self.filter._normalize_quaternion(q)
        self.assertAlmostEqual(np.linalg.norm(q_norm), 1.0, places=5)

    def test_process_model(self):
        """Test process model."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        gyro = np.array([0.1, 0.0, 0.0])
        q_new = self.filter._process_model(q, gyro, 0.01)
        
        # Should remain normalized
        self.assertAlmostEqual(np.linalg.norm(q_new), 1.0, places=5)

    def test_batch_processing(self):
        """Test batch processing."""
        n_samples = 100
        accel_data = np.tile([0.0, 0.0, -9.81], (n_samples, 1))
        gyro_data = np.zeros((n_samples, 3))
        timestamps = np.linspace(0, 1, n_samples)

        orientations, quaternions, covariances = self.filter.process_batch(
            accel_data, gyro_data, timestamps
        )

        self.assertEqual(orientations.shape, (n_samples, 3))
        self.assertEqual(quaternions.shape, (n_samples, 4))
        self.assertEqual(covariances.shape, (n_samples, 4, 4))

    def test_step(self):
        """Test single EKF step."""
        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.array([0.0, 0.0, 0.0])

        # Process bias estimation
        for _ in range(101):
            state = self.filter.step(accel, gyro)

        # State should be valid
        self.assertIsNotNone(state)
        self.assertAlmostEqual(np.linalg.norm(state.quaternion), 1.0, places=5)


class TestFilterComparison(unittest.TestCase):
    """Test cases comparing both filters."""

    def test_same_input(self):
        """Test both filters with same input."""
        n_samples = 50
        accel_data = np.tile([0.0, 0.0, -9.81], (n_samples, 1))
        gyro_data = np.zeros((n_samples, 3))
        timestamps = np.linspace(0, 0.5, n_samples)

        comp_filter = ComplementaryFilter(alpha=0.96, dt=0.01)
        ekf_filter = ExtendedKalmanFilter(dt=0.01)

        comp_orientations, _ = comp_filter.process_batch(
            accel_data, gyro_data, timestamps
        )
        ekf_orientations, _, _ = ekf_filter.process_batch(
            accel_data, gyro_data, timestamps
        )

        # Both should produce valid outputs
        self.assertEqual(comp_orientations.shape, ekf_orientations.shape)
        self.assertFalse(np.any(np.isnan(comp_orientations)))
        self.assertFalse(np.any(np.isnan(ekf_orientations)))


if __name__ == "__main__":
    unittest.main()

