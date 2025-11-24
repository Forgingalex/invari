#!/usr/bin/env python3
"""
Parameter sweep for filter tuning.

Sweeps alpha parameter for complementary filter and Q/R matrices for EKF,
evaluating performance with RMSE metrics.

Usage:
    python scripts/sweep_parameters.py <input_file> [--output output_dir]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters import ComplementaryFilter, ExtendedKalmanFilter
from src.io import IMULoader
from src.utils import compute_rmse


def sweep_complementary(data, alpha_range, output_dir):
    """Sweep alpha parameter for complementary filter."""
    print("Sweeping complementary filter alpha parameter...")
    
    results = []
    dt = 1.0 / data.sample_rate

    for alpha in alpha_range:
        filter_obj = ComplementaryFilter(alpha=alpha, dt=dt)
        orientations, quaternions = filter_obj.process_batch(
            data.accelerometer,
            data.gyroscope,
            data.timestamps
        )

        # Compute metrics (using variance as proxy for stability)
        roll_std = np.std(orientations[:, 0])
        pitch_std = np.std(orientations[:, 1])
        yaw_std = np.std(orientations[:, 2])

        results.append({
            "alpha": alpha,
            "roll_std": roll_std,
            "pitch_std": pitch_std,
            "yaw_std": yaw_std,
            "mean_std": (roll_std + pitch_std + yaw_std) / 3
        })

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "complementary_sweep.csv", index=False)
    print(f"✓ Saved results to {output_dir / 'complementary_sweep.csv'}")
    
    return df


def sweep_ekf(data, Q_scales, R_scales, output_dir):
    """Sweep Q and R parameters for EKF."""
    print("Sweeping EKF Q and R parameters...")
    
    results = []
    dt = 1.0 / data.sample_rate
    base_Q = np.eye(4)
    base_R = np.eye(3)

    for Q_scale, R_scale in product(Q_scales, R_scales):
        Q = base_Q * Q_scale
        R = base_R * R_scale

        filter_obj = ExtendedKalmanFilter(dt=dt, Q=Q, R=R)
        orientations, quaternions, covariances = filter_obj.process_batch(
            data.accelerometer,
            data.gyroscope,
            data.timestamps
        )

        # Compute metrics
        roll_std = np.std(orientations[:, 0])
        pitch_std = np.std(orientations[:, 1])
        yaw_std = np.std(orientations[:, 2])
        
        # Average covariance trace (uncertainty measure)
        mean_trace = np.mean([np.trace(cov) for cov in covariances])

        results.append({
            "Q_scale": Q_scale,
            "R_scale": R_scale,
            "roll_std": roll_std,
            "pitch_std": pitch_std,
            "yaw_std": yaw_std,
            "mean_std": (roll_std + pitch_std + yaw_std) / 3,
            "mean_cov_trace": mean_trace
        })

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "ekf_sweep.csv", index=False)
    print(f"✓ Saved results to {output_dir / 'ekf_sweep.csv'}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for filter tuning"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to IMU CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sweep_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.90,
        help="Minimum alpha for complementary filter"
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.99,
        help="Maximum alpha for complementary filter"
    )
    parser.add_argument(
        "--alpha-steps",
        type=int,
        default=10,
        help="Number of alpha steps"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading IMU data from {args.input_file}...")
    data = IMULoader.load_csv(args.input_file)
    print(f"Loaded {len(data)} samples at {data.sample_rate:.1f} Hz")

    # Sweep complementary filter
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    comp_results = sweep_complementary(data, alpha_range, output_dir)

    # Sweep EKF
    Q_scales = [0.001, 0.01, 0.1, 1.0, 10.0]
    R_scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    ekf_results = sweep_ekf(data, Q_scales, R_scales, output_dir)

    # Print summary
    print("\n=== Complementary Filter Summary ===")
    best_alpha = comp_results.loc[comp_results["mean_std"].idxmin()]
    print(f"Best alpha: {best_alpha['alpha']:.3f}")
    print(f"Mean std: {best_alpha['mean_std']:.4f} rad")

    print("\n=== EKF Summary ===")
    best_ekf = ekf_results.loc[ekf_results["mean_std"].idxmin()]
    print(f"Best Q_scale: {best_ekf['Q_scale']:.3f}")
    print(f"Best R_scale: {best_ekf['R_scale']:.3f}")
    print(f"Mean std: {best_ekf['mean_std']:.4f} rad")

    print(f"\n✓ Parameter sweep complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

