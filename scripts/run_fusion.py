#!/usr/bin/env python3
"""
Run sensor fusion on IMU data file.

Usage:
    python scripts/run_fusion.py <input_file> [--filter COMPLEMENTARY|EKF|BOTH] [--alpha 0.96] [--output output_dir]
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters import ComplementaryFilter, ExtendedKalmanFilter
from src.io import IMULoader
from src.utils import plot_orientation, plot_signals, plot_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run sensor fusion on IMU data"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to IMU CSV file"
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["COMPLEMENTARY", "EKF", "BOTH"],
        default="BOTH",
        help="Filter to use"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.96,
        help="Complementary filter alpha parameter"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading IMU data from {args.input_file}...")
    data = IMULoader.load_csv(args.input_file)
    print(f"Loaded {len(data)} samples at {data.sample_rate:.1f} Hz")
    print(f"Duration: {data.get_duration():.2f} seconds")

    dt = 1.0 / data.sample_rate
    results = {}

    # Process with complementary filter
    if args.filter in ["COMPLEMENTARY", "BOTH"]:
        print("\nProcessing with Complementary Filter...")
        comp_filter = ComplementaryFilter(alpha=args.alpha, dt=dt)
        comp_orientations, comp_quaternions = comp_filter.process_batch(
            data.accelerometer,
            data.gyroscope,
            data.timestamps
        )
        results["complementary"] = {
            "orientations": comp_orientations,
            "quaternions": comp_quaternions
        }
        print("✓ Complementary filter complete")

    # Process with EKF
    if args.filter in ["EKF", "BOTH"]:
        print("\nProcessing with Extended Kalman Filter...")
        ekf_filter = ExtendedKalmanFilter(dt=dt)
        ekf_orientations, ekf_quaternions, ekf_covariances = ekf_filter.process_batch(
            data.accelerometer,
            data.gyroscope,
            data.timestamps
        )
        results["ekf"] = {
            "orientations": ekf_orientations,
            "quaternions": ekf_quaternions,
            "covariances": ekf_covariances
        }
        print("✓ EKF complete")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    for filter_name, result in results.items():
        np.save(
            output_dir / f"{filter_name}_orientations.npy",
            result["orientations"]
        )
        np.save(
            output_dir / f"{filter_name}_quaternions.npy",
            result["quaternions"]
        )
        if "covariances" in result:
            np.save(
                output_dir / f"{filter_name}_covariances.npy",
                result["covariances"]
            )

    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        plot_signals(
            data.timestamps,
            data.accelerometer,
            data.gyroscope,
            save_path=str(output_dir / "raw_signals.png")
        )

        if "complementary" in results:
            plot_orientation(
                data.timestamps,
                results["complementary"]["orientations"],
                title="Complementary Filter Orientation",
                save_path=str(output_dir / "complementary_orientation.png")
            )

        if "ekf" in results:
            plot_orientation(
                data.timestamps,
                results["ekf"]["orientations"],
                title="EKF Orientation",
                save_path=str(output_dir / "ekf_orientation.png")
            )

        if "complementary" in results and "ekf" in results:
            plot_comparison(
                data.timestamps,
                None,
                results["complementary"]["orientations"],
                results["ekf"]["orientations"],
                title="Filter Comparison",
                save_path=str(output_dir / "filter_comparison.png")
            )

        print("✓ Plots saved")

    print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()

