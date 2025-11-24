"""
FastAPI endpoint for IMU log processing and sensor fusion.

Provides REST API endpoints for:
- Uploading IMU CSV logs
- Processing with complementary filter
- Processing with Extended Kalman Filter
- Retrieving results and visualizations
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.filters import ComplementaryFilter, ExtendedKalmanFilter
from src.io import IMULoader
from src.utils import compute_rmse, compute_orientation_error

app = FastAPI(
    title="INVARI API",
    description="Sensor Fusion Platform for IMU Data",
    version="0.1.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FilterConfig(BaseModel):
    """Filter configuration parameters."""
    alpha: Optional[float] = 0.96
    dt: Optional[float] = None
    Q: Optional[List[List[float]]] = None
    R: Optional[List[List[float]]] = None


class ProcessingResult(BaseModel):
    """Result from filter processing."""
    orientations: List[List[float]]  # (N, 3) - roll, pitch, yaw
    quaternions: List[List[float]]  # (N, 4) - w, x, y, z
    timestamps: List[float]
    sample_rate: float
    duration: float


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "INVARI Sensor Fusion API",
        "version": "0.1.0",
        "endpoints": {
            "upload": "/upload",
            "process_complementary": "/process/complementary",
            "process_ekf": "/process/ekf",
            "compare": "/compare"
        }
    }


@app.post("/upload")
async def upload_imu_log(file: UploadFile = File(...)):
    """
    Upload IMU CSV log file.

    Expected format: timestamp, ax, ay, az, gx, gy, gz
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load IMU data
        data = IMULoader.load_csv(tmp_path)
        
        return {
            "filename": file.filename,
            "samples": len(data),
            "duration": data.get_duration(),
            "sample_rate": data.sample_rate,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/process/complementary")
async def process_complementary(
    file: UploadFile = File(...),
    alpha: float = 0.96,
    dt: Optional[float] = None
):
    """
    Process IMU log with complementary filter.

    Parameters:
    -----------
    file : UploadFile
        IMU CSV file.
    alpha : float, default=0.96
        Complementary filter blending parameter.
    dt : float, optional
        Sampling period. If None, estimated from data.
    """
    # Save and load file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        data = IMULoader.load_csv(tmp_path)
        
        if dt is None:
            dt = 1.0 / data.sample_rate

        # Process with complementary filter
        filter_obj = ComplementaryFilter(alpha=alpha, dt=dt)
        orientations, quaternions = filter_obj.process_batch(
            data.accelerometer,
            data.gyroscope,
            data.timestamps
        )

        return ProcessingResult(
            orientations=orientations.tolist(),
            quaternions=quaternions.tolist(),
            timestamps=data.timestamps.tolist(),
            sample_rate=data.sample_rate,
            duration=data.get_duration()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/process/ekf")
async def process_ekf(
    file: UploadFile = File(...),
    dt: Optional[float] = None,
    Q: Optional[List[List[float]]] = None,
    R: Optional[List[List[float]]] = None
):
    """
    Process IMU log with Extended Kalman Filter.

    Parameters:
    -----------
    file : UploadFile
        IMU CSV file.
    dt : float, optional
        Sampling period. If None, estimated from data.
    Q : List[List[float]], optional
        Process noise covariance (4x4).
    R : List[List[float]], optional
        Measurement noise covariance (3x3).
    """
    # Save and load file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        data = IMULoader.load_csv(tmp_path)
        
        if dt is None:
            dt = 1.0 / data.sample_rate

        Q_array = np.array(Q) if Q else None
        R_array = np.array(R) if R else None

        # Process with EKF
        filter_obj = ExtendedKalmanFilter(dt=dt, Q=Q_array, R=R_array)
        orientations, quaternions, _ = filter_obj.process_batch(
            data.accelerometer,
            data.gyroscope,
            data.timestamps
        )

        return ProcessingResult(
            orientations=orientations.tolist(),
            quaternions=quaternions.tolist(),
            timestamps=data.timestamps.tolist(),
            sample_rate=data.sample_rate,
            duration=data.get_duration()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/compare")
async def compare_filters(
    file: UploadFile = File(...),
    alpha: float = 0.96,
    dt: Optional[float] = None,
    Q: Optional[List[List[float]]] = None,
    R: Optional[List[List[float]]] = None
):
    """
    Compare complementary filter and EKF on the same data.

    Returns results from both filters for comparison.
    """
    # Save and load file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        data = IMULoader.load_csv(tmp_path)
        
        if dt is None:
            dt = 1.0 / data.sample_rate

        # Process with complementary filter
        comp_filter = ComplementaryFilter(alpha=alpha, dt=dt)
        comp_orientations, comp_quaternions = comp_filter.process_batch(
            data.accelerometer,
            data.gyroscope,
            data.timestamps
        )

        # Process with EKF
        Q_array = np.array(Q) if Q else None
        R_array = np.array(R) if R else None
        ekf_filter = ExtendedKalmanFilter(dt=dt, Q=Q_array, R=R_array)
        ekf_orientations, ekf_quaternions, _ = ekf_filter.process_batch(
            data.accelerometer,
            data.gyroscope,
            data.timestamps
        )

        # Compute comparison metrics
        orientation_diff = np.abs(comp_orientations - ekf_orientations)
        rmse_roll = compute_rmse(comp_orientations[:, 0], ekf_orientations[:, 0])
        rmse_pitch = compute_rmse(comp_orientations[:, 1], ekf_orientations[:, 1])
        rmse_yaw = compute_rmse(comp_orientations[:, 2], ekf_orientations[:, 2])

        quat_error = compute_orientation_error(comp_quaternions, ekf_quaternions)
        mean_quat_error = np.mean(quat_error)

        return {
            "complementary": ProcessingResult(
                orientations=comp_orientations.tolist(),
                quaternions=comp_quaternions.tolist(),
                timestamps=data.timestamps.tolist(),
                sample_rate=data.sample_rate,
                duration=data.get_duration()
            ),
            "ekf": ProcessingResult(
                orientations=ekf_orientations.tolist(),
                quaternions=ekf_quaternions.tolist(),
                timestamps=data.timestamps.tolist(),
                sample_rate=data.sample_rate,
                duration=data.get_duration()
            ),
            "metrics": {
                "rmse_roll": float(rmse_roll),
                "rmse_pitch": float(rmse_pitch),
                "rmse_yaw": float(rmse_yaw),
                "mean_quaternion_error_rad": float(mean_quat_error),
                "mean_quaternion_error_deg": float(np.degrees(mean_quat_error))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

