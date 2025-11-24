# INVARI

**Sensor Fusion Platform for IMU Data**

INVARI is a production-grade sensor fusion platform for analyzing IMU logs, applying complementary and Extended Kalman filters, and visualizing orientation in 3D.

## Overview

INVARI provides:
- **Complementary Filter**: Simple, efficient orientation estimation combining accelerometer and gyroscope data
- **Extended Kalman Filter**: Optimal state estimation with uncertainty quantification
- **3D Visualization**: Real-time orientation viewer using Three.js
- **Comparative Analysis**: RMSE metrics and parameter tuning tools
- **REST API**: FastAPI backend for processing IMU logs
- **Web Dashboard**: React frontend with interactive plots and controls

## Architecture

```
INVARI/
├── src/                    # Core Python modules
│   ├── filters/           # Filter implementations
│   │   ├── complementary.py
│   │   └── ekf.py
│   ├── io/                # Data I/O
│   │   └── imu_loader.py
│   └── utils/             # Utilities
│       ├── metrics.py
│       └── visualization.py
├── api/                   # FastAPI backend
│   └── main.py
├── web/                   # React frontend
│   └── src/
│       ├── components/
│       │   ├── OrientationViewer.js
│       │   ├── SignalPlots.js
│       │   └── FileUpload.js
│       └── App.js
├── notebooks/             # Analysis notebooks
│   └── fusion_comparison.ipynb
├── scripts/               # Utility scripts
│   ├── run_fusion.py
│   └── sweep_parameters.py
├── tests/                 # Test suite
│   └── test_filters.py
└── data/                  # Sample data
    └── sample_imu.csv
```

## Mathematical Foundation

### Complementary Filter

The complementary filter combines accelerometer and gyroscope measurements using a tunable blending parameter α:

**Accelerometer-based angles (low-frequency):**
```
φ_acc = atan2(ay, az)
θ_acc = atan2(-ax, sqrt(ay² + az²))
```

**Gyroscope integration (high-frequency):**
```
φ_gyro = φ_prev + gx * dt
θ_gyro = θ_prev + gy * dt
```

**Complementary fusion:**
```
φ = α * φ_gyro + (1 - α) * φ_acc
θ = α * θ_gyro + (1 - α) * θ_acc
ψ = ψ_prev + gz * dt  (yaw from gyro only)
```

**Parameters:**
- `α`: Blending parameter (0-1), typically 0.95-0.98
  - Higher α: More trust in gyroscope (better dynamic response)
  - Lower α: More trust in accelerometer (better long-term stability)

### Extended Kalman Filter

The EKF provides optimal state estimation by modeling system dynamics and measurement uncertainties.

**State Vector:**
```
x = [q0, q1, q2, q3]ᵀ  (quaternion [w, x, y, z])
```

**Process Model (gyroscope integration):**
```
x_k = f(x_{k-1}, u_k, w_k)
```
where `u_k` is gyroscope input and `w_k` is process noise.

**Measurement Model (accelerometer):**
```
z_k = h(x_k) + v_k
```
where `v_k` is measurement noise.

**Prediction Step:**
```
P_k|k-1 = F_k P_{k-1|k-1} F_kᵀ + Q_k
x_k|k-1 = f(x_{k-1|k-1}, u_k)
```

**Update Step:**
```
K_k = P_k|k-1 H_kᵀ (H_k P_k|k-1 H_kᵀ + R_k)⁻¹
x_k|k = x_k|k-1 + K_k (z_k - h(x_k|k-1))
P_k|k = (I - K_k H_k) P_k|k-1
```

**Parameters:**
- `Q`: Process noise covariance (4x4) - models gyroscope uncertainty
- `R`: Measurement noise covariance (3x3) - models accelerometer uncertainty

## Installation

### Prerequisites

- Python 3.8+
- Node.js 18+ (for web frontend)
- Docker & Docker Compose (optional)

### Python Environment

```bash
# Install dependencies
make install
# or
pip install -r requirements.txt

# For development
make install-dev
# or
pip install -r requirements-dev.txt
```

### Web Frontend

```bash
cd web
npm install
```

## Usage

### Command Line

**Process IMU data with both filters:**
```bash
python scripts/run_fusion.py data/sample_imu.csv --filter BOTH --plot
```

**Parameter sweep:**
```bash
python scripts/sweep_parameters.py data/sample_imu.csv --output sweep_results
```

### Python API

```python
from src.filters import ComplementaryFilter, ExtendedKalmanFilter
from src.io import IMULoader

# Load data
data = IMULoader.load_csv("data/sample_imu.csv")

# Complementary filter
comp_filter = ComplementaryFilter(alpha=0.96, dt=1.0/data.sample_rate)
comp_orientations, comp_quaternions = comp_filter.process_batch(
    data.accelerometer,
    data.gyroscope,
    data.timestamps
)

# Extended Kalman Filter
ekf_filter = ExtendedKalmanFilter(dt=1.0/data.sample_rate)
ekf_orientations, ekf_quaternions, ekf_covariances = ekf_filter.process_batch(
    data.accelerometer,
    data.gyroscope,
    data.timestamps
)
```

### REST API

**Start API server:**
```bash
make run-api
# or
uvicorn api.main:app --reload
```

**Endpoints:**
- `POST /upload` - Upload IMU CSV file
- `POST /process/complementary` - Process with complementary filter
- `POST /process/ekf` - Process with EKF
- `POST /compare` - Compare both filters

**Example:**
```bash
curl -X POST "http://localhost:8000/process/complementary" \
  -F "file=@data/sample_imu.csv" \
  -F "alpha=0.96"
```

### Web Dashboard

**Start web frontend:**
```bash
make run-web
# or
cd web && npm start
```

Navigate to `http://localhost:3000` to access the dashboard.

### Docker Compose

**Start all services:**
```bash
make run-docker
# or
docker-compose up --build
```

This starts:
- API server at `http://localhost:8000`
- Web dashboard at `http://localhost:3000`

## Data Format

IMU CSV files should have the following columns:
- `timestamp`: Time in seconds
- `ax, ay, az`: Accelerometer readings in m/s²
- `gx, gy, gz`: Gyroscope readings in rad/s

Example:
```csv
timestamp,ax,ay,az,gx,gy,gz
0.000,0.12,-0.05,-9.78,0.001,-0.002,0.001
0.010,0.11,-0.04,-9.79,0.002,-0.001,0.000
...
```

## Evaluation Metrics

### RMSE (Root Mean Square Error)

Measures the difference between estimated and true orientations:
```
RMSE = sqrt(mean((θ_true - θ_estimated)²))
```

### Orientation Error

Quaternion-based angular error:
```
θ_error = 2 * arccos(|q_true · q_estimated|)
```

### Interpretation

- **Lower RMSE**: Better accuracy
- **Lower orientation error**: Better alignment
- **Stable covariance (EKF)**: Consistent uncertainty estimates

## Benchmark Examples

### Example 1: Static IMU

For a static IMU (gravity only), both filters should converge to near-zero orientation:
- Complementary filter: α = 0.96 typically works well
- EKF: Q = 0.01*I, R = 0.1*I provides good balance

### Example 2: Dynamic Motion

For dynamic motion:
- Complementary filter: Higher α (0.98) for better tracking
- EKF: Tune Q/R based on sensor noise characteristics

### Example 3: Parameter Tuning

Use `sweep_parameters.py` to find optimal parameters:
```bash
python scripts/sweep_parameters.py data/sample_imu.csv \
  --alpha-min 0.90 \
  --alpha-max 0.99 \
  --alpha-steps 20
```

## Testing

```bash
make test
# or
pytest tests/ -v
```

## Development

**Code formatting:**
```bash
make format
```

**Linting:**
```bash
make lint
```

**Clean generated files:**
```bash
make clean
```

## Project Structure Details

### Core Modules

- **`src/filters/complementary.py`**: Complementary filter implementation with bias estimation
- **`src/filters/ekf.py`**: Extended Kalman Filter with quaternion state representation
- **`src/io/imu_loader.py`**: CSV loader with flexible column mapping
- **`src/utils/metrics.py`**: RMSE, MAE, and orientation error calculations
- **`src/utils/visualization.py`**: Matplotlib plotting utilities

### API Endpoints

- **`/upload`**: Validate and parse IMU CSV files
- **`/process/complementary`**: Run complementary filter with configurable α
- **`/process/ekf`**: Run EKF with configurable Q/R matrices
- **`/compare`**: Run both filters and return comparison metrics

### Web Components

- **`OrientationViewer`**: Three.js 3D visualization with playback controls
- **`SignalPlots`**: Plotly.js charts for raw signals and fused orientations
- **`FileUpload`**: Drag-and-drop CSV upload with parameter controls

## Performance Considerations

- **Batch processing**: Use `process_batch()` for efficiency
- **Vectorization**: All operations use NumPy for speed
- **Memory**: Large datasets (>100k samples) may require chunking
- **Real-time**: Single `update()`/`step()` calls are optimized for <1ms latency

## Limitations

- **Yaw drift**: Without magnetometer, yaw estimation drifts over time
- **Gimbal lock**: Euler angles have singularities (quaternions avoid this)
- **Ground truth**: RMSE requires known reference trajectories
- **Sensor calibration**: Assumes calibrated IMU (bias removal included)

## Future Enhancements

- [ ] Magnetometer integration for yaw correction
- [ ] SLAM integration for absolute positioning
- [ ] Real-time streaming API
- [ ] Mobile app for live IMU capture
- [ ] Machine learning-based filter tuning
- [ ] Multi-IMU sensor fusion

## License

MIT License - see LICENSE file for details

## Citation

If you use INVARI in your research, please cite:

```bibtex
@software{invari2024,
  title={INVARI: Sensor Fusion Platform for IMU Data},
  author={Forgingalex},
  year={2024},
  url={https://github.com/Forgingalex/invari}
}
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/Forgingalex/invari/issues)
- Documentation: See README.md and QUICKSTART.md

---

**Built with ❤️ for the sensor fusion community**

