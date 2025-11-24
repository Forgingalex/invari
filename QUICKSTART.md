# Quick Start Guide

Get INVARI up and running in 5 minutes!

## Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/invari.git
cd invari

# Start all services
docker-compose up --build
```

Access:
- Web Dashboard: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Option 2: Local Development

### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start API server
make run-api
# or
uvicorn api.main:app --reload
```

### Frontend Setup

```bash
# Install Node dependencies
cd web
npm install

# Start React dev server
npm start
```

## Test with Sample Data

```bash
# Process sample IMU data
python scripts/run_fusion.py data/sample_imu.csv --filter BOTH --plot

# View results in output/ directory
ls output/
```

## Using the Web Dashboard

1. Navigate to http://localhost:3000
2. Click "Upload IMU Data" and select a CSV file
3. Adjust filter parameters (alpha for complementary filter)
4. Click "Process IMU Data"
5. View 3D orientation visualization and plots

## Using the Python API

```python
from src.filters import ComplementaryFilter
from src.io import IMULoader

# Load data
data = IMULoader.load_csv("data/sample_imu.csv")

# Process
filter_obj = ComplementaryFilter(alpha=0.96, dt=1.0/data.sample_rate)
orientations, quaternions = filter_obj.process_batch(
    data.accelerometer,
    data.gyroscope,
    data.timestamps
)

print(f"Processed {len(orientations)} samples")
```

## Next Steps

- Read the [README.md](README.md) for detailed documentation
- Check out [notebooks/fusion_comparison.ipynb](notebooks/fusion_comparison.ipynb) for analysis examples
- Run tests: `make test`
- Explore parameter tuning: `python scripts/sweep_parameters.py data/sample_imu.csv`

