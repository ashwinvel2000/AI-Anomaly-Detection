# ML Service API

A FastAPI web service for anomaly detection on time-series CSV data using pre-trained ONNX models. The service processes multi-sensor CSV files and runs inference on 10 different models including isolation forest and residual regression models.

## Key Features

- Anomaly detection on time-series sensor data via `/predict/all` endpoint
- Supports 9 ONNX models: delta_temp_open, full_vectors_if, pressure_pair_open, and 6 residual models
- CSV file processing with automatic timestamp joining across multiple sensor streams
- Model retraining capabilities with version management
- Built-in data quality checks and feature engineering
- RESTful API with automatic documentation

## How It Works

Upload CSV files containing sensor data with timestamp and value columns. The service:

1. Reads and joins CSV files by matching timestamps
2. Applies feature engineering (derived features like DeltaTemperature, IsOpen, ToolStateNum)
3. Maps CSV column names to ONNX model input features automatically
4. Runs inference on all available models
5. Returns anomalies with scores and metadata

For residual models, the service compares predictions to actual values using MAD (Median Absolute Deviation) thresholds to detect anomalies.

## Quick Setup

### Requirements
- Python 3.9+
- Dependencies in requirements.txt

### Installation
```bash
pip install -r requirements.txt
```

### Run the server
```bash
python run_dev.py
```

Visit http://127.0.0.1:8000/docs for interactive API documentation.

## File Structure

### Minimal Required Files
```
ml-service/
├── requirements.txt
├── run_dev.py
├── src/
│   ├── __init__.py
│   ├── main.py                    # Core API endpoints
│   ├── config.py                  # Configuration settings
│   ├── schemas.py                 # Data models
│   ├── versioning.py              # Model metadata utilities
│   ├── predictor.py               # Model predictor class
│   ├── trainer.py                 # Training utilities
│   ├── metrics.py                 # Metrics storage
│   └── pipeline/
│       └── csharp_compat/
│           ├── __init__.py
│           ├── registry.py        # ONNX model loader
│           ├── onnx_infer.py      # ONNX inference logic
│           ├── loaders.py         # CSV processing
│           ├── joins.py           # Time-series joining
│           ├── dq.py              # Data quality checks
│           ├── events.py          # Anomaly events
│           └── features.py        # Feature engineering
└── models/                        # ONNX model files
    ├── delta_temp_open.onnx
    ├── full_vectors_if.onnx
    ├── pressure_pair_open.onnx
    ├── residual_battery.onnx
    ├── residual_downP.onnx
    ├── residual_downT.onnx
    ├── residual_upP.onnx
    ├── residual_upT.onnx
    ├── target_pos_residual.onnx
    └── residual_mad.json           # MAD thresholds
```

## CSV Data Format

Each CSV file should contain two columns:
- Column 1: Timestamp (ISO 8601 format recommended)
- Column 2: Sensor value with descriptive header

Example:
```csv
Timestamp,Battery Voltage (Volts)
2025-08-11T12:00:00Z,12.4
2025-08-11T12:01:00Z,12.6
```

### Supported Sensor Types
- Battery Voltage (Volts)
- Annulus Pressure (PSI) / Upstream Pressure
- Bore Pressure (PSI) / Downstream Pressure  
- Annulus Temperature (Deg.C) / Upstream Temperature
- Bore Temperature (Deg.C) / Downstream Temperature
- Position (%) / Choke Position
- Tool State
- Target Position

## Configuration

The service uses environment variables or defaults for configuration:

- `EXTERNAL_MODEL_DIR`: Directory for production ONNX models (default: wwwroot/models)
- `REGISTRY_DIR`: Directory for model training registry (default: models)
- `ALLOW_PROMOTE`: Enable model promotion (default: false)

## Model Information

### Isolation Forest Models
- delta_temp_open.onnx
- full_vectors_if.onnx  
- pressure_pair_open.onnx

### Residual Regression Models
- residual_battery.onnx (Battery Voltage prediction)
- residual_downP.onnx (Downstream Pressure prediction)
- residual_upP.onnx (Upstream Pressure prediction)
- residual_downT.onnx (Downstream Temperature prediction)
- residual_upT.onnx (Upstream Temperature prediction)
- target_pos_residual.onnx (Choke Position from Target Position)

Residual models use XGBoost regression with MAD-based anomaly thresholds stored in residual_mad.json.

## API Usage Examples

### Run All Models
```bash
curl -X POST "http://localhost:8000/predict/all" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sensor1.csv" \
  -F "files=@sensor2.csv"
```

### Check Available Models
```bash
curl -X GET "http://localhost:8000/models/available"
```

### Retrain Specific Model
```bash
curl -X POST "http://localhost:8000/retrain/residual_battery" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@training_data.csv"
```

## Development Notes

The service automatically discovers available ONNX models in the models directory, so adding new models is as simple as placing the .onnx file in the models folder. The system handles feature mapping between CSV column names and ONNX model input features automatically.

Model loading is happens on first request for faster startup times. The startup process takes about 2-3 seconds instead of pre-loading all models.

## Common Issues

- Missing python-multipart: `pip install python-multipart`
- Model not found errors: Ensure ONNX files exist in models directory
- CSV format issues: Verify timestamp format and column headers match expected sensor names
- Memory issues with large files: Process data in smaller batches

