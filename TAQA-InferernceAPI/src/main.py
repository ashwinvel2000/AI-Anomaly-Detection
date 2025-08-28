from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .schemas import BatchPredictRequest, RetrainConfig, StatusResponse
from .predictor import predictor
from .trainer import trainer, TrainJob
from .versioning import read_pointer, load_meta, compute_schema_hash, model_dir, save_meta
from .metrics import metrics_store
from .config import EXTERNAL_MODEL_DIR, REGISTRY_DIR, ALLOW_PROMOTE
import hashlib

# Residual model target mapping (ONNX feature names)
RESIDUAL_TARGETS = {
    "residual_battery": "Battery-Voltage",
    "residual_downP": "Downstream-Pressure",
    "residual_upP": "Upstream-Pressure",
    "residual_downT": "Downstream-Temperature",
    "residual_upT": "Upstream-Temperature",
    "target_pos_residual": "Choke-Position",  # special case (predicted Choke-Position from Target-Position + ToolStateNum)
}

IF_FULL_FEATS = [
    "Battery-Voltage",
    "Upstream-Pressure",
    "Downstream-Pressure",
    "Downstream-Upstream-Difference",
    "Upstream-Temperature",
    "Downstream-Temperature",
    "Choke-Position",
]
LOG_FEATURES = ["Upstream-Pressure", "Downstream-Pressure", "Downstream-Upstream-Difference"]

def _rebuild_model_manifest(base_dir: Path | None = None) -> None:
    """Rebuild a model_manifest.json similar to notebook output.
    Scans provided directory (defaults to EXTERNAL_MODEL_DIR if exists else REGISTRY_DIR) for *.onnx and residual_mad.json."""
    target_dir = base_dir or (EXTERNAL_MODEL_DIR if EXTERNAL_MODEL_DIR.exists() else REGISTRY_DIR)
    try:
        files = list(target_dir.glob("*.onnx"))
        manifest = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "files": {}
        }
        for fp in sorted(files):
            try:
                h = hashlib.sha256(fp.read_bytes()).hexdigest()[:12]
                manifest["files"][fp.name] = h
            except Exception:
                continue
        mad_path = target_dir / "residual_mad.json"
        if mad_path.exists():
            manifest["files"][mad_path.name] = hashlib.sha256(mad_path.read_bytes()).hexdigest()[:12]
        (target_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    except Exception as e:
        print(f"WARN: Failed to rebuild model_manifest.json: {e}")

def _update_residual_mad(stats: Dict[str, Dict[str, float]]) -> None:
    """Merge provided MAD stats into registry root residual_mad.json."""
    mad_path = REGISTRY_DIR / "residual_mad.json"
    try:
        current = json.loads(mad_path.read_text()) if mad_path.exists() else {}
    except Exception:
        current = {}
    current.update(stats)
    mad_path.write_text(json.dumps(current, indent=2))
    # also try to mirror into EXTERNAL_MODEL_DIR if exists (optional)
    try:
        if EXTERNAL_MODEL_DIR.exists():
            (EXTERNAL_MODEL_DIR / "residual_mad.json").write_text(json.dumps(current, indent=2))
    except Exception as e:
        print(f"WARN: Could not mirror residual_mad.json to external dir: {e}")
    _rebuild_model_manifest()  # refresh manifest after update

def _retrain_residual(model_name: str, df: pd.DataFrame, cfg: RetrainConfig, sess, files, dq):
    """Train residual regression model replicating notebook logic (XGBRegressor + MAD).
    Returns JSONResponse."""
    try:
        from xgboost import XGBRegressor  # type: ignore
        import onnxmltools  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Residual retrain dependencies missing: {e}. Install xgboost and onnxmltools.")

    # Apply same feature derivations
    df = _apply_derived_features(df.copy())

    # Map CSV columns to ONNX names for convenience
    all_possible_mappings = {
        'Bore Pressure (PSI)': 'Downstream-Pressure',
        'Annulus Pressure (PSI)': 'Upstream-Pressure',
        'Bore Temperature (Deg.C)': 'Downstream-Temperature',
        'Annulus Temperature (Deg.C)': 'Upstream-Temperature',
        'Position (%)': 'Choke-Position',
        'Battery Voltage (Volts)': 'Battery-Voltage',
        'Bore-Annulus Difference (PSI)': 'Downstream-Upstream-Difference',
        'Target Position': 'Target-Position',
        'Tool State': 'ToolStateNum'
    }
    df_onnx = df.rename(columns=all_possible_mappings)

    # Determine target and feature list
    if model_name == "target_pos_residual":
        target_feature = RESIDUAL_TARGETS[model_name]  # Choke-Position
        feature_list = ["Target-Position", "ToolStateNum"]
        mad_key = "Target-Position"
    else:
        target_feature = RESIDUAL_TARGETS[model_name]
        feature_list = [f for f in IF_FULL_FEATS if f != target_feature]
        mad_key = target_feature

    # Validate presence
    missing = [c for c in feature_list + [target_feature] if c not in df_onnx.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns for residual retrain: {missing}")

    X_df = df_onnx[feature_list].copy()
    # Apply log1p to LOG_FEATURES except if column is the target or not in X
    for c in LOG_FEATURES:
        if c in X_df.columns and c != target_feature:
            X_df[c] = np.log1p(np.clip(X_df[c], 0, None))
    X = X_df.astype(np.float32).values
    y = df_onnx[target_feature].astype(np.float32).values

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='hist',
        random_state=cfg.random_state,
    ).fit(X, y)

    # Export ONNX
    try:
        onnx_model = onnxmltools.convert_xgboost(
            model,
            name=model_name,
            initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=12,
        )
        meta = onnx_model.metadata_props.add()
        meta.key, meta.value = "feature_names", ",".join(feature_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX export failed for residual model: {e}")

    # Version directory
    from .versioning import model_dir, save_meta, compute_schema_hash
    tdir = model_dir(model_name)
    os.makedirs(tdir, exist_ok=True)
    onnx_path = tdir / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    # Residual stats
    y_pred = model.predict(X).astype(np.float32)
    resid = np.abs(y - y_pred)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    cutoff = float(np.percentile(resid, 99))
    _update_residual_mad({mad_key: {"mad": mad, "cutoff": cutoff}})

    # Metadata
    s_hash = compute_schema_hash(pd.DataFrame(X_df))
    job_id = uuid.uuid4().hex
    meta = {
        "model_name": model_name,
        "model_version": tdir.name,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "features": feature_list,
        "target_feature": target_feature,
        "mad_key": mad_key,
        "schema_hash": s_hash,
        "n_rows": int(X.shape[0]),
        "retrained_type": "xgb_regression_residual",
        "job_id": job_id,
        "data_quality": dq,
        "onnx_model": str(onnx_path),
        "residual_stats": {"mad": mad, "cutoff": cutoff},
    }
    save_meta(tdir, meta)
    _rebuild_model_manifest()

    return JSONResponse(status_code=200, content={
        "job_id": job_id,
        "model_version": tdir.name,
        "model_name": model_name,
        "trained_at": meta["trained_at"],
        "training_samples": int(X.shape[0]),
        "features_mapped": feature_list,
        "target_feature": target_feature,
        "mad_key": mad_key,
        "mad": mad,
        "cutoff": cutoff,
        "saved_to": str(tdir),
        "status": "completed"
    })


app = FastAPI(title="ML Service")

# CORS default open; adjust as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

started_at = datetime.utcnow()


@app.middleware("http")
async def add_logging(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = (time.perf_counter() - start) * 1000
        path = request.url.path
        method = request.method
        print(json.dumps({
            "level": "info",
            "request_id": rid,
            "route": path,
            "method": method,
            "duration_ms": round(dur_ms, 2),
        }))


@app.on_event("startup")
async def _startup():
    print(" Starting ML Service...")
    
    # Skip problematic startup code for now
    print("  Skipping predictor.load_current() - may cause issues with some endpoints")
    print("  Skipping trainer.start() - retrain endpoints should still work")
    
    # Quick ONNX registry check (skip pre-warming for faster startup)
    try:
        from .pipeline.csharp_compat.registry import registry
        print(" ONNX registry imported successfully")
        print("  Skipping ONNX model pre-warming for faster startup")
                
    except Exception as e:
        print(f" ONNX registry issue: {e}")
    
    print(" Startup complete - CSV processing pipeline is working!")


@app.get("/status", response_model=StatusResponse)
async def status():
    cur = read_pointer()
    meta = load_meta(cur) if cur else None
    return StatusResponse(
        status="ok",
        model_name=(meta or {}).get("model_name") if meta else None,
        model_version=(meta or {}).get("model_version") if meta else None,
        started_at=started_at,
    )


def _df_from_json_rows(req: BatchPredictRequest) -> pd.DataFrame:
    rows = [r.root for r in req.rows]
    return pd.DataFrame(rows)


def _validate_schema(df: pd.DataFrame) -> Optional[str]:
    meta_dir = read_pointer()
    if not meta_dir:
        return None
    meta = load_meta(meta_dir)
    if not meta:
        return None
    expected_hash = meta.get("schema_hash")
    got_hash = compute_schema_hash(df)
    if expected_hash and got_hash != expected_hash:
        return f"Schema hash mismatch"
    expected_cols = set(meta.get("features", meta.get("schema", [])))
    got_cols = set(df.columns)
    if expected_cols and expected_cols != got_cols:
        return f"Columns mismatch. expected={sorted(expected_cols)} got={sorted(got_cols)}"
    return None


def _read_multi_csv(files: List[UploadFile]):
    from .pipeline.csharp_compat.loaders import read_two_col_csv
    streams = {}
    for f in files:
        content = f.file.read()
        df = pd.read_csv(io.BytesIO(content))
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail=f"CSV {f.filename} must have at least two columns")
        tag = str(df.columns[1])
        # re-read strictly via two-col reader
        f.file.seek(0)
        series = list(read_two_col_csv(io.BytesIO(f.file.read())))
        streams[tag] = series
    return streams


def _build_joined_rows(streams):
    from .pipeline.csharp_compat.joins import inner_join_identical_timestamps
    return list(inner_join_identical_timestamps(streams))


def _dq_report(rows):
    from .pipeline.csharp_compat.dq import get_breaches, scan_window, FLAT_WINDOW
    if not rows:
        return {"states": {}, "breaches": []}
    win = rows[-FLAT_WINDOW:] if len(rows) >= FLAT_WINDOW else rows[:]
    states = scan_window(win)
    breaches = [e.__dict__ for e in get_breaches(win)]
    return {"states": states, "breaches": breaches}


def _rows_to_df(rows):
    if not rows:
        return pd.DataFrame()
    data = [r.Values for r in rows]
    df = pd.DataFrame(data)
    return df


@app.get("/models/available")
async def list_available_models():
    """List all available ONNX models"""
    models_dir = EXTERNAL_MODEL_DIR if EXTERNAL_MODEL_DIR.exists() else REGISTRY_DIR
    available_models = []
    
    try:
        for model_file in models_dir.glob("*.onnx"):
            model_name = model_file.stem
            available_models.append(model_name)
        available_models.sort()
    except Exception as e:
        return {"error": f"Failed to discover models: {e}", "models": []}
    
    return {
        "models_directory": str(models_dir),
        "external_model_dir_exists": EXTERNAL_MODEL_DIR.exists(),
        "registry_dir": str(REGISTRY_DIR),
        "available_models": available_models,
        "total_count": len(available_models)
    }


@app.post("/predict/all")
async def predict_all_models(files: List[UploadFile] = File(...)):
    """Run predictions on all available models with the same dataset"""
    
    # Dynamically discover available models from the actual models directory
    available_models = []
    models_dir = EXTERNAL_MODEL_DIR if EXTERNAL_MODEL_DIR.exists() else REGISTRY_DIR
    
    try:
        for model_file in models_dir.glob("*.onnx"):
            model_name = model_file.stem  # Remove .onnx extension
            available_models.append(model_name)
        available_models.sort()  # Sort for consistent ordering
        print(f"DEBUG: Discovered {len(available_models)} available models: {available_models}")
    except Exception as e:
        print(f"ERROR: Failed to discover models: {e}")
        # Fallback to known working models if discovery fails
        available_models = [
            "delta_temp_open", "full_vectors_if", "pressure_pair_open",
            "residual_battery", "residual_downP", "residual_downT", "residual_upP", 
            "residual_upT", "target_pos_residual"
        ]
    
    results = {}
    errors = {}
    
    for model_name in available_models:
        try:
            # Reset file positions for each model
            for f in files:
                try:
                    f.file.seek(0)
                except Exception:
                    pass
            
            # Call the individual predict function with model_name
            response = await predict(files, None, model_name, False, 50, 10)
            if hasattr(response, 'body'):
                import json
                results[model_name] = json.loads(response.body.decode())
            else:
                results[model_name] = response
        except Exception as e:
            errors[model_name] = str(e)
    
    # Print summary of anomaly counts
    print("=" * 80)
    print("ANOMALY COUNT SUMMARY:")
    for model_name in available_models:
        if model_name in results:
            anomaly_count = len(results[model_name].get("anomalies", []))
            print(f"   {model_name:<20}: {anomaly_count:>4} anomalies")
        elif model_name in errors:
            print(f"   {model_name:<20}: ERROR - {errors[model_name]}")
    print("=" * 80)

    return JSONResponse(status_code=200, content={
        "status": "completed",
        "predictions": results,
        "errors": errors,
        "successful_models": list(results.keys()),
        "failed_models": list(errors.keys()),
        "total_models": len(available_models),
        "success_count": len(results),
        "error_count": len(errors)
    })


@app.post("/predict/datahub")
async def predict_from_datahub(
    streams: Dict[str, List[Dict[str, Any]]],
    model_name: str = "choke_position"
):
    """Run prediction using DataHub streams data (time-series streams that need joining)"""
    import pandas as pd
    from datetime import datetime
    
    # Convert DataHub streams to DataFrame (similar to MultiCsvJoiner.cs logic)
    print(f"DEBUG DATAHUB: Received {len(streams)} streams for model '{model_name}'")
    
    # DataHub streams come as: 
    # {
    #   "Battery-Voltage": [{"Voltage": 12.5, "Timestamp": "2025-08-12T10:30:00Z"}, ...],
    #   "Upstream-Pressure": [{"Pressure": 150.2, "Timestamp": "2025-08-12T10:30:00Z"}, ...],
    #   ...
    # }
    
    # Convert each stream to DataFrame and join by timestamp
    dataframes = {}
    
    for stream_name, stream_data in streams.items():
        if not stream_data:
            continue
            
        df_stream = pd.DataFrame(stream_data)
        
        # Standardize timestamp column
        if 'Timestamp' in df_stream.columns:
            df_stream['Timestamp'] = pd.to_datetime(df_stream['Timestamp'])
        elif 'timestamp' in df_stream.columns:
            df_stream['Timestamp'] = pd.to_datetime(df_stream['timestamp'])
            df_stream = df_stream.drop('timestamp', axis=1)
        else:
            raise HTTPException(status_code=400, detail=f"No timestamp found in stream '{stream_name}'")
        
        # Map DataHub property names to our expected column names
        property_mapping = {
            'Voltage': f'{stream_name}',
            'Pressure': f'{stream_name}', 
            'Temperature': f'{stream_name}',
            'Position': f'{stream_name}',
            'State': f'{stream_name}',
            'Cycles': f'{stream_name}',
            'Runtime': f'{stream_name}',
            'Resets': f'{stream_name}',
            'Lifetime': f'{stream_name}'
        }
        
        # Rename the value column to match our naming convention
        for prop_name, new_name in property_mapping.items():
            if prop_name in df_stream.columns:
                df_stream = df_stream.rename(columns={prop_name: new_name})
                break
        
        dataframes[stream_name] = df_stream
        print(f"DEBUG DATAHUB: Stream '{stream_name}' has {len(df_stream)} rows")
    
    if not dataframes:
        raise HTTPException(status_code=400, detail="No valid streams provided")
    
    # Join all dataframes by timestamp (similar to MultiCsvJoiner logic)
    result_df = None
    for stream_name, df_stream in dataframes.items():
        if result_df is None:
            result_df = df_stream
        else:
            result_df = pd.merge(result_df, df_stream, on='Timestamp', how='outer', suffixes=('', f'_{stream_name}'))
    
    # Sort by timestamp
    result_df = result_df.sort_values('Timestamp')
    
    print(f"DEBUG DATAHUB: Joined data has {len(result_df)} rows and columns: {list(result_df.columns)}")
    
    # Apply DataHub stream name to CSV column name mapping
    datahub_to_csv_mapping = {
        'Battery-Voltage': 'Battery Voltage (Volts)',
        'Upstream-Pressure': 'Annulus Pressure (PSI)',
        'Downstream-Pressure': 'Bore Pressure (PSI)', 
        'Upstream-Temperature': 'Annulus Temperature (Deg.C)',
        'Downstream-Temperature': 'Bore Temperature (Deg.C)',
        'Choke-Position': 'Position (%)',
        'Tool-State': 'Tool State',
        'Target-Position': 'Target Position',
        'Downstream-Upstream-Difference': 'Bore-Annulus Difference (PSI)'
    }
    
    # Rename columns to match CSV format
    rename_dict = {}
    for datahub_name, csv_name in datahub_to_csv_mapping.items():
        if datahub_name in result_df.columns:
            rename_dict[datahub_name] = csv_name
    
    result_df = result_df.rename(columns=rename_dict)
    
    # Drop timestamp as it's not needed for prediction
    if 'Timestamp' in result_df.columns:
        result_df = result_df.drop('Timestamp', axis=1)
    
    print(f"DEBUG DATAHUB: Final DataFrame columns: {list(result_df.columns)}")
    
    # Apply validation and preprocessing (same as CSV processing)
    err = _validate_schema(result_df)
    if err:
        raise HTTPException(status_code=400, detail=err)
    
    # Apply derived feature calculations and feature mapping
    result_df = _apply_derived_features(result_df)
    result_df = _apply_feature_mapping(result_df, model_name)
    
    # Use global predictor (ONNX-based routes handle session separately)
    if predictor.model is None:
        predictor.load_current()
    if predictor.model is None:
        raise HTTPException(status_code=500, detail="Predictor has no loaded model (train/promote first)")
    preds = predictor.predict(result_df)
    meta = predictor.meta or {}
    
    return JSONResponse({
        "predictions": preds,
        "model_name": model_name,
        "model_version": meta.get("model_version"),
        "data_source": "datahub",
        "streams_processed": list(streams.keys()),
        "rows_processed": len(result_df),
        "joined_data_shape": result_df.shape
    })


@app.post("/retrain/datahub")
async def retrain_from_datahub(
    streams: Dict[str, List[Dict[str, Any]]],
    model_name: str,
    config: Optional[str] = None
):
    """Retrain model using DataHub streams data"""
    import pandas as pd
    
    print(f"DEBUG DATAHUB RETRAIN: Processing {len(streams)} streams for model '{model_name}'")
    
    # Use same joining logic as prediction
    dataframes = {}
    
    for stream_name, stream_data in streams.items():
        if not stream_data:
            continue
            
        df_stream = pd.DataFrame(stream_data)
        
        # Standardize timestamp
        if 'Timestamp' in df_stream.columns:
            df_stream['Timestamp'] = pd.to_datetime(df_stream['Timestamp'])
        elif 'timestamp' in df_stream.columns:
            df_stream['Timestamp'] = pd.to_datetime(df_stream['timestamp'])
            df_stream = df_stream.drop('timestamp', axis=1)
        
        # Map property names
        property_mapping = {
            'Voltage': f'{stream_name}',
            'Pressure': f'{stream_name}', 
            'Temperature': f'{stream_name}',
            'Position': f'{stream_name}',
            'State': f'{stream_name}',
            'Cycles': f'{stream_name}',
            'Runtime': f'{stream_name}',
            'Resets': f'{stream_name}',
            'Lifetime': f'{stream_name}'
        }
        
        for prop_name, new_name in property_mapping.items():
            if prop_name in df_stream.columns:
                df_stream = df_stream.rename(columns={prop_name: new_name})
                break
        
        dataframes[stream_name] = df_stream
    
    # Join all dataframes by timestamp
    result_df = None
    for stream_name, df_stream in dataframes.items():
        if result_df is None:
            result_df = df_stream
        else:
            result_df = pd.merge(result_df, df_stream, on='Timestamp', how='outer', suffixes=('', f'_{stream_name}'))
    
    result_df = result_df.sort_values('Timestamp')
    
    # Map to CSV column names
    datahub_to_csv_mapping = {
        'Battery-Voltage': 'Battery Voltage (Volts)',
        'Upstream-Pressure': 'Annulus Pressure (PSI)',
        'Downstream-Pressure': 'Bore Pressure (PSI)', 
        'Upstream-Temperature': 'Annulus Temperature (Deg.C)',
        'Downstream-Temperature': 'Bore Temperature (Deg.C)',
        'Choke-Position': 'Position (%)',
        'Tool-State': 'Tool State',
        'Target-Position': 'Target Position',
        'Downstream-Upstream-Difference': 'Bore-Annulus Difference (PSI)'
    }
    
    rename_dict = {}
    for datahub_name, csv_name in datahub_to_csv_mapping.items():
        if datahub_name in result_df.columns:
            rename_dict[datahub_name] = csv_name
    
    result_df = result_df.rename(columns=rename_dict)
    
    # Drop timestamp
    if 'Timestamp' in result_df.columns:
        result_df = result_df.drop('Timestamp', axis=1)
    
    # Apply preprocessing and train model (same as other retrain methods)
    err = _validate_schema(result_df)
    if err:
        raise HTTPException(status_code=400, detail=err)
    
    result_df = _apply_derived_features(result_df)
    training_df = _apply_feature_mapping(result_df, model_name)
    
    # Parse config and train
    cfg: RetrainConfig
    if config and config.strip():
        try:
            cfg = RetrainConfig.parse_raw(config)
        except Exception:
            cfg = RetrainConfig()
    else:
        cfg = RetrainConfig()
    
    from sklearn.ensemble import IsolationForest
    import joblib
    
    contamination = cfg.contamination if cfg.contamination is not None else 'auto'
    
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        contamination=contamination,
        random_state=cfg.random_state,
    )
    
    model.fit(training_df)
    
    # Save model
    from .versioning import model_dir, save_meta
    tdir = model_dir(model_name)
    os.makedirs(tdir, exist_ok=True)
    
    joblib.dump(model, tdir / "model.pkl")
    
    # Save metadata
    meta = {
        "model_name": model_name,
        "model_version": tdir.name,
        "trained_at": datetime.now().isoformat(),
        "data_source": "datahub",
        "training_samples": len(training_df),
        "streams_processed": list(streams.keys())
    }
    save_meta(tdir, meta)
    
    return JSONResponse(status_code=200, content={
        "job_id": f"datahub_{int(datetime.now().timestamp())}",
        "model_name": model_name,
        "model_version": tdir.name,
        "training_samples": len(training_df),
        "data_source": "datahub",
        "streams_processed": list(streams.keys()),
        "status": "completed"
    })


@app.post("/predict/json")
async def predict_from_json(
    data: List[Dict[str, Any]],
    model_name: str = "choke_position"
):
    """Run prediction using JSON data (for AVEVA Datahub integration)"""
    import pandas as pd
    
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    
    # Apply validation and preprocessing
    err = _validate_schema(df)
    if err:
        raise HTTPException(status_code=400, detail=err)
    
    # Apply derived feature calculations and feature mapping (same as CSV processing)
    df = _apply_derived_features(df)
    
    # Ensure predictor loaded
    if predictor.model is None:
        predictor.load_current()
    if predictor.model is None:
        raise HTTPException(status_code=500, detail="Predictor has no loaded model (train/promote first)")
    # Apply feature mapping
    df = _apply_feature_mapping(df, model_name)
    preds = predictor.predict(df)
    meta = predictor.meta or {}
    
    return JSONResponse({
        "predictions": preds,
        "model_name": model_name,
        "model_version": meta.get("model_version"),
        "data_source": "json",
        "rows_processed": len(df)
    })


@app.post("/retrain/json")
async def retrain_from_json(
    data: List[Dict[str, Any]],
    model_name: str,
    config: Optional[str] = None
):
    """Retrain model using JSON data (for AVEVA Datahub integration)"""
    import pandas as pd
    
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    
    # Apply same processing as file-based retrain
    err = _validate_schema(df)
    if err:
        raise HTTPException(status_code=400, detail=err)
    
    # Apply derived features and feature mapping
    df = _apply_derived_features(df)
    training_df = _apply_feature_mapping(df, model_name)
    
    # Parse config
    cfg: RetrainConfig
    if config and config.strip():
        try:
            cfg = RetrainConfig.parse_raw(config)
        except Exception:
            cfg = RetrainConfig()
    else:
        cfg = RetrainConfig()
    
    # Train model (same logic as file-based retrain)
    from sklearn.ensemble import IsolationForest
    import joblib
    
    contamination = cfg.contamination if cfg.contamination is not None else 'auto'
    
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        contamination=contamination,
        random_state=cfg.random_state,
    )
    
    model.fit(training_df)
    
    # Save model
    from .versioning import model_dir, save_meta
    tdir = model_dir(model_name)
    os.makedirs(tdir, exist_ok=True)
    
    joblib.dump(model, tdir / "model.pkl")
    
    # Save metadata
    meta = {
        "model_name": model_name,
        "model_version": tdir.name,
        "trained_at": datetime.now().isoformat(),
        "data_source": "json",
        "training_samples": len(training_df)
    }
    save_meta(tdir, meta)
    
    return JSONResponse(status_code=200, content={
        "job_id": f"json_{int(datetime.now().timestamp())}",
        "model_name": model_name,
        "model_version": tdir.name,
        "training_samples": len(training_df),
        "data_source": "json",
        "status": "completed"
    })


# Helper functions for feature processing
def _apply_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply derived feature calculations using original CSV column names"""
    # DeltaTemperature = Annulus Temperature - Bore Temperature
    if 'Annulus Temperature (Deg.C)' in df.columns and 'Bore Temperature (Deg.C)' in df.columns:
        df['DeltaTemperature'] = df['Annulus Temperature (Deg.C)'] - df['Bore Temperature (Deg.C)']
        print(f"DEBUG: Calculated DeltaTemperature for {len(df)} rows")

    # IsOpen = 1.0 if Position > 10 else 0.0
    if 'Position (%)' in df.columns:
        df['IsOpen'] = (df['Position (%)'] > 10).astype(float)
        print(f"DEBUG: Calculated IsOpen for {len(df)} rows")

    # ToolStateNum = numeric encoding of Tool State
    if 'Tool State' in df.columns:
        mapping = {
            'Closed': 0,
            'Open': 1,
            'Opening': 2,
            'Closing': 3,
            'Unknown': -1,
        }
        # If already numeric just copy; else map strings
        if pd.api.types.is_numeric_dtype(df['Tool State']):
            df['ToolStateNum'] = df['Tool State']
        else:
            df['ToolStateNum'] = df['Tool State'].astype(str).map(mapping).fillna(-1).astype(float)
        print(f"DEBUG: Encoded ToolStateNum for {len(df)} rows")
    
    return df


def _apply_feature_mapping(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Apply feature mapping for specific model"""
    from .pipeline.csharp_compat.registry import registry
    from .pipeline.csharp_compat import onnx_infer as oi
    
    # Load model to get required features
    onnx_file = f"{model_name}.onnx"
    try:
        sess = registry.get_session(onnx_file)
        required_features = set(oi._feature_names(sess))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Feature mapping
    all_possible_mappings = {
        'Bore Pressure (PSI)': 'Downstream-Pressure',
        'Annulus Pressure (PSI)': 'Upstream-Pressure', 
        'Bore Temperature (Deg.C)': 'Downstream-Temperature',
        'Annulus Temperature (Deg.C)': 'Upstream-Temperature',
        'Position (%)': 'Choke-Position',
        'Battery Voltage (Volts)': 'Battery-Voltage',
        'Bore-Annulus Difference (PSI)': 'Downstream-Upstream-Difference',
        'Target Position': 'Target-Position',
        'Tool State': 'ToolStateNum'
    }
    
    # Filter mapping for this model
    model_specific_mapping = {
        csv_name: onnx_name 
        for csv_name, onnx_name in all_possible_mappings.items()
        if onnx_name in required_features and csv_name in df.columns
    }
    
    # Apply mapping
    df_mapped = df.rename(columns=model_specific_mapping)
    
    # Filter to only required features
    available_features = [col for col in required_features if col in df_mapped.columns]
    return df_mapped[available_features]


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    config: str = Form(None),
    model_name: str = Form("iforest"),
    sliding_window: Optional[bool] = Form(False),
    window_size: Optional[int] = Form(50),
    step: Optional[int] = Form(10),
):
    t0 = time.perf_counter()
    try:
        streams = _read_multi_csv(files)
        rows = _build_joined_rows(streams)
        dq = _dq_report(rows)
        df = _rows_to_df(rows)
    finally:
        for f in files:
            try:
                f.file.seek(0)
            except Exception:
                pass
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data provided")
    err = _validate_schema(df)
    if err:
        raise HTTPException(status_code=400, detail=err)
    # Dynamically run ONNX model specified by model_name
    from fastapi import HTTPException
    from .pipeline.csharp_compat.registry import registry
    from .pipeline.csharp_compat import onnx_infer as oi
    from .pipeline.csharp_compat.events import make_residual_event, severity_if
    onnx_file = f"{model_name}.onnx"
    try:
        sess = registry.get_session(onnx_file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in EXTERNAL_MODEL_DIR.")

    # Residual models are handled later in this function (regression + MAD logic)

    anomalies: List[Dict[str, Any]] = []
    
    # Debug: show all rows before filtering
    print(f"DEBUG: Total rows before filtering: {len(rows)}")
    if rows:
        print(f"DEBUG: Sample row keys: {list(rows[0].Values.keys())}")
    
    # Get the features this specific ONNX model expects
    required_features = set(oi._feature_names(sess))
    print(f"DEBUG: ONNX model '{model_name}' requires these features: {required_features}")
    
    # Create comprehensive CSV column name to ONNX feature name mapping
    all_possible_mappings = {
        # CSV column name -> ONNX feature name
        'Bore Pressure (PSI)': 'Downstream-Pressure',
        'Annulus Pressure (PSI)': 'Upstream-Pressure', 
        'Bore Temperature (Deg.C)': 'Downstream-Temperature',
        'Annulus Temperature (Deg.C)': 'Upstream-Temperature',
        'Position (%)': 'Choke-Position',
        'Battery Voltage (Volts)': 'Battery-Voltage',
        'Bore-Annulus Difference (PSI)': 'Downstream-Upstream-Difference',
        'Target Position': 'Target-Position',
        'Tool State': 'ToolStateNum'  # This may need numeric conversion
    }
    
    # Filter the mapping to only include features this model actually needs
    model_specific_mapping = {
        csv_name: onnx_name 
        for csv_name, onnx_name in all_possible_mappings.items() 
        if onnx_name in required_features
    }
    
    print(f"DEBUG: Model-specific feature mapping: {model_specific_mapping}")
    
    # Debug: Show sample Tool State values if this model needs ToolStateNum
    if 'ToolStateNum' in required_features and 'Tool State' in (rows[0].Values.keys() if rows else []):
        sample_tool_states = set()
        for i, row in enumerate(rows[:100]):  # Check first 100 rows
            if 'Tool State' in row.Values:
                sample_tool_states.add(str(row.Values['Tool State']))
        print(f"DEBUG: Sample Tool State values found: {sample_tool_states}")
    
    # Simple row class for mapped data
    class MappedRow:
        def __init__(self, timestamp, values):
            self.Timestamp = timestamp
            self.Values = values
    
    # Apply model-specific feature mapping and calculate derived features
    mapped_rows = []
    for row in rows:
        mapped_values = {}
        
        # First, apply basic feature mapping
        for csv_name, onnx_name in model_specific_mapping.items():
            if csv_name in row.Values:
                value = row.Values[csv_name]
                
                # Special handling for Tool State -> ToolStateNum conversion
                if csv_name == 'Tool State' and onnx_name == 'ToolStateNum':
                    # Convert tool state to numeric value
                    # Common mappings (may need adjustment based on your data)
                    tool_state_mapping = {
                        'Closed': 0,
                        'Open': 1,
                        'Opening': 2,
                        'Closing': 3,
                        'Unknown': -1,
                        # Add more mappings as needed
                    }
                    
                    if isinstance(value, str):
                        mapped_value = tool_state_mapping.get(value, -1)  # Default to -1 for unknown states
                        print(f"DEBUG: Converting Tool State '{value}' to ToolStateNum {mapped_value}")
                    else:
                        # If it's already numeric, use as-is
                        mapped_value = float(value)
                    
                    mapped_values[onnx_name] = mapped_value
                else:
                    # Regular mapping - just copy the value
                    mapped_values[onnx_name] = value
        
        # Calculate derived features (matching C# logic exactly)
        # 1. DeltaTemperature = Upstream-Temperature - Downstream-Temperature
        if ('DeltaTemperature' in required_features):
            # Check if we have the required source features
            has_upstream_temp = False
            has_downstream_temp = False
            upstream_temp_value = None
            downstream_temp_value = None
            
            # Look for mapped upstream temperature
            if 'Upstream-Temperature' in mapped_values:
                has_upstream_temp = True
                upstream_temp_value = mapped_values['Upstream-Temperature']
            elif 'Annulus Temperature (Deg.C)' in row.Values:
                has_upstream_temp = True
                upstream_temp_value = row.Values['Annulus Temperature (Deg.C)']
            
            # Look for mapped downstream temperature  
            if 'Downstream-Temperature' in mapped_values:
                has_downstream_temp = True
                downstream_temp_value = mapped_values['Downstream-Temperature']
            elif 'Bore Temperature (Deg.C)' in row.Values:
                has_downstream_temp = True
                downstream_temp_value = row.Values['Bore Temperature (Deg.C)']
            
            if has_upstream_temp and has_downstream_temp:
                delta_temp = upstream_temp_value - downstream_temp_value
                mapped_values['DeltaTemperature'] = delta_temp
        
        # 2. IsOpen = 1.0 if Choke-Position > 10, else 0.0
        if 'IsOpen' in required_features and 'Choke-Position' in mapped_values:
            choke_position = mapped_values['Choke-Position']
            is_open = 1.0 if choke_position > 10 else 0.0
            mapped_values['IsOpen'] = is_open
            print(f"DEBUG: Calculated IsOpen = {is_open} (Choke-Position = {choke_position})")
        
        # Only create mapped row if we have at least some required features
        if mapped_values:
            mapped_row = MappedRow(
                timestamp=row.Timestamp,
                values=mapped_values
            )
            mapped_rows.append(mapped_row)
    
    available_features = set(mapped_rows[0].Values.keys()) if mapped_rows else set()
    print(f"DEBUG: Mapped {len(mapped_rows)} rows with features: {available_features}")
    print(f"DEBUG: Missing features for this model: {required_features - available_features}")
    print(f"DEBUG: Extra features (not needed): {available_features - required_features}")
    
    # Skip the filter_by_features step since we already have correctly mapped features
    # The mapped_rows should contain exactly the features the ONNX model needs
    ok_rows = mapped_rows
    
    print(f"DEBUG: Using mapped rows directly: {len(ok_rows)}")
    
    # If model_name indicates a residual model, run regression + MAD threshold
    if 'residual' in model_name.lower():
        print(f"DEBUG: Using {len(mapped_rows)} mapped rows for residual model")
        
        if len(mapped_rows) == 0:
            print("DEBUG: No rows available for processing after mapping")
        else:
            try:
                yhat = oi.run_regression(sess, mapped_rows)
                print(f"DEBUG: Successfully ran regression, got {len(yhat)} predictions")
            except Exception as e:
                print(f"DEBUG: Failed to run regression on mapped rows: {e}")
                # Fallback to original approach
                ok_rows = oi.filter_by_features(sess, rows)
                print(f"DEBUG: Fallback - rows after ONNX feature filtering: {len(ok_rows)}")
                if len(ok_rows) == 0:
                    print("DEBUG: No rows match ONNX model features even in fallback")
                else:
                    yhat = oi.run_regression(sess, ok_rows)
                    mapped_rows = ok_rows
            
            # Only proceed if we have predictions
            if len(mapped_rows) > 0 and 'yhat' in locals() and len(yhat) > 0:
                # Map model names to observed tags and MAD cutoff keys (matching C# exactly)
                model_to_observed_and_cutoff = {
                    'residual_battery': ('Battery-Voltage', 'Battery-Voltage'),
                    'residual_upP': ('Upstream-Pressure', 'Upstream-Pressure'),
                    'residual_downP': ('Downstream-Pressure', 'Downstream-Pressure'),
                    'residual_upT': ('Upstream-Temperature', 'Upstream-Temperature'),
                    'residual_downT': ('Downstream-Temperature', 'Downstream-Temperature'),
                    'target_pos_residual': ('Choke-Position', 'Target-Position')
                }
                
                # Map ONNX feature names back to CSV column names for observed values
                onnx_to_csv = {
                    'Battery-Voltage': 'Battery Voltage (Volts)',
                    'Upstream-Pressure': 'Annulus Pressure (PSI)',
                    'Downstream-Pressure': 'Bore Pressure (PSI)',
                    'Upstream-Temperature': 'Annulus Temperature (Deg.C)',
                    'Downstream-Temperature': 'Bore Temperature (Deg.C)',
                    'Choke-Position': 'Position (%)'
                }
                
                observed_tag, cutoff_key = model_to_observed_and_cutoff.get(model_name, (None, None))
                
                # Debug: print mapping info
                print(f"DEBUG: Model {model_name}, observed_tag: {observed_tag}, cutoff_key: {cutoff_key}")
                
                if observed_tag and cutoff_key:
                    cut = registry.residual_mad.get(cutoff_key, {}).get("cutoff", float("inf"))
                    print(f"DEBUG: MAD cutoff for {cutoff_key}: {cut}")
                    print(f"DEBUG: Available MAD keys: {list(registry.residual_mad.keys())}")
                    
                    # Debug: show some prediction values
                    if len(yhat) > 0:
                        print(f"DEBUG: Sample predictions: {yhat[:5]}")
                    
                    # Get CSV column name for observed value
                    csv_column = onnx_to_csv.get(observed_tag)
                    if not csv_column:
                        print(f"DEBUG: No CSV mapping found for observed_tag {observed_tag}")
                    else:
                        anomaly_count = 0
                        for i, (orig_row, mapped_row, ph) in enumerate(zip(rows, mapped_rows, yhat)):
                            # Get observed value from original CSV data
                            obs = orig_row.Values.get(csv_column)
                            if obs is None:
                                continue
                                
                            resid = abs(obs - float(ph))
                            
                            if resid > cut:
                                anomaly_count += 1
                                ae = make_residual_event(orig_row.Timestamp, observed_tag, resid, float(ph), float(obs), cut)
                                anomalies.append({**ae.__dict__, "Timestamp": orig_row.Timestamp.isoformat()})
                        
                        print(f"DEBUG: Found {anomaly_count} anomalies out of {len(mapped_rows)} rows")
                else:
                    print(f"DEBUG: No mapping found for model {model_name}, skipping residual calculation")
    else:
        # Default: treat as IF or other anomaly model
        labels, scores = oi.run_if(sess, ok_rows)
        for i, (lab, sc) in enumerate(zip(labels, scores)):
            if lab == -1:
                anomalies.append({
                    "Timestamp": ok_rows[i].Timestamp.isoformat(),
                    "Detector": model_name,
                    "RawValue": 0.0,
                    "Score": float(sc),
                    "Threshold": 0.0,
                    "Severity": severity_if(model_name, float(sc)),
                })

    meta = predictor.meta or {}
    # Internal model prediction (IsolationForest) via registry interface
    internal_preds = None
    internal_scores = None
    try:
        if predictor.model is not None:
            if sliding_window:
                sw = predictor.model.sliding_window_predict(df, window_size or 50, step or 10)
                internal_preds = sw
            else:
                p = predictor.predict(df)
                internal_preds = p
                try:
                    internal_scores = predictor.model.score_samples(df)
                except Exception:
                    internal_scores = None
    except Exception:
        internal_preds = None
    t2 = time.perf_counter()
    return JSONResponse({
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "preprocess": {
            "dq_report": dq,
            "stats": {"n_rows": len(rows), "n_tags": len(streams)},
        },
        "model_version": meta.get("model_version"),
        "internal_model": {
            "name": predictor.model.model_name if predictor.model else None,
            "predictions": internal_preds,
            "scores": list(internal_scores) if internal_scores is not None else None,
        },
        "timings": {"total_ms": round((t2 - t0) * 1000, 2)},
    })


@app.post("/predict_json")
async def predict_json(request: Request):
    try:
        data = await request.json()
        payload = BatchPredictRequest(**data)
    except Exception:
        raise HTTPException(status_code=400, detail="Provide JSON {rows: [...]} body")

    df = _df_from_json_rows(payload)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data provided")

    # schema validation
    err = _validate_schema(df)
    if err:
        raise HTTPException(status_code=400, detail=err)

    # predict timings
    t1 = time.perf_counter()
    # ensure model loaded
    if predictor.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    preds = predictor.predict(df)
    t2 = time.perf_counter()

    meta = predictor.meta or {}
    return JSONResponse({
        "predictions": preds,
        "model_version": meta.get("model_version"),
        "timings": {
            "predict_ms": round((t2 - t1) * 1000, 2),
        },
    })


@app.post("/retrain/{model_name}")
async def retrain(model_name: str, files: List[UploadFile] = File(...), config: Optional[str] = Form(None)):
    # Parse config JSON if provided; accept Swagger's placeholder 'string' as empty
    cfg: RetrainConfig
    if config and config.strip() and config.strip().lower() != "string":
        try:
            cfg = RetrainConfig.parse_raw(config)
        except Exception:
            cfg = RetrainConfig()
    else:
        cfg = RetrainConfig()

    # Use the model_name from the path parameter
    # Special case: if user calls /retrain/all, delegate to bulk retrain endpoint
    if model_name.lower() == "all":
        return await retrain_all_models(files=files, config=config)
    cfg.model_name = model_name

    # Process data through the same pipeline as inference
    try:
        streams = _read_multi_csv(files)
        rows = _build_joined_rows(streams)
        dq = _dq_report(rows)
        df = _rows_to_df(rows)
    finally:
        for f in files:
            try:
                f.file.seek(0)
            except Exception:
                pass
    
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data provided")
    
    err = _validate_schema(df)
    if err:
        raise HTTPException(status_code=400, detail=err)

    # Load the existing ONNX model to understand its feature requirements
    from .pipeline.csharp_compat.registry import registry
    from .pipeline.csharp_compat import onnx_infer as oi
    onnx_file = f"{model_name}.onnx"
    try:
        sess = registry.get_session(onnx_file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in EXTERNAL_MODEL_DIR.")
    
    # Apply the same data preparation pipeline as prediction endpoints
    df = _apply_derived_features(df)
    training_df = _apply_feature_mapping(df, model_name)
    
    # Get the feature mapping for metadata (recreate since helper function doesn't return it)
    from .pipeline.csharp_compat.registry import registry
    from .pipeline.csharp_compat import onnx_infer as oi
    onnx_file = f"{model_name}.onnx"
    try:
        sess = registry.get_session(onnx_file)
        required_features = set(oi._feature_names(sess))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in EXTERNAL_MODEL_DIR.")
    
    all_possible_mappings = {
        'Bore Pressure (PSI)': 'Downstream-Pressure',
        'Annulus Pressure (PSI)': 'Upstream-Pressure', 
        'Bore Temperature (Deg.C)': 'Downstream-Temperature',
        'Annulus Temperature (Deg.C)': 'Upstream-Temperature',
        'Position (%)': 'Choke-Position',
        'Battery Voltage (Volts)': 'Battery-Voltage',
        'Bore-Annulus Difference (PSI)': 'Downstream-Upstream-Difference',
        'Target Position': 'Target-Position',
        'Tool State': 'ToolStateNum'
    }
    
    # Filter mapping for this model (for metadata only)
    model_specific_mapping = {
        csv_name: onnx_name 
        for csv_name, onnx_name in all_possible_mappings.items() 
        if onnx_name in required_features and csv_name in df.columns
    }
    print(f"DEBUG RETRAIN: Training data shape: {training_df.shape}")
    print(f"DEBUG RETRAIN: Training features: {list(training_df.columns)}")
    
    # Detect and route residual models to dedicated retrain logic
    if model_name.startswith("residual_") or model_name.endswith("_residual") or model_name == "target_pos_residual":
        # rewind file pointers (already read) so residual retrain can access originals if needed
        try:
            return _retrain_residual(model_name, df, cfg, sess, files, dq)
        finally:
            for f in files:
                try:
                    f.file.seek(0)
                except Exception:
                    pass

    # Train a new IsolationForest model with the processed data (non-residual models)
    from sklearn.ensemble import IsolationForest
    import joblib
    
    # Set default contamination if None
    contamination = cfg.contamination if cfg.contamination is not None else 'auto'
    
    max_samples = cfg.max_samples if cfg.max_samples is not None else 'auto'
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=cfg.random_state,
    )
    
    print(f"DEBUG RETRAIN: Training IsolationForest with {len(training_df)} samples. Dtypes: {training_df.dtypes.to_dict()}")
    model.fit(training_df)
    
    # Save the retrained model to REGISTRY_DIR
    from .versioning import model_dir, save_meta, compute_schema_hash
    
    tdir = model_dir(model_name)
    os.makedirs(tdir, exist_ok=True)
    
    # Save the sklearn model
    model_path = tdir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"DEBUG RETRAIN: Saved model to {model_path}")

    # Attempt ONNX export for IsolationForest so inference can stay ONNX-based
    onnx_export_error = None
    onnx_path = tdir / "model.onnx"
    try:
        from skl2onnx import convert_sklearn  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore
        import onnx  # type: ignore

        initial_type = [("input", FloatTensorType([None, training_df.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Inject feature name metadata (used by _feature_names in inference)
        meta_kv = {"feature_names": ",".join(training_df.columns)}
        # Append / overwrite metadata_props
        existing = {p.key: p.value for p in onnx_model.metadata_props}
        existing.update(meta_kv)
        # Clear and repopulate
        onnx_model.metadata_props.clear()
        for k, v in existing.items():
            prop = onnx.helper.StringStringEntryProto(key=k, value=v)  # type: ignore
            onnx_model.metadata_props.append(prop)

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"DEBUG RETRAIN: Exported ONNX model to {onnx_path}")
    except Exception as e:
        onnx_export_error = str(e)
        print(f"WARN RETRAIN: Could not export ONNX model for {model_name}: {e}")
    
    # Test the saved model
    loaded_model = joblib.load(model_path)
    test_predictions = loaded_model.predict(training_df.head(5))
    print(f"DEBUG RETRAIN: Test predictions: {test_predictions}")
    
    # Create metadata
    s_hash = compute_schema_hash(training_df)
    job_id = uuid.uuid4().hex
    
    meta = {
        "model_name": model_name,
        "model_version": tdir.name,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "features": list(training_df.columns),
        "original_csv_features": list(df.columns),
        "feature_mapping": model_specific_mapping,
        "schema_hash": s_hash,
        "n_rows": len(training_df),
        "features_dtypes": {c: str(training_df[c].dtype) for c in training_df.columns},
        "thresholds": cfg.thresholds or {},
        "training_args": {
            "n_estimators": cfg.n_estimators,
            "max_samples": cfg.max_samples,
            "contamination": cfg.contamination,
            "random_state": cfg.random_state,
        },
        "job_id": job_id,
        "data_quality": dq,
        "source_model": f"EXTERNAL_MODEL_DIR/{onnx_file}",
        "retrained_type": "sklearn_isolation_forest"
    }
    if onnx_path.exists():
        meta["onnx_model"] = str(onnx_path)
    if onnx_export_error:
        meta["onnx_export_error"] = onnx_export_error
    
    save_meta(tdir, meta)
    print(f"DEBUG RETRAIN: Saved metadata to {tdir / 'meta.json'}")

    return JSONResponse(status_code=200, content={
        "job_id": job_id,
        "model_version": tdir.name,
        "model_name": model_name,
        "trained_at": meta["trained_at"],
        "training_samples": len(training_df),
        "features_mapped": list(training_df.columns),
        "original_features": list(df.columns),
        "feature_mapping": model_specific_mapping,
        "saved_to": str(tdir),
        "status": "completed"
    })


def _list_available_models() -> List[str]:
    """Discover available base model names (without extension) from EXTERNAL_MODEL_DIR and local models dir.
    Filters to .onnx files only and skips duplicates."""
    discovered: set[str] = set()
    candidate_dirs = [EXTERNAL_MODEL_DIR, Path("models")]  # external inference dir + local models
    for d in candidate_dirs:
        try:
            if d.exists():
                for p in d.glob("*.onnx"):
                    discovered.add(p.stem)
        except Exception:
            pass
    return sorted(discovered)


@app.post("/retrain/all")
async def retrain_all_models(files: List[UploadFile] = File(...), config: Optional[str] = Form(None)):
    """Retrain all discovered ONNX models with the same uploaded dataset.

    This scans both EXTERNAL_MODEL_DIR and the local models/ folder for .onnx files.
    For each base model name found, the single-model retrain pipeline is executed.
    """
    available_models = _list_available_models()
    if not available_models:
        return JSONResponse(status_code=404, content={
            "status": "error",
            "detail": "No .onnx models found in EXTERNAL_MODEL_DIR or models/ to retrain"
        })

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for mname in available_models:
        # Reset file handles for each iteration (UploadFile objects are reused)
        for f in files:
            try:
                f.file.seek(0)
            except Exception:
                pass
        try:
            resp = await retrain(mname, files, config)
            # Attempt to parse JSON body if it's a Response
            body: Any
            if hasattr(resp, 'body'):
                try:
                    import json as _json
                    body = _json.loads(resp.body.decode())  # type: ignore[attr-defined]
                except Exception:
                    body = getattr(resp, 'body', b'').decode() if hasattr(resp, 'body') else "Success"
            else:
                body = resp
            results[mname] = body
        except Exception as e:
            errors[mname] = str(e)

    summary = {
        "status": "completed" if not errors else ("partial" if results else "failed"),
        "successful_models": list(results.keys()),
        "failed_models": list(errors.keys()),
        "total_models": len(available_models),
        "success_count": len(results),
        "error_count": len(errors),
        "registry_dir": str(REGISTRY_DIR),
        "models_examined": available_models,
        "results": results,
        "errors": errors,
    }
    return JSONResponse(status_code=200, content=summary)


class PromoteRequest(BaseModel):
    model_name: str
    version: str


@app.post("/promote")
async def promote(req: PromoteRequest):
    if not ALLOW_PROMOTE:
        raise HTTPException(status_code=403, detail="Promotion is disabled")
    src = REGISTRY_DIR / req.model_name / req.version
    if not src.exists() or not src.is_dir():
        raise HTTPException(status_code=404, detail=f"Version not found: {src}")
    # Copy all files from the version folder to EXTERNAL_MODEL_DIR root
    copied = []
    EXTERNAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for p in src.glob("**/*"):
        if p.is_file():
            dest = EXTERNAL_MODEL_DIR / p.name
            shutil.copy2(p, dest)
            copied.append(dest.name)
    return {"promoted_from": str(src), "to": str(EXTERNAL_MODEL_DIR), "files": copied}


@app.get("/metrics")
async def get_metrics():
    m = metrics_store.get()
    if not m:
        raise HTTPException(status_code=404, detail="No metrics available")
    return m
