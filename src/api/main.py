from fastapi import FastAPI            # Web framework for APIs
from pathlib import Path               # For handling file paths cleanly
from typing import List, Dict, Any     # For type hints (clarity in endpoints)
import pandas as pd                    # To handle incoming JSON as DataFrames
import boto3, os                       # AWS SDK for Python + env variables
import math
import traceback
from joblib import load as _load_model
 
# Import inference pipeline
from src.inference_pipeline.inference import predict
 
 
# ----------------------------
# Sanitize NaN/Inf
# ----------------------------
def sanitize_floats(obj):
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
 
 
# ----------------------------
# Config
# ----------------------------
S3_BUCKET = os.getenv(
    "S3_BUCKET",
    "housing-regression-data-chunmunchoubey-2026"
)
REGION = os.getenv(
    "AWS_REGION",
    "ap-south-1"
)
s3 = boto3.client("s3", region_name=REGION)
 
 
# Ensures your app always has the latest model/data locally,
# but avoids re-downloading every time it starts.
def load_from_s3(key, local_path):
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"📥 Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)
 
 
# ----------------------------
# Paths
# ----------------------------
# Downloads model from S3 if not cached.
MODEL_PATH = Path(load_from_s3("models/xgb_best_model.pkl", "models/xgb_best_model.pkl"))
 
# Load expected training features directly from the model (always in sync with the model)
_model_for_schema = _load_model(MODEL_PATH)
TRAIN_FEATURE_COLUMNS = list(_model_for_schema.get_booster().feature_names)
 
# Download encoders (note: freq encoder is a .pkl file, not .json)
FREQ_ENCODER_PATH = Path(load_from_s3("models/freq_encoder.pkl", "models/freq_encoder.pkl"))
TARGET_ENCODER_PATH = Path(load_from_s3("models/target_encoder.pkl", "models/target_encoder.pkl"))
 
 
# ----------------------------
# App
# ----------------------------
# Instantiates the FastAPI app.
app = FastAPI(title="Housing Regression API")
 
 
# / → simple landing endpoint to confirm API is alive.
@app.get("/")
def root():
    return {"message": "Housing Regression API is running 🚀"}
 
 
# /health → checks if model exists, returns status info (like expected feature count).
@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    return status
 
 
# Prediction Endpoint: This is the core ML serving endpoint.
@app.post("/predict")
def predict_batch(data: List[dict]):
    try:
        if not MODEL_PATH.exists():
            return {"error": f"Model not found at {str(MODEL_PATH)}"}
 
        df = pd.DataFrame(data)
        if df.empty:
            return {"error": "No data provided"}
 
        preds_df = predict(
            df,
            model_path=MODEL_PATH,
            freq_encoder_path=FREQ_ENCODER_PATH,
            target_encoder_path=TARGET_ENCODER_PATH,
        )
 
        resp = {"predictions": preds_df["predicted_price"].astype(float).tolist()}
        if "actual_price" in preds_df.columns:
            resp["actuals"] = preds_df["actual_price"].astype(float).tolist()
 
        return sanitize_floats(resp)
 
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
 
 
# Batch runner
from src.batch.run_monthly import run_monthly_predictions
 
 
# Trigger a monthly batch job via API.
@app.post("/run_batch")
def run_batch():
    preds = run_monthly_predictions()
    return {
        "status": "success",
        "rows_predicted": int(len(preds)),
        "output_dir": "data/predictions/"
    }
 
 
# Returns a preview of the most recent batch predictions.
@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found"}
 
    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }