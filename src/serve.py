from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu GCS hoac DagsHub/Local fallback ve may khi server khoi dong.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 1. Thu tai tu GCS neu co cau hinh
    if GCS_BUCKET and not GCS_BUCKET.startswith("<"):
        try:
            print(f"Connecting to GCS bucket: {GCS_BUCKET}...")
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_MODEL_KEY)
            blob.download_to_filename(MODEL_PATH)
            print("Model downloaded successfully from GCS.")
            return
        except Exception as e:
            print(f"Failed to download from GCS: {e}. Falling back to DagsHub...")

    # 2. Thu tai tu DagsHub raw URL
    url = "https://dagshub.com/tphamdinh583/CI-CD_AI/raw/main/models/model.pkl"
    print(f"Downloading model from DagsHub raw URL: {url}")
    import urllib.request
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded successfully from DagsHub.")
        return
    except Exception as e:
        print(f"Failed to download from DagsHub: {e}. Falling back to local workspace model...")

    # 3. Thu copy tu local workspace neu co
    local_ws_path = os.path.join(os.getcwd(), "models", "model.pkl")
    if os.path.exists(local_ws_path):
        import shutil
        try:
            print(f"Copying model from workspace: {local_ws_path} -> {MODEL_PATH}")
            shutil.copy(local_ws_path, MODEL_PATH)
            print("Model copied successfully from local workspace.")
            return
        except Exception as copy_err:
            print(f"Failed to copy from local workspace: {copy_err}")

    # Neu tat ca deu that bai va khong co file cu, raise loi
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Khong tim thay file model.pkl o bat ky nguon nao.")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 12 features (wine quality), got {len(req.features)}",
        )

    # Du doan voi model
    try:
        pred = int(model.predict([req.features])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Nhan tuong ung: 0 -> "thap", 1 -> "trung_binh", 2 -> "cao"
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = labels.get(pred, "unknown")

    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
