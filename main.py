from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import io
import os
import base64

from model_work import CatDogModel
from models.schema import PredictionResponse


app = FastAPI()

# paths
MODEL_PATH = os.path.join("ml_models", "unet_model_ml020.keras")
CLASS_NAMES_PATH = os.path.join("ml_models", "class_names.json")

# load model
cat_dog_model = CatDogModel(MODEL_PATH, CLASS_NAMES_PATH)

# ensure uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Cat vs Dog API is running!"}

@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        # save uploaded file
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # load and preprocess image
        img = Image.open(input_path).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0

        # predict
        result = cat_dog_model.predict(img_array)

        # encode image as base64
        with open(input_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return PredictionResponse(
            class_name=result["class"],
            confidence=result["confidence"],
            image_base64=img_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/show/{filename}")
async def show_image(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on file extension
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    elif ext == ".gif":
        media_type = "image/gif"
    else:
        media_type = "application/octet-stream"  # fallback for other files

    return FileResponse(file_path, media_type=media_type)
