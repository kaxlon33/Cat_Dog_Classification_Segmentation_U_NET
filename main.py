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

MODEL_PATH = os.getcwd() + "/ml_models/unet_model_ml020.h5"
CLASS_NAMES_PATH = os.getcwd() + "/ml_models/class_names.json"

cat_dog_model = CatDogModel(MODEL_PATH, CLASS_NAMES_PATH)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Keep track of last uploaded file
last_uploaded_file = None

@app.get("/")
def root():
    return {"message": "Cat vs Dog API is running!"}

@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    global last_uploaded_file, last_segmentation_file
    try:
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        last_uploaded_file = input_path

        img = Image.open(input_path).convert("RGB")
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0

        preds = cat_dog_model.model.predict(np.expand_dims(img_array, axis=0))[0]  
        mask = np.argmax(preds, axis=-1)  # shape (128,128), values 0=bg, 1=cat, 2=dog

        colors = {
            0: (0, 0, 0),       # black
            1: (0, 255, 0),     # green
            2: (0, 0, 255)      # blue
        }
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for k, color in colors.items():
            mask_rgb[mask == k] = color

        # Save mask
        mask_img = Image.fromarray(mask_rgb)
        mask_filename = f"mask_{file.filename}"
        mask_path = os.path.join(UPLOAD_DIR, mask_filename)
        mask_img.save(mask_path)
        last_segmentation_file = mask_path

        # Convert prediction to one main class (highest area)
        unique, counts = np.unique(mask, return_counts=True)
        main_class_index = unique[np.argmax(counts)]
        main_class = cat_dog_model.class_names[main_class_index]
        confidence = float(counts[np.argmax(counts)] / mask.size)

        with open(input_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return PredictionResponse(
            class_name=main_class,
            confidence=confidence,
            image_base64=img_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/show")
async def show_image(segmentation: bool = False):
    file_to_show = last_segmentation_file if segmentation else last_uploaded_file

    if not file_to_show or not os.path.exists(file_to_show):
        raise HTTPException(status_code=404, detail="No image has been uploaded yet")

    ext = os.path.splitext(file_to_show)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    else:
        media_type = "application/octet-stream"

    return FileResponse(file_to_show, media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
