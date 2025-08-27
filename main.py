from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import shutil
from models.schema import ImagePredRequestModel, ImagePredResponseModel
from model_work import CatAndDogModel

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    cat_dog_model = CatAndDogModel()
    cat_dog_model.load_model()  # synchronous
    ml_models['cat_dog_model'] = cat_dog_model
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Welcome to Pet Segmentation & Classification API"}

# JSON prediction endpoint
@app.post("/predict", response_model=ImagePredResponseModel)
def predict(body: ImagePredRequestModel):
    cat_dog_model = ml_models["cat_dog_model"]
    img_array = cat_dog_model.preprocess_image(body.image)
    class_name, confidence = cat_dog_model.predict(img_array)

    return {
        "class_name": class_name,
        "confidence": round(confidence, 2)
    }

# Image prediction + segmentation mask endpoint
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    cat_dog_model = ml_models["cat_dog_model"]

    # Ensure uploads folder exists
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    input_path = os.path.join(upload_dir, file.filename)

    # Save uploaded file temporarily
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Preprocess + predict + save mask
    img_array = cat_dog_model.preprocess_image(input_path)
    output_path = cat_dog_model.predict_and_save_mask(img_array, input_path)

    return FileResponse(output_path, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
