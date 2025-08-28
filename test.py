import os
from model_work import CatDogModel

MODEL_PATH = os.path.join("ml_models", "unet_model_ml020.h5")
CLASS_NAMES_PATH = os.path.join("ml_models", "class_names.json")

# Load model
try:
    cat_dog_model = CatDogModel(MODEL_PATH, CLASS_NAMES_PATH)
    print("✅ Model and class names loaded successfully!")
    print("Class names:", cat_dog_model.class_names)
except Exception as e:
    print("❌ Error loading model or class names:", e)
