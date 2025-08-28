import tensorflow as tf
import numpy as np
import os
import json

class CatDogModel:
    def __init__(self, model_path: str, class_names_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # load Keras model (.keras file)
        self.model = tf.keras.models.load_model(model_path)

        # load class names
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"Class names file not found: {class_names_path}")
        with open(class_names_path, "r") as f:
            self.class_names = json.load(f)

    def predict(self, image_array: np.ndarray) -> dict:
        image_array = np.expand_dims(image_array, axis=0)
        preds = self.model.predict(image_array)
        predicted_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        return {
            "class": self.class_names[predicted_class],
            "confidence": confidence
        }
