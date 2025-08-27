import os
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
class CatAndDogModel:
    def __init__(self):
        self.model_path = os.path.join(os.getcwd(), "ml_models", "cat_dog_model.h5")
        self.class_path = os.path.join(os.getcwd(), "ml_models", "class_names.txt")
        self.input_img_size = (128, 128)
        self.model = None
        self.class_indices = None

# model loading
    def load_model(self):
        try:
            self.model = load_model(self.model_path)
            with open(self.class_path , 'r') as f:
                self.class_indices = json.load(f)
            print("Model and class indices loaded successfully.")
        except Exception as e:
            print(f"Error loading model or class indices: {e}")
            return False
        
        return True
    
    def preprocess_image(self , img_path):
        img = image.load_img(img_path, target_size=self.input_img_size)
        img_array = image.img_to_array(img)
        img_array  = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension    
        return img_array 

    def predict(self, image_array):
        prediction = self.model.predict(image_array)[0][0]
        class_name = "dogs" if prediction > 0.5 else "cats" # 0.5 lok loz m ya / classfile kho phat p yay ya mhr 
        return class_name , prediction
    
    def predict_and_save_mask(self, image_array, input_path):
        mask = self.model.predict(image_array)[0]
        mask = (mask > 0.5).astype(np.uint8) * 255
        output_path = input_path.replace(".jpg", "_mask.png").replace(".jpeg", "_mask.png").replace(".png", "_mask.png")
        cv2.imwrite(output_path, mask)

        return output_path