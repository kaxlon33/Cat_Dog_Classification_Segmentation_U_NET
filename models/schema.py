from pydantic import BaseModel

class ImagePredRequestModel(BaseModel):
    image: str  # Can be image path or base64 string

class ImagePredResponseModel(BaseModel):
    class_name: str
    confidence: float
