from fastapi import FastAPI,UploadFile, File 
import shutil
import os 

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Brain Tumor API Running"}

from fastapi import FastAPI
from backend.predict import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Brain Tumor API Running"}

@app.get("/test")
def test_prediction():
    result = predict_image("test_images/sample.jpg")
    return result


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return predict_image(file_path)

