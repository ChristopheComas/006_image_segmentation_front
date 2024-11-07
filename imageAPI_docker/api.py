from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse,JSONResponse
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm

from utils import preprocess_image, postprocess_mask

app = FastAPI()

# Load your U-Net model with custom loss function
MODEL_PATH = './Unet-efficientnetb2.weights.h5'
BACKBONE = 'efficientnetb2'

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=8, activation='softmax')
model.load_weights(MODEL_PATH) 

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Segmentation API"}

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Preprocess the image
    input_image = preprocess_image(image)

    # Predict the mask
    prediction = model.predict(input_image)
    mask = postprocess_mask(prediction)
    
    return JSONResponse(content={"mask": mask.tolist()})