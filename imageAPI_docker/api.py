from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
from fastapi.responses import JSONResponse


app = FastAPI()

# Load your U-Net model with custom loss function
MODEL_PATH = './Unet-efficientnetb2.weights.h5'
# MODEL_PATH = './Unet-efficientnetb0.weights.h5'
BACKBONE = 'efficientnetb2'

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=8, activation='softmax')
# model = sm.Unet(BACKBONE, classes=8, activation='softmax')
model.load_weights(MODEL_PATH) 
             

# Preprocess the input image to fit model requirements
def preprocess_image(image: Image.Image, target_size=(512, 512)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize image
    if image_array.ndim == 2:  # Handle grayscale images
        image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Post-process the prediction to create a mask image
def postprocess_mask(prediction):
    mask = np.squeeze(prediction)  # Remove batch dimension
    return mask

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