from PIL import Image
import numpy as np

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
    mask = np.argmax(mask, axis=-1)
    print("\n-------------------\n")
    print(mask.shape)
    print("\n-------------------\n")

    return mask