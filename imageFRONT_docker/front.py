import streamlit as st
import requests
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cv2
import matplotlib.gridspec as gridspec

# Define FastAPI URL
API_URL = "https://apisegwafg-f8bnemaqbthdfedr.francecentral-01.azurewebsites.net/segment"  # Modify the URL if your API is deployed elsewhere

def upload_and_segment(image):
    # Encode image as JPEG
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        st.error("Error in image encoding")
        return None

    # Convert the image to bytes
    img_bytes = encoded_image.tobytes()

    # Send the image to the API
    files = {'file': img_bytes}
    response = requests.post(API_URL, files={"file": ("image.jpg", img_bytes, "image/jpeg")})

    if response.status_code == 200:
        # Get the mask from the API response
        result = response.json()
        mask = np.array(result["mask"])
        return mask
    else:
        st.error("Error in API request")
        return None

def plot_color_mask(mask_array):
    """
    Display a color-coded segmentation mask without axes and provide a legend with class colors.

    Args:
    - mask_array: The predicted mask array, either in one-hot encoded format or as class indices.
    """
    # Initialize the color array with zeros (shape: H, W, 3 for RGB channels)
    color_array = np.zeros((*mask_array.shape[0:2], 3), dtype=np.uint8)
    
    # If mask has multiple channels (one-hot encoded), convert it to class indices
    if mask_array.shape[-1] == 8:
        mask_array = np.argmax(mask_array, axis=-1)
    
   
    # Dictionary mapping class names to colors (BGR format)
    class_colors = {
        'void': [0, 0, 255],       # Blue
        'road': [255, 255, 0],     # Yellow
        'construction': [255, 0, 0], # Red
        'object': [0, 255, 255],   # Cyan
        'vegetation': [0, 255, 0], # Green
        'sky': [255, 0, 255],      # Magenta
        'human': [255, 100, 100],  # Light pink
        'vehicle': [255, 125, 0]   # Orange
    }

    # Apply colors to each class label in the mask
    for class_idx, (class_name, color) in enumerate(class_colors.items()):
        color_array[mask_array == class_idx] = color

    
    st.image(color_array, caption="masked Image", use_column_width=True)

    # Create a figure for displaying the mask
    fig, ax = plt.subplots(figsize=(7, 7))

    # Create a legend for class colors
    legend_patches = [Patch(color=np.array(color) / 255, label=class_name) for class_name, color in class_colors.items()]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.2, 1), title="Classes")
    ax.axis('off')
    # Display the plot in Streamlit
    st.pyplot(fig)

# Streamlit UI
st.title("Image Segmentation")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded file as bytes and decode it using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # Decode the image from bytes (1 for color image)

    # Display original image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
    # Send the image to API and get the mask
    with st.spinner("Segmenting the image..."):
        mask = upload_and_segment(image)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if mask is not None:
        # Display the mask using the plot_color_mask function
        plot_color_mask(mask)
        