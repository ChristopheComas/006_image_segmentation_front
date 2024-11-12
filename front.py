import streamlit as st
import requests
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cv2
import matplotlib.gridspec as gridspec

# Define FastAPI URL
BASE_URL = "https://apisegwafg-f8bnemaqbthdfedr.francecentral-01.azurewebsites.net/"  # Adjust to your API URL
API_URL_seg = BASE_URL+"segment"

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
    response = requests.post(API_URL_seg, files={"file": ("image.jpg", img_bytes, "image/jpeg")})

    if response.status_code == 200:
        # Get the mask from the API response
        result = response.json()
        mask = np.array(result["mask"])
        return mask
    else:
        st.error("Error in API request")
        return None
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
    
   


    # Apply colors to each class label in the mask
    for class_idx, (class_name, color) in enumerate(class_colors.items()):
        color_array[mask_array == class_idx] = color

    
    st.image(color_array, caption="Masked image", use_column_width=True)

    # Create a figure for displaying the mask
    fig, ax = plt.subplots(figsize=(7, 1))

    # Create a legend for class colors
    legend_patches = [Patch(color=np.array(color) / 255, label=class_name) for class_name, color in class_colors.items()]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.2, 1), title="Classes")
    ax.axis('off')
    # Display the plot in Streamlit
    st.pyplot(fig)


class_names = list(class_colors.keys())

def calculate_iou(mask1, mask2, num_classes=8):
    """
    Calculate the Intersection over Union (IoU) score for a masked image with multiple categories.
    
    Parameters:
        mask1 (np.ndarray): First mask array of shape (height, width).
        mask2 (np.ndarray): Second mask array of shape (height, width).
        num_classes (int): Number of classes/categories in the masks.
        
    Returns:
        float: Mean IoU score across all classes.
        dict: IoU score per class.
    """
    iou_per_class = {}
    for cls in range(num_classes):
        # Create binary masks for the current class
        mask1_cls = (mask1 == cls)
        mask2_cls = (mask2 == cls)

        # Calculate intersection and union
        intersection = np.logical_and(mask1_cls, mask2_cls).sum()
        union = np.logical_or(mask1_cls, mask2_cls).sum()

        # Handle the case where union is zero to avoid division by zero
        if union == 0:
            iou_per_class[class_names[cls]] = float('nan')  # If union is 0, IoU is undefined for this class
        else:
            iou_per_class[class_names[cls]] = round(intersection / union, 2)

    # Calculate mean IoU, ignoring classes with undefined IoU
    mean_iou = round(np.nanmean(list(iou_per_class.values())), 2)

    return mean_iou, iou_per_class

# Streamlit UI
st.title("Image Segmentation GUI")

response = requests.get(f"{BASE_URL}list_files")

print(f"{BASE_URL}list_files")

if response.status_code == 200:
    data = response.json()
    image_filenames = data.get("images", [])
    mask_filenames = data.get("masks", [])
else:
    st.error("Failed to load data from the API.")
    image_filenames, mask_filenames = [], []


selected_image = st.selectbox("Select an image file:", image_filenames, index=None, placeholder="Select an image ...")

# Find the associated mask file for the selected image
if selected_image:
    # Assuming the mask filenames have a predictable structure
    
    selected_mask = next((mask for mask in mask_filenames if '_'.join(selected_image.split('_')[:3])+"_gtFine_labelIds_cat.png" in mask), None)

    st.write(selected_mask)
    st.subheader("Inputs image and mask", divider=True)

    # Fetch and display the selected image
    if selected_mask:
        image_response = requests.get(f"{BASE_URL}get_file", params={"filename": selected_image})
        mask_response = requests.get(f"{BASE_URL}get_file", params={"filename": selected_mask})

        if image_response.status_code == 200 and mask_response.status_code == 200:
            # Decode and display the image
            image_bytes = np.asarray(bytearray(image_response.content), dtype=np.uint8)
            image = cv2.imdecode(image_bytes, 1)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Selected Image: {selected_image}", use_column_width=True)

            # Decode and display the mask
            mask_bytes = np.asarray(bytearray(mask_response.content), dtype=np.uint8)
            mask = cv2.imdecode(mask_bytes, 0)  # 0 for grayscale
            if mask is not None:
                # Display the mask using the plot_color_mask function
                plot_color_mask(mask)        
        else:
            st.error("Failed to load the selected image or mask.")
    else:
        st.warning("No matching mask found for the selected image.")

if st.button("Segment Image"):
        st.subheader("Output mask", divider=True)

        with st.spinner("Segmenting the image..."):
            # Perform segmentation and resize the mask to match the original image
            mask_pred = upload_and_segment(image)
            mask_pred = cv2.resize(mask_pred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if mask_pred is not None:
            # Display the mask using the plot_color_mask function
            plot_color_mask(mask_pred)
            
            mean_iou, iou_per_class = calculate_iou(mask, mask_pred)
            mean_iou = round(mean_iou, 2)
            # Round each class IoU to 2 decimal places
            iou_per_class = {cls: round(iou, 2) if not np.isnan(iou) else iou for cls, iou in iou_per_class.items()}

            st.write("### IoU Calculation Results")
            st.write("Mean IoU:", mean_iou)
            st.write("IoU per Class:")
            for cls, iou in iou_per_class.items():
                st.write(f"Class {cls}: {iou if not np.isnan(iou) else 'Undefined'}")