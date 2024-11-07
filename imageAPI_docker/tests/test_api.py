import pytest
import numpy as np
from PIL import Image
from utils import preprocess_image, postprocess_mask  # Only import these two functions

# Test preprocess_image function
def test_preprocess_image():
    """Test that preprocess_image resizes, normalizes, and shapes the image correctly."""
    # Create a dummy RGB image
    image = Image.new("RGB", (600, 600), color=(255, 0, 0))

    # Preprocess the image
    processed_image = preprocess_image(image, target_size=(512, 512))

    # Check the processed image properties
    assert processed_image.shape == (1, 512, 512, 3)  # (batch, height, width, channels)
    assert processed_image.min() >= 0.0 and processed_image.max() <= 1.0  # Normalization

# Test postprocess_mask function
def test_postprocess_mask():
    """Test that postprocess_mask removes the batch dimension correctly."""
    # Create a dummy prediction with a batch dimension
    dummy_prediction = np.random.rand(1, 512, 512, 8)  # (batch, height, width, classes)

    # Postprocess the mask
    mask = postprocess_mask(dummy_prediction)

    # Check the mask shape after squeezing
    assert mask.shape == (512, 512, 8)  # (height, width, classes)
