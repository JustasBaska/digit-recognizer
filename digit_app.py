import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass as nd_center_of_mass


def preprocess_canvas_image(raw_img):
    """
    Converts canvas image data into a 28x28 MNIST-like input.
    Steps:
    - Extract grayscale digit
    - Crop black edges
    - Resize and pad
    - Center the digit
    - Normalize to 0–1
    """
    # Step 1: Convert to grayscale (take red channel)
    gray = raw_img[:, :, 0].astype("uint8")

    # Step 2: Invert (MNIST = white digit on black bg)
    inverted = gray

    # Step 3: Threshold to binary image
    _, binary = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)

    # Step 4: Crop the digit region (bounding box)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((1, 28, 28))  # Return blank if no content
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]

    # Step 5: Resize to 20x20 (keep aspect ratio)
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 6: Pad to 28x28
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    padded = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

    # Step 7: Center the digit by shifting to center of mass
    cy, cx = nd_center_of_mass(padded)
    shift_x = int(14 - cx)
    shift_y = int(14 - cy)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered = cv2.warpAffine(padded, M, (28, 28))

    # Step 8: Normalize and reshape
    final = centered.astype("float32") / 255.0
    return final.reshape(1, 28, 28)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

st.title("✍️ Draw a Digit (0–9)")
st.write("Draw a digit below and let the model guess it!")

# Create canvas component
canvas_result = st_canvas(
    fill_color="black",  # Background color
    stroke_width=15,
    stroke_color="white",  # Pen color
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Only show Predict button if there's something drawn
if canvas_result.image_data is not None:
    # Extract image from canvas
    img = canvas_result.image_data


    
if st.button("Predict"):
    import cv2

    # Step 1: Extract canvas image
    img = canvas_result.image_data

    img_array = preprocess_canvas_image(img)

    # Visual debug: What the model actually sees

    # Step 6: Predict
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    predicted_digit = int(np.argmax(prediction))
    st.subheader(f"Prediction: {predicted_digit} ({confidence:.2f}% confidence)")

