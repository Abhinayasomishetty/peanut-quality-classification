import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("peanut_model.h5")

# Class names (same order as folders)
classes = ['broken', 'kernel', 'unshelled']

# Load test image
img = image.load_img(r"C:\Users\somis\OneDrive\Desktop\peanut_project\test.jpg", target_size=(150,150)) # image name
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
predicted_class = classes[np.argmax(prediction)]

print("Prediction:", predicted_class)