import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
# Load VGG16 pre-trained model with cuDNN optimization
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False
# Add custom layers for face recognition
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification (face or not)
model = Model(inputs=base_model.input, outputs=output)
# Compile model with cuDNN-optimized backend
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Prepare data (e.g., using OpenCV for image preprocessing)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    return np.expand_dims(image / 255.0, axis=0)
# Example: Load and predict on a test image
test_image = preprocess_image("test_face.jpg")
prediction = model.predict(test_image)
print("Face detected" if prediction[0] > 0.5 else "No face detected")
