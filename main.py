import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess images
def load_images(data_path, label):
    images = []
    labels = []

    for filename in os.listdir(data_path):
        img_path = os.path.join(data_path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(label)

    return images, labels


# Load real and DeepFake images
real_images, real_labels = load_images('dataset/real', label=0)
fake_images, fake_labels = load_images('dataset/fake', label=1)

# Concatenate and shuffle the data
images = np.concatenate((real_images, fake_images), axis=0)
labels = np.concatenate((real_labels, fake_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the CNN model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, accuracy: {accuracy:.4f}')

# Save the model
model.save('deepfake_detection_model.h5')
