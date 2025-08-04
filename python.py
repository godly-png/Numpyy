import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set dataset path (Ensure you have fingerprint images with labeled subfolders)
DATASET_PATH = "./fingerprint_dataset"  # Example path

# Define categories
CATEGORIES = ["Blood_Group_A", "Blood_Group_B", "Blood_Group_AB", "Blood_Group_O",
              "Drug_Positive", "Drug_Negative"]


# Image Processing function
def load_and_preprocess_images(dataset_path, img_size=(128, 128)):
    images, labels = [], []
    for category in CATEGORIES:
        path = os.path.join(dataset_path, category)
        label = CATEGORIES.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, img_size)  # Resize
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1)  # Reshape for CNN
    labels = np.array(labels)
    return images, labels


# Load and preprocess dataset
X, y = load_and_preprocess_images(DATASET_PATH)

# Split dataset into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define CNN Model (LeNet-5 inspired)
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(CATEGORIES), activation='softmax')  # Multi-class classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Create and train the model
model = create_cnn_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate model
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# Save model
model.save("fingerprint_model.h5")


# Load and test on a new fingerprint
def predict_fingerprint(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)

    prediction = np.argmax(model.predict(img))
    return CATEGORIES[prediction]


# Test on a sample fingerprint
sample_image = "./fingerprint_dataset/sample_test.jpg"
model = tf.keras.models.load_model("fingerprint_model.h5")
result = predict_fingerprint(sample_image, model)
print("Predicted Class:", result)