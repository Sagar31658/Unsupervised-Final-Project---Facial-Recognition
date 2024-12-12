# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:35:51 2024

@author: leor7
"""

import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras import layers, regularizers, Sequential, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ignore warnings
warnings.filterwarnings('ignore')

# Read the CSV file
df = pd.read_csv(r'Dataset/umist_faces.csv')

# Display the first few rows of the DataFrame
print(df.shape)
print(df.head())

# Check for duplicates
counter = 0
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if df.iloc[i, 0] == df.iloc[j, 0]:  # Check for the same image if labels are the same
            if df.iloc[i, 1:].equals(df.iloc[j, 1:]):  # Check if the rows are the same
                print("Duplicate found")
                print(df.iloc[i, 0])
                print(df.iloc[j, 0])
                print("Row 1: ", df.iloc[i, 1:])
                print("Row 2: ", df.iloc[j, 1:])
                counter += 1
                print("\n\n")
print("Total duplicates found: ", counter)

df = df.drop_duplicates()
print(df.shape)

X = df.drop('label', axis=1).values
y = df['label'].values

# Reshape images (assuming 112x92 image size)
image_size = (112, 92)
X = X.reshape(-1, *image_size)

# Split data into training, testing, and validation sets using stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Normalize data
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Balance the training dataset using ImageDataGenerator
TARGET_IMAGES_PER_PERSON = 48

# Split the dataset by label
unique_labels = np.unique(y_train)
label_to_images = {label: X_train[y_train == label] for label in unique_labels}

# Create an ImageDataGenerator instance for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Before balancing: plot the number of images per person
initial_counts = [np.sum(y_train == label) for label in unique_labels]
plt.figure(figsize=(10, 6))
plt.bar(unique_labels, initial_counts, color='skyblue')
plt.title('Number of Images per Person (Before Balancing)')
plt.xlabel('Person (Label)')
plt.ylabel('Number of Images')
plt.show()

balanced_X = []
balanced_y = []

for label, images in label_to_images.items():
    num_images = images.shape[0]
    if num_images < TARGET_IMAGES_PER_PERSON:
        # Add channel dimension to images (assuming grayscale images)
        images = np.expand_dims(images, axis=-1)  # Shape becomes (num_images, height, width, 1)
        
        # Generate augmented images
        augmented_images = []
        for x in datagen.flow(images, batch_size=1, seed=42):
            augmented_images.append(x[0])  # x is a batch, take the first image
            if len(augmented_images) + num_images >= TARGET_IMAGES_PER_PERSON:
                break
        combined_images = np.concatenate([images, np.array(augmented_images)])
    else:
        # Randomly select TARGET_IMAGES_PER_PERSON images
        combined_images = images[np.random.choice(num_images, TARGET_IMAGES_PER_PERSON, replace=False)]

    balanced_X.extend(combined_images)
    balanced_y.extend([label] * TARGET_IMAGES_PER_PERSON)

# Convert balanced dataset to numpy arrays
balanced_X = np.array(balanced_X)
balanced_y = np.array(balanced_y)

print(f"Balanced dataset shape: {balanced_X.shape}, {balanced_y.shape}")

# After balancing: plot the number of images per person
balanced_counts = [np.sum(balanced_y == label) for label in unique_labels]
plt.figure(figsize=(10, 6))
plt.bar(unique_labels, balanced_counts, color='lightcoral')
plt.title('Number of Images per Person (After Balancing)')
plt.xlabel('Person (Label)')
plt.ylabel('Number of Images')
plt.show()

# Flatten balanced_X for PCA and autoencoder
balanced_X_flat = balanced_X.reshape(balanced_X.shape[0], -1)

# Apply PCA to retain 99% variance
pca = PCA(0.99)
X_train_pca = pca.fit_transform(balanced_X_flat)

# Define the autoencoder model
def create_autoencoder(input_dim, hidden_units_1, hidden_units_2, learning_rate, reg_param):
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units_1, activation='relu', kernel_regularizer=regularizers.l2(reg_param)),
        layers.Dense(hidden_units_2, activation='relu', kernel_regularizer=regularizers.l2(reg_param)),
        layers.Dense(hidden_units_1, activation='relu', kernel_regularizer=regularizers.l2(reg_param)),
        layers.Dense(input_dim, activation='sigmoid')  # Output layer with sigmoid for normalized values
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError())
    return model

# Train the autoencoder
input_dim = X_train_pca.shape[1]
hidden_units_1 = 128
hidden_units_2 = 64
learning_rate = 0.001
reg_param = 0.001

autoencoder = create_autoencoder(input_dim, hidden_units_1, hidden_units_2, learning_rate, reg_param)
history = autoencoder.fit(X_train_pca, X_train_pca,
                          epochs=100,
                          batch_size=32,
                          validation_split=0.2,
                          verbose=1)

# Visualize training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Encode the data using the bottleneck layer
encoder = Sequential(autoencoder.layers[:2])  # Extract the encoder part
X_encoded = encoder.predict(X_train_pca)

# Loop through 1 to 21 clusters and find the optimal number based on silhouette score
# Initialize variables
silhouette_scores = []
cluster_range = range(2, 30)  # Start from 2 as silhouette is undefined for 1 cluster
best_silhouette_score = -1
best_num_clusters = None

# Iterate through cluster numbers
for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X_encoded)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_encoded, y_pred)
    silhouette_scores.append(silhouette_avg)
    
    # Track the best score
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_num_clusters = num_clusters

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

print(f"Optimal number of clusters: {best_num_clusters}, with silhouette score: {best_silhouette_score}")