# -*- coding: utf-8 -*-
# Import necessary libraries
import os
import random
import gdown
import tensorflow as tf
from tensorflow.keras import layers, Model, metrics, optimizers
from tensorflow.keras.applications import resnet
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# URL for the dataset and directory for downloading
data_url = "https://drive.google.com/file/d/13hSwP2O4pd3NVVnWj2Fcah_r-jkf8uiv/view?usp=drive_link"
download_dir = "./classification_data/"

# Define the target shape for image resizing
image_shape = (200, 200)

# Function to load and process a single image
def load_and_process_image(file_path):
    image_data = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, image_shape)

# Function to process image triplets (anchor, positive, negative)
def process_triplets(anchor, pos, neg):
    return (load_and_process_image(anchor),
            load_and_process_image(pos),
            load_and_process_image(neg))

# Function to create a dataset from a directory
def create_dataset(directory):
    anchor_list, positive_list, negative_list = [], [], []

    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        images = os.listdir(category_path)

        # Loop through each image, creating triplets
        for anchor_img in images:
            anchor_img_path = os.path.join(category_path, anchor_img)
            anchor_list.append(anchor_img_path)

            positive_img = random.choice(images)
            positive_list.append(os.path.join(category_path, positive_img))

            different_category = random.choice([c for c in os.listdir(directory) if c != category])
            negative_img = random.choice(os.listdir(os.path.join(directory, different_category)))
            negative_list.append(os.path.join(directory, different_category, negative_img))

    # Create and return the dataset
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(anchor_list),
                                   tf.data.Dataset.from_tensor_slices(positive_list),
                                   tf.data.Dataset.from_tensor_slices(negative_list)))
    dataset = dataset.shuffle(1024).map(process_triplets).batch(64).prefetch(8)
    return dataset

# Create training and validation datasets
train_data = create_dataset(download_dir + "train_data/")
val_data = create_dataset(download_dir + "val_data/")

# Build the base CNN model using ResNet50
base_model = resnet.ResNet50(weights="imagenet", input_shape=image_shape + (3,), include_top=False)
flattened_output = layers.Flatten()(base_model.output)
dense_layer1 = layers.Dense(512, activation="relu")(flattened_output)
normalized1 = layers.BatchNormalization()(dense_layer1)
dense_layer2 = layers.Dense(256, activation="relu")(normalized1)
normalized2 = layers.BatchNormalization()(dense_layer2)
final_output = layers.Dense(256)(normalized2)

embedding_model = Model(inputs=base_model.input, outputs=final_output, name="Image_Embedding")

# Make specific layers trainable
for layer in base_model.layers:
    layer.trainable = layer.name >= "conv5_block1_out"

# Define a custom Distance Layer for the Siamese Network
class DistanceLayer(layers.Layer):
    def call(self, anchor_embedding, positive_embedding, negative_embedding):
        distance_pos = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), -1)
        distance_neg = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), -1)
        return distance_pos, distance_neg

# Inputs for the Siamese Network
anchor_input = layers.Input(name="anchor_input", shape=image_shape + (3,))
positive_input = layers.Input(name="positive_input", shape=image_shape + (3,))
negative_input = layers.Input(name="negative_input", shape=image_shape + (3,))

# Build the Siamese Network
siamese_network = Model(inputs=[anchor_input, positive_input, negative_input],
                        outputs=DistanceLayer()(
                            embedding_model(resnet.preprocess_input(anchor_input)),
                            embedding_model(resnet.preprocess_input(positive_input)),
                            embedding_model(resnet.preprocess_input(negative_input))
                        ))

# Custom Siamese Model class
class CustomSiameseModel(Model):
    def __init__(self, network, margin=0.5):
        super().__init__()
        self.network = network
        self.margin = margin
        self.loss_metric = metrics.Mean(name="loss")
        self.accuracy_metric = metrics.Mean(name="accuracy")

    def call(self, inputs):
        return self.network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, accuracy = self.compute_loss_and_accuracy(data)
        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(accuracy)
        return {"loss": self.loss_metric.result(), "accuracy": self.accuracy_metric.result()}

    def test_step(self, data):
        loss, accuracy = self.compute_loss_and_accuracy(data)
        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(accuracy)
        return {"loss": self.loss_metric.result(), "accuracy": self.accuracy_metric.result()}

    def compute_loss_and_accuracy(self, data):
        ap_dist, an_dist = self.network(data)
        loss = tf.maximum(ap_dist - an_dist + self.margin, 0.0)

        # Calculate accuracy as top-1 accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.less(ap_dist, an_dist), tf.float32))
        return loss, accuracy

    @property
    def metrics(self):
        return [self.loss_metric, self.accuracy_metric]

# Compile and train the Siamese model
siamese_model = CustomSiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adadelta())

# Callback for early stopping
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Training the model
history = siamese_model.fit(train_data, epochs=30, validation_data=val_data)

# Save model weights
embedding_model.save_weights("siamese_model_weights.h5")

# Plot training history (loss and accuracy)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()

# Compute and display cosine similarity
cosine_similarity = metrics.CosineSimilarity()
sample = next(iter(train_data))
anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding_model(resnet.preprocess_input(anchor)),
    embedding_model(resnet.preprocess_input(positive)),
    embedding_model(resnet.preprocess_input(negative)),
)
positive_similarity = cosine_similarity(anchor_embedding, positive_embedding).numpy()
negative_similarity = cosine_similarity(anchor_embedding, negative_embedding).numpy()
print("Positive similarity:", positive_similarity)
print("Negative similarity", negative_similarity)

# Compute ROC and AUC
actual_labels = [1] * positive_similarity.size + [0] * negative_similarity.size
import numpy as np
predicted_scores = np.vstack([positive_similarity, negative_similarity]).squeeze()
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(actual_labels, predicted_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Save the Siamese model as a TensorFlow SavedModel
embedding_model.save("embedding_model.h5")
