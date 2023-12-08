import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Flatten, MaxPooling2D, PReLU, Dense, Dropout, BatchNormalization,
                                     GlobalAveragePooling2D, GaussianNoise)
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation,RandomTranslation, RandomZoom, RandomContrast, RandomHeight, RandomWidth
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import cv2
import matplotlib.pyplot as plt

# Constants
BASE_PATH = "C:/Users/101231186/Desktop/AML"
TRAIN_PATH = os.path.join(BASE_PATH, "train_data")
TEST_PATH = os.path.join(BASE_PATH, "test_data")
VAL_PATH = os.path.join(BASE_PATH, "val_data")
num_classes = 4000

def count_directories(path):
    return len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

def create_labels_file(dataset_path, file_name):
    image_paths, image_labels = [], []
    class_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    for class_label, class_dir in enumerate(class_dirs):
        class_dir_path = os.path.join(dataset_path, class_dir)
        image_files = os.listdir(class_dir_path)

        for image_file in image_files:
            image_file_path = os.path.join(class_dir_path, image_file)
            if os.path.isfile(image_file_path):
                image_paths.append(image_file_path)
                image_labels.append(class_label)

    with open(file_name, 'w') as f:
        for path, label in zip(image_paths, image_labels):
            f.write(f"{path} {label}\n")

def load_labels(file_path, base_img_path):
    list_IDs, labels = [], {}
    with open(file_path, 'r') as file:
        for line in file:
            relative_image_path, label = line.strip().rsplit(' ', 1)
            full_image_path = os.path.join(base_img_path, relative_image_path)
            list_IDs.append(full_image_path)
            labels[full_image_path] = int(label)
    return list_IDs, labels

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=4000, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load and preprocess the image
            img = self.load_and_preprocess_image(ID, self.dim)
            X[i,] = img

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


    def load_and_preprocess_image(self, image_path, target_size):
        # Load the image file
        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)  # Resize image
        # No need to convert to RGB as preprocess_input will handle it
        image = preprocess_input(image)  # Use ResNet50V2's preprocess_input
        return image

# Count directories
train_classes_count = count_directories(TRAIN_PATH)
test_classes_count = count_directories(TEST_PATH)
val_classes_count = count_directories(VAL_PATH)
print('Total training classes:', train_classes_count)
print('Total testing classes:', test_classes_count)
print('Total validation classes:', val_classes_count)

# Create labels files
create_labels_file(TRAIN_PATH, 'labels_train.txt')
create_labels_file(TEST_PATH, 'labels_test.txt')
create_labels_file(VAL_PATH, 'labels_val.txt')

# Load labels and initialize generators
train_list_IDs, train_labels = load_labels('labels_train.txt', TRAIN_PATH)
test_list_IDs, test_labels = load_labels('labels_test.txt', TEST_PATH)
val_list_IDs, val_labels = load_labels('labels_val.txt', VAL_PATH)

training_generator = DataGenerator(train_list_IDs, train_labels)
testing_generator = DataGenerator(test_list_IDs, test_labels)
validation_generator = DataGenerator(val_list_IDs, val_labels)

# Load pre-trained ResNet50V2 model (excluding the top layer)
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Data Augmentation
data_augmentation = Sequential([
    RandomFlip('horizontal_and_vertical'),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.1),
    RandomTranslation(height_factor=0.1, width_factor=0.1),
    RandomHeight(0.1),
    RandomWidth(0.1)
])

# Unfreezing some layers of the base model for fine-tuning
for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
    layer.trainable = True

# Define the model
model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Learning Rate Scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
lr_scheduler = LearningRateScheduler(scheduler)

# Define metrics
top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1)

# Compile the model with a different optimizer or learning rate
model.compile(optimizer=Adam(learning_rate=1e-4),  # Adjusted learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with callbacks
history = model.fit(training_generator,
                    epochs=40,  # Increased epochs
                    validation_data=validation_generator,
                    callbacks=[EarlyStopping(verbose=1, patience=3), checkpoint, lr_scheduler])

# Evaluate the model
evaluation_results = model.evaluate(validation_generator)
print(f"Validation Loss: {evaluation_results[0]}, Validation Accuracy: {evaluation_results[1]}")

from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity

# After training and evaluation, compute the average accuracy per class
def average_accuracy_per_class(model, generator, num_classes):
    # Initialize a list to store correct counts for each class
    correct_counts = [0] * num_classes
    total_counts = [0] * num_classes

    # Loop through the batches in the generator
    for images, labels in generator:
        predictions = model.predict(images)
        predicted_classes = tf.argmax(predictions, axis=1)
        true_classes = tf.argmax(labels, axis=1)

        # Update the correct count and total count for each class
        for i in range(len(true_classes)):
            true_class_index = true_classes[i]
            total_counts[true_class_index] += 1
            if predicted_classes[i] == true_class_index:
                correct_counts[true_class_index] += 1

    # Compute the average accuracy for each class
    average_accuracies = [correct / total if total != 0 else 0 for correct, total in zip(correct_counts, total_counts)]

    # Return the overall average accuracy across classes
    return sum(average_accuracies) / len(average_accuracies)

# Compute ROC curve and AUC for each class
def roc_auc_per_class(model, generator, num_classes):
    all_fpr = []
    all_tpr = []
    all_auc = []

    for images, labels in generator:
        predictions = model.predict(images)

        # Compute ROC curve and AUC for each class
        for i in range(num_classes):
            try:
                # Check if both positive and negative examples are present
                if len(np.unique(labels[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
                    roc_auc = auc(fpr, tpr)
                else:
                    # If only one class is present, assign arbitrary values
                    fpr, tpr, roc_auc = [0], [0], 0.5

                all_fpr.append(fpr)
                all_tpr.append(tpr)
                all_auc.append(roc_auc)
            except Exception as e:
                print(f"Error in computing ROC for class {i}: {e}")
                fpr, tpr, roc_auc = [0], [0], 0.5  # Default values in case of an error
                all_fpr.append(fpr)
                all_tpr.append(tpr)
                all_auc.append(roc_auc)

    return all_fpr, all_tpr, all_auc

# Compute average cosine similarity between predicted embeddings and true embeddings
def average_cosine_similarity(model, generator):
    cosine_similarities = []

    for images, labels in generator:
        predictions = model.predict(images)
        
        # Ensure that labels are one-hot encoded
        one_hot_labels = keras.utils.to_categorical(labels, num_classes=model.output_shape[-1])

        # Compute cosine similarity for each pair of prediction and true label
        for i in range(predictions.shape[0]):
            sim = cosine_similarity([predictions[i]], [one_hot_labels[i]])[0][0]
            cosine_similarities.append(sim)

    return np.mean(cosine_similarities)

# Then call this function as before
train_avg_acc_per_class = average_accuracy_per_class(model, training_generator, num_classes)
val_avg_acc_per_class = average_accuracy_per_class(model, validation_generator, num_classes)

print("\nAverage Accuracy Per Class (Training): {:.2f}%".format(train_avg_acc_per_class * 100))
print("Average Accuracy Per Class (Validation): {:.2f}%".format(val_avg_acc_per_class * 100))

# Assess the performance on the train and validation sets
train_evaluation = model.evaluate(training_generator)
validation_evaluation = model.evaluate(validation_generator)

# Unpack the evaluation metrics for both train and validation
train_loss, train_acc = train_evaluation
val_loss, val_acc = validation_evaluation

# Displaying the performance metrics as percentages
print('\nTraining Metrics:')
print('Loss:', train_loss)
print('Accuracy: {:.2f}%'.format(train_acc * 100))

print('\nValidation Metrics:')
print('Loss:', val_loss)
print('Accuracy: {:.2f}%'.format(val_acc * 100))

# Retrieve the history object from model training
training_stats = history.history

# Derive accuracy and loss metrics from the history
train_acc_values = training_stats['accuracy']
val_acc_values = training_stats['val_accuracy']
train_loss_values = training_stats['loss']
val_loss_values = training_stats['val_loss']

# Visualize the training accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(train_acc_values, label='Training Accuracy')
plt.plot(val_acc_values, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy for Training vs. Validation')

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(train_loss_values, label='Training Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss for Training vs. Validation')

plt.tight_layout()
plt.show()

# Compute ROC curves and plot them
train_fpr, train_tpr, train_auc = roc_auc_per_class(model, training_generator, num_classes)
val_fpr, val_tpr, val_auc = roc_auc_per_class(model, validation_generator, num_classes)

import random

def plot_subset_roc_curves(fpr, tpr, auc, num_classes, subset_size=10):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')

    selected_classes = random.sample(range(num_classes), subset_size)
    for i in selected_classes:
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {auc[i]:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Subset of Classes)')
    plt.legend()
    plt.show()

# Plot for Training and Validation Set
plot_subset_roc_curves(train_fpr, train_tpr, train_auc, num_classes=4000, subset_size=10)
plot_subset_roc_curves(val_fpr, val_tpr, val_auc, num_classes=4000, subset_size=10)

# Compute and print average cosine similarity
train_avg_cosine_similarity = average_cosine_similarity(model, training_generator)
val_avg_cosine_similarity = average_cosine_similarity(model, validation_generator)

print("\nAverage Cosine Similarity (Training): {:.4f}".format(train_avg_cosine_similarity))
print("Average Cosine Similarity (Validation): {:.4f}".format(val_avg_cosine_similarity))
plt.show()

# Save the model
model.save_weights("final_weights.h5")
model.save("final_model.h5")