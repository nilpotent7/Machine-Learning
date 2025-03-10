import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def z_score_standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    standardized_data = (data - mean) / std
    return standardized_data

def FindFiles(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

# Define the 26 classes (letters A-Z).
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
num_classes = len(letters)

# Find all fonts in the given directory
fonts = FindFiles("Fonts")  # Adjust the path as needed

# Image parameters
img_width, img_height = 50, 50    # Image dimensions
symbol_size = (30,50)                  # Font size (adjust as needed)
version = 25

# Lists to collect images and labels
data_images = []  # Each image will be stored as a (2500, 1) numpy array (flattened from 50x50)
data_labels = []  # One-hot encoded vector of length 26

# Calculate total iterations for progress tracking:
# For each letter (26), we render two versions (uppercase and lowercase),
# for each font and for 2 variations.
total_iterations = num_classes * 2 * len(fonts) * version

with tqdm(total=total_iterations, desc="Generating dataset") as pbar:
    for i, letter in enumerate(letters):
        for case in ["upper", "lower"]:
            # Choose symbol based on case; both cases get the same label.
            symbol = letter if case == "upper" else letter.lower()
            for fontpath in fonts:
                for v in range(version):  # Two variations per letter per font per case
                    # Create a blank grayscale image (one channel) with a black background (0)
                    img = np.zeros((img_height, img_width), dtype=np.uint8)
                    img_pil = Image.fromarray(img, mode='L')
                    draw = ImageDraw.Draw(img_pil)
                    
                    font = ImageFont.truetype(fontpath, random.randrange(symbol_size[0], symbol_size[1]))

                    # Compute bounding box of the symbol using textbbox to determine text dimensions.
                    bbox = draw.textbbox((0, 0), symbol, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Compute position to center the symbol in the image.
                    # Subtract bbox[0] and bbox[1] to compensate for any offsets.
                    x = (img_width - text_width) / 2 - bbox[0]
                    y = (img_height - text_height) / 2 - bbox[1]
                    position = (x, y)

                    # Draw the symbol in white (255)
                    draw.text(position, symbol, font=font, fill=255)

                    # Convert the PIL image back to a numpy array and reshape to (2500, 1)
                    image_array = np.array(img_pil).reshape(-1, 1)
                    data_images.append(image_array)

                    # Create one-hot encoded label for the letter (both cases share the same label)
                    one_hot = np.zeros(num_classes, dtype=np.uint8)
                    one_hot[i] = 1
                    one_hot.reshape(one_hot.shape[0], 1)
                    data_labels.append(one_hot)

                    pbar.update(1)

# Convert lists to numpy arrays.
data_images = np.array(data_images)  # Shape: (num_samples, 2500, 1)
data_labels = np.array(data_labels)  # Shape: (num_samples, 26)

data_images = z_score_standardize(data_images)

# Save the complete dataset as a compressed NPZ file.
np.savez_compressed("dataset.npz", images=data_images, labels=data_labels)
print("Complete dataset saved to 'dataset.npz'.")

# -----------------------------------------------------------------------------
# Function to split the dataset with a customizable train_ratio.
def split_dataset(images, labels, train_ratio=0.8, shuffle=True):
    num_samples = images.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    train_size = int(num_samples * train_ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    train_images = images[train_idx]
    train_labels = labels[train_idx]
    test_images = images[test_idx]
    test_labels = labels[test_idx]
    return train_images, train_labels, test_images, test_labels

# Customize the training split (e.g., 80% training, 20% testing)
train_ratio = 0.8
train_images, train_labels, test_images, test_labels = split_dataset(data_images, data_labels, train_ratio=train_ratio)
print(f"Dataset split into {train_images.shape[0]} training samples and {test_images.shape[0]} testing samples.")

# Save the split dataset into a separate NPZ file.
np.savez_compressed("dataset.npz", train_images=train_images, train_labels=train_labels, 
                    test_images=test_images, test_labels=test_labels)
print("Split dataset saved to 'dataset_split.npz'.")
