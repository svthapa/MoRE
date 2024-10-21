import os
import numpy as np
from PIL import Image
from skimage import exposure
from concurrent.futures import ProcessPoolExecutor

def preprocess_image(image_path):
    """
    Preprocess an image: Convert to grayscale, resize, apply adaptive histogram equalization,
    and scale pixel values.
    """
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((224, 224))  # Resize to 224x224
        image_np = np.array(image) / 255.0  # Convert to numpy array and scale pixel values to [0, 1]
        image_eq = exposure.equalize_adapthist(image_np)  # Apply adaptive histogram equalization
        image_eq = (image_eq * 255).astype(np.uint8)  # Scale back to [0, 255] and convert to uint8
        return image_eq.flatten()  # Return flattened image for easier aggregation
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.array([])  # Return an empty array in case of an error

def calculate_mean_std(image_arrays):
    """
    Calculate the mean and standard deviation across all pixel values of the preprocessed images.
    """
    all_pixels = np.concatenate(image_arrays, axis=0)
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)
    return mean, std

def main():
    """
    Main function to preprocess images and calculate mean and std of the dataset.
    """
    with open('../data/cxr_paths.txt', 'r') as file:
    # Read lines from the file
        image_paths = [line.strip() for line in file]

    # Use ProcessPoolExecutor to process images in parallel
    with ProcessPoolExecutor(max_workers=7) as executor:
        image_arrays = list(executor.map(preprocess_image, image_paths))

    # Filter out any empty arrays returned due to errors
    image_arrays = [arr for arr in image_arrays if arr.size > 0]
    
    if image_arrays:
        mean, std = calculate_mean_std(image_arrays)
        print(f"Mean: {mean}, Std: {std}")
    else:
        print("No images were processed successfully.")

if __name__ == "__main__":
    main()