import cv2
import os
import numpy as np
from scipy.stats import pearsonr


def resize_image(image, target_size=(256, 256)):
    """Resizes image to target size if it is not already the correct size."""
    if image.shape != target_size:
        print(f"🔄 Resizing image from {image.shape} to {target_size}")
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


def find_best_horizontal_shift(ref_image, target_image):
    """Finds best horizontal shift using Pearson correlation."""
    max_corr, best_shift = -1, 0

    # Resize images if necessary
    ref_image = resize_image(ref_image)
    target_image = resize_image(target_image)

    height, width = ref_image.shape
    ref_crop, target_crop = ref_image[:height // 3, :], target_image[:height // 3, :]

    for shift in range(-width // 2, width // 2):
        shifted_image = np.roll(target_crop, shift, axis=1)
        corr_coeff, _ = pearsonr(ref_crop.flatten(), shifted_image.flatten())
        if corr_coeff > max_corr:
            max_corr, best_shift = corr_coeff, shift
    return best_shift, max_corr


def align_and_save_images(ref_image_path, input_root_dir, output_root_dir):
    """Aligns all images in the directory while preserving directory structure."""
    ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)

    if ref_image is None:
        print(f"❌ Error: Failed to load reference image from {ref_image_path}")
        return

    _, ref_image = cv2.threshold(ref_image, 127, 255, cv2.THRESH_BINARY)
    ref_image = resize_image(ref_image)  # Ensure reference image is 512x512

    for subject in range(1, 125):
        subject = f"{subject:03d}"
        subject = str(subject)  # Convert subject number to string
        subject_path = os.path.join(input_root_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        for sequence in os.listdir(subject_path):
            sequence_path = os.path.join(subject_path, sequence)
            if not os.path.isdir(sequence_path):
                continue

            output_sequence_path = os.path.join(output_root_dir, subject, sequence)
            os.makedirs(output_sequence_path, exist_ok=True)

            for image_file in os.listdir(sequence_path):
                if image_file.endswith('.png'):
                    image_path = os.path.join(sequence_path, image_file)
                    target_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if target_image is None:
                        print(f"❌ Error: Failed to load target image: {image_path}")
                        continue

                    _, target_image = cv2.threshold(target_image, 127, 255, cv2.THRESH_BINARY)

                    # Resize target image if necessary
                    target_image = resize_image(target_image)

                    best_shift, _ = find_best_horizontal_shift(ref_image, target_image)
                    aligned_image = np.roll(target_image, best_shift, axis=1)

                    output_image_path = os.path.join(output_sequence_path, image_file)
                    cv2.imwrite(output_image_path, aligned_image)
                    print(f"✅ Aligned and saved: {output_image_path} with shift {best_shift}")


# Paths
ref_image_path = r"G:\CASIA_B\Aligned_Images2\refined_average_image.png"
input_root_dir = r"G:\CASIA_B\GaitDatasetB-silh_090_Normalized"
output_root_dir = r"G:\CASIA_B\GaitDatasetB-silh_090_Normalized_Alinged"

# Align images
align_and_save_images(ref_image_path, input_root_dir, output_root_dir)