import cv2
import os
import numpy as np
from scipy.stats import pearsonr


def count_white_pixels(image_path):
    """Count white pixels in an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return np.sum(image == 255) if image is not None else 0


def extract_direction(filename):
    """Extracts direction from filename."""
    if "left_to_right" in filename:
        return "left_to_right"
    elif "right_to_left" in filename:
        return "right_to_left"
    return None


def process_sequences_for_subject(subject, subject_dir, save_dir):
    """Processes each sequence and saves the best image."""
    os.makedirs(save_dir, exist_ok=True)
    for sequence in os.listdir(subject_dir):
        sequence_path = os.path.join(subject_dir, sequence)
        if not os.path.isdir(sequence_path):
            continue
        image_files = sorted(os.listdir(sequence_path))
        if len(image_files) <= 20:
            continue
        max_white_pixels, best_image, best_image_direction = 0, None, ""
        for image_file in image_files[5:-5]:  # Ignore first and last 5 frames
            image_path = os.path.join(sequence_path, image_file)
            white_pixels = count_white_pixels(image_path)
            if white_pixels > max_white_pixels:
                max_white_pixels = white_pixels
                best_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                best_image_direction = extract_direction(image_file)
        if best_image is not None:
            if best_image_direction == "left_to_right":
                best_image = cv2.flip(best_image, 1)
            save_path = os.path.join(save_dir, f"{subject}_{sequence}_right_to_left.png")
            cv2.imwrite(save_path, best_image)
            print(f"Saved: {save_path}")


def compute_refined_average_image(image_dir, save_path):
    """Computes refined average image after filtering based on correlation."""
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    if not image_files:
        print("No images found in directory.")
        return
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)
    height, width = first_image.shape
    sum_image = np.zeros((height, width), dtype=np.float64)
    image_list = []
    for image_file in image_files:
        image = cv2.imread(os.path.join(image_dir, image_file), cv2.IMREAD_GRAYSCALE)
        if image.shape != (height, width):
            continue
        sum_image += image
        image_list.append(image)
    average_image = np.uint8(sum_image / len(image_list))
    correlations = {img: np.corrcoef(image.flatten(), average_image.flatten())[0, 1] for img, image in
                    zip(image_files, image_list)}
    avg_corr = np.mean(list(correlations.values()))
    filtered_images = [image_list[idx] for idx, img_name in enumerate(image_files) if correlations[img_name] > avg_corr]
    if filtered_images:
        refined_avg = np.uint8(np.sum(filtered_images, axis=0) / len(filtered_images))
        cv2.imwrite(save_path, refined_avg)
        print(f"Refined average image saved: {save_path}")


def find_best_horizontal_shift(ref_image, target_image):
    """Finds best horizontal shift using Pearson correlation."""
    max_corr, best_shift = -1, 0
    height, width = ref_image.shape
    ref_crop, target_crop = ref_image[:height // 3, :], target_image[:height // 3, :]
    for shift in range(-width // 2, width // 2):
        shifted_image = np.roll(target_crop, shift, axis=1)
        corr_coeff, _ = pearsonr(ref_crop.flatten(), shifted_image.flatten())
        if corr_coeff > max_corr:
            max_corr, best_shift = corr_coeff, shift
    return best_shift, max_corr


def align_images(ref_image_path, target_dir, save_dir):
    """Aligns images based on the refined average image."""
    os.makedirs(save_dir, exist_ok=True)
    ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    _, ref_image = cv2.threshold(ref_image, 127, 255, cv2.THRESH_BINARY)
    for file_name in os.listdir(target_dir):
        if file_name.endswith('.png') and file_name != 'refined_average_image.png':
            target_image = cv2.imread(os.path.join(target_dir, file_name), cv2.IMREAD_GRAYSCALE)
            _, target_image = cv2.threshold(target_image, 127, 255, cv2.THRESH_BINARY)
            best_shift, _ = find_best_horizontal_shift(ref_image, target_image)
            aligned_image = np.roll(target_image, best_shift, axis=1)
            cv2.imwrite(os.path.join(save_dir, file_name), aligned_image)
            print(f"Aligned and saved: {file_name}, Shift: {best_shift}")


# Main execution
base_subject_dir = r"G:\GaitDatasetB-silh_090_Normalized"
best_images_dir = 'G:\\Best_Images_Right_to_Left'
refined_avg_path1 = os.path.join(best_images_dir, 'refined_average_image.png')
aligned_images_dir1 = 'G:\\Aligned_Images'
refined_avg_path2 = os.path.join(aligned_images_dir1, 'refined_average_image.png')
aligned_images_dir2 = 'G:\\Aligned_Images2'
refined_avg_path3 = os.path.join(aligned_images_dir2, 'refined_average_image.png')

# Step 1: Process subjects to extract best images
for subject in os.listdir(base_subject_dir):
    subject_dir = os.path.join(base_subject_dir, subject)
    if os.path.isdir(subject_dir):
        process_sequences_for_subject(subject, subject_dir, best_images_dir)

# Step 2: Compute refined average image from best images
compute_refined_average_image(best_images_dir, refined_avg_path1)

# Step 3: Align images using refined average
align_images(refined_avg_path1, best_images_dir, aligned_images_dir1)

# Step 4: Compute refined average image from first set of aligned images
compute_refined_average_image(aligned_images_dir1, refined_avg_path2)

# Step 5: Align images again using the second refined average image
align_images(refined_avg_path2, aligned_images_dir1, aligned_images_dir2)

# Step 6: Compute refined average image from first set of aligned images
compute_refined_average_image(aligned_images_dir2, refined_avg_path3)