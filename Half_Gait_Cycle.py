import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Paths
reference_path = r"G:\CASIA_B\Aligned_Images2\refined_average_image.png"
base_folder = r"G:\CASIA_B\GaitDatasetB-silh_090_Normalized_Alinged"
output_folder = r"G:\CASIA_B\Half_Gait_Cycles"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)


def compute_similarity(frame1, frame2):
    """Computes normalized cross-correlation similarity between two grayscale images."""
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)
    norm_factor = np.linalg.norm(frame1) * np.linalg.norm(frame2)

    if norm_factor == 0:
        return 0  # Avoid division by zero

    return np.sum(frame1 * frame2) / norm_factor


def simple_moving_average(data, window_size=5):
    """Applies simple moving average (SMA) smoothing to a list of coefficients."""
    if len(data) < window_size:  # If not enough data points, return original
        return np.array(data)

    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def find_nearest_peaks(original_signal, smoothed_peaks):
    """Finds the nearest peaks in the original signal corresponding to the smoothed peaks."""
    original_peaks, _ = find_peaks(original_signal)
    nearest_peaks = []
    for sp in smoothed_peaks:
        nearest = min(original_peaks, key=lambda x: abs(x - sp))
        nearest_peaks.append(nearest)
    return nearest_peaks


def extract_half_gait_cycles():
    """Extracts half gait cycles based on similarity to reference frame."""
    # Load reference frame
    reference_frame = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if reference_frame is None:
        print(f"Error: Could not read reference frame at {reference_path}")
        return

    for subject in os.listdir(base_folder):  # Loop through subjects
        subject_path = os.path.join(base_folder, subject)

        if not os.path.isdir(subject_path):
            continue

        for sequence in os.listdir(subject_path):  # Loop through sequences
            sequence_path = os.path.join(subject_path, sequence)

            if not os.path.isdir(sequence_path):
                continue

            frame_files = sorted(os.listdir(sequence_path))  # Ensure correct order
            coefficients = []
            frame_paths = []

            # Compute similarity for each frame
            for frame_name in frame_files:
                frame_path = os.path.join(sequence_path, frame_name)
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

                if frame is None:
                    continue

                coeff = compute_similarity(reference_frame, frame)
                coefficients.append(coeff)
                frame_paths.append(frame_path)

            if not coefficients:
                print(f"Skipping {sequence} (No valid frames)")
                continue

            # Apply simple moving average (SMA) smoothing
            smoothed_coeffs = simple_moving_average(coefficients, window_size=5)

            # Find peaks in smoothed similarity coefficients
            smoothed_peaks, _ = find_peaks(smoothed_coeffs, height=np.mean(smoothed_coeffs))

            if len(smoothed_peaks) < 2:
                print(f"Skipping {sequence} (Not enough peaks in smoothed data)")
                continue

            # Adjust smoothed peak indices to match original data length
            offset = (len(coefficients) - len(smoothed_coeffs)) // 2
            smoothed_peaks = [p + offset for p in smoothed_peaks if p + offset < len(coefficients)]

            # Find the nearest peaks in original (non-smoothed) coefficients
            refined_peaks = find_nearest_peaks(coefficients, smoothed_peaks)

            if len(refined_peaks) < 2:
                print(f"Skipping {sequence} (Not enough refined peaks)")
                continue

            # Select the two middle consecutive peaks
            mid_index = len(refined_peaks) // 2
            if mid_index == 0 or mid_index >= len(refined_peaks) - 1:
                print(f"Skipping {sequence} (Not enough consecutive peaks)")
                continue

            peak1, peak2 = refined_peaks[mid_index], refined_peaks[mid_index + 1]
            selected_frames = frame_paths[peak1:peak2 + 1]

            # Output directory (mirrors input structure)
            output_subject_path = os.path.join(output_folder, subject)
            output_sequence_path = os.path.join(output_subject_path, sequence)
            os.makedirs(output_sequence_path, exist_ok=True)

            # Save selected frames
            for frame_path in selected_frames:
                frame_name = os.path.basename(frame_path)
                output_path = os.path.join(output_sequence_path, frame_name)
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(output_path, frame)

            print(f"Saved half gait cycle frames for {sequence} in {output_sequence_path}")

# Run extraction
extract_half_gait_cycles()