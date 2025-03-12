import cv2
import numpy as np
import os

# Base directory containing all subjects
base_dir = r"G:\GaitDatasetB-silh_090"
output_dir = r"G:\GaitDatasetB-silh_090_Normalized"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Step 1: Iterate through each subject in the base directory
for subject in range(1, 125):
    subject_str = f"{subject:03d}"  # Convert subject number to zero-padded string
    subject_path = os.path.join(base_dir, subject_str)
    if not os.path.isdir(subject_path):
        continue  # Skip if not a directory

    # Step 2: Iterate through each sequence of the subject
    for sequence in os.listdir(subject_path):
        sequence_path = os.path.join(subject_path, sequence)

        if not os.path.isdir(sequence_path):
            continue  # Skip if not a directory

        # Ensure output path for subject & sequence exists
        output_sequence_path = os.path.join(output_dir, subject_str, sequence)
        os.makedirs(output_sequence_path, exist_ok=True)

        # Step 3: Iterate through each image in the sequence
        for image_file in os.listdir(sequence_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip non-image files

            image_path = os.path.join(sequence_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Skipping {image_path}, unable to load image.")
                continue

            # **Threshold and find contours (blobs)**
            _, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # **Filter blobs based on area**
            big_blobs = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            big_blobs = sorted(big_blobs, key=cv2.contourArea, reverse=True)[:10]

            # **Calculate centroids**
            centroids = []
            for cnt in big_blobs:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    centroids.append((cX, cY))

            # **Calculate distance stats**
            def calculate_distance_stats(centroids):
                stats = []
                for i, (x1, y1) in enumerate(centroids):
                    distances = []
                    sum_distance = 0
                    for j, (x2, y2) in enumerate(centroids):
                        if i != j:
                            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            distances.append(dist)
                            sum_distance += dist
                    avg_distance = sum_distance / (len(centroids) - 1) if len(centroids) > 1 else 0
                    deviation = np.std(distances) if len(distances) > 1 else 0
                    stats.append((sum_distance, avg_distance, deviation))
                return stats

            distance_stats = calculate_distance_stats(centroids)

            if not distance_stats:
                print(f"No valid blobs found in {image_path}. Skipping.")
                continue

            # **Find centroid with the smallest sum of distances**
            min_idx = np.argmin([stat[0] for stat in distance_stats])
            selected_centroid = centroids[min_idx]
            selected_avg = distance_stats[min_idx][1]
            selected_deviation = distance_stats[min_idx][2]
            threshold_distance = selected_avg + 1.5 * selected_deviation

            # **Select blobs within the threshold**
            selected_blobs = []
            selected_indices = []
            for i, centroid in enumerate(centroids):
                dist = np.sqrt((centroid[0] - selected_centroid[0]) ** 2 + (centroid[1] - selected_centroid[1]) ** 2)
                if dist <= threshold_distance:
                    selected_blobs.append(big_blobs[i])
                    selected_indices.append(i)

            # **Include all blobs up to the max selected index**
            if selected_indices:
                max_selected_index = max(selected_indices)
                selected_blobs = big_blobs[: max_selected_index + 1]

            # **Find the bounding box of all selected blobs**
            x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
            for blob in selected_blobs:
                x, y, w, h = cv2.boundingRect(blob)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x + w), max(y_max, y + h)

            blob_width, blob_height = x_max - x_min, y_max - y_min

            # **Resize while maintaining aspect ratio**
            new_height = 224  # Changed from 512 to 448
            aspect_ratio = blob_width / blob_height if blob_height > 0 else 1
            new_width = int(new_height * aspect_ratio)
            new_width = max(new_width, 1)

            final_size = 256  # Ensure final frame is 512x512

            # **Ensure new_width does not exceed 512**
            if new_width > final_size:
                new_width = final_size
                new_height = int(new_width / aspect_ratio)  # Adjust height to maintain aspect ratio

            # **Create blank 512x512 square frame**
            square_frame = np.zeros((final_size, final_size), dtype=np.uint8)

            # **Extract region of interest (ROI)**
            roi = image[y_min:y_max, x_min:x_max]

            # **Resize ROI to fit within limits**
            resized_roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # **Calculate vertical padding**
            top_padding = (final_size - new_height) // 2
            bottom_padding = final_size - (top_padding + new_height)

            # **Calculate horizontal offset (adjusted)**
            x_offset = (final_size - new_width) // 2

            # **Ensure correct slicing before assigning ROI**
            square_frame[top_padding:top_padding + new_height, x_offset:x_offset + new_width] = resized_roi

            # **Save the processed image**
            output_path = os.path.join(output_sequence_path, image_file)
            cv2.imwrite(output_path, square_frame)

            # **Print information**
            print(f"Processed and saved: {output_path}")

print("Processing completed for all subjects and sequences.")
