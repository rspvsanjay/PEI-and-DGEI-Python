import os
import shutil
import cv2
import numpy as np
from collections import defaultdict

# Paths
half_cycle_folder = r"G:\CASIA_B\Half_Gait_Cycles"
output_folder = r"G:\CASIA_B\Selected_Half_Cycles"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)


def collect_half_cycles():
    """Collects all half-cycle sequences with 'right_to_left' in the name and groups them by frame count."""
    frame_count_dict = defaultdict(list)

    # Loop through subjects
    for subject in os.listdir(half_cycle_folder):
        subject_path = os.path.join(half_cycle_folder, subject)

        if not os.path.isdir(subject_path):
            continue

        # Loop through sequences
        for sequence in os.listdir(subject_path):

            sequence_path = os.path.join(subject_path, sequence)

            if not os.path.isdir(sequence_path):
                continue

            # Count frames in this half-cycle sequence
            frame_files = sorted(os.listdir(sequence_path))
            frame_count = len(frame_files)

            if frame_count > 0:
                frame_count_dict[frame_count].append((subject, sequence, sequence_path))

    return frame_count_dict


def display_frame_counts(frame_count_dict):
    """Displays the distinct number of frames found across sequences."""
    print("\nDistinct frame counts found in sequences:")
    for frame_count, sequences in sorted(frame_count_dict.items()):
        print(f"- {frame_count} frames: {len(sequences)} sequences")


def compute_average_frames(sequences, frame_count, frame_count_output_folder):
    """Computes average frames when there are more than two sequences."""
    num_sequences = len(sequences)
    if num_sequences < 2:
        return  # No averaging needed

    print(f"Computing average frames for {num_sequences} sequences with {frame_count} frames...")

    # Load frames for averaging
    frame_sums = []
    for i in range(frame_count):
        frame_sums.append(None)  # Initialize empty list

    for subject, sequence, sequence_path in sequences:
        frame_files = sorted(os.listdir(sequence_path))
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(sequence_path, frame_file)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

            if frame is None:
                continue  # Skip corrupted images

            if frame_sums[i] is None:
                frame_sums[i] = frame.astype(np.float32)
            else:
                frame_sums[i] += frame.astype(np.float32)

    # Compute the average
    averaged_frames = [frame / num_sequences for frame in frame_sums if frame is not None]

    # Save the averaged frames
    avg_output_path = os.path.join(frame_count_output_folder, "Averaged_Frames")
    os.makedirs(avg_output_path, exist_ok=True)

    for i, avg_frame in enumerate(averaged_frames):
        output_frame_path = os.path.join(avg_output_path, f"frame_{i + 1:03d}.png")
        cv2.imwrite(output_frame_path, avg_frame.astype(np.uint8))  # Convert back to uint8

    print(f"Saved averaged frames for {frame_count} frames in {avg_output_path}")


def select_and_save_half_cycles(frame_count_dict):
    """Selects and saves 'right_to_left' sequences, grouping by frame count and computing averages if needed."""
    for frame_count, sequences in frame_count_dict.items():
        # Output folder for this specific frame count
        frame_count_output_folder = os.path.join(output_folder, f"Frames_{frame_count}")
        os.makedirs(frame_count_output_folder, exist_ok=True)

        # Save original sequences
        for subject, sequence, sequence_path in sequences:
            new_output_path = os.path.join(frame_count_output_folder, f"{subject}_{sequence}")
            os.makedirs(new_output_path, exist_ok=True)

            for frame_file in sorted(os.listdir(sequence_path)):
                src_path = os.path.join(sequence_path, frame_file)
                dst_path = os.path.join(new_output_path, frame_file)
                shutil.copy2(src_path, dst_path)

            print(f"Saved {subject}/{sequence} (Frame count: {frame_count}) to {new_output_path}")

        # If more than two sequences exist, compute averaged frames
        if len(sequences) > 1:
            compute_average_frames(sequences, frame_count, frame_count_output_folder)


# Collect and process half cycles
frame_count_dict = collect_half_cycles()
display_frame_counts(frame_count_dict)  # Display distinct frame counts
select_and_save_half_cycles(frame_count_dict)

print("\nSelection process completed.")