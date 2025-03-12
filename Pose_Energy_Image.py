import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

def get_file_list(path):
    """Returns a sorted list of valid image files in the given directory."""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return sorted([f for f in os.listdir(path) if f.lower().endswith(valid_extensions)])

def get_folder_list(path):
    """Returns a sorted list of directories in the given path."""
    return sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

def find_shortest_path(G, start, end):
    path = nx.shortest_path(G, source=start, target=end, weight='weight')
    length = nx.shortest_path_length(G, source=start, target=end, weight='weight')
    return path, length


def draw_graph(G):
    pos = nx.spring_layout(G)  # Positioning nodes

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=12)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("Graph Representation")
    plt.show()

def create_graph(pose_length, correlations, start, end):
    G = nx.DiGraph()
    edges = []
    last_frame = str(f"{len(correlations[0]):03d}")
    frames = len(correlations[0])
    for pose_index in range(pose_length):
        pose = str(f"{(pose_index+1):03d}")
        frame = str(f"{1:03d}")
        node = pose + "," + frame
        # print("node: ", node)
        edges.append((start, node, correlations[pose_index, 0]))
        node = pose + "," + last_frame
        # print("node: ", node)
        edges.append((node, end, 0.99))

        for frame_index in range(1, frames):
            frame1 = str(f"{(frame_index):03d}")
            frame2 = str(f"{(frame_index + 1):03d}")
            node1 = pose + "," + frame1
            # print("node1: ", node1)
            node2 = pose + "," + frame2
            edges.append((node1, node2, correlations[pose_index, frame_index]))
            frame1 = str(f"{(frame_index):03d}")
            pose1 = str(f"{(pose_index + 1):03d}")
            node1 = pose1 + "," + frame1
            # print("node1: ", node1)
            frame2 = str(f"{(frame_index + 1):03d}")
            pose2 = str(f"{(pose_index + 2):03d}")
            if pose_index == (pose_length - 1):
                pose2 = str(f"{1:03d}")
            node2 = pose2 + "," + frame2
            # print("node2: ", node2)
            if pose_index < (pose_length - 1):
                # print("Weight: ", correlations[pose_index+1, frame_index])
                edges.append((node1, node2, correlations[pose_index+1, frame_index]))
            else:
                # print("Weight: ", correlations[0, frame_index])
                edges.append((node1, node2, correlations[0, frame_index]))

    G.add_weighted_edges_from(edges)
    return G


# Function to compute and display the average image for each pose group
def compute_and_display_avg_images(pose_groups, image_folder, save_path, subject, sequence):
    for pose, img_list in pose_groups.items():
        images = []

        for img_name in img_list:
            img_path = os.path.join(image_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
            if img is not None:
                images.append(img.astype(np.float64))  # Convert to float for averaging

        if images:
            avg_img = np.mean(images, axis=0)  # Compute the average image
            avg_img = np.uint8(avg_img)  # Convert back to uint8 for display
            # Save the averaged image
            save_filepath_dir = os.path.join(save_path, subject, sequence)
            os.makedirs(save_filepath_dir, exist_ok=True)
            save_filename = f"pose{pose}.png"
            save_filepath = os.path.join(save_filepath_dir, save_filename)
            cv2.imwrite(save_filepath, avg_img)


# Path settings
pose_path = r'G:\CASIA_B\Selected_Half_Cycles\Frames_10\Averaged_Frames'
path1 = r"G:\CASIA_B\GaitDatasetB-silh_090_Normalized_Alinged"
save_path = r"G:\CASIA_B\GaitDatasetB-silh_090_Normalized_Alinged_PEI"

pose_files = get_file_list(pose_path)
poses = []

for pose_file in pose_files:
    file_path = os.path.join(pose_path, pose_file)
    if not os.path.exists(file_path):
        print(f"Skipping missing file: {file_path}")
        continue

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {file_path}")
        continue

    img = cv2.resize(img, (256, 256))
    poses.append(img.astype(np.float64))


folder_list = get_folder_list(path1)
print("folder_list len: ", len(folder_list))

#for folder in folder_list:
for subject in range(1, 125):#125):
    subject = f"{subject:03d}"
    subject = str(subject)  # Convert subject number to string
    path2 = os.path.join(path1, subject)#path2 = os.path.join(path1, folder)
    subfolders = get_folder_list(path2)
    print("subject: ", subject)

    #for subfolder in subfolders:
    for sequence in range(0,len(subfolders)):
        path3 = os.path.join(path2, subfolders[sequence])
        print("subfolders[sequence]: ", subfolders[sequence])
        image_files = get_file_list(path3)
        print("image_files: ", image_files)

        if len(image_files) < 3:
            print(f"Skipping {path3}, not enough images.")
            continue

        correlations = np.zeros((len(poses), len(image_files)))

        for pose_idx, pose_img in enumerate(poses):
            for img_idx, img_file in enumerate(image_files):
                img_path = os.path.join(path3, img_file)
                if not os.path.exists(img_path):
                    continue

                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = image.astype(np.float64)
                correlations[pose_idx, img_idx] = np.corrcoef(pose_img.flatten(), image.flatten())[0, 1]


        correlations = 1-correlations
        pose_length = len(correlations)
        # print("pose_length: ", pose_length)
        start = "start"
        end = "end"
        graph = create_graph(pose_length, correlations, start, end)
        # draw_graph(graph)
        path, length = find_shortest_path(graph, start, end)
        print("path: ", path)
        print("path length: ", len(path))
        print("Total weight over the path: ", length)

        # Remove "start" and "end"
        filtered_path = [p for p in path if p not in ["start", "end"]]
        # print("filtered_path: ", filtered_path)

        pose_groups = defaultdict(list)

        for img_file, pose_frame in zip(image_files, filtered_path):
            pose, _ = pose_frame.split(',')  # Extract the pose number
            pose_groups[pose].append(img_file)

        print("pose_groups: ", pose_groups)
        compute_and_display_avg_images(pose_groups, path3, save_path, subject, subfolders[sequence])