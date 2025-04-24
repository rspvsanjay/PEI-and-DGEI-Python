import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from collections import defaultdict

# Path to your data
root_path = r"E:\CASIA_B\GaitDatasetB-silh_090_Normalized_Alinged_DGEI13"

def load_data(root_path):
    subjects = []
    num_poses = 11
    skip_subjects = []

    for subject_name in os.listdir(root_path):
        subject_path = os.path.join(root_path, subject_name)
        if os.path.isdir(subject_path):
            sequences = []
            for seq_name in sorted(os.listdir(subject_path)):
                seq_path = os.path.join(subject_path, seq_name)
                poses = {}
                print(f"Loading: {subject_name} with {seq_name}")
                for pose_idx in range(1, num_poses + 1):
                    pose_file = f"pose{pose_idx:03d}.png"
                    pose_path = os.path.join(seq_path, pose_file)

                    if os.path.exists(pose_path):
                        image = Image.open(pose_path).convert('L')
                        pose_data = np.array(image).astype(np.float32) / 255.0
                        poses[pose_idx - 1] = pose_data.flatten()
                    else:
                        print(f"Missing: {pose_path}")
                sequences.append(poses)

            if len(sequences) >= 3:
                subjects.append((subject_name, sequences))
            else:
                print(f"Skipping Subject '{subject_name}' — Only {len(sequences)} sequences found.")
                skip_subjects.append((subject_name, sequences))

    if skip_subjects:
        print("\nSubjects skipped due to insufficient sequences:")
        for subject_name, sequences in skip_subjects:
            print(f"Subject: {subject_name} — Found {len(sequences)} sequences")

    return subjects

def train_classifiers(subjects):
    num_poses = 13
    classifiers = {}

    for pose_idx in range(num_poses):
        X_train = []
        y_train = []

        for subject_id, (subject_name, sequences) in enumerate(subjects):
            train_sequences = []
            for i, seq_name in enumerate(sorted(os.listdir(os.path.join(root_path, subject_name)))):
                if seq_name.startswith('nm') and seq_name in ['nm-03', 'nm-04', 'nm-05', 'nm-06']:
                    train_sequences.append(sequences[i])

            for seq_index, pose_dict in enumerate(train_sequences):
                if pose_idx in pose_dict:
                    X_train.append(pose_dict[pose_idx])
                    y_train.append(subject_id)
                else:
                    print(f"Training skip: Subject {subject_name} | Sequence {seq_index + 1} — Pose {pose_idx + 1} missing.")

        if len(X_train) == 0:
            print(f"Skipping training for Pose {pose_idx + 1} — no samples found.")
            continue

        X_train = np.array(X_train, dtype=np.float64)

        # PCA
        pca = PCA()
        score = pca.fit_transform(X_train)
        explained = pca.explained_variance_ratio_ * 100

        sm, no_components = 0, 0
        for k in range(len(explained)):
            sm += explained[k]
            no_components += 1
            if sm >= 99.4029:
                break

        mat1 = score[:, :no_components]
        m = np.mean(X_train, axis=0)

        # LDA
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(mat1, y_train)

        classifiers[pose_idx] = (pca, lda, m, no_components)

    return classifiers

def classify_test(subjects, classifiers):
    num_poses = 13
    correct = 0
    total = 0

    for subject_id, (subject_name, sequences) in enumerate(subjects):
        test_sequences = []
        for i, seq_name in enumerate(sorted(os.listdir(os.path.join(root_path, subject_name)))):
            if seq_name.startswith('bg') or seq_name.startswith('cl') or seq_name in ['nm-01', 'nm-02']:
                test_sequences.append(sequences[i])

        for seq_index, pose_dict in enumerate(test_sequences):
            pose_predictions = []

            for pose_idx in range(num_poses):
                if pose_idx in pose_dict and pose_idx in classifiers:
                    pose_sample = pose_dict[pose_idx].reshape(1, -1)
                    pca, lda, m, no_components = classifiers[pose_idx]

                    img_mean = pose_sample - m
                    img_proj = np.dot(img_mean, pca.components_.T)
                    test_features = img_proj[:, :no_components]

                    proba = lda.predict_proba(test_features)[0]
                    classes = lda.classes_
                    max_prob = np.max(proba)
                    max_index = np.argmax(proba)
                    predicted_class = classes[max_index]

                    pose_predictions.append({"pose": pose_idx + 1, "predicted_class": predicted_class, "probability": max_prob})
                    print(f"Pose {pose_idx + 1} Prediction for Subject '{subject_name}' | Sequence {seq_index + 4}: {predicted_class + 1} with Probability: {max_prob:.4f}")
                else:
                    if pose_idx not in pose_dict:
                        print(f"Testing skip: Subject {subject_name} | Sequence {seq_index + 4} — Pose {pose_idx + 1} missing.")

            probability_sum = defaultdict(float)
            for entry in pose_predictions:
                cls = entry['predicted_class']
                prob = entry['probability']
                probability_sum[cls] += prob

            if probability_sum:
                max_class = max(probability_sum, key=probability_sum.get)
                max_value = probability_sum[max_class]
                print(f"Class with max probability: {max_value}: ", (max_class + 1))
                if max_class == subject_id:
                    correct += 1
                total += 1
            else:
                print("No predictions available.")

    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\nTotal Sequences: {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Sequence-Level Accuracy: {accuracy:.2f}%")
    else:
        print("No valid sequences were classified.")

if __name__ == "__main__":
    subjects = load_data(root_path)
    print(f"Total number of subjects: {len(subjects)}")
    classifiers = train_classifiers(subjects)
    classify_test(subjects, classifiers)