import cv2
import numpy as np
import csv
from pathlib import Path
import torch
from l2cs import Pipeline  # Ensure this module is correctly installed and available

# Initialize gaze estimation pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gaze_pipeline = Pipeline(
    weights="Models/L2CSNet_gaze360.pkl",
    arch='ResNet50',  # Ensure this matches the available architecture options
    device=device
)

# Load the pre-trained facial landmark model
landmark_model = "Models/lbfmodel.yaml"  # Ensure this path and file type are correct
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(landmark_model)
print("Model loaded")

# Define the path to Haar cascade XML file
haarcascade_path = "Models/haarcascade_frontalface_default.xml"

# Load a pre-trained face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Define headers for CSV file
face_header = ['Unique_ID', 'Emotion'] + [f"Point_{i}_x" for i in range(68)] + [f"Point_{i}_y" for i in range(68)] + ['Pitch', 'Yaw', 'Gaze_Confidence']

# Define emotion mapping
emotion_map = {
    0: 'Neutral',
    1: 'Anger',
    2: 'Disgust',
    3: 'Fear',
    4: 'Happiness',
    5: 'Sadness',
    6: 'Surprise',
    7: 'Other'
}

# Paths to the annotation directory and image folders
annotations_dir = Path("AFFWILD/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set")
primary_image_folder = Path("AFFWILD/Faces/cropped_aligned")
secondary_image_folder = Path("AFFWILD/Faces/cropped_aligned_new_50_vids")
output_csv_file = Path("output_gaze_landmarks_ALL.csv")

# Prepare CSV writer
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(face_header)

    # Iterate through all annotation files in the directory
    for annotation_file in annotations_dir.glob("*.txt"):
        # Read the annotation file
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()[1:]  # Skip header

        # Iterate through images and annotations
        for i, annotation in enumerate(annotations):
            # Try to find the image in the primary folder first
            image_folder = primary_image_folder / annotation_file.stem
            if not image_folder.exists():
                # If not found in the primary folder, try the secondary folder
                image_folder = secondary_image_folder / annotation_file.stem
                if not image_folder.exists():
                    print(f"{image_folder} not found in any of the directories. Skipping.")
                    continue

            image_file = image_folder / f"{i + 1:05d}.jpg"  # Image filenames like 00001.jpg            
            emotion_index = int(annotation.strip())  # Corresponding emotion index
            emotion = emotion_map.get(emotion_index, 'Unknown')  # Map emotion index to emotion string

            # Create a unique ID by combining annotation file name and image name
            unique_id = f"{annotation_file.stem}_{image_file.name}"

            print(f"Processing {unique_id}")

            # Read the image
            frame = cv2.imread(str(image_file))
            if frame is None:
                print(f"Could not read image {image_file}. Skipping.")
                continue

            # Convert image to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Initialize landmarks with (-1, -1) for missing data
            face_landmarks = [(-1, -1)] * 68

            if len(faces) == 0:
                print("No faces detected. Skipping image.")
                continue

            for (x, y, w, h) in faces:
                try:
                    # Define the bounding box for each detected face
                    bbox = np.array([[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]])

                    # Detect facial landmarks within the bounding box
                    _, landmarks = landmark_detector.fit(gray_frame, bbox)

                    if landmarks:
                        landmarks = landmarks[0][0]  # Extract landmarks for the first face detected
                        for idx, (x, y) in enumerate(landmarks):
                            if idx < 68:
                                face_landmarks[idx] = (int(x), int(y))
                except Exception as e:
                    print(f"Error detecting landmarks for {unique_id}: {e}")
                    # If an error occurs, face_landmarks will remain as [(-1, -1)] * 68

            try:
                # Gaze estimation
                gaze_results = gaze_pipeline.step(frame)

                if len(gaze_results.pitch) > 0:  # Ensure gaze results are available
                    pitch = gaze_results.pitch[0]
                    yaw = gaze_results.yaw[0]
                    gaze_confidence = gaze_results.scores[0]
                else:
                    pitch, yaw, gaze_confidence = -1, -1, -1
                    print(f"No gaze results for image {unique_id}.")
            except Exception as e:
                print(f"Error during gaze estimation for {unique_id}: {e}")
                pitch, yaw, gaze_confidence = -1, -1, -1

            # Flatten landmarks into separate x and y lists
            x_coords = [x for (x, y) in face_landmarks]
            y_coords = [y for (x, y) in face_landmarks]

            # Write the landmarks and gaze results for the current frame to the CSV file
            face_row = [unique_id, emotion] + x_coords + y_coords + [pitch, yaw, gaze_confidence]
            writer.writerow(face_row)

print(f"Processing completed. Results saved to {output_csv_file}")
