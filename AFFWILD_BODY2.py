import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Load YOLO
yolo_config_path = 'Models/yolov3.cfg'
yolo_weights_path = '/Models/yolov3.weights'
yolo_classes_path = '/Models/coco.names'

net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
with open(yolo_classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define body parts
body_parts = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle']
body_header = ['Frame', 'Emotion'] + [f"{part}_x" for part in body_parts] + [f"{part}_y" for part in body_parts]

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

# Paths to the annotation directory and video folders
annotations_dir = Path("AFFWILD/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set")
primary_video_folder = Path("AFFWILD/Videos/batch1")
secondary_video_folder = Path("40018022/AFFWILD/Videos/batch2")
tertiary_video_folder = Path("AFFWILD/Videos/newvids")
output_csv_file = Path("output_body_ALL2.csv")

# Load pose detection model
pose_proto = "Models/pose_deploy_linevec.prototxt"
pose_model = "Models/pose_iter_440000.caffemodel"
pose_net = cv2.dnn.readNetFromCaffe(pose_proto, pose_model)

extensions = ['.mp4', '.avi']

# Initialize a list to store results
results = []

# Iterate through all annotation files in the directory
for annotation_file in annotations_dir.glob("*.txt"):
    # Read the annotation file
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()[1:]  # Skip header

    video_file_found = False
    for ext in extensions:
        # Try to find the video file in the primary folder first
        video_file = primary_video_folder / (annotation_file.stem + ext)
        if video_file.exists():
            video_file_found = True
            break  # Exit the extensions loop if video file is found

        # If not found in the primary folder, try the secondary folder
        video_file = secondary_video_folder / (annotation_file.stem + ext)
        if video_file.exists():
            video_file_found = True
            break  # Exit the extensions loop if video file is found

        # If not found in the secondary folder, try the tertiary folder
        video_file = tertiary_video_folder / (annotation_file.stem + ext)
        if video_file.exists():
            video_file_found = True
            break  # Exit the extensions loop if video file is found

    if not video_file_found:
        print(f"Video file {annotation_file.stem} not found in any of the directories. Skipping.")
        continue  # Skip to the next annotation file

    cap = cv2.VideoCapture(str(video_file))

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        continue  # Skip to the next annotation file

    frame_number = 0

    # Iterate through images and annotations
    for i, annotation in enumerate(annotations):
        emotion_index = int(annotation.strip())  # Corresponding emotion index
        emotion = emotion_map.get(emotion_index, 'Unknown')  # Map emotion index to emotion string

        ret, frame = cap.read()
        if not ret:
            # No more frames to read
            break

        frame_number += 1
        unique_id = f"{annotation_file.stem}_{frame_number:05d}"

        height, width = frame.shape[:2]

        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Run the detection
        outs = net.forward(output_layers)

        # Process the detection results
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == 'person':  # Class ID for 'person'
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(f"indices: {indices}")

        # Check if indices is not empty and process correctly
        if len(indices) > 0:
            # Flatten the indices array and ensure it's a 1D array
            indices = indices.flatten() if len(indices.shape) > 1 else indices
            
            # Access the largest box from the indices
            if indices.size > 0:
                largest_box = max((boxes[i] for i in indices), key=lambda box: box[2] * box[3])
                points = [(-1, -1)] * len(body_parts)
                
                # Extract box coordinates
                x, y, w, h = largest_box

                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(width - x, w)
                h = min(height - y, h)

                # Crop the image to the largest bounding box
                cropped_image = frame[y:y+h, x:x+w]
                
                frameHeight, frameWidth = cropped_image.shape[:2]
                
                # Check if cropped image is valid
                if cropped_image.size == 0:
                    print(f"Error: Cropped image is empty for frame {frame_number}.")
                    continue  # Skip to the next frame

                # Prepare the cropped image for pose detection
                inWidth = 368
                inHeight = 368
                pose_blob = cv2.dnn.blobFromImage(cropped_image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
                pose_net.setInput(pose_blob)
                pose_output = pose_net.forward()

                # Process pose detection results
                H, W = pose_output.shape[2], pose_output.shape[3]

                for i in range(pose_output.shape[1]):
                    prob_map = pose_output[0, i, :, :]
                    _, conf, _, point = cv2.minMaxLoc(prob_map) 
                    x = int((frameWidth * point[0]) / W)
                    y = int((frameHeight * point[1]) / H)
                    if conf > 0.1:
                        points.append((x, y))
                    else:
                        points.append((-1, -1))

                # Prepare data for DataFrame
                row = [unique_id, emotion] + [coord for point in points for coord in point]

                # Append to results list
                results.append(row)
            else:
                print("No valid indices found after NMS. Skipping frame.")

    cap.release()  # Release the video capture object after processing the video

# Create a DataFrame and save to CSV
df = pd.DataFrame(results, columns=body_header)
df.to_csv(output_csv_file, index=False)
