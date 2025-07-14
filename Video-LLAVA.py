import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import numpy as np
from decord import VideoReader, cpu
import logging
import os
import av
import pandas as pd
import time
from collections import Counter
emotions = [
    "happiness",  # hap
    "sadness",    # sad
    "anger",      # ang
    "fear",       # fea
    "surprise",   # sur
    "disgust",    # dis
    "excitement", # exc
    "frustration",# fru
    "neutral",    # neu
    "boredom",    # bor
    "embarrassment", # emb
    "contentment",   # con
    "relief",        # rel
    "pride",         # pri
    "shame",         # sha
    "guilt",         # gui
    "admiration",    # adm
    "love",          # lov
    "confusion",     # con
    "other"          # oth
]

# Mapping from numbers to emotions
emotion_mapping = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
    7: "other"
}

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def read_video_decord(video_path, indices):
    logging.debug(f"Reading frames for indices: {indices}")
    
    try:
        # Initialize the video reader
        vr = VideoReader(video_path)
        
        # Ensure indices are within the valid range
        valid_indices = [i for i in indices if i < len(vr)]
        
        if valid_indices:
            # Read the frames for the given indices
            frames = vr.get_batch(valid_indices).asnumpy()
            frames = np.array(frames)  # Convert to a single NumPy array
            logging.debug(f"Number of frames read: {len(frames)}")
        else:
            frames = np.empty((0,))  # Empty array if no valid indices
            logging.warning("No valid frames found for the given indices.")
    
    except Exception as e:
        logging.error(f"Error reading video: {e}")
        frames = np.empty((0,))
    
    return frames

def parse_annotations(file_path, video_base_name):
    logging.info(f"Reading annotations from {file_path} for video {video_base_name}")
    df = pd.read_csv(file_path)

    filtered_df = df[(df['video_file'].str.startswith(video_base_name))]
    
    # Convert true emotion using the mapping and filter out -1 values
    annotations = []
    for _, row in filtered_df.iterrows():
        true_emotion = row['emotion']
        if true_emotion != -1:  # Ignore entries with true emotion -1
            emotion_string = emotion_mapping.get(true_emotion)
            annotations.append({
                'video_file': row['video_file'],
                'segment_name': row['segment_name'],
                'start_frame': row['start_frame'],
                'end_frame': row['end_frame'],
                'emotion': emotion_string  # Use the mapped emotion string
            })

    logging.debug(f"Found {len(annotations)} annotations.")
    return annotations

def process_batch(batch_frames, model, processor, prompt_template):
    if not batch_frames.size:
        logging.error("The batch_frames provided are empty.")
        raise ValueError("The batch_frames provided are empty.")
    
    logging.debug(f"Processing batch with {batch_frames.shape[0]} frames")
    
    # Convert to a single numpy array if batch_frames is a list
    if isinstance(batch_frames, list):
        batch_frames = np.array(batch_frames)
    
    inputs = processor(text=prompt_template, videos=batch_frames, return_tensors="pt")
    
    start_time = time.time()
    try:
        generate_ids = model.generate(**inputs, max_length=400)
    except Exception as e:
        logging.error(f"Error during model generation: {e}")
        raise
    end_time = time.time()
    
    predicted_emotions = processor.batch_decode(generate_ids, skip_special_tokens=True)
    predicted_emotions = [text.split("ASSISTANT:")[1].strip().strip("' ") if "ASSISTANT:" in text else text for text in predicted_emotions]
    logging.debug(f"Predicted emotions: {predicted_emotions}")

    return predicted_emotions, end_time - start_time


# Load the model and processor
logging.info("Loading the model and processor")
model = VideoLlavaForConditionalGeneration.from_pretrained("Models/Video-LLaVA-7B-hf")
processor = VideoLlavaProcessor.from_pretrained("Video-LLaVA-7B-hf")

#annotation_file_path = "pre-processed/df_iemocap.csv"
#base_video_dir_path = "IEMOCAP_full_release/"

annotation_file_path = "AFFWILD/combined_segments.csv"

video_dir_path = "AFFWILD/Videos/Full"

emotion_counts = Counter()  # To keep track of all chosen emotions



batch_size = 16  # Define batch size
frame_size = (224, 224)  # Define the desired frame size

# Get all video files in the current session
video_files = [f for f in os.listdir(video_dir_path) if f.startswith('9-15-1920x1080')]

results = []



for video_file in video_files:
    video_base_name = os.path.splitext(video_file)[0]  # Base name of the video file without extension
    video_path = os.path.join(video_dir_path, video_file)
    logging.info(f"Processing video file: {video_file}")
    
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()  # FPS
    logging.info(f"Video FPS: {fps}")
    
    # Retrieve the matching annotations for this video
    annotations = parse_annotations(annotation_file_path, video_base_name)
    
    for annotation in annotations:
        file, segment, startframe, endframe, true_emotion = annotation["video_file"], annotation["segment_name"], annotation["start_frame"], annotation['end_frame'],annotation['emotion']
        indices = list(range(startframe, endframe + 1))  # Including end_idx
        
        # Read frames for the current segment
        clip = read_video_decord(video_path, indices)
        
        if clip.size == 0:
            logging.warning(f"No frames extracted for annotation: start_time={start_time}, end_time={end_time}")
            continue  # Skip if no frames were extracted

        # Initialize time tracker for segment
        segment_start_time = time.time()
        
        all_predictions = []
        for start_idx in range(0, clip.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, clip.shape[0])
            batch_frames = clip[start_idx:end_idx]
            
            try:
                #prompt_template = "USER: <video> Please specify the predominant emotion shown in this video segment. Choose from: 'neu' (neutral), 'fru' (frustration), 'sad' (sadness), 'sur' (surprise), 'ang' (anger), 'hap' (happiness), 'exc' (excitement), 'fea' (fear), 'dis' (disgust), 'oth' (other). Only generate the emotion, no other text. ASSISTANT:"
                prompt_template = "USER: <video> Please describe this video ASSISTANT: "
                predicted_emotions, time_taken = process_batch(batch_frames, model, processor, prompt_template)
                prompt_template = "USER: <video> " + predicted_emotions + " Please specify the predominant emotion shown in this video segment. Choose from: 'neu' (neutral), 'fru' (frustration), 'sad' (sadness), 'sur' (surprise), 'ang' (anger), 'hap' (happiness), 'exc' (excitement), 'fea' (fear), 'dis' (disgust), 'oth' (other). Only generate the emotion, no other text. ASSISTANT:"
                predicted_emotions, time_taken = process_batch(batch_frames, model, processor, prompt_template)

            except Exception as e:
                logging.error(f"Failed to process batch starting at index {start_idx}: {e}")
                continue

            all_predictions.extend(predicted_emotions)
            
        # Calculate total time for segment
        segment_end_time = time.time()
        segment_time_taken = segment_end_time - segment_start_time
        logging.info(f"Time taken for segment: {segment_time_taken:.2f}s")

        Average_Time = segment_time_taken / clip.shape[0] if clip.shape[0] > 0 else 0  # Avoid division by zero
        if all_predictions:
            frame_accuracy = [1.0 if true_emotion in emotion else 0.0 for emotion in all_predictions]
            segment_accuracy = sum(frame_accuracy) / len(frame_accuracy)
            segment_accuracy_percentage = segment_accuracy * 100
            
            # Aggregate predictions for the segment
            most_common_emotion = Counter(all_predictions).most_common(1)[0][0]
            accuracy = 1.0 if true_emotion in most_common_emotion else 0.0
            logging.info(f"Most common predicted emotion: {most_common_emotion}")
            
            results.append({
                "True Emotion": true_emotion,
                "Predicted Emotion": most_common_emotion,
                "Time Taken (s)": segment_time_taken,  # Total time for segment
                "Average Time(s)": Average_Time,  # Time for 1 frame
                "Frame Accuracy (%)": segment_accuracy_percentage,  # Accuracy of the segment as a percentage
                "Segment Accuracy": accuracy,
                "Count": len(clip),
                "Segment": segment,
            })


# Save results to CSV
results_df = pd.DataFrame(results)
output_file_path = "processed_results_LLAVA.csv"
results_df.to_csv(output_file_path, index=False)

logging.info(f"Results saved to {output_file_path}")

# Log the counts of all chosen emotions
logging.info(f"Emotion counts: {dict(emotion_counts)}")
