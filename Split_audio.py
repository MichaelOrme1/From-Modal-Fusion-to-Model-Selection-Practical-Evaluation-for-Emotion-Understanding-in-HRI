import os
import pandas as pd
import librosa
import moviepy.editor as mp
import soundfile as sf  
import numpy as np

def load_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        # Skip the first line and read the remaining lines
        return [int(line.strip()) for idx, line in enumerate(file) if idx > 0]

# Function to extract audio from video and return the audio array
def extract_audio_from_video(video_path):
    video = mp.VideoFileClip(video_path)
    audio_array = video.audio.to_soundarray(fps=22000)  # Set an appropriate sample rate
    return audio_array, video.fps  # Return audio array and FPS

# Set the directory containing videos and annotations
video_directory = "AFFWILD/Videos"  
annotation_directory = "AFFWILD/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set"
audio_directory = "AFFWILD/Audio"

# List to hold all CSV data across videos
combined_csv_data = []

# Iterate over all files in the directory
for root, dirs, files in os.walk(video_directory):
    for filename in files:
        if filename.endswith(".mp4"):
            full_path = os.path.join(root, filename)
            video_name = filename[:-4]  # Get the video name without extension
            video_path = os.path.join(root, filename)
            annotation_path = os.path.join(annotation_directory, f"{video_name}.txt")

            # Check if the annotation file exists
            if not os.path.exists(annotation_path):
                print(f"Annotation file not found for {video_name}. Skipping...")
                continue

            # Extract audio from video
            audio, fps = extract_audio_from_video(video_path)

            # Load annotations from the text file
            annotations = load_annotations(annotation_path)

            # Initialize segment counter and data for this video
            segment_counter = 1
            grouped_frames = []
            current_emotion = annotations[0]
            indices = []

            # Group frames by emotion
            for i, emotion in enumerate(annotations):
                if emotion == current_emotion:
                    indices.append(i)
                else:
                    # Record the previous segment information
                    if indices:
                        grouped_frames.append((current_emotion, indices))

                    current_emotion = emotion
                    indices = [i]  # Start a new group

            # Append the last group if any
            if indices:
                grouped_frames.append((current_emotion, indices))

            # Process each emotion group to extract audio segments and write to CSV
            sr = 22000  # Sample rate for the audio
            for emotion, indices in grouped_frames:
                #print(f"Emotion: {emotion}, Start Frame: {indices[0]}, End Frame: {indices[-1]}, Total Frames: {len(indices)}")
                start_time = indices[0] / fps  # Convert frame index to time
                end_time = indices[-1] / fps
                start_sample = int(start_time * sr)  # Convert time to audio samples
                end_sample = int(end_time * sr)

                # Extract the corresponding audio segment
                audio_segment = audio[start_sample:end_sample]
                #print(f"Min value: {audio_segment.min()}, Max value: {audio_segment.max()}")

                
                audio_segment = audio_segment.astype(np.float32)
                
                if audio_segment.ndim == 1:  # Mono audio
                    audio_segment = audio_segment.reshape(-1, 1)

                #print(f"Audio segment shape after: {audio_segment.shape}, dtype: {audio_segment.dtype}")
                # Create the audio save directory if it doesn't exist
                segment_dir = os.path.join(audio_directory, "Segment")
                os.makedirs(segment_dir, exist_ok=True)  # Create directory if it doesn't exist

                # Prepare the segment filename and write the audio segment
                segment_filename = os.path.join(segment_dir, f'{video_name}_{segment_counter}.wav')
                sf.write(segment_filename, audio_segment, sr)  # Transpose if necessary

                # Append to combined CSV data
                combined_csv_data.append({
                    'video_file': video_name,
                    'segment_name': f'{video_name}_{segment_counter}',
                    'start_frame': indices[0],
                    'end_frame': indices[-1],
                    'emotion': emotion
                })

                # Increment the segment counter after saving
                segment_counter += 1

            print(f"Processed {video_name}: Extracted {segment_counter - 1} audio segments.")

# Save all collected CSV data to a single CSV file
if combined_csv_data:
    combined_csv_df = pd.DataFrame(combined_csv_data)
    combined_csv_path = os.path.join(annotation_directory, 'combined_segments.csv')
    combined_csv_df.to_csv(combined_csv_path, index=False)
    print(f"All segments saved to {combined_csv_path}.")
else:
    print("No segments were extracted.")

