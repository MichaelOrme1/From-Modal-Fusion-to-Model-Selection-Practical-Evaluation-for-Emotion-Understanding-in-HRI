import pandas as pd
from pathlib import Path
from moviepy.editor import VideoFileClip
import opensmile

# Paths to directories
annotations_dir = Path("AFFWILD/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set")
primary_video_folder = Path("AFFWILD/Videos/batch1")
secondary_video_folder = Path("AFFWILD/Videos/batch2")
tertiary_video_folder = Path("AFFWILD/Videos/newvids")
audio_folder = Path("AFFWILD/Audio/")
combined_features_file = Path("AFFWILD/Audio/features/audio.csv")

# Initialize OpenSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
)

def extract_audio_from_video(video_file, output_audio_path):
    try:
        video = VideoFileClip(str(video_file))
        audio = video.audio
        audio.write_audiofile(str(output_audio_path))
    except Exception as e:
        print(f"Error extracting audio from {video_file}: {e}")

def extract_features_with_opensmile(audio_file_path):
    try:
        features = smile.process_file(str(audio_file_path))
        if features.empty:
            return features  # Return empty DataFrame if no features were extracted
        # Replace NaN values with -1
        features.fillna(-1, inplace=True)
        return features
    except Exception as e:
        print(f"Error extracting features from {audio_file_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def process_annotation(annotation_file):
    extensions = ['.mp4', '.avi']
    video_file_found = False
    
    try:
        # Try to find the video file with the given extensions
        for ext in extensions:
            for folder in [primary_video_folder, secondary_video_folder, tertiary_video_folder]:
                video_file = folder / (annotation_file.stem + ext)
                if video_file.exists():
                    video_file_found = True
                    break  # Exit the extensions loop if video file is found
            if video_file_found:
                break  # Exit the folders loop if video file is found

        if not video_file_found:
            print(f"Video file {annotation_file.stem} not found in any of the directories. Skipping.")
            return pd.DataFrame()  # Return an empty DataFrame to signal no features

        # Extract audio if not already present
        audio_file = audio_folder / (video_file.stem + '.wav')
        if not audio_file.exists():
            extract_audio_from_video(video_file, audio_file)

        # Extract features using OpenSMILE
        features = extract_features_with_opensmile(audio_file)
        if not features.empty:
            # Calculate total number of features and add frame numbers
            num_features = len(features)
            
            # Define the number of digits in frame numbers (5 for zero-padded to 5 digits)
            num_digits = 5
            
            # Add formatted frame numbers to features
            features['frame_number'] = [f"{i+1:0{num_digits}d}" for i in range(num_features)]
            features['video_id'] = video_file.stem  # Add a column for the video ID
            features['video_frame'] = features['video_id'] + "_" + features['frame_number']  # Combine video_id with frame_number
            return features
        else:
            print(f"No features extracted for {audio_file}")
            return pd.DataFrame()  # Return an empty DataFrame if no features were extracted

    except Exception as e:
        print(f"Error processing annotation {annotation_file}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

# Accumulate all features in a list
all_features = []

# Process each annotation file
for annotation_file in annotations_dir.glob("*.txt"):
    features = process_annotation(annotation_file)
    if not features.empty:
        all_features.append(features)

# Combine all features into a single DataFrame
if all_features:
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_features.to_csv(combined_features_file, index=False)
else:
    print("No features to save.")



# Print a sample of the merged data
sample_data = pd.read_csv(combined_features_file)
print("\nSample of merged data:")
print(sample_data.head())  # Display the first few rows

# After saving the combined features to CSV
print("\nDataFrame checks:")
print("Columns:", combined_features.columns)
print("Data Types:\n", combined_features.dtypes)
print("Missing Values:\n", combined_features.isnull().sum())
print("Number of duplicate rows:", combined_features.duplicated().sum())