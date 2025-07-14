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

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from videollama2 import model_init, mm_infer
import traceback

cache_dir = os.getenv('TRANSFORMERS_CACHE', '~/.cache/huggingface')

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

frame_sizes = [
    (16,16),
    (32,32),
    (64,64),
    (128, 128),
    (224,224),

]
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logging.info(f"Hugging Face cache directory: {os.path.expanduser(cache_dir)}")

model1_path = 'MiniCPM-V-2_6'
model2_path = 'sharegpt4video-8b'
model3_path = 'Models/VideoLLaMA2-7B'


logging.info('Loading models and tokenizers...')
model1 = AutoModel.from_pretrained(model1_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model1 = model1.eval().cuda()
tokenizer1 = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True)

# Load model2 and tokenizer2 correctly
model_name = get_model_name_from_path(model2_path)
tokenizer2, model2, processor2, context_len = load_pretrained_model(
    model2_path, None, model_name)
model2 = model2.cuda().eval()

model3, processor3, tokenizer3 = model_init(model3_path)

model_list = [
    {
        'model_name': "MiniCPM-V-2_6",
        'model': model1,
        'tokenizer': tokenizer1,
        'processor': None  
    },
    {
        'model_name': "sharegpt4video-8b",
        'model': model2,
        'tokenizer': tokenizer2,
        'processor': processor2
    },
    {
        'model_name': "VideoLLaMA2-7B",
        'model': model3,
        'tokenizer': tokenizer3,
        'processor': processor3
    }
]



modal = 'video'


def encode_video(video_path, frame_indices,FRAME_SIZE):
    try:
        # Initialize the VideoReader with the given video path
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)  # Total number of frames in the video

        # Filter frame_indices to ensure they are within bounds
        valid_indices = [idx for idx in frame_indices if idx < num_frames]
        if not valid_indices:
            logging.warning(f"No valid frame indices found in {frame_indices}")
            return []

        # Log the number of valid frames being read
        logging.info(f'Reading {len(valid_indices)} frames based on provided indices.')
        
        # Read and convert valid frames to numpy arrays
        original_frames = vr.get_batch(valid_indices).asnumpy()

        # Resize all frames to consistent size (specified by FRAME_SIZE)
        frames = [Image.fromarray(v.astype('uint8')).resize(FRAME_SIZE, Image.LANCZOS) for v in original_frames]

        # Check if all frames have the same size
        sizes = set(frame.size for frame in frames)
        if len(sizes) > 1:
            raise ValueError(f"Frames have inconsistent sizes: {sizes}")

        logging.info(f'Number of frames read and resized: {len(frames)}')
        return frames
    
    except IndexError as e:
        logging.error(f"IndexError: {e}. One or more frame indices are out of bounds.")
    except Exception as e:
        logging.error(f"An error occurred while encoding video: {e}")
    
    # Return an empty list if an error occurred or if no frames were processed
    return []


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

    
    
def process_frames_with_model(model_name, model, tokenizer, frames, processor,prompt_template):
    #prompt_template = "What is the emotion in this video?"
    #prompt_template = "Please specify the predominant emotion shown in this video segment. Choose from: 'neu' (neutral), 'fru' (frustration), 'sad' (sadness), 'sur' (surprise), 'ang' (anger), 'hap' (happiness), 'exc' (excitement), 'fea' (fear), 'dis' (disgust), 'oth' (other). Only generate the emotion, no other text."
    
    start_time = time.time()
    if model_name == "MiniCPM-V-2_6":
        msgs = [
            {'role': 'user', 'content': [frame for frame in frames] + [prompt_template]}, 
        ]

        params = {
            "use_image_id": False,
            "max_slice_nums": 2  
        }
        try:
            logging.info('Processing frames with MiniCPM-V-2_6 model...')
            logging.debug(f"Number of frames to process: {len(frames)}")

            answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                **params
            )
            
            end_time = time.time()
            if "ASSISTANT:" in answer:
                answer = answer.split("ASSISTANT:")[1].strip()
                
            logging.info(f"Predicted Emotion: {answer}")
            return answer, end_time - start_time
        except Exception as e:
            logging.error(f"Error processing with MiniCPM-V-2_6 model: {e}")
            return [],0
            
    elif model_name == "sharegpt4video-8b":
        #prompt_template = "What is the emotion in this video?"

        try:
            logging.info('Processing frames with sharegpt4video-8b...')
            logging.debug(f"Number of frames to process: {len(frames)}")
            
            image_size = (224, 224)
            image_tensor = process_images(frames, processor, model.config)[0]

            input_ids = tokenizer_image_token(
                prompt_template, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).to(device=model.device, non_blocking=True)

            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device=model.device, non_blocking=True),
                    image_sizes=[image_size],
                    do_sample=False,
                    top_p=0.7,  # Adjusted
                    temperature=0.5,  # Adjusted
                    num_beams=5,
                    max_new_tokens=100,  # Adjusted
                    pad_token_id=pad_token_id,
                    use_cache=True
                )
                

                answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


                if "ASSISTANT:" in answer:
                    answer = answer.split("ASSISTANT:")[1].strip()
                    

                logging.info(f"Predicted Emotion: {answer}")
            
            end_time = time.time()
            return answer, end_time - start_time

        except Exception as e:
            logging.error(f"Error processing with {model_name}: {e}\n{traceback.format_exc()}")

       
    elif model_name == "VideoLLaMA2-7B":
        logging.info('Processing frames with VideoLLaMA2-7B model...')
        logging.debug(f"Number of frames to process: {len(frames)}")
        try:
            answer = mm_infer(
                processor[modal](frames),
                prompt_template,
                model=model,
                tokenizer=tokenizer,
                modal=modal
            )
            
            end_time = time.time()
            if "ASSISTANT:" in answer:
                answer = answer.split("ASSISTANT:")[1].strip()
                
            
            logging.info(f"Predicted Emotion: {answer}")
            return answer, end_time - start_time
        except Exception as e:
            logging.error(f"Error processing with VideoLLaMA2-7B model: {e}")
            return [],0
            
annotation_file_path = "AFFWILD/combined_segments.csv"


# Dictionary to hold emotion counts for each model
emotion_counts_per_model = {model_info['model_name']: Counter() for model_info in model_list}

batch_size = 64  # Define batch size

video_dir_path = "AFFWILD/Videos/Full"

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
        
        for FRAME_SIZE in frame_sizes:
            # Read frames for the current segment
            clip = encode_video(video_path, indices,FRAME_SIZE)

            if len(clip) == 0:
                logging.warning(f"No frames found in folder: {segment}")
                continue

            # Process frames with each model
            for model_info in model_list:
                model_name = model_info['model_name']
                model = model_info['model']
                tokenizer = model_info['tokenizer']
                processor = model_info['processor']
                
                logging.info(f"Processing with model: {model_name}")
            
                
                segment_start_time = time.time()
                all_predictions = []
                for start_idx in range(0, len(clip), batch_size):
                    batch_frames = clip[start_idx:start_idx + batch_size]
                    
                    try:
                        prompt = "Please specify the predominant emotion shown in this video segment. Choose from: 'neu' (neutral), 'fru' (frustration), 'sad' (sadness), 'sur' (surprise), 'ang' (anger), 'hap' (happiness), 'exc' (excitement), 'fea' (fear), 'dis' (disgust), 'oth' (other). Only generate the emotion, no other text."
                        if model_name == "sharegpt4video-8b":
                            prompt = "What is the emotion in this video?"
                        #prompt = "Please describe this video"
                        predicted_emotions, time_taken = process_frames_with_model(model_name, model, tokenizer, batch_frames, processor,prompt)
                        # Prepare the new prompt for the second prediction
                        #prompt = predicted_emotions + " Please specify the predominant emotion shown in this video segment. Choose from: 'neu' (neutral), 'fru' (frustration), 'sad' (sadness), 'sur' (surprise), 'ang' (anger), 'hap' (happiness), 'exc' (excitement), 'fea' (fear), 'dis' (disgust), 'oth' (other). Only generate the emotion, no other text."
                        
                        # Second prediction with the updated prompt
                        #predicted_emotions, time_taken = process_frames_with_model(model_name, model, tokenizer, batch_frames, processor, prompt)
                        # Count emotions for this specific model
                        for emotion in emotions:
                            if emotion in predicted_emotions:
                                emotion_counts_per_model[model_name][emotion] += 1
                    except Exception as e:
                        logging.error(f"Failed to process batch starting at index {start_idx}: {e}")
                        continue

                    all_predictions.append(predicted_emotions)
                # Calculate total time for segment
                segment_end_time = time.time()
                segment_time_taken = segment_end_time - segment_start_time
                logging.info(f"Time taken for segment with {model_name}: {segment_time_taken:.2f}s")
                
                Average_Time = segment_time_taken / len(clip) if len(clip) > 0 else 0  # Avoid division by zero
                if all_predictions:
                    frame_accuracy = [1.0 if true_emotion in emotion else 0.0 for emotion in all_predictions]
                    segment_accuracy = sum(frame_accuracy) / len(frame_accuracy)
                    segment_accuracy_percentage = segment_accuracy * 100
                    
                    # Aggregate predictions for the segment
                    most_common_emotion = Counter(all_predictions).most_common(1)[0][0]

                    accuracy = 1.0 if true_emotion in most_common_emotion else 0.0
                    logging.info(f"Most common predicted emotion with {model_name}: {most_common_emotion}")

                    results.append({
                        "True Emotion": true_emotion,
                        "Predicted Emotion": most_common_emotion,
                        "Time Taken (s)": segment_time_taken,  # Total time for segment
                        "Average Time(s)": Average_Time,  # Time for 1 frame
                        "Frame Accuracy (%)": segment_accuracy_percentage,  # Accuracy of the segment as a percentage
                        "Segment Accuracy": accuracy,
                        "Count": len(clip),
                        "Model Name": model_name,  # Keep track of the model used
                        "Segment": segment,
                        "Frame Size": FRAME_SIZE
                    })

                    # Update emotion counts for this model
                    #emotion_counts.update(all_predictions)

# Print or log the emotion counts per model
for model_name, emotion_counts in emotion_counts_per_model.items():
    logging.info(f"Emotion counts for {model_name}: {emotion_counts}")

# Save results to CSV
results_df = pd.DataFrame(results)
output_file_path = "processed_results.csv"
results_df.to_csv(output_file_path, index=False)

logging.info(f"Results saved to {output_file_path}")

