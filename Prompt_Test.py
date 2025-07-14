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

cache_dir = os.getenv('TRANSFORMERS_CACHE', '~/.cache/huggingface')





# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logging.info(f"Hugging Face cache directory: {os.path.expanduser(cache_dir)}")

model1_path = 'MiniCPM-V-2_6'
model2_path = 'sharegpt4video-8b'
model3_path = 'VideoLLaMA2-7B'


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


FRAME_SIZE = (224, 224)  # Consistent frame size
modal = 'video'

def encode_video(video_path, frame_indices):
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
    
    # Detect gender from the video base name
    if 'F' in video_base_name:
        video_gender = 'F'
    elif 'M' in video_base_name:
        video_gender = 'M'
    else:
        logging.warning(f"No gender information found in video base name: {video_base_name}")
        video_gender = None

    if video_gender:
        # Filter annotations for matching video base name and gender
        filtered_df = df[
            (df['wav_file'].str.startswith(video_base_name)) & 
            (df['wav_file'].str.contains(f"_{video_gender}"))
        ]
        
        # Extract annotations and wav_file
        annotations = filtered_df[['start_time', 'end_time', 'emotion', 'wav_file']].to_dict('records')
    else:
        # If no gender detected, return an empty list or handle accordingly
        annotations = []
    
    logging.debug(f"Found {len(annotations)} annotations.")
    return annotations, video_gender

    
    
def process_frames_with_model(model_name, model, tokenizer, frames, processor,prompt_template):
    
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

        try:
            logging.info('Processing frames with sharegpt4video-8b...')
            logging.debug(f"Number of frames to process: {len(frames)}")
            
            image_size = frames[0].size
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
            logging.error(f"Error processing with sharegpt4video-8b: {e}")

       
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
            




annotation_file_path = "df_iemocap.csv"
base_video_dir_path = "IEMOCAP_full_release/"

# Define the number of sessions
num_sessions = 1  

results = []
emotion_counts = Counter()  # To keep track of all chosen emotions

prompt_templates = [
    "What is the emotion in this video?",
    "Please specify the predominant emotion shown in this video segment.",
    "How does the character feel in this video?",
    "What is the main emotion displayed in this clip?",
    "Can you describe the emotion expressed in this video?",
    "What feelings are being portrayed in this video segment?",
    "How is the person in the video emotionally reacting?",
    "What is the emotional tone of this video?",
    "What emotion is primarily displayed in this scene?",
    "How does the person in the video seem to be feeling?",
    "What is the emotional state of the character in this video?",
    "Can you identify the emotion shown in this video?",
    "What emotion is being conveyed by the person in this clip?",
    "How would you describe the mood of this video?",
    "What is the dominant emotion in this video?",
    "What emotions are noticeable in this video?",
    "What is the emotional expression of the person in the video?",
    "Which emotion is the person in this video exhibiting?",
    "What does the person in the video seem to be feeling emotionally?",
    "What is the mood of the person in this video?",
    "What kind of emotion does the person in the video exhibit?",
    "How would you summarize the emotion in this video?",
    "What does the body language and facial expression convey emotionally in this video?",
    "Can you identify the emotional message in this video?",
    "What emotion is the person on the left feeling?",
    "Please specify the predominant emotion shown in this video segment. Choose from: 'neu' (neutral), 'fru' (frustration), 'sad' (sadness), 'sur' (surprise), 'ang' (anger), 'hap' (happiness), 'exc' (excitement), 'fea' (fear), 'dis' (disgust), 'oth' (other). Only generate the emotion, no other text.",
    "You are an expert in recognising emotions, what emotions are present here?",
    "Give a score for the emotion most present in this video",
    "What emotion do you think an expert would see in this video?",
    "Describe the video",
    "What is the predominant emotion shown by the subject in this video segment? Please choose one of the following: adoration, amusement, anxiety, disgust, emphatic pain, fear, surprise",
    "You are an expert in recognising emotions, please specify the predominant emotion shown in this video segment. Choose from: 'neu' (neutral), 'fru' (frustration), 'sad' (sadness), 'sur' (surprise), 'ang' (anger), 'hap' (happiness), 'exc' (excitement), 'fea' (fear), 'dis' (disgust), 'oth' (other). Only generate the emotion, no other text."
]
batch_size = 64  # Define batch size

for i in range(1, num_sessions + 1):
    session_dir = f"Session{i}"
    video_dir_path = os.path.join(base_video_dir_path, session_dir, "dialog", "avi", "DivX")

    if os.path.isdir(video_dir_path):
        logging.info(f"Processing video files in {video_dir_path}")

        # Get all video files in the current session
        video_files = [f for f in os.listdir(video_dir_path) if f.endswith('.avi') and not f.startswith('._') and f.startswith('Ses01F_impro01')]

        for video_file in video_files:
            video_base_name = os.path.splitext(video_file)[0]  # Base name of the video file without extension
            video_path = os.path.join(video_dir_path, video_file)
            logging.info(f"Processing video file: {video_file}")
            
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()  # FPS
            logging.info(f"Video FPS: {fps}")

            # Retrieve the matching annotations for this video
            annotations, video_gender = parse_annotations(annotation_file_path, video_base_name)

            for annotation in annotations:
                start_time, end_time, true_emotion, wav_file = annotation["start_time"], annotation["end_time"], annotation["emotion"], annotation["wav_file"]
                logging.info(f"Processing annotation: start_time={start_time}, end_time={end_time}, true_emotion={true_emotion}")
                
                # Calculate frame indices for the current segment
                start_idx = int(start_time * fps)
                end_idx = int(end_time * fps)
                indices = list(range(start_idx, end_idx + 1))  # Including end_idx
                
                # Read frames for the current segment
                clip = encode_video(video_path, indices)

                if len(clip) == 0:
                    logging.warning(f"No frames extracted for annotation: start_time={start_time}, end_time={end_time}")
                    continue

                # Initialize lists to hold predictions, average times, and accuracies for each prompt across models
                all_predictions = [[[] for _ in range(len(prompt_templates))] for _ in range(len(model_list))]
                average_times = [[[] for _ in range(len(prompt_templates))] for _ in range(len(model_list))]
                model_accuracies = [[[] for _ in range(len(prompt_templates))] for _ in range(len(model_list))]

                # Loop through each model in the model list
                for model_idx, model_info in enumerate(model_list):
                    model_name = model_info['model_name']
                    model = model_info['model']
                    tokenizer = model_info['tokenizer']
                    processor = model_info['processor']
                    logging.info(f"Processing with model: {model_name}")

                    for prompt_idx, prompt_template in enumerate(prompt_templates):
                        # Initialize time tracker for segment
                        segment_start_time = time.time()

                        # Process the clip with the model
                        for start_idx in range(0, len(clip), batch_size):
                            end_idx = min(start_idx + batch_size, len(clip))
                            batch_frames = clip[start_idx:end_idx]
                            
                            try:
                                # Assuming predicted_emotions is a single string or non-list type
                                predicted_emotions, time_taken = process_frames_with_model(model_name, model, tokenizer, batch_frames, processor, prompt_template)
                                average_times[model_idx][prompt_idx].append(time_taken / len(batch_frames))  # Average time for this batch
                                all_predictions[model_idx][prompt_idx].append(predicted_emotions)  # Store predicted emotion string
                            except Exception as e:
                                logging.error(f"Failed to process batch starting at index {start_idx}: {e}")
                                continue

                        # Calculate accuracy for this model
                        frame_accuracy = [1.0 if true_emotion in emotion else 0.0 for emotion in all_predictions[model_idx][prompt_idx]]
                        segment_accuracy = sum(frame_accuracy) / len(frame_accuracy) if frame_accuracy else 0
                        model_accuracies[model_idx][prompt_idx].append(segment_accuracy * 100)

                # Calculate most common emotions for each model and each prompt
                most_common_emotions = [
                    [
                        Counter(all_predictions[model_idx][prompt_idx]).most_common(1)[0][0] if all_predictions[model_idx][prompt_idx] else None
                        for prompt_idx in range(len(prompt_templates))
                    ]
                    for model_idx in range(len(model_list))
                ]
                # Shape of most_common_emotions
                most_common_emotions_shape = (len(most_common_emotions), len(most_common_emotions[0]) if most_common_emotions else 0)

                # Log or print the shape
                logging.info(f"Most common emotions shape: {most_common_emotions_shape}")

                for prompt_idx, prompt_template in enumerate(prompt_templates):
                    results.append({
                        "Prompt": prompt_template,
                        "Actual Emotion": true_emotion,
                        "Predicted Emotion 1": most_common_emotions[0][prompt_idx] if prompt_idx < len(most_common_emotions[0]) else None,  # Most common for Model 1
                        "Predicted Emotion 2": most_common_emotions[1][prompt_idx] if prompt_idx < len(most_common_emotions[1]) else None,  # Most common for Model 2
                        "Predicted Emotion 3": most_common_emotions[2][prompt_idx] if prompt_idx < len(most_common_emotions[2]) else None,  # Most common for Model 3
                        "Average Time 1": sum(average_times[0][prompt_idx]) / len(average_times[0][prompt_idx]) if average_times[0][prompt_idx] else None,
                        "Average Time 2": sum(average_times[1][prompt_idx]) / len(average_times[1][prompt_idx]) if average_times[1][prompt_idx] else None,
                        "Average Time 3": sum(average_times[2][prompt_idx]) / len(average_times[2][prompt_idx]) if average_times[2][prompt_idx] else None,
                        "Accuracy Model 1": model_accuracies[0][prompt_idx][-1] if model_accuracies[0][prompt_idx] else None,
                        "Accuracy Model 2": model_accuracies[1][prompt_idx][-1] if model_accuracies[1][prompt_idx] else None,
                        "Accuracy Model 3": model_accuracies[2][prompt_idx][-1] if model_accuracies[2][prompt_idx] else None,
                        "Gender": video_gender,
                        "WAV File": wav_file  # Include the wav_file in the results
                    })

    else:
        logging.warning(f"Directory {video_dir_path} does not exist.")

# Save results to CSV
results_df = pd.DataFrame(results)
output_file_path = "processed_results_prompts.csv"
results_df.to_csv(output_file_path, index=False)

logging.info(f"Results saved to {output_file_path}")





# for i in range(1, num_sessions + 1):
    # session_dir = f"Session{i}"
    # video_dir_path = os.path.join(base_video_dir_path, session_dir, "dialog", "avi", "DivX")

    # if os.path.isdir(video_dir_path):
        # logging.info(f"Processing video files in {video_dir_path}")

        # # Get all video files in the current session
        # video_files = [f for f in os.listdir(video_dir_path) if f.endswith('.avi') and not f.startswith('._')]

        # for video_file in video_files:
            # video_base_name = os.path.splitext(video_file)[0]  # Base name of the video file without extension
            # video_path = os.path.join(video_dir_path, video_file)
            # logging.info(f"Processing video file: {video_file}")
            
            # vr = VideoReader(video_path, ctx=cpu(0))
            # fps = vr.get_avg_fps()  # FPS
            # logging.info(f"Video FPS: {fps}")

            # # Retrieve the matching annotations for this video
            # annotations, video_gender = parse_annotations(annotation_file_path, video_base_name)

            # for annotation in annotations:
                # start_time, end_time, true_emotion, wav_file = annotation["start_time"], annotation["end_time"], annotation["emotion"], annotation["wav_file"]
                # logging.info(f"Processing annotation: start_time={start_time}, end_time={end_time}, true_emotion={true_emotion}")
                
                # # Calculate frame indices for the current segment
                # start_idx = int(start_time * fps)
                # end_idx = int(end_time * fps)
                # indices = list(range(start_idx, end_idx + 1))  # Including end_idx
                
                # # Read frames for the current segment
                # clip = encode_video(video_path, indices)

                # if len(clip) == 0:
                    # logging.warning(f"No frames extracted for annotation: start_time={start_time}, end_time={end_time}")
                    # continue

                # # Loop through each model in the model list
                # for model_info in model_list:
                    # model_name = model_info['model_name']
                    # model = model_info['model']
                    # tokenizer = model_info['tokenizer']
                    # processor = model_info['processor']
                    # logging.info(f"Processing with model: {model_name}")

                    # # Loop through each prompt template
                    # for prompt_template in prompt_templates:
                        # # Initialize time tracker for prompt
                        # segment_start_time = time.time()

                        # all_predictions = []
                        # for start_idx in range(0, len(clip), batch_size):
                            # end_idx = min(start_idx + batch_size, len(clip))
                            # batch_frames = clip[start_idx:end_idx]
                            
                            # try:
                                # predicted_emotions, time_taken = process_frames_with_model(model_name, model, tokenizer, batch_frames, processor, prompt_template)
                            # except Exception as e:
                                # logging.error(f"Failed to process batch starting at index {start_idx}: {e}")
                                # continue

                            # all_predictions.append(predicted_emotions)  

                        # # Calculate total time for prompt processing
                        # segment_end_time = time.time()
                        # segment_time_taken = segment_end_time - segment_start_time
                        # logging.info(f"Time taken for prompt '{prompt_template}' with model {model_name}: {segment_time_taken:.2f}s")

                        # Average_Time = segment_time_taken / len(clip) if len(clip) > 0 else 0  # Avoid division by zero
                        # if all_predictions:
                            # frame_accuracy = [1.0 if true_emotion in emotion else 0.0 for emotion in all_predictions]
                            # segment_accuracy = sum(frame_accuracy) / len(frame_accuracy)
                            # segment_accuracy_percentage = segment_accuracy * 100
                            
                            # # Aggregate predictions for the segment
                            # most_common_emotion = Counter(all_predictions).most_common(1)[0][0]

                            # accuracy = 1.0 if true_emotion in most_common_emotion else 0.0
                            # logging.info(f"Most common predicted emotion with {model_name}: {most_common_emotion}")

                            # results.append({
                                # "True Emotion": true_emotion,
                                # "Predicted Emotion": most_common_emotion,
                                # "Time Taken (s)": segment_time_taken,  # Total time for prompt processing
                                # "Average Time(s)": Average_Time,  # Time for 1 frame
                                # "Frame Accuracy (%)": segment_accuracy_percentage,  # Accuracy of the segment as a percentage
                                # "Segment Accuracy": accuracy,
                                # "Count": len(clip),
                                # "Gender": video_gender,
                                # "Model Name": model_name,  # Keep track of the model used
                                # "WAV File": wav_file,  # Include the wav_file in the results
                                # "Prompt": prompt_template  # Include the prompt in the results
                            # })

    # else:
        # logging.warning(f"Directory {video_dir_path} does not exist.")

# # Save results to CSV
# results_df = pd.DataFrame(results)
# output_file_path = "/processed_results_prompts2.csv"
# results_df.to_csv(output_file_path, index=False)

# logging.info(f"Results saved to {output_file_path}")