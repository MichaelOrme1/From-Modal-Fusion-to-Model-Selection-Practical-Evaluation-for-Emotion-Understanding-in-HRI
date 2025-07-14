import whisperx
import gc
import torch
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8" if device == "cpu" else "float16"

# Log device and compute type
logging.info(f"Using device: {device}")
logging.info(f"Compute type: {compute_type}")

audio_file = "AFFWILD/Audio/9-15-1920x1080.wav"
batch_size = 16  # Reduce if low on GPU memory

logging.info("Loading Model")
# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

logging.info("Loading Audio")
audio = whisperx.load_audio(audio_file)

logging.info("Transcribing")
result = model.transcribe(audio, batch_size=batch_size)
logging.info(f"Transcription segments (before alignment): {result['segments']}")

# Optional: Clean up GPU memory if needed
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
logging.info("Aligning")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

logging.info(f"Transcription segments (after alignment): {result['segments']}")

# Optional: Clean up GPU memory if needed
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
logging.info("Diarizing")
diarize_model = whisperx.DiarizationPipeline(use_auth_token="", device=device)

# Add min/max number of speakers if known
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

logging.info(f"Diarization segments: {diarize_segments}")
logging.info(f"Segments with speaker IDs: {result['segments']}")

# Log the type and sample content of result["segments"]
logging.info(f"Type of result['segments']: {type(result['segments'])}")
logging.info(f"Sample of result['segments']: {result['segments'][:5]}")

