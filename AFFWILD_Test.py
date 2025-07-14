import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.utils import resample
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from itertools import combinations
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from joblib import Parallel, delayed
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
import time
import os
import logging

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,                # Set the minimum log level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    handlers=[
        logging.StreamHandler()        # Output logs to the console
        # You can also add FileHandler to log to a file
    ]
)

logger = logging.getLogger(__name__)

# Set the NLTK data path
nltk_data_path = 'Models' 
nltk.data.path.append(nltk_data_path)


# Define SMOTE object
smote = SMOTE(random_state=42)

def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove punctuation
    tokens = [token for token in tokens if token not in punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens
    
    
def generate_autoencoder_config(feature_size):
    encoding_dim = max(8, int(feature_size // 3))
    first_layer_size = max(8, int(2 * feature_size // 3))
    
    layer_sizes = [first_layer_size]
    current_size = first_layer_size
    while current_size > encoding_dim:
        next_size = max(encoding_dim, current_size - int(feature_size // 5))
        if next_size == current_size:
            break
        layer_sizes.append(next_size)
        current_size = next_size
    
    return {
        'encoding_dim': encoding_dim,
        'layers': layer_sizes
    }

def autoencoder(X, save_path):
    # Normalize features across all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and validation sets
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

    # Generate dynamic configuration based on feature size
    config = generate_autoencoder_config(X_train.shape[1])

    # Define the autoencoder model based on dynamic configuration
    input_layer = Input(shape=(X_train.shape[1],))
    encoded = input_layer
    for units in config['layers']:
        encoded = Dense(units, activation='relu')(encoded)
    encoded = Dense(config['encoding_dim'], activation='relu')(encoded)

    # Decoder
    decoded = encoded
    for units in reversed(config['layers']):
        decoded = Dense(units, activation='relu')(decoded)
    decoded = Dense(X_train.shape[1], activation='linear')(decoded)

    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    # Train the autoencoder with callback to monitor NaNs
    class TerminateOnNaN(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs.get('loss') is not None and np.isnan(logs.get('loss')):
                print("NaN loss encountered. Terminating training.")
                self.model.stop_training = True

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # Number of epochs with no improvement to wait before stopping
        restore_best_weights=True
    )
    
    autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    callbacks=[TerminateOnNaN(), early_stopping])

    # Save the model
    autoencoder.save(save_path)
    print(f"Autoencoder model saved to {save_path}")


def adjust_features(features, target_shape):

    num_features = features.shape[1]
    if num_features < target_shape[0]:
        # Pad with zeros
        adjusted_features = np.pad(features, ((0, 0), (0, target_shape[0] - num_features)), mode='constant', constant_values=0)
    else:
        # Truncate excess features
        adjusted_features = features[:, :target_shape[0]]

    return adjusted_features

#face_gaze_audio_text_body.csv
#face+body+gaze+audio+transcriptions2.csv

#combined_df = pd.read_csv('IEMOCAP+AFFWILD.csv')

# # Load combined dataset
combined_df = pd.read_csv('face_gaze_audio_text_body.csv')
combined_df2 = pd.read_csv('face+body+gaze+audio+transcriptions2.csv')

combined_df.rename(columns={'Emotion_x':'Emotion'}, inplace=True)
combined_df2.rename(columns={'Emotion_x':'Emotion'}, inplace=True)
combined_df2.rename(columns={'gaze_pitch':'Pitch'}, inplace=True)
combined_df2.rename(columns={'gaze_yaw':'Yaw'}, inplace=True)
combined_df2.rename(columns={'transcription':'text'}, inplace=True)

combined_df = combined_df[combined_df['Emotion'] != 'Unknown']
# combined_df = combined_df[combined_df['Emotion'] != 'oth']
# combined_df = combined_df[combined_df['Emotion'] != 'xxx']
# combined_df = combined_df[combined_df['Emotion'] != 'dis']

# combined_df2 = combined_df2[combined_df2['Emotion'] != 'Unknown']
# combined_df2 = combined_df2[combined_df2['Emotion'] != 'oth']
combined_df2 = combined_df2[combined_df2['Emotion'] != 'xxx']
# combined_df2 = combined_df2[combined_df2['Emotion'] != 'dis']

emotion_mapping = {
    'ang': 'Anger',
    'dis': 'Disgust',
    'fea': 'Fear',
    'hap': 'Happiness',
    'neu': 'Neutral',
    'sad': 'Sadness',
    'sur': 'Surprise',
    'oth': 'Other',   
    'exc': 'Happiness',
    'fru': 'Anger',
}

# Apply the mapping to the 'Emotion' column in both DataFrames
combined_df['Emotion'] = combined_df['Emotion'].map(emotion_mapping).fillna(combined_df['Emotion'])
combined_df2['Emotion'] = combined_df2['Emotion'].map(emotion_mapping).fillna(combined_df2['Emotion'])




# Define the columns for facial landmarks and body pose
facial_keypoints_columns = [
    'Point_0_x', 'Point_1_x', 'Point_2_x', 'Point_3_x', 'Point_4_x', 'Point_5_x', 'Point_6_x', 'Point_7_x',
    'Point_8_x', 'Point_9_x', 'Point_10_x', 'Point_11_x', 'Point_12_x', 'Point_13_x', 'Point_14_x', 'Point_15_x',
    'Point_16_x', 'Point_17_x', 'Point_18_x', 'Point_19_x', 'Point_20_x', 'Point_21_x', 'Point_22_x', 'Point_23_x',
    'Point_24_x', 'Point_25_x', 'Point_26_x', 'Point_27_x', 'Point_28_x', 'Point_29_x', 'Point_30_x', 'Point_31_x',
    'Point_32_x', 'Point_33_x', 'Point_34_x', 'Point_35_x', 'Point_36_x', 'Point_37_x', 'Point_38_x', 'Point_39_x',
    'Point_40_x', 'Point_41_x', 'Point_42_x', 'Point_43_x', 'Point_44_x', 'Point_45_x', 'Point_46_x', 'Point_47_x',
    'Point_48_x', 'Point_49_x', 'Point_50_x', 'Point_51_x', 'Point_52_x', 'Point_53_x', 'Point_54_x', 'Point_55_x',
    'Point_56_x', 'Point_57_x', 'Point_58_x', 'Point_59_x', 'Point_60_x', 'Point_61_x', 'Point_62_x', 'Point_63_x',
    'Point_64_x', 'Point_65_x', 'Point_66_x', 'Point_67_x',
    'Point_0_y', 'Point_1_y', 'Point_2_y', 'Point_3_y', 'Point_4_y', 'Point_5_y', 'Point_6_y', 'Point_7_y',
    'Point_8_y', 'Point_9_y', 'Point_10_y', 'Point_11_y', 'Point_12_y', 'Point_13_y', 'Point_14_y', 'Point_15_y',
    'Point_16_y', 'Point_17_y', 'Point_18_y', 'Point_19_y', 'Point_20_y', 'Point_21_y', 'Point_22_y', 'Point_23_y',
    'Point_24_y', 'Point_25_y', 'Point_26_y', 'Point_27_y', 'Point_28_y', 'Point_29_y', 'Point_30_y', 'Point_31_y',
    'Point_32_y', 'Point_33_y', 'Point_34_y', 'Point_35_y', 'Point_36_y', 'Point_37_y', 'Point_38_y', 'Point_39_y',
    'Point_40_y', 'Point_41_y', 'Point_42_y', 'Point_43_y', 'Point_44_y', 'Point_45_y', 'Point_46_y', 'Point_47_y',
    'Point_48_y', 'Point_49_y', 'Point_50_y', 'Point_51_y', 'Point_52_y', 'Point_53_y', 'Point_54_y', 'Point_55_y',
    'Point_56_y', 'Point_57_y', 'Point_58_y', 'Point_59_y', 'Point_60_y', 'Point_61_y', 'Point_62_y', 'Point_63_y',
    'Point_64_y', 'Point_65_y', 'Point_66_y', 'Point_67_y'
]

body_keypoints_columns = [
    'Nose_x', 'Neck_x', 'RShoulder_x', 'RElbow_x', 'RWrist_x', 
    'LShoulder_x', 'LElbow_x', 'LWrist_x', 'RHip_x', 'RKnee_x', 
    'RAnkle_x', 'LHip_x', 'LKnee_x', 'LAnkle_x', 'REye_x', 
    'LEye_x', 'REar_x', 'LEar_x', 'Nose_y', 'Neck_y', 'RShoulder_y', 
    'RElbow_y', 'RWrist_y', 'LShoulder_y', 'LElbow_y', 'LWrist_y', 
    'RHip_y', 'RKnee_y', 'RAnkle_y', 'LHip_y', 'LKnee_y', 'LAnkle_y', 
    'REye_y', 'LEye_y', 'REar_y', 'LEar_y']

gaze_keypoints_columns = ['Pitch', 'Yaw']

audio_keypoints_columns = [
    'F0final_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma', 'jitterDDP_sma', 
    'shimmerLocal_sma', 'logHNR_sma', 'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma', 
    'pcm_RMSenergy_sma', 'pcm_zcr_sma', 'audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 
    'audSpec_Rfilt_sma[2]', 'audSpec_Rfilt_sma[3]', 'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]', 
    'audSpec_Rfilt_sma[6]', 'audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]', 'audSpec_Rfilt_sma[9]', 
    'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]', 'audSpec_Rfilt_sma[12]', 'audSpec_Rfilt_sma[13]', 
    'audSpec_Rfilt_sma[14]', 'audSpec_Rfilt_sma[15]', 'audSpec_Rfilt_sma[16]', 'audSpec_Rfilt_sma[17]', 
    'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]', 'audSpec_Rfilt_sma[21]', 
    'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]', 'audSpec_Rfilt_sma[24]', 'audSpec_Rfilt_sma[25]', 
    'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma', 'pcm_fftMag_spectralRollOff25.0_sma', 
    'pcm_fftMag_spectralRollOff50.0_sma', 'pcm_fftMag_spectralRollOff75.0_sma', 'pcm_fftMag_spectralRollOff90.0_sma', 
    'pcm_fftMag_spectralFlux_sma', 'pcm_fftMag_spectralCentroid_sma', 'pcm_fftMag_spectralEntropy_sma', 
    'pcm_fftMag_spectralVariance_sma', 'pcm_fftMag_spectralSkewness_sma', 'pcm_fftMag_spectralKurtosis_sma', 
    'pcm_fftMag_spectralSlope_sma', 'pcm_fftMag_psySharpness_sma', 'pcm_fftMag_spectralHarmonicity_sma', 
    'mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]', 'mfcc_sma[5]', 'mfcc_sma[6]', 
    'mfcc_sma[7]', 'mfcc_sma[8]', 'mfcc_sma[9]', 'mfcc_sma[10]', 'mfcc_sma[11]', 'mfcc_sma[12]', 
    'mfcc_sma[13]', 'mfcc_sma[14]'
]

# # Combine all the keypoints columns and add 'text' and 'Emotion' into the list
# columns_to_select = (
    # facial_keypoints_columns +
    # body_keypoints_columns +
    # gaze_keypoints_columns +
    # audio_keypoints_columns +
    # ['text', 'Emotion']  # Adding 'text' and 'Emotion' as a list
# )

# # Select only the specified columns from both DataFrames
# df1_keypoints = combined_df[columns_to_select]
# df2_keypoints = combined_df2[columns_to_select]
# # Concatenate the keypoints DataFrames
# combined_keypoints_df = pd.concat([df1_keypoints, df2_keypoints], ignore_index=True)

# y = combined_keypoints_df['Emotion'].values  

# # Check the distribution of each class
# classes, counts = np.unique(y, return_counts=True)
# class_distribution = dict(zip(classes, counts))
# print("Class distribution before balancing:", class_distribution)

# combined_keypoints_df.to_csv('IEMOCAP+AFFWILD.csv', index=False)



# Extract features and labels
X_facial = combined_df[facial_keypoints_columns].values
X_body = combined_df[body_keypoints_columns].values
X_gaze = combined_df[gaze_keypoints_columns].values
X_audio = combined_df[audio_keypoints_columns].values
X_text = combined_df['text'].values.astype(str)


y = combined_df['Emotion'].values  

# # Extract features and labels
# X_facial = combined_df2[facial_keypoints_columns].values
# X_body = combined_df2[body_keypoints_columns].values
# X_gaze = combined_df2[gaze_keypoints_columns].values
# X_audio = combined_df2[audio_keypoints_columns].values
# X_text = combined_df2['text'].values.astype(str)


# y = combined_df2['Emotion'].values  


# Handle NaNs in X by replacing with -1
X_facial[np.isnan(X_facial)] = -1
X_body[np.isnan(X_body)] = -1
X_gaze[np.isnan(X_gaze)] = -1
X_audio[np.isnan(X_audio)] = -1



vectorizer_filename = 'Encoders/AFFWILD/AFFWILD_tfidf_vectorizer.pkl'

if os.path.exists(vectorizer_filename):
    # Load the saved TF-IDF vectorizer
    with open(vectorizer_filename, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
else:
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)

start_time = time.time()
# Apply preprocessing to each transcript
X_text_preprocessed = [preprocess_text(text) for text in X_text]

# Convert preprocessed text back to string for TF-IDF
text_strings = [' '.join(tokens) for tokens in X_text_preprocessed]


# Fit-transform to extract TF-IDF features
tfidf_features = tfidf_vectorizer.fit_transform(text_strings)

tfidf_features_array = tfidf_features.toarray()

text_time = time.time() - start_time

average_text_processing = text_time / len(X_text) if len(X_text) > 0 else 0

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

y_one_hot = to_categorical(y_encoded)

# Normalize data 
scaler_facial = StandardScaler()
X_facial_scaled = scaler_facial.fit_transform(X_facial)

scaler_body = StandardScaler()
X_body_scaled = scaler_body.fit_transform(X_body)

scaler_gaze = StandardScaler()
X_gaze_scaled = scaler_gaze.fit_transform(X_gaze)

scaler_audio = StandardScaler()
X_audio_scaled = scaler_audio.fit_transform(X_audio)

# keras_model_path = 'encoders/autoencoder_facelandmarks_IEMOCAP+AFFWILD.keras'
# autoencoder(X_facial_scaled, keras_model_path)

# keras_model_path = 'encoders/autoencoder_body_IEMOCAP+AFFWILD.keras'
# autoencoder(X_body_scaled, keras_model_path)

# keras_model_path = 'encoders/autoencoder_audio_IEMOCAP+AFFWILD.keras'
# autoencoder(X_audio_scaled, keras_model_path)

# keras_model_path = 'encoders/autoencoder_text_IEMOCAP+AFFWILD.keras'
# autoencoder(tfidf_features_array, keras_model_path)



# # Load the pre-trained autoencoder models
# autoencoder_facial_AFFWILD = load_model('encoders/autoencoder_facelandmarks_AFFWILD.keras')
# autoencoder_body_AFFWILD = load_model('encoders/autoencoder_body_AFFWILD.keras')
# autoencoder_audio_AFFWILD = load_model('encoders/autoencoder_audio_AFFWILD.keras')
# autoencoder_text_AFFWILD = load_model('encoders/autoencoder_text_AFFWILD.keras')


# autoencoder_facial_IEMOCAP = load_model('encoders/autoencoder_facelandmarks_REDUCED.keras')
# autoencoder_body_IEMOCAP = load_model('encoders/autoencoder_bodypose_REDUCED.keras')
# autoencoder_audio_IEMOCAP = load_model('encoder_models/autoencoder_model_ComParE_2016_LowLevelDescriptors_Fraction_0.3333333333333333.keras')
# autoencoder_text_IEMOCAP = load_model('encoders/autoencoder_text.keras')


# # Extract the encoder parts of the autoencoders
# encoder_facial_AFFWILD = Model(inputs=autoencoder_facial_AFFWILD.input, outputs=autoencoder_facial_AFFWILD.layers[3].output)
# encoder_body_AFFWILD = Model(inputs=autoencoder_body_AFFWILD.input, outputs=autoencoder_body_AFFWILD.layers[3].output)
# encoder_audio_AFFWILD = Model(inputs=autoencoder_audio_AFFWILD.input, outputs=autoencoder_audio_AFFWILD.layers[3].output)
# encoder_text_AFFWILD = Model(inputs=autoencoder_text_AFFWILD.input, outputs=autoencoder_text_AFFWILD.layers[3].output)

# text_AFFWILD_target_shape = encoder_text_AFFWILD.input.shape[1:]


# encoder_facial_IEMOCAP = Model(inputs=autoencoder_facial_IEMOCAP.input, outputs=autoencoder_facial_IEMOCAP.layers[3].output)
# encoder_body_IEMOCAP = Model(inputs=autoencoder_body_IEMOCAP.input, outputs=autoencoder_body_IEMOCAP.layers[3].output)
# encoder_audio_IEMOCAP = Model(inputs=autoencoder_audio_IEMOCAP.input, outputs=autoencoder_audio_IEMOCAP.layers[3].output)
# encoder_text_IEMOCAP = Model(inputs=autoencoder_text_IEMOCAP.input, outputs=autoencoder_text_IEMOCAP.layers[3].output)

# text_IEMOCAP_target_shape = encoder_text_IEMOCAP.input.shape[1:]

# # Load the pre-trained autoencoder models
# autoencoder_facial = load_model('encoders/autoencoder_facelandmarks_IEMOCAP+AFFWILD.keras')
# autoencoder_body = load_model('encoders/autoencoder_body_IEMOCAP+AFFWILD.keras')
# autoencoder_audio = load_model('encoders/autoencoder_audio_IEMOCAP+AFFWILD.keras')
# autoencoder_text = load_model('encoders/autoencoder_text_IEMOCAP+AFFWILD.keras')

# # Load the pre-trained autoencoder models
# autoencoder_facial = load_model('encoders/autoencoder_facelandmarks_REDUCED.keras')
# autoencoder_body = load_model('encoders/autoencoder_bodypose_REDUCED.keras')
# autoencoder_audio = load_model('encoder_models/autoencoder_model_ComParE_2016_LowLevelDescriptors_Fraction_0.3333333333333333.keras')
# autoencoder_text = load_model('encoders/autoencoder_text.keras')

# Load the pre-trained autoencoder models
autoencoder_facial = load_model('encoders/autoencoder_facelandmarks_AFFWILD.keras')
autoencoder_body = load_model('encoders/autoencoder_body_AFFWILD.keras')
autoencoder_audio = load_model('encoders/autoencoder_audio_AFFWILD.keras')
autoencoder_text = load_model('encoders/autoencoder_text_AFFWILD.keras')

encoder_facial = Model(inputs=autoencoder_facial.input, outputs=autoencoder_facial.layers[3].output)
encoder_body = Model(inputs=autoencoder_body.input, outputs=autoencoder_body.layers[3].output)
encoder_audio = Model(inputs=autoencoder_audio.input, outputs=autoencoder_audio.layers[3].output)
encoder_text = Model(inputs=autoencoder_text.input, outputs=autoencoder_text.layers[3].output)

# Measure time for facial data encoding
start_time_facial = time.time()
encoded_X_facial = encoder_facial.predict(X_facial_scaled)
time_facial_encoding = time.time() - start_time_facial
average_time_facial_encoding = time_facial_encoding / len(X_facial_scaled)

# Measure time for body data encoding
start_time_body = time.time()
encoded_X_body = encoder_body.predict(X_body_scaled)
time_body_encoding = time.time() - start_time_body
average_time_body_encoding = time_body_encoding / len(X_body_scaled)

# Measure time for audio data encoding
start_time_audio = time.time()
encoded_X_audio = encoder_audio.predict(X_audio_scaled)
time_audio_encoding = time.time() - start_time_audio
average_time_audio_encoding = time_audio_encoding / len(X_audio_scaled)

# Measure time for text data encoding
start_time_text = time.time()
encoded_X_text = encoder_text.predict(tfidf_features_array)
time_text_encoding = time.time() - start_time_text
average_time_text_encoding = time_text_encoding / len(tfidf_features_array)

average_time_text_encoding = average_time_text_encoding + average_text_processing

#Create a dictionary to store feature subsets
feature_subsets = {
    # #Original features
    'Original Gaze': X_gaze_scaled,

    'Face': encoded_X_facial,
    'Body': encoded_X_body,
    'Audio': encoded_X_audio,
    'Text': encoded_X_text,

    # #Combinations
    'Face+Body': np.concatenate((encoded_X_facial, encoded_X_body), axis=1),
    'Face+OriginalGaze': np.concatenate((encoded_X_facial, X_gaze_scaled), axis=1),
    'Face+Audio': np.concatenate((encoded_X_facial, encoded_X_audio), axis=1),
    'Face+Text': np.concatenate((encoded_X_facial, encoded_X_text), axis=1),
    'Body+OriginalGaze': np.concatenate((encoded_X_body, X_gaze_scaled), axis=1),
    'Body+Audio': np.concatenate((encoded_X_body, encoded_X_audio), axis=1),
    'Body+Text': np.concatenate((encoded_X_body, encoded_X_text), axis=1),
    'OriginalGaze+Audio': np.concatenate((X_gaze_scaled, encoded_X_audio), axis=1),
    'OriginalGaze+Text': np.concatenate((X_gaze_scaled, encoded_X_text), axis=1),
    'Audio+Text': np.concatenate((encoded_X_audio, encoded_X_text), axis=1),
    'Face+Body+OriginalGaze': np.concatenate((encoded_X_facial, encoded_X_body, X_gaze_scaled), axis=1),
    'Face+Body+Audio': np.concatenate((encoded_X_facial, encoded_X_body, encoded_X_audio), axis=1),
    'Face+Body+Text': np.concatenate((encoded_X_facial, encoded_X_body, encoded_X_text), axis=1),
    'Face+OriginalGaze+Audio': np.concatenate((encoded_X_facial, X_gaze_scaled, encoded_X_audio), axis=1),
    'Face+OriginalGaze+Text': np.concatenate((encoded_X_facial, X_gaze_scaled, encoded_X_text), axis=1),
    'Face+Audio+Text': np.concatenate((encoded_X_facial, encoded_X_audio, encoded_X_text), axis=1),
    'Body+OriginalGaze+Audio': np.concatenate((encoded_X_body, X_gaze_scaled, encoded_X_audio), axis=1),
    'Body+OriginalGaze+Text': np.concatenate((encoded_X_body, X_gaze_scaled, encoded_X_text), axis=1),
    'Body+Audio+Text': np.concatenate((encoded_X_body, encoded_X_audio, encoded_X_text), axis=1),
    'OriginalGaze+Audio+Text': np.concatenate((X_gaze_scaled, encoded_X_audio, encoded_X_text), axis=1),
    'Face+Body+OriginalGaze+Audio': np.concatenate((encoded_X_facial, encoded_X_body, X_gaze_scaled, encoded_X_audio), axis=1),
    'Face+Body+OriginalGaze+Text': np.concatenate((encoded_X_facial, encoded_X_body, X_gaze_scaled, encoded_X_text), axis=1),
    'Face+Body+Audio+Text': np.concatenate((encoded_X_facial, encoded_X_body, encoded_X_audio, encoded_X_text), axis=1),
    'Face+OriginalGaze+Audio+Text': np.concatenate((encoded_X_facial, X_gaze_scaled, encoded_X_audio, encoded_X_text), axis=1),
    'Body+OriginalGaze+Audio+Text': np.concatenate((encoded_X_body, X_gaze_scaled, encoded_X_audio, encoded_X_text), axis=1),
    'Face+Body+OriginalGaze+Audio+Text': np.concatenate((encoded_X_facial, encoded_X_body, X_gaze_scaled, encoded_X_audio, encoded_X_text), axis=1),
}



# # Encode data using the pre-trained autoencoders
# encoded_X_facial_AFFWILD = encoder_facial_AFFWILD.predict(X_facial_scaled)
# encoded_X_body_AFFWILD = encoder_body_AFFWILD.predict(X_body_scaled)
# encoded_X_audio_AFFWILD = encoder_audio_AFFWILD.predict(X_audio_scaled)

# if tfidf_features_array.shape[1] != text_AFFWILD_target_shape[0]:
    # tfidf_features_array_padded = adjust_features(tfidf_features_array, text_AFFWILD_target_shape)
# else:
    # tfidf_features_array_padded = tfidf_features_array


# encoded_X_text_AFFWILD = encoder_text_AFFWILD.predict(tfidf_features_array_padded)


# encoded_X_facial_IEMOCAP = encoder_facial_IEMOCAP.predict(X_facial_scaled)
# encoded_X_body_IEMOCAP = encoder_body_IEMOCAP.predict(X_body_scaled)
# encoded_X_audio_IEMOCAP = encoder_audio_IEMOCAP.predict(X_audio_scaled)


# if tfidf_features_array.shape[1] != text_IEMOCAP_target_shape[0]:
    # tfidf_features_array_padded = adjust_features(tfidf_features_array, text_IEMOCAP_target_shape)
# else:
    # tfidf_features_array_padded = tfidf_features_array

# encoded_X_text_IEMOCAP = encoder_text_IEMOCAP.predict(tfidf_features_array_padded)


# # Define the variable for the suffix
# suffix = '(AFFWILD)'

# # Create a dictionary to store feature subsets
# feature_subsets = {
    # # Original features
    # f'Original Gaze {suffix}': X_gaze_scaled,


    # # AFFWILD features
    # f'AFFWILD Face {suffix}': encoded_X_facial_AFFWILD,
    # f'IEMOCAP Face {suffix}': encoded_X_facial_IEMOCAP,
    # f'AFFWILD Body {suffix}': encoded_X_body_AFFWILD,
    # f'IEMOCAP Body {suffix}': encoded_X_body_IEMOCAP,
    # f'AFFWILD Audio {suffix}': encoded_X_audio_AFFWILD,
    # f'IEMOCAP Audio {suffix}': encoded_X_audio_IEMOCAP,
    # f'AFFWILD Text {suffix}': encoded_X_text_AFFWILD,
    # f'IEMOCAP Text {suffix}': encoded_X_text_IEMOCAP,

    # # Combinations
    # f'AFFWILD Face+Body {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_body_AFFWILD), axis=1),
    # f'IEMOCAP Face+Body {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_body_IEMOCAP), axis=1),
    # f'AFFWILD Face+OriginalGaze {suffix}': np.concatenate((encoded_X_facial_AFFWILD, X_gaze_scaled), axis=1),
    # f'IEMOCAP Face+OriginalGaze {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, X_gaze_scaled), axis=1),
    # f'AFFWILD Face+Audio {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_audio_AFFWILD), axis=1),
    # f'IEMOCAP Face+Audio {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_audio_IEMOCAP), axis=1),
    # f'AFFWILD Face+Text {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Face+Text {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Body+OriginalGaze {suffix}': np.concatenate((encoded_X_body_AFFWILD, X_gaze_scaled), axis=1),
    # f'IEMOCAP Body+OriginalGaze {suffix}': np.concatenate((encoded_X_body_IEMOCAP, X_gaze_scaled), axis=1),
    # f'AFFWILD Body+Audio {suffix}': np.concatenate((encoded_X_body_AFFWILD, encoded_X_audio_AFFWILD), axis=1),
    # f'IEMOCAP Body+Audio {suffix}': np.concatenate((encoded_X_body_IEMOCAP, encoded_X_audio_IEMOCAP), axis=1),
    # f'AFFWILD Body+Text {suffix}': np.concatenate((encoded_X_body_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Body+Text {suffix}': np.concatenate((encoded_X_body_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD OriginalGaze+Audio {suffix}': np.concatenate((X_gaze_scaled, encoded_X_audio_AFFWILD), axis=1),
    # f'IEMOCAP OriginalGaze+Audio {suffix}': np.concatenate((X_gaze_scaled, encoded_X_audio_IEMOCAP), axis=1),
    # f'AFFWILD OriginalGaze+Text {suffix}': np.concatenate((X_gaze_scaled, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP OriginalGaze+Text {suffix}': np.concatenate((X_gaze_scaled, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Audio+Text {suffix}': np.concatenate((encoded_X_audio_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Audio+Text {suffix}': np.concatenate((encoded_X_audio_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Face+Body+OriginalGaze {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_body_AFFWILD, X_gaze_scaled), axis=1),
    # f'IEMOCAP Face+Body+OriginalGaze {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_body_IEMOCAP, X_gaze_scaled), axis=1),
    # f'AFFWILD Face+Body+Audio {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_body_AFFWILD, encoded_X_audio_AFFWILD), axis=1),
    # f'IEMOCAP Face+Body+Audio {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_body_IEMOCAP, encoded_X_audio_IEMOCAP), axis=1),
    # f'AFFWILD Face+Body+Text {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_body_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Face+Body+Text {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_body_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Face+OriginalGaze+Audio {suffix}': np.concatenate((encoded_X_facial_AFFWILD, X_gaze_scaled, encoded_X_audio_AFFWILD), axis=1),
    # f'IEMOCAP Face+OriginalGaze+Audio {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, X_gaze_scaled, encoded_X_audio_IEMOCAP), axis=1),
    # f'AFFWILD Face+OriginalGaze+Text {suffix}': np.concatenate((encoded_X_facial_AFFWILD, X_gaze_scaled, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Face+OriginalGaze+Text {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, X_gaze_scaled, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Face+Audio+Text {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_audio_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Face+Audio+Text {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_audio_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Body+OriginalGaze+Audio {suffix}': np.concatenate((encoded_X_body_AFFWILD, X_gaze_scaled, encoded_X_audio_AFFWILD), axis=1),
    # f'IEMOCAP Body+OriginalGaze+Audio {suffix}': np.concatenate((encoded_X_body_IEMOCAP, X_gaze_scaled, encoded_X_audio_IEMOCAP), axis=1),
    # f'AFFWILD Body+OriginalGaze+Text {suffix}': np.concatenate((encoded_X_body_AFFWILD, X_gaze_scaled, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Body+OriginalGaze+Text {suffix}': np.concatenate((encoded_X_body_IEMOCAP, X_gaze_scaled, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Body+Audio+Text {suffix}': np.concatenate((encoded_X_body_AFFWILD, encoded_X_audio_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Body+Audio+Text {suffix}': np.concatenate((encoded_X_body_IEMOCAP, encoded_X_audio_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD OriginalGaze+Audio+Text {suffix}': np.concatenate((X_gaze_scaled, encoded_X_audio_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP OriginalGaze+Audio+Text {suffix}': np.concatenate((X_gaze_scaled, encoded_X_audio_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Face+Body+OriginalGaze+Audio {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_body_AFFWILD, X_gaze_scaled, encoded_X_audio_AFFWILD), axis=1),
    # f'IEMOCAP Face+Body+OriginalGaze+Audio {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_body_IEMOCAP, X_gaze_scaled, encoded_X_audio_IEMOCAP), axis=1),
    # f'AFFWILD Face+Body+OriginalGaze+Text {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_body_AFFWILD, X_gaze_scaled, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Face+Body+OriginalGaze+Text {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_body_IEMOCAP, X_gaze_scaled, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Face+Body+Audio+Text {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_body_AFFWILD, encoded_X_audio_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Face+Body+Audio+Text {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_body_IEMOCAP, encoded_X_audio_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Face+OriginalGaze+Audio+Text {suffix}': np.concatenate((encoded_X_facial_AFFWILD, X_gaze_scaled, encoded_X_audio_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Face+OriginalGaze+Audio+Text {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, X_gaze_scaled, encoded_X_audio_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Body+OriginalGaze+Audio+Text {suffix}': np.concatenate((encoded_X_body_AFFWILD, X_gaze_scaled, encoded_X_audio_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Body+OriginalGaze+Audio+Text {suffix}': np.concatenate((encoded_X_body_IEMOCAP, X_gaze_scaled, encoded_X_audio_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
    # f'AFFWILD Face+Body+OriginalGaze+Audio+Text {suffix}': np.concatenate((encoded_X_facial_AFFWILD, encoded_X_body_AFFWILD, X_gaze_scaled, encoded_X_audio_AFFWILD, encoded_X_text_AFFWILD), axis=1),
    # f'IEMOCAP Face+Body+OriginalGaze+Audio+Text {suffix}': np.concatenate((encoded_X_facial_IEMOCAP, encoded_X_body_IEMOCAP, X_gaze_scaled, encoded_X_audio_IEMOCAP, encoded_X_text_IEMOCAP), axis=1),
# }




classifiers = {
    #Faster models
    'Logistic Regression': LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', max_iter=2000),  # Generally fast
    'Gaussian Naive Bayes': GaussianNB(),  # Very fast
    'KNN': KNeighborsClassifier(n_neighbors=5),  # Very fast in training but can be slow in prediction
    'LDA': LinearDiscriminantAnalysis(),  # Fast

    #Moderate models
    'Decision Tree': DecisionTreeClassifier(),  # Moderate
    'SGD': SGDClassifier(max_iter=1000, tol=1e-3),  # Generally fast for large datasets
    'Extra Trees': ExtraTreesClassifier(n_estimators=100),  # Moderate to fast due to high randomness
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100),  # Moderate, depends on base estimator
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100),  # Moderate, faster variant of gradient boosting

    #Slower models
    'Random Forest': RandomForestClassifier(n_estimators=100),  # Slower, builds multiple decision trees
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100,learning_rate=0.1, max_depth=3),  # Slower, sequential tree building
    'SVM': SVC(kernel='linear', max_iter=2000,C=1.0),  # Slow for large datasets
    'MLP': MLPClassifier(max_iter=2000),  # Generally slow, involves neural network training
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, alpha=1.0, lambda_=1.0),  # Slower, optimized gradient boosting
    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, lambda_l1=1.0, lambda_l2=1.0),  # Moderate to slow, optimized gradient boosting
    'CatBoost': CatBoostClassifier(iterations=100, learning_rate=0.1, l2_leaf_reg=1.0, verbose=0),  # Moderate to slow, handles categorical features
    'QDA': QuadraticDiscriminantAnalysis(),  # Moderate, more complex than LDA
}

def compute_confusion_matrix(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm

def evaluate_model(classifier, X_test, y_test):
    # Start the timer
    start_time = time.time()
    
    # Make predictions for the entire dataset at once
    y_pred = classifier.predict(X_test)
    
    # End the timer
    total_prediction_time = time.time() - start_time
    
    # Compute the average time per prediction
    average_time = total_prediction_time / len(X_test) if len(X_test) > 0 else 0
    
    # Return the evaluation metrics and the average time per prediction
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='weighted', zero_division=0),
        recall_score(y_test, y_pred, average='weighted', zero_division=0),
        f1_score(y_test, y_pred, average='weighted', zero_division=0),
        average_time)
    

def cross_validate_model(classifier, X, y, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=23)
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    cv_results = cross_validate(classifier, X, y, cv=kf, scoring=scoring)

    accuracy = cv_results['test_accuracy'].mean()
    precision = cv_results['test_precision_weighted'].mean()
    recall = cv_results['test_recall_weighted'].mean()
    f1 = cv_results['test_f1_weighted'].mean()

    return accuracy, precision, recall, f1
    
    
def compute_class_importances(tree, n_classes, n_features):
    importances = np.zeros((n_classes, n_features))
    # Loop over all nodes
    for node_id in range(tree.tree_.node_count):
        if tree.tree_.children_left[node_id] != tree.tree_.children_right[node_id]:  # not a leaf node
            feature = tree.tree_.feature[node_id]
            for class_index in range(n_classes):
                # Accumulate importance by class based on impurity reduction and samples for that class
                class_samples = tree.tree_.value[node_id][0][class_index]
                total_samples = tree.tree_.weighted_n_node_samples[node_id]
                class_proportion = class_samples / total_samples
                importances[class_index, feature] += class_proportion * tree.tree_.impurity[node_id] * total_samples
    return importances

def process_tree(tree, n_classes, n_features):
    return compute_class_importances(tree, n_classes, n_features)
    
    
# Function to calculate the total encoding time based on modalities present in the subset
def calculate_encoding_time(subset_name):
    encoding_time = 0
    if 'Face' in subset_name:
        encoding_time += average_time_facial_encoding
    if 'Body' in subset_name:
        encoding_time += average_time_body_encoding
    if 'Audio' in subset_name:
        encoding_time += average_time_audio_encoding
    if 'Text' in subset_name:
        encoding_time += average_time_text_encoding
    # Gaze encoding time is 0
    return encoding_time

# Ensure the output directory exists
output_dir = '/mnt/scratchevaluation_results'
os.makedirs(output_dir, exist_ok=True)

# Initialize combined_results to collect results
combined_results = []

# Evaluate each classifier on each feature subset
for subset_name, subset_features in feature_subsets.items():
    X_train, X_test, y_train, y_test = train_test_split(subset_features, y_encoded, test_size=0.2, random_state=21)
    
    result = []
    for clf_name, clf in classifiers.items():
        filename = f"/Encoders/AFFWILD/{clf_name}_{subset_name}_model.pkl"
    
        if os.path.exists(filename):
            continue
        print(f"Training {clf_name} with {subset_name}...")
        with tqdm(total=1, desc=f"{clf_name} Training Progress") as pbar:
            if isinstance(clf, (LogisticRegression, SVC)):
                # Apply scaling if the classifier is Logistic Regression or SVM
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                clf.fit(X_train_scaled, y_train)
            else:
                clf.fit(X_train, y_train)
            pbar.update(1)
        
        # Save the trained classifier
        filename = f"Encoders/AFFWILD/{clf_name}_{subset_name}_model.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(clf, file)
        
        print(f"Saved {clf_name} trained on {subset_name} as {filename}.")

# Evaluate each classifier on each feature subset
for subset_name, subset_features in feature_subsets.items():
    X_train, X_test, y_train, y_test = train_test_split(subset_features, y_encoded, test_size=0.2, random_state=21)
    encoding_time = calculate_encoding_time(subset_name)
    num_samples, num_features = subset_features.shape
    
    result = []
    for clf_name, clf in classifiers.items():
        filename = f"Encoders/AFFWILD/{clf_name}_{subset_name}_model.pkl"
    
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)/ (1024 * 1024)  # Size in megabytes
            # Load the saved classifier
            with open(filename, 'rb') as file:
                loaded_clf = pickle.load(file)
                logger.info(f"Loaded {filename}")
                
            logger.info(f"Evaluating {clf_name} with {subset_name} on test set...")
            test_accuracy, test_precision, test_recall, test_f1, prediction_time = evaluate_model(loaded_clf, X_test, y_test)
            
            Latency = encoding_time + prediction_time
        
            result.append({
                'Feature Subset': subset_name,
                'Classifier': clf_name,
                'Test_Accuracy': test_accuracy,
                'Test_Precision': test_precision,
                'Test_Recall': test_recall,
                'Test_F1 Score': test_f1,
                'Encoding Time': encoding_time,
                'Prediction Time': prediction_time,
                'Latency': Latency,
                'File Size': file_size,
                'Feature Size': num_features
            })

            logger.info(f"{clf_name} with {subset_name} Results:")
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Test Precision: {test_precision:.4f}")
            logger.info(f"Test Recall: {test_recall:.4f}")
            logger.info(f"Test F1 Score: {test_f1:.4f}")
            logger.info(f"Encoding Time: {encoding_time:.6f} seconds")
            logger.info(f"Prediction Time: {prediction_time:.6f} seconds")
            logger.info(f"Latency: {Latency:.6f} seconds")
            logger.info(f"File size: {file_size:.2f} MB")
            logger.info(f"Feature Size: {num_features}")


    # Append results for the current subset to combined_results
    combined_results.extend(result)

# Convert combined_results to a DataFrame
combined_results_df = pd.DataFrame(combined_results)

# Save combined results to CSV
combined_results_df.to_csv(os.path.join(output_dir, 'combined_evaluation_results_AFFWILD.csv'), index=False)

logger.info("File Saved")



# # Evaluate each classifier on each feature subset
# for subset_name, subset_features in feature_subsets.items():
    # X_train, X_test, y_train, y_test = train_test_split(subset_features, y_encoded, test_size=0.2, random_state=21)
    
    # result = []
    # for clf_name, clf in classifiers.items():
        # print(f"Training {clf_name} with {subset_name}...")
        # with tqdm(total=1, desc=f"{clf_name} Training Progress") as pbar:
            # if isinstance(clf, (LogisticRegression, SVC)):
                # # Apply scaling if the classifier is Logistic Regression or SVM
                # scaler = StandardScaler()
                # X_train_scaled = scaler.fit_transform(X_train)
                # X_test_scaled = scaler.transform(X_test)
                # clf.fit(X_train_scaled, y_train)
            # else:
                # clf.fit(X_train, y_train)
            # pbar.update(1)
            
        # print(f"Cross-validating {clf_name} with {subset_name}...")
        # cv_accuracy, cv_precision, cv_recall, cv_f1 = cross_validate_model(clf, subset_features, y_encoded)
        
        # # print(f"Evaluating {clf_name} with {subset_name} with distribution analysis...")
        # # metrics_distribution = evaluate_model_with_distribution(clf, subset_features, y_encoded)
        # # avg_metrics = np.mean(metrics_distribution, axis=0)
        # # std_metrics = np.std(metrics_distribution, axis=0)
        
        # print(f"Evaluating {clf_name} with {subset_name} on test set...")
        # test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(clf, X_test, y_test)
        
        # # Get feature importances if available
        # if hasattr(clf, 'feature_importances_'):
            
                
                
            # importances = clf.feature_importances_
            # # Ensure the importances array is of float type
            # importances = importances.astype(float)
            
            # # Normalize feature importances if the classifier is LightGBM or CatBoost
            # if isinstance(clf, (LGBMClassifier,CatBoostClassifier)):
                # total_importance = importances.sum()
                # if total_importance > 0:
                    # importances /= total_importance
                    
            # feature_importances = dict(zip(range(len(subset_features[0])), importances))  
        # else:
            # feature_importances = None
        
        # # Output feature importances if available
        # if feature_importances:
            # all_feature_importances.append({
                # 'Feature Subset': subset_name,
                # 'Classifier': clf_name,
                # **feature_importances  # Unpack feature importances into dictionary
            # })
            
            
            

        # if isinstance(clf, (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier)):

            # if isinstance(clf, BaggingClassifier):
                # # Ensure we're handling Bagging with DecisionTreeClassifier as base estimator
                # if not isinstance(clf.estimator_, DecisionTreeClassifier):
                    # raise ValueError("BaggingClassifier must use DecisionTreeClassifier as the base estimator")

            # # Determine the number of classes and features
            # n_classes = len(clf.classes_)
            # n_features = X_train.shape[1]

            # # Use joblib for parallel processing
            # if isinstance(clf, BaggingClassifier):
                # # Process each base estimator (DecisionTreeClassifier) in the BaggingClassifier
                # importances = Parallel(n_jobs=-1)(delayed(process_tree)(tree, n_classes, n_features) for tree in clf.estimators_)
            # else:
                # # Process each tree in RandomForestClassifier, ExtraTreesClassifier, or DecisionTreeClassifier
                # importances = Parallel(n_jobs=-1)(delayed(process_tree)(tree, n_classes, n_features) for tree in clf.estimators_) if hasattr(clf, 'estimators_') else [compute_class_importances(clf, n_classes, n_features)]

            # # Sum the importances from all trees
            # class_importances = np.sum(importances, axis=0)

            # # Normalize by the number of trees
            # num_trees = len(clf.estimators_) if hasattr(clf, 'estimators_') else 1
            # class_importances /= num_trees  # Average by number of trees
            
            # # Normalize so that the sum of importances for each feature across all classes is 1
            # # Transpose to get features as rows
            # class_importances_transposed = class_importances.T
            # total_importances = class_importances_transposed.sum(axis=1, keepdims=True)
            # feature_importances_normalized = class_importances_transposed / total_importances

            # feature_names = [f'Feature_{i}' for i in range(n_features)]

            # # Create a DataFrame for class-specific feature importances
            # class_names = clf.classes_  # Get the class names
            # class_importances_df = pd.DataFrame(feature_importances_normalized, columns=classes)
            # class_importances_df['Feature'] = feature_names  # Add feature names
            # class_importances_df['Classifier'] = clf_name
            # class_importances_df['Feature Subset'] = subset_name

            # # Append the DataFrame to the list
            # all_class_importances.append(class_importances_df)
        
        # # # Efficient SHAP calculation with sampling
        # # explainer = shap.TreeExplainer(clf)
        
        # # # Use only a subset of X_test for SHAP value calculation to save time
        # # X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)  
        # # shap_values = explainer.shap_values(X_sample)
        
        # # for class_index, emotion in enumerate(clf.classes_):
            # # shap_class_df = pd.DataFrame(shap_values[class_index], columns=feature_names)
            # # shap_class_df['True Label'] = y_test.loc[X_sample.index]
            # # shap_class_df['Predicted Label'] = clf.predict(X_sample)
            # # shap_class_df['Feature Subset'] = subset_name
            
            # # # Append to the combined SHAP list
            # # shap_combined.append(shap_class_df)
        # # Compute the confusion matrix
        # #cm = compute_confusion_matrix(clf, X_test, y_test)
        
        # # Save the confusion matrix to a file
        # #cm_filename = f'{clf_name}_{subset_name}_confusion_matrix_AFFWILD.csv'
        # #cm_df = pd.DataFrame(cm, index=[f'True_{class_name}' for class_name in classes],
                     # #columns=[f'Pred_{class_name}' for class_name in classes])
        # #cm_df.to_csv(os.path.join(output_dir, cm_filename))
        
        # result.append({
            # 'Feature Subset': subset_name,
            # 'Classifier': clf_name,
            # 'Cross_Validation_Accuracy': cv_accuracy,
            # 'Cross_Validation_Precision': cv_precision,
            # 'Cross_Validation_Recall': cv_recall,
            # 'Cross_Validation_F1 Score': cv_f1,
            # # # 'Avg_Distribution_Accuracy': avg_metrics[0],
            # # # 'Avg_Distribution_Precision': avg_metrics[1],
            # # # 'Avg_Distribution_Recall': avg_metrics[2],
            # # # 'Avg_Distribution_F1 Score': avg_metrics[3],
            # # # 'Std_Distribution_Accuracy': std_metrics[0],
            # # # 'Std_Distribution_Precision': std_metrics[1],
            # # # 'Std_Distribution_Recall': std_metrics[2],
            # # # 'Std_Distribution_F1 Score': std_metrics[3],
            # 'Test_Accuracy': test_accuracy,
            # 'Test_Precision': test_precision,
            # 'Test_Recall': test_recall,
            # 'Test_F1 Score': test_f1
        # })

        # print(f"{clf_name} with {subset_name} Results:")
        # print(f"Cross Validation Accuracy: {cv_accuracy}")
        # print(f"Cross Validation Precision: {cv_precision}")
        # print(f"Cross Validation Recall: {cv_recall}")
        # print(f"Cross Validation F1 Score: {cv_f1}")
        # # # print(f"Avg Distribution Accuracy: {avg_metrics[0]}")
        # # # print(f"Avg Distribution Precision: {avg_metrics[1]}")
        # # # print(f"Avg Distribution Recall: {avg_metrics[2]}")
        # # # print(f"Avg Distribution F1 Score: {avg_metrics[3]}")
        # # # print(f"Std Distribution Accuracy: {std_metrics[0]}")
        # # # print(f"Std Distribution Precision: {std_metrics[1]}")
        # # # print(f"Std Distribution Recall: {std_metrics[2]}")
        # # # print(f"Std Distribution F1 Score: {std_metrics[3]}")
        # print(f"Test Accuracy: {test_accuracy}")
        # print(f"Test Precision: {test_precision}")
        # print(f"Test Recall: {test_recall}")
        # print(f"Test F1 Score: {test_f1}")
        # print()

    # # Append results for the current subset to combined_results
    # combined_results.extend(result)

# # Convert combined_results to a DataFrame
# combined_results_df = pd.DataFrame(combined_results)

# # Save combined results to CSV
# combined_results_df.to_csv(os.path.join(output_dir, 'combined_evaluation_results_COMBINED_ALLCLASSIFIER.csv'), index=False)

# # Convert list of feature importances to DataFrame 
# feature_importances_df = pd.DataFrame(all_feature_importances)

# # Define output directory and filename for saving CSV
# output_file = os.path.join(output_dir, 'feature_importances_COMBINED_ALLCLASSIFIER.csv')

# # Save feature importances to CSV
# feature_importances_df.to_csv(output_file, index=False)

# print(f"Feature importances saved to: {output_file}")

# # Combine all class-specific feature importances into a single DataFrame
# combined_class_importances_df = pd.concat(all_class_importances, ignore_index=True)

# # Save the combined DataFrame to CSV
# class_importances_file = os.path.join(output_dir, 'COMBINED_combined_class_specific_feature_importances.csv')
# combined_class_importances_df.to_csv(class_importances_file, index=False)
# print(f"Class-specific feature importances saved to '{class_importances_file}'")

# # # Combine all SHAP values into a single DataFrame
# # shap_combined_df = pd.concat(shap_combined, ignore_index=True)

# # # Save consolidated SHAP values to CSV
# # shap_csv_file = os.path.join(output_dir, 'IEMOCAP_consolidated_shap_values_SAMPLE.csv')
# # shap_combined_df.to_csv(shap_csv_file, index=False)
# # print(f"Consolidated SHAP values saved to '{shap_csv_file}'")
