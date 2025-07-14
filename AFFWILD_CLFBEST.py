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
from sklearn.model_selection import GridSearchCV
import json
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Embedding, Flatten, Attention, LSTM, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from keras_tuner import Hyperband
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping




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
nltk_data_path = '/users/40018022/Multimodal/Models' 
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
    
    



combined_df = pd.read_csv('IEMOCAP+AFFWILD.csv')

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





# Extract features and labels
X_facial = combined_df[facial_keypoints_columns].values
X_body = combined_df[body_keypoints_columns].values
X_gaze = combined_df[gaze_keypoints_columns].values
X_audio = combined_df[audio_keypoints_columns].values
X_text = combined_df['text'].values.astype(str)

y = combined_df['Emotion'].values  

# Handle NaNs in X by replacing with -1
X_facial[np.isnan(X_facial)] = -1
X_body[np.isnan(X_body)] = -1
X_gaze[np.isnan(X_gaze)] = -1
X_audio[np.isnan(X_audio)] = -1

# Check the distribution of each class
classes, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(classes, counts))
print("Class distribution before balancing:", class_distribution)
num_classes = len(np.unique(y))


vectorizer_filename = 'Encoders/AFFWILD+IEMOCAP_tfidf_vectorizer.pkl'

# Load the saved TF-IDF vectorizer
with open(vectorizer_filename, 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
    
start_time = time.time()
# Apply preprocessing to each transcript
X_text_preprocessed = [preprocess_text(text) for text in X_text]

# Convert preprocessed text back to string for TF-IDF
text_strings = [' '.join(tokens) for tokens in X_text_preprocessed]


# Fit-transform to extract TF-IDF features
tfidf_features = tfidf_vectorizer.transform(text_strings)

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



# Load the pre-trained autoencoder models
autoencoder_facial = load_model('encoders/autoencoder_facelandmarks_IEMOCAP+AFFWILD.keras')
autoencoder_body = load_model('encoders/autoencoder_body_IEMOCAP+AFFWILD.keras')
autoencoder_audio = load_model('encoders/autoencoder_audio_IEMOCAP+AFFWILD.keras')
autoencoder_text = load_model('encoders/autoencoder_text_IEMOCAP+AFFWILD.keras')

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
    #Original features
    'Original Gaze': X_gaze_scaled,

    'Face': encoded_X_facial,
    'Body': encoded_X_body,
    'Audio': encoded_X_audio,
    'Text': encoded_X_text,

    # #Combinations
    # 'Face+Body': np.concatenate((encoded_X_facial, encoded_X_body), axis=1),
    # 'Face+OriginalGaze': np.concatenate((encoded_X_facial, X_gaze_scaled), axis=1),
    # 'Face+Audio': np.concatenate((encoded_X_facial, encoded_X_audio), axis=1),
    # 'Face+Text': np.concatenate((encoded_X_facial, encoded_X_text), axis=1),
    # 'Body+OriginalGaze': np.concatenate((encoded_X_body, X_gaze_scaled), axis=1),
    # 'Body+Audio': np.concatenate((encoded_X_body, encoded_X_audio), axis=1),
    # 'Body+Text': np.concatenate((encoded_X_body, encoded_X_text), axis=1),
    # 'OriginalGaze+Audio': np.concatenate((X_gaze_scaled, encoded_X_audio), axis=1),
    # 'OriginalGaze+Text': np.concatenate((X_gaze_scaled, encoded_X_text), axis=1),
    # 'Audio+Text': np.concatenate((encoded_X_audio, encoded_X_text), axis=1),
    # 'Face+Body+OriginalGaze': np.concatenate((encoded_X_facial, encoded_X_body, X_gaze_scaled), axis=1),
    # 'Face+Body+Audio': np.concatenate((encoded_X_facial, encoded_X_body, encoded_X_audio), axis=1),
    # 'Face+Body+Text': np.concatenate((encoded_X_facial, encoded_X_body, encoded_X_text), axis=1),
    # 'Face+OriginalGaze+Audio': np.concatenate((encoded_X_facial, X_gaze_scaled, encoded_X_audio), axis=1),
    # 'Face+OriginalGaze+Text': np.concatenate((encoded_X_facial, X_gaze_scaled, encoded_X_text), axis=1),
    # 'Face+Audio+Text': np.concatenate((encoded_X_facial, encoded_X_audio, encoded_X_text), axis=1),
    # 'Body+OriginalGaze+Audio': np.concatenate((encoded_X_body, X_gaze_scaled, encoded_X_audio), axis=1),
    # 'Body+OriginalGaze+Text': np.concatenate((encoded_X_body, X_gaze_scaled, encoded_X_text), axis=1),
    # 'Body+Audio+Text': np.concatenate((encoded_X_body, encoded_X_audio, encoded_X_text), axis=1),
    # 'OriginalGaze+Audio+Text': np.concatenate((X_gaze_scaled, encoded_X_audio, encoded_X_text), axis=1),
    # 'Face+Body+OriginalGaze+Audio': np.concatenate((encoded_X_facial, encoded_X_body, X_gaze_scaled, encoded_X_audio), axis=1),
    # 'Face+Body+OriginalGaze+Text': np.concatenate((encoded_X_facial, encoded_X_body, X_gaze_scaled, encoded_X_text), axis=1),
    # 'Face+Body+Audio+Text': np.concatenate((encoded_X_facial, encoded_X_body, encoded_X_audio, encoded_X_text), axis=1),
    # 'Face+OriginalGaze+Audio+Text': np.concatenate((encoded_X_facial, X_gaze_scaled, encoded_X_audio, encoded_X_text), axis=1),
    # 'Body+OriginalGaze+Audio+Text': np.concatenate((encoded_X_body, X_gaze_scaled, encoded_X_audio, encoded_X_text), axis=1),
    'Face+Body+OriginalGaze+Audio+Text': np.concatenate((encoded_X_facial, encoded_X_body, X_gaze_scaled, encoded_X_audio, encoded_X_text), axis=1),
}








# Define classifiers without hyperparameters
classifiers = {
    # Faster models
    'Logistic Regression': LogisticRegression(),  # Default parameters
    'Gaussian Naive Bayes': GaussianNB(),  # Default parameters
    'KNN': KNeighborsClassifier(),  # Default parameters
    'LDA': LinearDiscriminantAnalysis(),  # Default parameters

    # Moderate models
    'Decision Tree': DecisionTreeClassifier(),  # Default parameters
    'SGD': SGDClassifier(),  # Default parameters
    'Extra Trees': ExtraTreesClassifier(),  # Default parameters
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier()),  # Default parameters
    'HistGradientBoosting': HistGradientBoostingClassifier(),  # Default parameters

    # Slower models
    'Random Forest': RandomForestClassifier(),  # Default parameters
    'Gradient Boosting': GradientBoostingClassifier(),  # Default parameters
    'SVM': SVC(),  # Default parameters
    'MLP': MLPClassifier(),  # Default parameters
    'XGBoost': XGBClassifier(),  # Default parameters
    'LightGBM': LGBMClassifier(),  # Default parameters
    'CatBoost': CatBoostClassifier(verbose=0),  # Default parameters
    'QDA': QuadraticDiscriminantAnalysis()  # Default parameters
}

# Define hyperparameter grids
param_grids = {
        'Logistic Regression': [
        {
            'penalty': ['l1'],
            'C': [0.1, 1, 10],
            'solver': ['liblinear'],  # 'l1' is only supported with 'liblinear'
            'max_iter': [100, 200, 500, 1000]
        },
        {
            'penalty': ['l2'],
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': [100, 200, 500, 1000]
        },
        {
            'penalty': ['elasticnet'],
            'l1_ratio': [0.1, 0.5, 0.9],
            'C': [0.1, 1, 10],
            'solver': ['saga'],
            'max_iter': [100, 200, 500, 1000]
        }
    ],
    'KNN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'LDA': {
        'solver': ['svd', 'lsqr', 'eigen']
    },
    'Decision Tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [ 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SGD': {
        'loss': ['hinge', 'perceptron', 'squared_error', 'huber', 'modified_huber', 'log_loss', 'epsilon_insensitive', 'squared_hinge', 'squared_epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [1e-4, 1e-3, 1e-2],
        'max_iter': [3000,4000,5000]
    },
    'Extra Trees': {
        'n_estimators': [50, 100, 150],
        'max_depth': [ 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [False, True]
    },
    'Bagging': {
        'n_estimators': [50, 100, 150],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0]
    },
    'HistGradientBoosting': {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_iter': [50, 100, 150],
        'max_depth': [3, 5, 7]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [False, True]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3, 4, 5], 
        'gamma': ['scale', 'auto'],  
        'max_iter': [100, 200, 500]
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [1e-4, 1e-3, 1e-2],
        'max_iter': [200, 500, 1000]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'alpha': [0, 0.5, 1],
        'lambda': [0, 0.5, 1]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'lambda_l1': [0, 0.5, 1],
        'lambda_l2': [0, 0.5, 1]
    },
    'CatBoost': {
        'iterations': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5]
    }
    
}

def build_model(hp):
    model = Sequential()
    
    # Input Layer
    model.add(Input(shape=(X_train.shape[1],)))
    


    # First Dense Layer
    model.add(Dense(
        units=hp.Int('units_input', min_value=32, max_value=512, step=32),
        activation=hp.Choice('activation_input', values=['relu', 'tanh', 'sigmoid']),
        kernel_initializer=hp.Choice('kernel_initializer', values=['glorot_uniform', 'he_normal', 'lecun_normal'])
    ))
    
    # Hidden Layers
    for i in range(hp.Int('num_layers', 1, 10)):  # Tune the number of hidden layers
        
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']),
            kernel_initializer=hp.Choice('kernel_initializer', values=['glorot_uniform', 'he_normal', 'lecun_normal'])
        ))
        if hp.Boolean(f'dropout_{i}'):
            model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1)))

   
    
    # Output Layer
    model.add(Dense(
        units=num_classes,
        activation='softmax'  # Use 'softmax' for multi-class classification
    ))
    
    # Optimizer Configuration
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='log')

    if optimizer == 'adam':
        selected_optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        momentum = hp.Float('momentum', min_value=0.0, max_value=0.9, step=0.1)
        selected_optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        selected_optimizer = RMSprop(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=selected_optimizer,
        loss='categorical_crossentropy',  # For multi-class classification
        metrics=['accuracy']
    )
    
    return model

# Initialize a dictionary to store best configurations
best_configs = {}

# Evaluate each classifier on each feature subset
for subset_name, subset_features in feature_subsets.items():
    X_train, X_test, y_train, y_test = train_test_split(subset_features, y_one_hot, test_size=0.2, random_state=21)
    
    # for clf_name, clf in classifiers.items():
        # filename = f"/mnt/scratch/users/40018022/Encoders/{clf_name}_{subset_name}_model_BEST.pkl"
        # logger.info(f"Evaluating {clf_name} on {subset_name}...")
        
        # try:
            # # Check if the classifier has a corresponding hyperparameter grid
            # if clf_name in param_grids:
                # param_grid = param_grids[clf_name]
                # grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=4, verbose=0)
                
                # if isinstance(clf, (LogisticRegression, SVC, MLPClassifier)):
                    # scaler = StandardScaler()
                    # X_train_scaled = scaler.fit_transform(X_train)
                    # X_test_scaled = scaler.transform(X_test)
                    # grid_search.fit(X_train_scaled, y_train)
                # else:
                    # grid_search.fit(X_train, y_train)
                
                # best_model = grid_search.best_estimator_
                # best_params = grid_search.best_params_
                # best_score = grid_search.best_score_
                
                # logger.info(f"Best parameters for {clf_name} on {subset_name}: {best_params}")
                # logger.info(f"Best score for {clf_name} on {subset_name}: {best_score}")

                # # Save the best model
                # with open(filename, 'wb') as file:
                    # pickle.dump(best_model, file)
                    # logger.info(f"Saved {clf_name} model for {subset_name} as {filename}")
                
                # # Store the best configuration
                # best_configs[f"{clf_name}_{subset_name}"] = {
                    # 'score': best_score,
                    # 'params': best_params
                # }

        # except Exception as e:
            # logger.error(f"Failed to train or save {clf_name} model for {subset_name}: {e}")

    # Define and train Keras model using Keras Tuner
    logger.info(f"Training Keras model with {subset_name} using Keras Tuner...")
    try:
        tuner = Hyperband(
            build_model,
            objective='val_accuracy',
            max_epochs=50,
            hyperband_iterations=1,
            overwrite = True
        )
        
        early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # Fixed patience value
        min_delta=1e-4  # Fixed min_delta value
        )
        
        tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1,callbacks=[early_stopping])

        # Get the best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_score = tuner.oracle.get_best_trials(num_trials=1)[0].score

        logger.info(f"Best model score with {subset_name}: {best_score}")
        
        # Save the best Keras model
        keras_model_filename = f"/Encoders/Keras_{subset_name}_model.keras"
        best_model.save(keras_model_filename)
        logger.info(f"Saved Keras model trained on {subset_name} as {keras_model_filename}.keras")

        # Store Keras configuration
        best_configs[f"Keras_{subset_name}"] = {
            'accuracy': best_score,
            'params': best_hp.values
        }

    except Exception as e:
        logger.error(f"Failed to train or save Keras model for {subset_name}: {e}")

# Save all best configurations to a JSON file
with open("/Encoders/best_configurations_KERAS.json", 'w') as file:
    json.dump(best_configs, file, indent=4)
    logger.info("Saved all best configurations to Encoders/best_configurations_KERAS.json")


