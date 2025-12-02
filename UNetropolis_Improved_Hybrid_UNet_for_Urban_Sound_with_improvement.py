"""
# ═════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY: UNetropolis - Urban Sound Classification
# ═════════════════════════════════════════════════════════════════════════════
#
# PROJECT TITLE: UNetropolis: Improved Hybrid Double-UNet for Urban Sound Classification
#
# OBJECTIVE:
#   To classify urban sounds into 10 distinct categories (e.g., siren, drilling,
#   dog bark) using a novel deep learning approach that adapts the UNet architecture
#   —traditionally used for image segmentation—for audio classification tasks.
#
# PROBLEM STATEMENT & CONTEXT:
#   Urban sound classification is critical for noise pollution monitoring, smart
#   city surveillance, and autonomous navigation. Traditional 1D audio models often
#   miss complex time-frequency patterns. This project leverages Mel Spectrograms
#   (2D representations of sound) and a Double-UNet architecture to capture both
#   local spectral details and high-level temporal features.
#
# MODEL OVERVIEW:
#   - Architecture: Hybrid Double-UNet (Two stacked U-Net encoders/decoders).
#   - Input: Mel Spectrograms (128 mel bands, ~4s duration).
#   - Key Features:
#     * Mask-aware augmentation (overlaying sounds).
#     * Spectrogram normalization (Min-Max).
#     * Refined classification head (GlobalAvgPool + Dense).
#
# DATASET DESCRIPTION:
#   - Name: UrbanSound8K
#   - Size: 8732 labeled sound excerpts (<= 4 seconds).
#   - Classes: 10 (Air Conditioner, Car Horn, Children Playing, Dog Bark, Drilling,
#     Engine Idling, Gun Shot, Jackhammer, Siren, Street Music).
#   - Format: WAV files, pre-sorted into 10 folds for cross-validation.
#
# EXPECTED OUTCOMES:
#   - A robust classifier capable of distinguishing complex urban sounds.
#   - Visualizations of spectral data and model performance (Confusion Matrix).
#   - A reusable, modular data generator for audio processing.
#
# PREREQUISITES:
#   - Python 3.7+
#   - Libraries: TensorFlow/Keras, Librosa, NumPy, Pandas, Matplotlib, Scikit-learn.
#   - Domain Knowledge: Basic understanding of digital signal processing (DSP)
#     and Convolutional Neural Networks (CNNs).
#
# ═════════════════════════════════════════════════════════════════════════════
"""

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: LIBRARY IMPORTS & ENVIRONMENT SETUP
# ═════════════════════════════════════════════════════════════════════════════

import os

# What: Set TensorFlow logging level to suppress non-critical messages.
# Why: Reduces console clutter from standard library warnings (like GPU availability),
#      allowing the user to focus on application-specific logs and outputs.
# How: Modifies the 'TF_CPP_MIN_LOG_LEVEL' environment variable. '2' filters out
#      INFO and WARNING logs, leaving only ERRORs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import sys
import numpy as np
import pandas as pd
import librosa # Library for audio analysis
import random
import matplotlib.pyplot as plt # For plotting
from IPython.display import Audio # For playing audio in Jupyter/IPython environments

# --- Deep Learning Framework (TensorFlow/Keras) ---
# Context: Keras is the high-level API for TensorFlow, chosen for its ease of use
# and modularity in building complex neural networks like UNet.
from tensorflow.keras.utils import Sequence # Base class for Keras data generators
from tensorflow.keras import Input, Model # For defining functional API models
from tensorflow.keras import layers, callbacks # Core layers and callback functions
from tensorflow.keras.layers import (
    Cropping2D, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose,
    Concatenate, GlobalAveragePooling2D, Activation, BatchNormalization, Dense
) # Specific layers used in the model
from tensorflow.keras.optimizers import Adam # Optimizer for model training
from tensorflow.keras.callbacks import ReduceLROnPlateau # Learning rate scheduler callback

# --- Evaluation Metrics ---
# Context: Scikit-learn provides robust tools for model evaluation.
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: DIRECTORY STRUCTURE INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════

# What: Define and create the necessary directory structure for the project.
# Why: Ensures a clean workspace where data, code, models, and outputs are organized logically.
#      This prevents file clutter and makes the project structure reproducible.
# How: Iterates through a list of paths and uses `os.makedirs` with `exist_ok=True`.
dirs = [
    "data/UrbanSound8K", # Placeholder for processed data or temporary files
    "data/masks",       # Placeholder for mask-related data
    "notebooks",        # For Jupyter notebooks
    "src",              # For Python source files (like datagen.py)
    "models",           # For saving trained Keras models
    "docs"              # For documentation or reports
]

for d in dirs:
    # Context: "/kaggle/working/" is the standard writable directory in Kaggle kernels.
    # If running locally, this path might need adjustment (e.g., "./").
    os.makedirs(f"/kaggle/working/{d}", exist_ok=True)
print("Working directories created/ensured:", dirs)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA LOADING & EXPLORATORY DATA ANALYSIS (EDA)
# ═════════════════════════════════════════════════════════════════════════════

# What: Verify dataset accessibility and list contents.
# Why: Confirms that the input data is mounted correctly before attempting to read it.
print("\n--- Listing Dataset Contents ---")
print("Kaggle input directory:", os.listdir("/kaggle/input/"))
print("UrbanSound8K dataset directory:", os.listdir("/kaggle/input/urbansound8k/"))

print("\n--- Loading Metadata ---")
# What: Load the dataset metadata CSV.
# Why: The CSV contains crucial mapping between filenames, folds, and class labels.
#      Pandas is used for efficient tabular data manipulation.
metadata_path = "/kaggle/input/urbansound8k/UrbanSound8K.csv"
df = pd.read_csv(metadata_path)
print("Metadata head:\n", df.head()) 

print("\n--- Class Distribution ---")
# What: Analyze the distribution of target classes.
# Why: To identify potential class imbalance. Severe imbalance can bias the model
#      towards majority classes.
# Interpretation: Ideally, counts should be roughly equal. UrbanSound8K is generally balanced.
print("Unique classes:", df['class'].unique())
print("Class counts:\n", df['class'].value_counts())

print("\n--- Fold Distribution ---")
# What: Check sample counts per fold.
# Why: The dataset is pre-partitioned into 10 folds. We must ensure these are preserved
#      for valid cross-validation (avoiding data leakage).
print("Fold counts:\n", df['fold'].value_counts())

print("\n--- Missing Values Check ---")
# What: Scan for null values in the metadata.
# Why: Missing data can crash training pipelines.
# Best Practice: Always validate data integrity early in the pipeline.
print("Any missing values in metadata:", df.isna().sum().sum())


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: AUDIO VISUALIZATION & SPECTROGRAM ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

print("\n--- Audio Sample Demonstration ---")
# What: Select and load a single random audio sample.
# Why: To sanity-check the audio loading process and visualize the raw waveform.
# Technical Deep-Dive:
#   - `librosa.load`: Decodes audio. We resample to 22050Hz (standard for speech/music tasks)
#     to reduce dimensionality while preserving relevant frequencies (Nyquist freq ~11kHz).
sample_meta = df.sample(1, random_state=7).iloc[0]
file_path = f"/kaggle/input/urbansound8k/fold{sample_meta['fold']}/{sample_meta['slice_file_name']}"

# Load audio (max 4 seconds)
y, sr = librosa.load(file_path, sr=22050, duration=4)
print(f"Loaded '{sample_meta['slice_file_name']}'; Class: {sample_meta['class']} | Sample Rate: {sr} | Samples: {len(y)}")

# Play audio
print("Playing audio sample:")
Audio(y, rate=sr) 

# What: Plot the amplitude waveform.
# Why: Visual inspection reveals signal density, silence, and potential noise.
plt.figure(figsize=(12, 3))
librosa.display.waveshow(y, sr=sr) 
plt.title(f"Waveform | {sample_meta['class']} ({sample_meta['slice_file_name']})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# --- Spectrogram Visualization Function ---
def show_urbansound8k_samples(
    df,
    audio_dir="/kaggle/input/urbansound8k",
    n_samples=6,
    random_state=None
):
    """
    Displays mel spectrograms for a given number of random UrbanSound8K samples.
    Includes min-max normalization for consistent visualization.

    Args:
        df (pd.DataFrame): The metadata DataFrame.
        audio_dir (str): Base directory where audio files are stored.
        n_samples (int): Number of random samples to display.
        random_state (int, optional): Seed for reproducibility. Defaults to None.
    """
    # What: Randomly sample `n_samples` from the dataframe.
    sample_df = df.sample(n_samples, random_state=random_state)
    plt.figure(figsize=(6 * n_samples, 6)) 
    
    for i, row in enumerate(sample_df.itertuples()): 
        file_path = f"{audio_dir}/fold{row.fold}/{row.slice_file_name}"
        try:
            # What: Load and pad audio.
            # Why: CNNs require fixed-size inputs. We pad shorter clips with silence (zeros).
            y, sr = librosa.load(file_path, sr=22050, duration=4)
            if len(y) < 4 * 22050:
                y = np.pad(y, (0, 4 * 22050 - len(y)), mode='constant')

            # What: Compute Mel Spectrogram.
            # Why: Mimics human hearing perception (Mel scale). 
            #      Converts 1D audio to 2D "image" features.
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # What: Min-Max Normalization.
            # Why: Scales values to [0, 1]. Crucial for neural network convergence.
            #      Prevents large values from dominating gradients.
            mel_db_normalized = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

            # Plotting
            plt.subplot(1, n_samples, i+1) 
            plt.imshow(mel_db_normalized, origin='lower', aspect='auto', cmap='magma') 
            label = getattr(row, 'class', getattr(row, 'classID', 'Unknown'))
            plt.title(f"Class: {label}")
            plt.axis('off') 
        except Exception as e:
            print(f"Error loading or plotting {file_path}: {e}")
            plt.subplot(1, n_samples, i+1)
            plt.axis('off')
            plt.title("Error")
            
    plt.suptitle(f"Random {n_samples} UrbanSound8K Spectrograms", fontsize=18) 
    plt.tight_layout() 
    plt.show()

print("\n--- Visualizing Sample Spectrograms ---")
show_urbansound8k_samples(df, audio_dir="/kaggle/input/urbansound8k", n_samples=6, random_state=42)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: CUSTOM DATA GENERATOR IMPLEMENTATION
# ═════════════════════════════════════════════════════════════════════════════

# What: Create a separate Python file for the data generator.
# Why: Modular design. Allows the generator to be imported and reused.
#      Also demonstrates how to manage project files programmatically.
os.makedirs("/kaggle/working/src", exist_ok=True)

with open("/kaggle/working/src/datagen.py", "w") as f:
    f.write("""
import numpy as np
import librosa
import tensorflow as tf # Required for tf.keras.utils.Sequence

class CustomDataGen(tf.keras.utils.Sequence):
    \"\"\"
    A custom Keras data generator for the UrbanSound8K dataset.
    
    Functionality:
    - Loads audio files on-the-fly (memory efficient).
    - Converts audio to Mel Spectrograms.
    - Applies padding to ensure fixed input size.
    - Performs Mask Overlay Augmentation (mixing sounds).
    - Normalizes inputs to [0, 1].
    \"\"\"
    def __init__(self, df, audio_dir, batch_size=8, shuffle=True, n_mels=128, duration=4, sr=22050, mask_overlay_df=None):
        self.df = df.reset_index(drop=True) 
        self.audio_dir = audio_dir 
        self.batch_size = batch_size 
        self.shuffle = shuffle # Important for training to prevent order bias
        self.n_mels = n_mels 
        self.duration = duration 
        self.sr = sr 
        self.mask_overlay_df = mask_overlay_df # Source for augmentation sounds
        self.indexes = np.arange(len(self.df)) 
        self.on_epoch_end() 

    def __len__(self):
        # What: Calculate number of batches per epoch.
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        # What: Generate one batch of data.
        # How: Slices the index array and calls __data_generation.
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch = self.df.iloc[batch_indexes]
        X, y = self.__data_generation(batch)
        return X, y

    def on_epoch_end(self):
        # What: Shuffle data after each epoch.
        # Why: Ensures the model doesn't learn the order of samples.
        if self.shuffle:
            np.random.shuffle(self.indexes) 

    def __load_audio(self, file_path):
        # What: Helper to load and pad audio.
        y, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
        if len(y) < int(self.sr * self.duration):
            y = np.pad(y, (0, int(self.sr * self.duration - len(y))), "constant")
        return y

    def __mel_spectrogram(self, y):
        # What: Helper to compute log-mel spectrogram.
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def __overlay_augment(self, base_audio, overlay_audio, snr_db=10):
        # What: Mix two audio signals.
        # Why: Data Augmentation. Simulates real-world overlapping sounds.
        # Technical: Adjusts overlay volume based on desired Signal-to-Noise Ratio (SNR).
        rms_base = np.sqrt(np.mean(base_audio ** 2)) 
        rms_overlay = np.sqrt(np.mean(overlay_audio ** 2)) 

        if rms_overlay == 0: 
            return base_audio

        desired_rms_overlay = rms_base / (10**(snr_db / 20))
        scaled_overlay = overlay_audio * (desired_rms_overlay / (rms_overlay + 1e-8))
        mixed = base_audio + scaled_overlay
        mixed = np.clip(mixed, -1.0, 1.0) # Prevent clipping distortion
        return mixed

    def __data_generation(self, batch_df):
        X = [] 
        y = [] 
        for _, row in batch_df.iterrows():
            file_path = f"{self.audio_dir}/fold{row['fold']}/{row['slice_file_name']}"
            base_audio = self.__load_audio(file_path)

            # Augmentation Logic: 50% chance to overlay another sound
            if self.mask_overlay_df is not None and np.random.rand() < 0.5:
                overlay_sample = self.mask_overlay_df.sample(1).iloc[0] 
                overlay_path = f"{self.audio_dir}/fold{overlay_sample['fold']}/{overlay_sample['slice_file_name']}"
                overlay_audio = self.__load_audio(overlay_path)
                base_audio = self.__overlay_augment(base_audio, overlay_audio, snr_db=10)

            mel_spec = self.__mel_spectrogram(base_audio)

            # Normalization: [0, 1] range
            mel_spec_normalized = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec) + 1e-8)

            # Reshape: Add channel dimension (H, W, 1) for CNN input
            mel_spec_final = np.expand_dims(mel_spec_normalized, axis=-1)
            X.append(mel_spec_final)
            y.append(row['classID']) 

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
""")

# Import the newly created module
sys.path.append("/kaggle/working/src/")
from datagen import CustomDataGen

# --- Test the Data Generator ---
print("\n--- Testing CustomDataGen ---")
# What: Verify the generator works as expected.
# Why: Debugging complex data pipelines is easier with a small sample batch.
mask_overlay_df = df[df['fold'] == 1].sample(20, random_state=42)
datagen = CustomDataGen(
    df=df,
    audio_dir="/kaggle/input/urbansound8k",
    batch_size=8,
    n_mels=128,
    duration=4,
    sr=22050,
    mask_overlay_df=mask_overlay_df 
)

X_batch, y_batch = datagen[0]
print("X_batch shape:", X_batch.shape) # Expected: (batch_size, n_mels, time_steps, 1)
print("y_batch:", y_batch) 

# Visualize one batch sample
plt.figure(figsize=(10, 4))
plt.imshow(X_batch[0][:, :, 0].T, aspect='auto', origin='lower', cmap='magma')
plt.title(f"Normalized Spectrogram for ClassID: {y_batch[0]}")
plt.xlabel("Time bins")
plt.ylabel("Mel bands")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()

# Check original dimensions for reference
y_test, sr_test = librosa.load(file_path, sr=22050, duration=4)
mel_test = librosa.feature.melspectrogram(y=y_test, sr=sr_test, n_mels=128)
mel_db_test = librosa.power_to_db(mel_test)
print("Original mel_db shape for a single sample:", mel_db_test.shape)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: MODEL ARCHITECTURE - DOUBLE UNET
# ═════════════════════════════════════════════════════════════════════════════

def crop_to_match(encoder_tensor, decoder_tensor):
    """
    Crops the encoder tensor to match the spatial dimensions of the decoder tensor.
    
    Why: In UNet, pooling reduces dimensions. Upsampling restores them, but sometimes
    dimensions don't match perfectly due to padding/rounding. Cropping ensures
    concatenation (skip connection) is possible.
    """
    crop_height = encoder_tensor.shape[1] - decoder_tensor.shape[1]
    crop_width = encoder_tensor.shape[2] - decoder_tensor.shape[2]
    cropping = ((0, max(0, crop_height)), (0, max(0, crop_width)))

    if crop_height != 0 or crop_width != 0:
        encoder_tensor = Cropping2D(cropping=cropping)(encoder_tensor)
    return encoder_tensor

def unet_block(inputs, filters, kernel_size=(3, 3), dropout=0.3):
    """
    Defines a single UNet encoder-decoder block.
    
    Architecture:
    - Encoder: Conv2D -> BatchNorm -> ReLU -> MaxPool
    - Bottleneck: Deepest feature extraction
    - Decoder: Conv2DTranspose (Upsample) -> Concat (Skip) -> Conv2D
    
    Design Pattern: 
    - Modular function allows stacking multiple UNets (Double UNet).
    - BatchNormalization added for faster convergence and stability.
    # Consists of two convolutional layers, each followed by BatchNormalization and ReLU,
    # then Dropout, and finally MaxPooling for downsampling.
    """
    # --- Encoder Path ---
    # First convolutional block in the encoder
    c1 = Conv2D(filters, kernel_size, padding='same')(inputs)
    c1 = BatchNormalization()(c1) 
    c1 = Activation('relu')(c1)

    c1 = Conv2D(filters, kernel_size, padding='same')(c1)
    c1 = BatchNormalization()(c1) 
    c1 = Activation('relu')(c1)
    c1 = Dropout(dropout)(c1) 
    p1 = MaxPooling2D((2, 2))(c1) 
    
    # Second convolutional block in the encoder
    c2 = Conv2D(filters*2, kernel_size, padding='same')(p1)
    c2 = BatchNormalization()(c2) 
    c2 = Activation('relu')(c2)

    c2 = Conv2D(filters*2, kernel_size, padding='same')(c2)
    c2 = BatchNormalization()(c2) 
    c2 = Activation('relu')(c2)
    c2 = Dropout(dropout)(c2) 
    p2 = MaxPooling2D((2, 2))(c2) 

    # --- Bottleneck ---
    # The deepest part of the UNet, capturing the most abstract features.
    b1 = Conv2D(filters*4, kernel_size, padding='same')(p2)
    b1 = BatchNormalization()(b1) 
    b1 = Activation('relu')(b1)

    b1 = Conv2D(filters*4, kernel_size, padding='same')(b1)
    b1 = BatchNormalization()(b1) 
    b1 = Activation('relu')(b1)
    b1 = Dropout(dropout)(b1) 

    # --- Decoder Path ---
    # Upsamples feature maps and concatenates them with corresponding encoder feature maps (skip connections).
    # Each upsampling block consists of Conv2DTranspose, Concatenation, and two Conv2D layers 
    # with BatchNormalization and ReLU.
    
    # First upsampling block (from bottleneck to c2 level)
    u1 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(b1) 
    c2_cropped = crop_to_match(c2, u1) 
    u1 = Concatenate()([u1, c2_cropped]) 

    u1 = Conv2D(filters*2, kernel_size, padding='same')(u1)
    u1 = BatchNormalization()(u1) 
    u1 = Activation('relu')(u1)

    u1 = Conv2D(filters*2, kernel_size, padding='same')(u1)
    u1 = BatchNormalization()(u1) 
    u1 = Activation('relu')(u1)
    
    # Second upsampling block (from u1 to c1 level)
    u2 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(u1) 
    c1_cropped = crop_to_match(c1, u2) 
    u2 = Concatenate()([u2, c1_cropped]) 

    u2 = Conv2D(filters, kernel_size, padding='same')(u2)
    u2 = BatchNormalization()(u2) 
    u2 = Activation('relu')(u2)

    u2 = Conv2D(filters, kernel_size, padding='same')(u2)
    u2 = BatchNormalization()(u2) 
    u2 = Activation('relu')(u2)

    return u2

def build_double_unet(input_shape, num_classes, dropout=0.3):
    """
    Builds the Double UNet model.
    
    Concept: Stacking two UNets allows the second network to refine the features
    extracted by the first, potentially capturing more complex hierarchical patterns.
    """
    inp = Input(shape=input_shape) 

    # First UNet block: Processes the raw input spectrogram.
    block1 = unet_block(inp, filters=32, dropout=dropout)

    # Second UNet block (takes output of first as input)
    block2 = unet_block(block1, filters=32, dropout=dropout)
    
    # --- IMPROVEMENT: Classification Head with Dense Layer ---
    # Instead of a 1x1 Conv2D followed by GlobalAveragePooling, a more typical 
    # classification head involves GlobalAveragePooling followed by one or more
    # Dense layers. This allows the model to learn complex non-linear mappings
    # from the pooled features to the final class probabilities.

    # --- Classification Head ---
    # What: Convert 2D feature maps to class probabilities.
    # How: GlobalAveragePooling2D flattens spatial dims -> Dense layer outputs logits.
    
    # Apply Global Average Pooling to reduce spatial dimensions to 1x1,
    # resulting in a feature vector for each sample in the batch.
    pooled_features = GlobalAveragePooling2D()(block2)
    
    # Add a Dense layer for classification.
    # The number of units equals `num_classes`, and 'softmax' activation
    # is used for multi-class classification, outputting probabilities for each class.
    classification_output = Dense(num_classes, activation='softmax', name='output')(pooled_features)

    
    # Define the Keras Model with the specified input and output.
    model = Model(inputs=inp, outputs=classification_output)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: TRAINING CONFIGURATION & EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

print("\n--- Preparing Data Splits ---")
metadata_path = "/kaggle/input/urbansound8k/UrbanSound8K.csv"
audio_dir = "/kaggle/input/urbansound8k"
df = pd.read_csv(metadata_path)

# Strategy: Leave-One-Fold-Out Cross-Validation (Simplified)
# We use Fold 1 for validation and Folds 2-10 for training.
# Why: Standard practice for UrbanSound8K to ensure comparable results.
val_fold = 1
train_df = df[df['fold'] != val_fold].reset_index(drop=True) 
val_df = df[df['fold'] == val_fold].reset_index(drop=True)   

# Augmentation Setup
mask_overlay_df = train_df.sample(20, random_state=42)

batch_size = 16 

# Generators
# Instantiate the CustomDataGen for training and validation datasets.

train_gen = CustomDataGen(train_df, audio_dir=audio_dir, batch_size=batch_size, shuffle=True, mask_overlay_df=mask_overlay_df)
val_gen = CustomDataGen(val_df, audio_dir=audio_dir, batch_size=batch_size, shuffle=False, mask_overlay_df=None) 

# Determine Input Shape
X_sample, y_sample = train_gen[0]
input_shape = X_sample.shape[1:]  
num_classes = 10  

print(f"Model Input Shape: {input_shape}")
print(f"Number of Classes: {num_classes}")

# Build Model
model = build_double_unet(input_shape=input_shape, num_classes=num_classes, dropout=0.3)
model.summary() 

# --- Model Compilation ---
# Optimizer: Adam (Adaptive Moment Estimation). Good default for deep learning.
# Loss: Sparse Categorical Crossentropy. Efficient for integer labels (no one-hot needed).
model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'] 
)

# --- Callbacks ---
# What: Automated interventions during training.
# 1. EarlyStopping: Prevents overfitting by stopping when validation loss stagnates.
# 2. ModelCheckpoint: Saves the best model version.
# 3. ReduceLROnPlateau: Fine-tunes learning rate when progress stalls.
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
mc = callbacks.ModelCheckpoint("/kaggle/working/models/double_unet_best.keras",
                                    save_best_only=True, monitor='val_loss', verbose=1)

# ReduceLROnPlateau: Reduces the learning rate when a metric (val_loss) has stopped improving.
# This helps the model converge better in later stages of training.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n--- Starting Model Training ---")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20, # Maximum number of epochs to train
    steps_per_epoch=len(train_gen), # Number of batches per training epoch
    validation_steps=len(val_gen), # Number of batches per validation epoch
    callbacks=[es, mc, reduce_lr], # List of callbacks to apply during training
    verbose=2 # Verbosity mode (1 = progress bar, 2 = one line per epoch) 
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: EVALUATION & RESULTS VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

print("\n--- Plotting Training History ---")
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1) 
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Double UNet Model Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2) 
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Double UNet Model Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout() 
plt.show()

# --- Confusion Matrix ---
print("\n--- Generating Confusion Matrix ---")

# Initialize val_preds and val_truth ONCE before the loop.
# This ensures that predictions and true labels are collected from all validation batches.
val_preds, val_truth = [], []

# What: Generate predictions for the entire validation set.
for batch_idx, (Xb, yb) in enumerate(val_gen):
    # Safety Check: Skip corrupted batches (NaN/Inf)
    if np.any(np.isnan(Xb)) or np.any(np.isinf(Xb)):
        print(f"Warning: Batch {batch_idx} contains NaN or Inf values. Skipping.")
        continue 

    if np.all(Xb == 0):
        print(f"Warning: Batch {batch_idx} is all zeros.")

    try:
        preds = model.predict(Xb, verbose=0)
        pred_labels = np.argmax(preds, axis=1) # Convert probs to class index
        val_preds.extend(pred_labels.tolist())
        val_truth.extend(yb.tolist())
    except Exception as e:
        print(f"Error predicting on batch {batch_idx}: {e}")

if val_truth and val_preds:
    cm = confusion_matrix(val_truth, val_preds, labels=list(range(num_classes)))
    # Map class IDs back to their original string names for better readability in the confusion matrix.
    label_names = [df[df['classID'] == i]['class'].values[0] for i in range(num_classes)]
    
# Plot the confusion matrix.
    plt.figure(figsize=(10,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(xticks_rotation=45, cmap='magma', colorbar=True) 
    plt.title("Validation Confusion Matrix (Double UNet)")
    plt.show()
else:
    print("Warning: No predictions available for confusion matrix. Check data loading and prediction loop.")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY & CONCLUSION
# ═════════════════════════════════════════════════════════════════════════════
"""
METHODOLOGY RECAP:
We successfully implemented a Double UNet architecture for Urban Sound Classification.
The pipeline included custom data generation with mask-aware augmentation, 
spectrogram normalization, and a robust training loop with callbacks.

VALIDATION:
- Data integrity checks passed (no missing values).
- Model trained with convergence (loss decreased, accuracy improved).
- Confusion matrix provides granular insight into per-class performance.

READINESS:
This script represents a "Proof of Concept" (PoC) suitable for research and 
portfolio demonstration. For production deployment, further optimization (e.g., 
quantization, serving infrastructure) and testing on the full dataset (all folds)
would be required.

KEY TAKEAWAYS:
1. Spectrograms allow applying powerful Vision techniques (UNet) to Audio.
2. Data Augmentation (mixing sounds) is crucial for robustness in noisy environments.
3. Double UNet captures hierarchical features effectively.

AUTHOR:
SACHIN PAUNIKAR
LINKEDIN : www.linkedin.com/in/sachin-paunikar-datascientists
"""
