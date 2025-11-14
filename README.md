# 🎵 UNetropolis: Hybrid Double UNet for Urban Sound Classification

This repository presents **UNetropolis**, a highly optimized deep learning solution for urban sound classification, leveraging a specialized Hybrid Double UNet architecture to achieve state-of-the-art performance.

| Metric | Result | Source |
| :--- | :--- | :--- |
| **Final Accuracy** | **95.89%** | |

## 🚀 Project Overview

The project addresses the challenge of accurately categorizing diverse, complex, and often overlapping environmental sounds present in urban settings. This is a critical task for applications ranging from smart city monitoring to autonomous systems.

The model is trained on the widely recognized **UrbanSound8K dataset**, utilizing an optimized data pipeline and a sophisticated convolutional architecture.

## ✨ Key Technical Excellence and Improvements

This implementation incorporates several crucial improvements for stability, robustness, and performance:

| Area | Improvement | Benefit | Source |
| :--- | :--- | :--- | :--- |
| **Preprocessing** | **Min-Max Normalization** | $\text{dB}$-scaled Mel Spectrograms are normalized to a $\text{[0, 1]}$ range, ensuring stable gradients and faster convergence. | |
| **Architecture** | **Batch Normalization** | Applied after *every* $\text{Conv2D}$ layer in the encoder and decoder to stabilize training and accelerate convergence. | |
| **Augmentation** | **Mask Overlay** | Randomly overlays "mask" audio samples with a specified $\text{SNR}$ to simulate complex, overlapping real-world sounds, drastically increasing model robustness. | |
| **Classification** | **$\text{GlobalAveragePooling2D}$ Head** | Uses $\text{GlobalAveragePooling2D}$ followed by a $\text{Dense}$ layer for a more flexible and powerful classification mapping from the feature vector. | |
| **Training** | **$\text{ReduceLROnPlateau}$** | Callback that automatically reduces the learning rate by a factor of $\text{0.5}$ if the validation loss plateaus, leading to better fine-tuning and convergence in later epochs. | |

## 📊 Dataset

The project uses the **UrbanSound8K dataset**, which contains $\text{8,732}$ labeled sound excerpts (less than $\text{4}$ seconds each) from $\text{10}$ urban sound classes.

The $\text{10}$ classes are:

  * air conditioner
  * car horn
  * children playing
  * dog bark
  * drilling
  * engine idling
  * gun shot
  * jackhammer
  * siren
  * street music

## 🛠️ Methodology: Core Components

### 1\. Data Preprocessing

Audio signals are converted into Mel Spectrograms, mimicking human auditory perception.

  * **Sampling Rate:** $\text{22050 Hz}$.
  * **Duration:** Fixed at $\text{4}$ seconds (truncated or padded).
  * **Mel Bands:** $\text{128}$ mel bands are computed.
  * **Decibel Conversion:** Power spectrograms are converted to a $\text{dB}$ scale using `librosa.power_to_db`.
  * **Min-Max Normalization (Improvement):** The $\text{dB}$-scaled spectrograms are min-max normalized to $\text{[0, 1]}$ for stable training.

### 2\. Data Augmentation

The model's generalization capabilities are enhanced through a specialized technique:

  * **Mask Overlay Augmentation (Improvement):** Randomly selected "mask" audio samples from the training set are overlaid onto the base audio at a specified $\text{SNR}$. This simulates real-world scenarios where sounds overlap, improving robustness.

### 3\. Model Architecture: Hybrid Double UNet

The solution is built on a custom **Hybrid Double UNet** structure, adapted from image segmentation for powerful feature extraction from mel spectrograms.

  * **Stacked UNets:** Two $\text{UNet}$-like encoder-decoder paths are stacked sequentially, allowing for hierarchical feature learning and refining representations.
  * **Batch Normalization (Improvement):** Applied after every $\text{Conv2D}$ layer in both the encoder and decoder to stabilize the training process.
  * **Skip Connections:** Critical skip connections are used to concatenate corresponding encoder feature maps with upsampled decoder features, preserving fine-grained details lost during downsampling.
  * **Classification Head (Improvement):** Replaces a simple final $\text{Conv2D}$ layer with:
      * $\text{GlobalAveragePooling2D}$: Reduces feature maps to a fixed-size vector.
      * $\text{Dense Layer}$: Fully connected layer with $\text{10}$ units ($\text{num\_classes}$) and $\text{softmax}$ activation for final probability prediction.

### 4\. Training Strategy

  * **Optimizer:** $\text{Adam}$ optimizer with an initial learning rate of $\text{0.001}$.
  * **Loss Function:** $\text{SparseCategorical Crossentropy}$, suitable for integer-encoded class labels.
  * **EarlyStopping:** Monitors $\text{val\_loss}$ and stops training if no improvement is observed for **$\text{5}$ consecutive epochs**.
  * **ReduceLROnPlateau (Improvement):** Monitors $\text{val\_loss}$ and reduces the learning rate by a factor of $\text{0.5}$ if no improvement is seen for **$\text{3}$ consecutive epochs**.

## 📈 Results and Visualizations

The main notebook generates key visualizations for analysis and validation:

  * **Accuracy and Loss Plots:** Visualizations of training and validation accuracy/loss per epoch to monitor learning and convergence.
  * **Validation Confusion Matrix:** A detailed breakdown of the model's classification performance for each urban sound class, helping to identify well-distinguished and commonly confused classes.
  * **Augmentation in Action:** A side-by-side comparison of an original Mel Spectrogram versus one augmented with the Mask Overlay technique, visually demonstrating the model's robustness training.

## ⚙️ Setup and Installation

### Prerequisites

  * Python 3.8+
  * A high-performance environment (GPU/TPU recommended for faster training).

### 1\. Clone the repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2\. Install dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
# Key dependencies include: tensorflow, numpy, pandas, librosa, scikit-learn, matplotlib
```

### 3\. Download the UrbanSound8K Dataset

  * Download the dataset from [https://urbansound8k.weebly.com/](https://urbansound8k.weebly.com/).
  * Extract the contents. Ensure the `audio` and `metadata` folders are placed such that the code can correctly locate them.

## 💻 Usage

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook unetropolis-improved-hybrid-unet-for-urban-sound.ipynb
    ```
2.  Execute all cells sequentially. The notebook will perform:
      * Metadata loading and exploration.
      * Writes the `CustomDataGen` class to `src/datagen.py`.
      * Builds, compiles, and trains the Double UNet model.
      * Plots training history and generates the confusion matrix.

## 📂 Project Structure

```
.
├── notebooks/
│   └── unetropolis-improved-hybrid-unet-for-urban-sound.ipynb # Main notebook
├── src/
│   └── datagen.py                           # CustomDataGen class
├── models/
│   └── double_unet_best.keras               # Saved best model weights
├── data/
│   └── UrbanSound8K/                        # UrbanSound8K dataset (audio/metadata)
│       ├── audio/
│       └── UrbanSound8K.csv
└── README.md                                # This file
```

## ☁️ Deployment Considerations

For production, the model can be deployed using several scalable approaches:

  * **TensorFlow Serving:** Convert the $\text{.keras}$ model to the $\text{Saved Model}$ format for high-performance serving via $\text{gRPC}$ or $\text{REST API}$.
  * **Web API (FastAPI/Flask):** Wrap the loaded model within a $\text{RESTful API}$ to allow external clients to send audio data and receive predictions.
  * **Edge Deployment:** Convert the model to **$\text{TensorFlow Lite}$** format for optimized inference on resource-constrained devices like mobile or embedded platforms.

## ⏭️ Future Work & Improvements

Areas for further enhancement include:

  * **Advanced Data Augmentation:** Implement $\text{SpecAugment}$ (time and frequency masking) directly on mel spectrograms.
  * **Model Architecture:** Integrate other attention layers (e.g., $\text{Squeeze-and-Excitation}$) or explore adding explicit $\text{Residual Connections}$.
  * **Cross-Validation:** Implement full $\text{k-fold}$ cross-validation across all $\text{10}$ folds of the $\text{UrbanSound8K}$ dataset for a more reliable estimate of generalization performance.
  * **Quantization & Pruning:** Explore model optimization techniques to reduce size and inference latency for deployment.

## ✍️ Contact

For any questions or collaborations, please reach out to:
\[sachin paunikar/https://github.com/ImdataScientistSachin)/imdatascientistsachin@gmial.com]

## 📄 License

This project is licensed under the MIT License - see the $\text{LICENSE}$ file for details.
