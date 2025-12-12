# 🏙️ UNetropolis: Hybrid Double-UNet for Urban Sound Classification

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio_Analysis-brightgreen?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research_PoC-purple?style=for-the-badge)

> **A novel Deep Learning approach adapting Image Segmentation architectures for robust Audio Classification.**

---

## 📖 Overview

**UNetropolis** represents a paradigm shift in audio classification. Instead of traditional 1D signal processing, this project treats sound as a visual texture. By converting audio into **Mel-Spectrograms**, we leverage the powerful spatial feature extraction capabilities of Computer Vision architectures.

Specifically, we utilize a **Double-UNet** architecture—stacking two U-Net (Encoder-Decoder) models sequentially—to capture both high-level temporal patterns and fine-grained spectral details. The output is refined through a Global Average Pooling classification head to distinguish between 10 complex urban sound categories.

## 🚀 Key Features

*   **Hybrid Double-UNet Architecture**: Two stacked U-Nets providing deep feature refinement, transitioning from segmentation-style maps to classification vectors.
*   **Visual Audio Processing**: Converts raw waveforms into 128-band Mel-Spectrograms (Log-Scale).
*   **Dynamic Data Augmentation**: Implements **Mask Overlay Augmentation**, mixing random noise samples with target audio on-the-fly to simulate real-world overlapping sounds (e.g., a dog barking over traffic).
*   **Robust Training Pipeline**: Includes Learning Rate Schedulers, Early Stopping, and Model Checkpointing.
*   **Self-Contained Executive Script**: The main script handles everything from directory setup and custom data generator creation (`src/datagen.py`) to training and evaluation.

---

## 🧠 Model Architecture

The model treats the spectrogram as a 2D image.

```mermaid
graph LR
    Input[Mel-Spectogram\n(128 x Time x 1)] --> U1[UNet Block 1\n(Encoder-Decoder)]
    U1 --> U2[UNet Block 2\n(Refinement)]
    U2 --> GAP[Global Avg Pooling]
    GAP --> Dense[Dense Prediction Layer\n(Softmax)]
    Dense --> Output[10 Class Probabilities]
```

1.  **Input**: Log-Mel Spectrograms.
2.  **UNet block 1**: Extracts coarse spectral features.
3.  **UNet block 2**: Refines features, focusing on subtle differences between similar sounds (e.g., *Siren* vs *Street Music*).
4.  **Head**: Flattens the feature map and maps it to specific classes.

---

## 📂 Dataset: UrbanSound8K

The model is designed for the **UrbanSound8K** dataset, consisting of 8,732 labeled sound excerpts (<= 4s) across 10 classes:

| ID | Class Name | ID | Class Name |
| :--- | :--- | :--- | :--- |
| 0 | Air Conditioner | 5 | Engine Idling |
| 1 | Car Horn | 6 | Gun Shot |
| 2 | Children Playing | 7 | Jackhammer |
| 3 | Dog Bark | 8 | Siren |
| 4 | Drilling | 9 | Street Music |

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.7+ and the following libraries installed:

```bash
pip install tensorflow numpy pandas librosa matplotlib scikit-learn ipython
```

*   **TensorFlow/Keras**: Deep Learning backend.
*   **Librosa**: Audio processing and spectral conversion.
*   **Pandas/NumPy**: Data manipulation.
*   **Matplotlib**: Visualization of waveforms and training history.

---

## ⚡ Usage

This project is designed to be plug-and-play. The main script automatically sets up its own environment structure.

1.  **Clone the Repository**
2.  **Prepare Data**: Ensure the `UrbanSound8K` dataset is available.
    *   *Note: The script defaults to a Kaggle input path `../input/urbansound8k`. You may need to adjust the `audio_dir` variable in the script if running locally.*
3.  **Run the Script**:
    ```bash
    python UNetropolis_Improved_Hybrid_UNet_for_Urban_Sound_with_improvement.py
    ```

**What happens next?**
1.  The script creates a `src/` directory and generates `datagen.py`.
2.  It loads metadata and visualizes random samples.
3.  Training begins (default 20 epochs).
4.  Training history (Accuracy/Loss) and a Confusion Matrix are plotted.

---

## 📊 Results & Visualization

The script utilizes `matplotlib` to provide rich insights during execution:

*   **Waveform Analysis**: Raw amplitude visualization of inputs.
*   **Spectrograms**: Heatmaps showing frequency intensity over time.
*   **Confusion Matrix**: A detailed breakdown of classification performance (True Label vs. Predicted Label).

---

## 👨‍💻 Author & Credits

**Sachin Paunikar**
*   [LinkedIn Profile](https://www.linkedin.com/in/sachin-paunikar-datascientists)

This project serves as a Proof of Concept (PoC) for applying advanced Computer Vision techniques to the domain of Audio Signal Processing.
