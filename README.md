# Radar Pulse Signal Detection using Fourier Transform and FT-CNN

**Course:** Mathematics for Computing (22MAT122) / Elements of Computing Systems (22AIE113)  
**Institution:** Amrita Vishwa Vidyapeetham, School of AI  


---

##  Detailed Project Overview

### The Challenge
[cite_start]Radar pulse signal detection is a cornerstone of modern electronic warfare, spectrum monitoring, and aerospace communication systems[cite: 14]. Traditional detection techniques often rely on:
* **Time-domain analysis:** Analyzing the raw signal amplitude over time.
* **Hand-crafted features:** Manually selecting specific signal attributes.
* **Threshold-based decisions:** Simple cut-off points for signal classification.

[cite_start]These methods frequently fail in complex environments or when the **Signal-to-Noise Ratio (SNR)** is low, making it difficult to distinguish true radar pulses from background noise[cite: 15].

### The Solution: Frequency Domain Learning
This project proposes a robust deep learning approach that shifts the analysis from the time domain to the **frequency domain**. [cite_start]By integrating **Fourier Transform (FT)** with a **Convolutional Neural Network (CNN)**, we create a model (FT-CNN) capable of learning complex, periodic patterns that are invisible in raw time-series data[cite: 17, 286].

[cite_start]The core innovation lies in the **FT-CNN architecture**, which utilizes the **Swish activation function** to overcome gradient limitations found in traditional ReLU-based networks, resulting in superior detection reliability[cite: 287].

---

##  Technical Architecture & Methodology

[cite_start]The system follows a strict four-stage pipeline to process signals and classify them as either "Good" (strong pulse) or "Bad" (weak/noisy) [cite: 352-367].

### 1. Data Preprocessing
* [cite_start]**Dataset:** We utilize the **Ionosphere Dataset** from the UCI Machine Learning Repository[cite: 20].
* [cite_start]**Input Dimensions:** 351 instances, each consisting of 34 continuous numerical features representing radar returns[cite: 22, 23].
* [cite_start]**Normalization:** Data is normalized to ensure all feature values fall within a consistent range, stabilizing the neural network training[cite: 45].
* [cite_start]**Handling Missing Data:** Any missing values are imputed using the mean of the respective feature column[cite: 46].

### 2. Feature Extraction (Fourier Transform)
[cite_start]Instead of feeding raw features into the network, we apply the **Fast Fourier Transform (FFT)**[cite: 48].
* [cite_start]**Transformation:** Converts the 34 time-domain features into the **frequency domain**[cite: 49].
* [cite_start]**Magnitude Spectrum:** We calculate the absolute value (magnitude) of the FFT output[cite: 51].
* [cite_start]**Why this matters:** This step isolates periodic signal components and patterns that are characteristic of high-quality radar returns, which are often obscured by noise in the time domain[cite: 50].

### 3. Deep Learning Model (FT-CNN)
[cite_start]The transformed frequency data is reshaped into a `1 x N x 1` format and passed through a Convolutional Neural Network designed with specific optimizations[cite: 56]:

* [cite_start]**Convolutional Layers (Conv1D):** These layers slide filters over the frequency data to detect local patterns and spectral features[cite: 58].
* **Swish Activation Function:** Unlike the standard ReLU (Rectified Linear Unit), we utilize **Swish** (`f(x) = x * sigmoid(x)`).
    * *Theory:* Swish is a smooth, non-monotonic function that allows a small amount of negative information to flow through. [cite_start]This solves the "Dying ReLU" problem where neurons become inactive and stop learning[cite: 287].
    * [cite_start]*Result:* Improved gradient flow and better learning of complex non-linear patterns[cite: 59].
* [cite_start]**Max Pooling:** Reduces the dimensionality of the feature maps, making the model computationally efficient and reducing the risk of overfitting[cite: 61, 62].

### 4. Classification
* [cite_start]**Fully Connected Layers:** Aggregates the local features learned by the CNN into global patterns[cite: 64].
* [cite_start]**Softmax/Sigmoid Layer:** Outputs the final probability, classifying the signal as either **Class 1 (Good)** or **Class 2 (Bad)**[cite: 65, 25].

---

##  Performance & Model Comparison

We benchmarked the **FT-CNN** against two other frequency-domain neural networks to validate its superiority.

| Model | Architecture Description | Accuracy Achieved |
| :--- | :--- | :--- |
| **FT-CNN** (Proposed) | **Convolutional Neural Network** with Swish activation. Learns spatial hierarchies in frequency data. | [cite_start]**98.57%** [cite: 282] |
| **FT-BPNN** | **Back-Propagation Neural Network**. A standard feedforward network that iteratively adjusts weights to minimize error. | [cite_start]**90.00%** [cite: 282] |
| **FT-PNN** | **Probabilistic Neural Network**. Uses Gaussian kernels and Bayesian decision theory to estimate class probability. | [cite_start]**88.57%** [cite: 282] |

[cite_start]**Conclusion:** The FT-CNN significantly outperforms traditional architectures (PNN and BPNN) in the frequency domain, proving that learnable convolutional filters are the most effective method for decoding radar signatures[cite: 12, 288].

---


## 🚀 How to Run the Project
This project uses **Marimo**, a reactive Python notebook format.

1.  **Install Dependencies:**
    ```bash
    pip install marimo numpy pandas matplotlib seaborn scipy scikit-learn tensorflow
    ```

2.  **Run the Main FT-CNN Model:**
    ```bash
    marimo run mfc_ft_cnn.py
    ```

3.  **Run Comparison Models:**
    ```bash
    marimo run MFC_ft_pnn.py
    marimo run MFC_ft_bpnn.py
    ```

---

## 📚 References
1.  [cite_start]**Dataset:** Ionosphere Dataset, UCI Machine Learning Repository[cite: 20].
2.  [cite_start]**Primary Research:** Fengyang Gu, et al., "Detection of Radar Pulse Signals Based on Deep Learning," IEEE[cite: 291].
3.  [cite_start]**Secondary Research:** Vincent G. Sigillito, et al., "Classification of Radar Returns from the Ionosphere using Neural Networks," JHU APL Technical Digest[cite: 293].
