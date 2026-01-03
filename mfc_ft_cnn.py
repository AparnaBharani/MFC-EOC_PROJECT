import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""



        # Radar Pulse Detection using Fourier Transform and Convolutional Neural Network (FT-CNN)

        ---
        #---
        #---
        #---


        # Group 3 Members:
        # Aparna Bharani - `CB.SC.U4AIE24304`
        # I Mahalakshmi - `CB.SC.U4AIE24322`
        # Maalika P - `CB.SC.U4AIE24332`
        # Parkavi R - `CB.SC.U4AIE24338`

        ---
        ## Abstract:
         This project explores radar pulse detection by applying the Fourier Transform (FT) to radar signals and 
        analyzing the resulting frequency components using three neural network models: FT-CNN (Convolutional 
        Neural Network), FT-PNN (Probabilistic Neural Network), and FT-BPNN (Backpropagation Neural 
        Network). We used the Ionosphere Dataset from the UCI repository to train and test these models. The 
        results show that FT-CNN delivers the highest accuracy, highlighting its superior capability in frequency-domain 
        signal classification

        ##Introduction:
         Radar pulse signal detection plays a crucial role in modern electronic warfare, spectrum monitoring, and 
        aerospace communication systems. Traditional signal detection techniques often rely on hand-crafted features 
        and threshold-based decision methods, which may falter under low signal-to-noise ratio (SNR) or complex 
        signal environments

        ## Objective:

        To design and implement a **radar pulse signal detection system** by integrating **Fourier Transform (FT)** with a **Convolutional Neural Network (CNN)**.

        - The **Fast Fourier Transform (FFT)** converts raw time-domain radar signal data into the frequency domain.
        - These transformed signals are then used to train a deep CNN model that **automatically learns frequency-based patterns** for accurate classification of radar pulse presence.
        - The model is evaluated using **precision**, **recall**, and **F1-score** to ensure **high detection reliability**, even in noisy or uncertain environments.

        ---

        ##Data Description:

        - Dataset: **Ionosphere Dataset** (from the UCI Machine Learning Repository)
        - Instances: **351**
        - Features: **34**
        - Description: Radar signal measurements.
        - Target Labels:
          - `'g'` → Good signal
          - `'b'` → Bad signal
        -Data Characteristics:
         -Real-Valued Data
         -No Missing Values
         -Imbalanced Dataset

        ##Model: FT - CNN
        Combining **Fourier-transformed features** with a **deep CNN** to detect the presence of radar pulses effectively.

        ##Methodology:

        ##Data Preprocessing
          -Split the dataset → 80% for training, 20% for testing
  
          -Normalized data → To ensure all feature values are in the same range
          -Missing values → Filled with mean values
  
        ##Feature Extraction — Fourier Transform (FT)

          -FT converts time domain to frequency domain
  
          -Better capture of signal patterns and periodic features
  
          -Used absolute value of FT output.
        CNN Flow
        ##1.Data Preprocessing:

          -FFT applied on Ionosphere Dataset
  
          -Data Normalization
  
          -Reshape into 1xNx1 format suitable for CNN
        ##2.Feature Extraction:

          -Convolution Layers detect local patterns
  
          -Swish Activation improves gradient flow
  
        ##3.Dimensionality Reduction:

          -MaxPoolingreduces feature size
  
          -Controls overfitting
  
        ##4.Classification:

          -Fully Connected Layers learn global patterns
  
          -SoftmaxLayer provides final class probability
        """
    )
    return




@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D,BatchNormalization
    from scipy.fft import fft
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
    import seaborn as sns
    from collections import Counter

    # Step 1: Load and Prepare Dataset
    dataset_path = r"C:\Users\parka\OneDrive\Documents\mfc_eoc_proj\mfc_eoc\ionosphere.csv"
    dataset = pd.read_csv(dataset_path)

    # Extract features (X) and labels (y)
    X = dataset.iloc[:, :-1].values  # Radar signal data
    y = dataset.iloc[:, -1].map({'g': 1, 'b': 0}).values  # Convert labels to binary

    # Step 2: Apply Fourier Transform (FFT)
    X_ft = np.abs(fft(X, axis=1))  # Convert to frequency domain

    # Step 3: Normalize Data
    scaler = StandardScaler()
    X_ft = scaler.fit_transform(X_ft)

    # Step 4: Train a CNN Model
    X_train, X_test, y_train, y_test = train_test_split(X_ft, y, test_size=0.2, random_state=42)
    print(len(X_test))  # Check how many test samples
    print(y_test)       # Class distribution
    print(Counter(y_test))

    # Reshape data for CNN input
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Build CNN Model using Swish activation
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='swish', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, kernel_size=3, activation='swish'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, kernel_size=3, activation='swish'),
        BatchNormalization(),
        MaxPooling1D(2),

        Flatten(),

        Dense(128, activation='swish'),
        Dropout(0.4),

        Dense(1, activation='sigmoid')

    ])

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train Model
    model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate Model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Step 5: Visualize Results
    plt.figure(figsize=(10, 4))
    plt.imshow(X_ft, aspect='auto', cmap='hot')
    plt.colorbar()
    plt.title("Fourier Transformed Radar Signal Data")
    plt.xlabel("Features (Frequency Components)")
    plt.ylabel("Samples")
    plt.show()

    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Compute Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['b (Bad)', 'g (Good)'],
                yticklabels=['b (Bad)', 'g (Good)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for FT-CNN with Swish Activation')
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred_prob)
    print(f"\nMean Squared Error (MSE): {mse:.6f}")

    # Plot a radar signal sample
    plt.figure(figsize=(10, 4))
    plt.plot(X_test[0], label="Radar Signal Sample")
    plt.title(f"Predicted: {y_pred[0][0]}, Actual: {y_test[0]}")
    plt.legend()
    plt.show()
    return (
        BatchNormalization,
        Conv1D,
        Counter,
        Dense,
        Dropout,
        Flatten,
        MaxPooling1D,
        Sequential,
        StandardScaler,
        X,
        X_ft,
        X_test,
        X_train,
        accuracy,
        classification_report,
        conf_matrix,
        confusion_matrix,
        dataset,
        dataset_path,
        fft,
        loss,
        mean_squared_error,
        model,
        mse,
        np,
        pd,
        plt,
        scaler,
        sns,
        tf,
        train_test_split,
        y,
        y_pred,
        y_pred_prob,
        y_test,
        y_train,
    )


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
