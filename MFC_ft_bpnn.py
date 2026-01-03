import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _(mo):
    mo.md(
        """
        ###FT-BPNN (Fourier Transform + Back-Propagation Neural Network)

        - The FT-BPNN model applies a **Feedforward Neural Network (FNN)** trained using the **backpropagation algorithm** to the Fourier-transformed radar signal data.
        - It **learns complex, non-linear patterns** in the frequency domain by **iteratively adjusting weights** to minimize classification error.
        - The model uses a **loss function** (like cross-entropy or MSE) to guide training.
        - Requires **sufficient training data** and **computational resources**, but is highly flexible and powerful.
        - Suitable for tasks involving complex, high-dimensional frequency features.
        """
    )
    return



@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.utils import to_categorical

    # 1. Load dataset
    data = pd.read_csv(r"C:\Users\parka\OneDrive\Documents\mfc_eoc_proj\mfc_eoc\ionosphere.csv")  # Replace with your file

    # 2. Features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 3. Apply Fourier Transform (magnitude)
    X_ft = np.abs(np.fft.fft(X, axis=1))

    # 4. Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ft)

    # 5. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

    # 7. Build BPNN model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_categorical.shape[1], activation='softmax'))

    # 8. Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 9. Train
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

    # 10. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n Test Accuracy: {accuracy:.4f}")

    # 11. Predict
    y_score = model.predict(X_test)
    y_pred = np.argmax(y_score, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 12. ROC Curve (Single line using macro-average AUC)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('FT-BPNN ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 13. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - FT + BPNN")
    plt.tight_layout()
    plt.show()
    return (
        ConfusionMatrixDisplay,
        Dense,
        LabelEncoder,
        Sequential,
        StandardScaler,
        X,
        X_ft,
        X_scaled,
        X_test,
        X_train,
        accuracy,
        auc,
        cm,
        confusion_matrix,
        data,
        disp,
        fpr,
        history,
        le,
        loss,
        model,
        np,
        pd,
        plt,
        roc_auc,
        roc_curve,
        scaler,
        to_categorical,
        tpr,
        train_test_split,
        y,
        y_categorical,
        y_encoded,
        y_pred,
        y_score,
        y_test,
        y_train,
        y_true,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ##RESULTS

        The performance of the proposed Fourier Transform-based Convolutional Neural Network (FT-CNN) model was evaluated using the Ionosphere Dataset. The model was trained using the Adam optimizer with over 60 epochs.

        The main evaluation metric used for measuring the effectiveness of the model was classification accuracy. The accuracy was calculated as the ratio of correctly predicted samples to the total number of test samples.

        Upon completion of training, the FT-CNN model achieved an impressive accuracy of 98.57%, outperforming conventional machine learning models like FT-PNN (88.57%) and FT-BPNN(90.00%).

        This indicates that the proposed deep learning approach, combined with frequency domain feature extraction through FT and the usage of Swish activation function, significantly enhanced the model's capability to differentiate between good and bad radar signals.

        ##CONCLUSION

        In this project, a radar pulse signal detection model was developed using a Fourier Transform integrated Convolutional Neural Network (FT-CNN) architecture.

         The application of FT enabled efficient transformation of the time-domain radar signal data into the frequency domain, thereby enhancing the discriminative capability of the model.
 
        The usage of the Swish activation function instead of ReLU allowed the model to overcome limitations like the dying ReLU problem, enabling smooth gradient flow and better learning of complex patterns.

        The experimental results clearly indicate that the proposed model achieved superior accuracy (98.57%) compared to conventional methods. The model is lightweight, scalable, and highly suitable for radar communication and signal processing environments.

        ##REFRENCES

        Detection of Radar Pulse Signals Based on Deep Learning FENGYANG GU,LUXIN ZHANG, SHILIAN ZHENG,JIECHEN ,KEQIANG YUE, ZHIJIN ZHAO AND XIAONIU YANG

        https://ieeexplore.ieee.org/document/10614929/

        VINCENT G. SIGILLITO, SIMON P. WING, LARRIE V. HUTTON, and KILE B. BAKER CLASSIFICATION OF RADAR RETURNS FROM THE IONOSPHERE USING NEURAL NETWORKS 

        https://secwww.jhuapl.edu/techdigest/content/techdigest/pdf/

        ##ACKNOWLEDGEMENT

        We sincerely thank professor Sunil Kumar, Faculty, School Of AI, Amrita Vishwa Vidyapeetham,Ettimadai for their continuous support, valuable insights, and expert guidance throughout the course of this project. Their encouragement and constructive suggestions greatly enhanced the quality of our work.

        We also extend our gratitude to the Department of AI for providing the necessary academic environment and facilities that enabled us to carry out this project successfully.

        Finally, we are grateful to all those who have directly or indirectly supported us during the development of this project.

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
