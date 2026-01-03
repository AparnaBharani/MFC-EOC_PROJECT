import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        ##Model Comparison

        This FT-CNN model is compared with two other models:

        - **FT-PNN**: Fourier Transform + Probabilistic Neural Network  
        - **FT-BPNN**: Fourier Transform + Back-propagation Neural Network

        ---

        ###FT-PNN (Fourier Transform + Probabilistic Neural Network)

        - The FT-PNN model applies a **Probabilistic Neural Network (PNN)** to the Fourier-transformed radar signal data.
        - It uses **Gaussian kernels** to estimate the probability density function (PDF) of each class.
        - Classification is based on **Bayesian decision theory**.
        - **Non-iterative**, making it **fast to train**.
        - Highly effective for **small to medium datasets** with high-dimensional features (like frequency domain data).
        """
    )
    return



@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.fft import fft
    from scipy.stats import multivariate_normal
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error

    # 1️⃣ Load Dataset
    df = pd.read_csv(r"C:\Users\parka\OneDrive\Documents\mfc_eoc_proj\mfc_eoc\ionosphere.csv")  # Use your actual dataset file

    # 2️⃣ Extract Features & Labels (Modify Column Names Accordingly)
    feature_columns = df.columns[:-1]  # All but last column as features
    label_column = df.columns[-1]      # Last column as label

    X = df[feature_columns].values
    y = df[label_column].values

    # 3️⃣ Compute Fourier Transform Features
    FT_features = np.abs(fft(X, axis=1))  # Compute magnitude spectrum
    FT_features = FT_features[:, :FT_features.shape[1] // 2]  # Keep only first half (FFT symmetry)

    # 4️⃣ Normalize Data
    scaler = StandardScaler()
    FT_features = scaler.fit_transform(FT_features)

    # 5️⃣ Split Data into Train & Test
    X_train, X_test, y_train, y_test = train_test_split(FT_features, y, test_size=0.2, random_state=42)

    # 6️⃣ Implement Probabilistic Neural Network (PNN)
    def pnn_predict(X_train, y_train, X_test, sigma=0.5):
        """ PNN Classification using Gaussian Kernel Density Estimation """
        classes = np.unique(y_train)
        predictions = []

        for x in X_test:
            probabilities = []
            for c in classes:
                X_class = X_train[y_train == c]  # Get samples of class c
                prob = np.mean([multivariate_normal.pdf(x, mean=xi, cov=sigma**2) for xi in X_class])
                probabilities.append(prob)

            predictions.append(classes[np.argmax(probabilities)])  # Class with highest probability

        return np.array(predictions)

    # 7️⃣ Train & Predict with PNN
    y_pred = pnn_predict(X_train, y_train, X_test, sigma=0.5)

    # 8️⃣ Evaluate Model Performance

    # 🔹 Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'🔹 FT-PNN Accuracy: {accuracy * 100:.2f}%')

    # 🔹 Mean Squared Error (MSE)
    y_test_num = (y_test == np.unique(y_test)[1]).astype(int)  # Convert to numeric (0/1)
    y_pred_num = (y_pred == np.unique(y_test)[1]).astype(int)
    mse = mean_squared_error(y_test_num, y_pred_num)
    print(f'🔹 Mean Squared Error (MSE): {mse:.4f}')

    # 🔹 Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('FT-PNN Confusion Matrix')
    plt.show()

    # 🔹 ROC Curve
    fpr, tpr, _ = roc_curve(y_test_num, y_pred_num)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('FT-PNN ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    return (
        FT_features,
        StandardScaler,
        X,
        X_test,
        X_train,
        accuracy,
        accuracy_score,
        auc,
        conf_matrix,
        confusion_matrix,
        df,
        feature_columns,
        fft,
        fpr,
        label_column,
        mean_squared_error,
        mse,
        multivariate_normal,
        np,
        pd,
        plt,
        pnn_predict,
        roc_auc,
        roc_curve,
        scaler,
        sns,
        tpr,
        train_test_split,
        y,
        y_pred,
        y_pred_num,
        y_test,
        y_test_num,
        y_train,
    )


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
