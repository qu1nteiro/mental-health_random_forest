# Predicting Mental Health Treatment in the Tech Workplace

This repository contains a series of machine learning models designed to predict whether an individual in the tech industry has sought treatment for a mental health condition. Using data from a 2014 survey, we explore, clean, and model the dataset to identify key predictive factors. This project was developed as part of the AIA course unit - University of Aveiro.

The project progresses from a simple interpretable model to more complex ensemble and deep learning methods, demonstrating a complete machine learning workflow.

## Dataset

The analysis is based on the `survey.csv` file, which contains responses from the 2014 OSMI (Open Sourcing Mental Illness) survey, available at Kagle. It includes 1259 responses with 27 columns covering demographic information, work-related factors, and attitudes towards mental health in the workplace.

The target variable for this analysis is `treatment`, a binary feature indicating whether the respondent has sought mental health treatment.

## Analysis and Modeling Notebooks

The project is structured into three Jupyter notebooks, each focusing on a different modeling approach.

### 1. `Notebook 1 – Logistic Regression.ipynb`

This notebook serves as the foundation for the project. It covers the initial data exploration, cleaning, and the implementation of a baseline Logistic Regression model.

**Key Steps:**
*   **Exploratory Data Analysis (EDA):** Initial inspection of data types, missing values, and distribution of key variables like `Age` and the target `treatment`.
*   **Data Cleaning:** Normalization of `Age` and `Gender` fields and encoding of binary "Yes/No" columns.
*   **Preprocessing Pipeline:** Construction of a robust `scikit-learn` pipeline using `ColumnTransformer` to handle numerical and categorical features. This includes imputing missing values, scaling numerical data with `StandardScaler`, and one-hot encoding categorical data.
*   **Modeling:**
    *   Training a `DummyClassifier` to establish a baseline accuracy.
    *   Training a `LogisticRegression` model with 5-fold stratified cross-validation.
    *   Hyperparameter tuning using `GridSearchCV` to find the optimal regularization strength (`C`) and penalty (`l1`, `l2`).
*   **Evaluation:** Analysis of the model's performance using classification reports, confusion matrices, ROC curves, and AUC scores.
*   **Feature Importance:** Interpretation of model coefficients to identify the most significant predictors for seeking treatment.

### 2. `Notebook 2 -Random Forest.ipynb`

This notebook builds on the initial analysis by exploring a more powerful ensemble model, the Random Forest.

**Key Steps:**
*   **Model Training:** A `RandomForestClassifier` is integrated into the preprocessing pipeline defined in the first notebook.
*   **Hyperparameter Tuning:** `GridSearchCV` is used to optimize `n_estimators` (number of trees) and `max_depth` (maximum tree depth), focusing on the F1-score as the primary metric.
*   **Model Evaluation:** The tuned model's performance is thoroughly evaluated on the test set using classification reports, confusion matrices, and ROC-AUC metrics.
*   **Feature Subset Experiment:** An additional experiment is conducted using a smaller set of features identified as important by the L1-regularized Logistic Regression model from the first notebook.

### 3. `Notebook 3 - MLP.ipynb`

This notebook explores deep learning solutions by implementing Multi-Layer Perceptrons (MLPs). It uniquely features both a from-scratch implementation and a high-level framework implementation.

**Key Steps:**
*   **NumPy "Raw" MLP Implementation:**
    *   A simple two-layer neural network is built from scratch using only NumPy.
    *   This includes functions for parameter initialization, forward propagation, backward propagation (with L2 regularization), and gradient descent.
    *   The model is trained on the full training set and evaluated.
*   **Refined MLP with K-Fold Cross-Validation:**
    *   The raw model is enhanced with early stopping to prevent overfitting.
    *   A manual grid search is performed over the hyperparameters (`n_h`, `lambda_reg`, `learning_rate`) using 5-fold cross-validation to find the best-performing combination.
    *   The final model is retrained on all training data using the best hyperparameters.
*   **TensorFlow/Keras Implementation:**
    *   A more modern and deeper MLP is built using the Keras API.
    *   The model leverages the Adam optimizer, `binary_crossentropy` loss, and Keras's built-in `EarlyStopping` callback.
    *   The final Keras model is trained and evaluated against the test set, providing a benchmark against the from-scratch implementation.

## Methodology

The core workflow across all notebooks includes the following stages:

1.  **Data Cleaning:** Invalid entries in the `Age` column are filtered, and the free-text `Gender` column is normalized into three categories: 'Male', 'Female', and 'Other'.
2.  **Preprocessing:** A `scikit-learn` `ColumnTransformer` pipeline is used to process features before modeling.
    *   **Numerical Features:** Missing values are imputed with the median, and data is scaled using `StandardScaler`.
    *   **Categorical Features:** Missing values are imputed with a constant 'Missing' value, and features are transformed using `OneHotEncoder`.
3.  **Data Splitting:** The data is split into training (80%) and test (20%) sets. For the MLP models, the training set is further divided to create a validation set for early stopping and hyperparameter tuning. Stratification is used to maintain the target class distribution in all splits.
4.  **Model Evaluation:** Models are evaluated based on a suite of metrics including:
    *   **F1-Score:** The primary metric for hyperparameter tuning, balancing precision and recall.
    *   **Classification Report:** Provides precision, recall, and F1-score for each class.
    *   **Confusion Matrix:** Visualizes the model's performance in terms of true/false positives and negatives.
    *   **ROC Curve and AUC Score:** Measures the model's ability to distinguish between the two classes.

## How to Run

To run this project, clone the repository and install the required libraries.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/qu1nteiro/mental-health_tech.git
    cd mental-health_tech
    ```

2.  **Install dependencies:**
    Ensure you have Python 3 installed. You can install the necessary libraries using pip.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```

3.  **Run the notebooks:**
    Launch Jupyter Notebook or JupyterLab and open the `.ipynb` files to explore the analysis and execute the code.
    ```bash
    jupyter notebook
