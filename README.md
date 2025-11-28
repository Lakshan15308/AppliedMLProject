# AML Project – Industrial Fault Detection Using Machine Learning

##### A complete pipeline for preprocessing, feature engineering, model training, evaluation, and interactive deployment.

### Project Overview

##### This project focuses on building a Machine Learning-Based Industrial Fault Detection System using real-world sensor data.
##### The system classifies equipment conditions into:

No Fault

Bearing Fault

Overheating

### The workflow includes:

#### Data preprocessing & visualization

Outlier handling

Class imbalance treatment (SMOTE)

Dimensionality reduction (PCA)

Model training (RandomForest, XGBoost, SVM)

Model evaluation (RMSE, MAE, Accuracy, Confusion Matrix)

Interactive prediction app using Streamlit

### Project Structure

AML_Project/
│── data/
│   ├── dataset.csv
│
│── notebooks/
│   ├── EDA_and_Preprocessing.ipynb
│   ├── Model_Training.ipynb
│   ├── PCA_Clustering.ipynb
│
│── models/
│   ├── fault_model.pkl
│   └── scaler.pkl
│
│── app/
│   ├── app.py                # Streamlit interface
│
│── README.md
│── requirements.txt
│── AML_Project.py            # (Converted from notebook)

### How to Run the Project


1. Clone the repository
git clone <your-repo-url>
cd AML_Project


2. Install requirements
pip install -r requirements.txt


If you don’t have nbconvert installed:

pip install nbconvert

3. Convert Notebook to Python Script (Optional)

If you want .py output:

jupyter nbconvert --to script AML_Project.ipynb

4. Run Streamlit App
cd app
streamlit run app.py


The interface will open in your browser.

### Features Implemented

#### ✔ Data Preprocessing

Missing value imputation

Outlier detection (IQR / Z-score)

Scaling (StandardScaler)

Normality testing (Shapiro–Wilk, QQ-plots)

#### ✔ Handling Imbalanced Data

Since the classes were in ratios like 60% / 30% / 8%, SMOTE was applied:

from imblearn.over_sampling import SMOTE

#### ✔ PCA Visualization

Explained variance plot

2D PCA scatter plot colored by fault type

Clustering: K-Means on PCA components

#### ✔ Model Training

Several models trained and compared:

Random Forest

XGBoost

SVM

Gaussian Mixture Models (for unsupervised)

Best model saved as:

fault_model.pkl

#### ✔ Model Evaluation

Confusion Matrix

Classification Report

Accuracy, Precision, Recall, F1-score

ROC–AUC

Cross-validation

#### ✔ Streamlit Prediction App

Interactive UI with sliders:

Vibration

Temperature

Pressure

Outputs:

Predicted fault type

Probability bar chart

### Example Prediction Output

Predicted Fault: Bearing Fault
Confidence:
 - No Fault: 0.04
 - Bearing Fault: 0.87
 - Overheating: 0.09


### Deployment & Usage
Run locally via Streamlit
streamlit run app/app.py


### Contributors

Lakshan Siriwardhana, Nishel Pirispulle, Tharanga Dissanayake, Lahiru Samaraweera

### License

This project is licensed under the MIT License.