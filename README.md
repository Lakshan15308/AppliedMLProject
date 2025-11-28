# AML Project – Industrial Fault Detection Using Machine Learning

##### A complete pipeline for preprocessing, feature engineering, model training, evaluation, and interactive deployment.

## Project Overview

##### This project focuses on building a Machine Learning-Based Industrial Fault Detection System using real-world sensor data.
##### The system classifies equipment conditions into:

###### No Fault

###### Bearing Fault

###### Overheating

## The workflow includes:

#### Data preprocessing & visualization

###### Outlier handling (Duplicates, Null Values)

###### Class imbalance treatment (SMOTE)

###### Dimensionality reduction (PCA, T-sne and UMAP)

###### Model training (RandomForest, XGBoost, SVM)

###### Model evaluation (RMSE, MAE, Accuracy, Confusion Matrix)

###### Interactive prediction app using Streamlit

### How to Run the Project

###### 1. Clone the repository
####### git clone <https://github.com/Lakshan15308/AppliedMLProject/tree/main>
####### cd AML_Project

###### 2. Install requirements
####### pip install -r requirements.tx

###### If you don’t have nbconvert installed:
####### pip install nbconvert

###### 3. Convert Notebook to Python Script (Optional)
####### If you want .py output:
####### jupyter nbconvert --to script AML_Project.ipynb

###### 4. Run Streamlit App
####### cd app
####### streamlit run app.py
####### The interface will open in your browser.


#### ✔ Model Training

###### Several models trained and compared:

####### Random Forest
####### XGBoost
####### SVM
####### Gaussian Mixture Models (for unsupervised)

###### Best model saved as: fault_model.pkl (Gaussian Mixture Model)

#### ✔ Model Evaluation

####### Confusion Matrix
####### Classification Report
####### ROC–AUC
####### Cross-validation

### Example Prediction Output

####### Predicted Fault: Bearing Fault
####### Confidence:
####### - No Fault: 0.04
####### - Bearing Fault: 0.87
####### - Overheating: 0.09


### Deployment & Usage
####### Run locally via Streamlit - streamlit run app/app.py


### Contributors

#### Lakshan Siriwardhana, Nishel Pirispulle, Tharanga Dissanayake, Lahiru Samaraweera

### License

##### This project is licensed under the MIT License.