#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Nish-hub94/AppliedMLProject/blob/main/AML_project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import normaltest
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# # **Data loading & Data Reading**

# In[2]:


df = pd.read_csv('industrial_fault_detection_data_1000.csv', index_col=0, skipinitialspace='True')


# In[3]:


df


# # **Data Pre-Processing and Feature Extraction**

# In[4]:


### First We should get an idea about the Data and their relationship ###
print(df.head())
print(df.shape)
print(df.columns)
df.describe()


# In[5]:


print(df.isnull().values.any())
df.count().isna()   # No Null values are available.


# In[6]:


# Hypothesis Testing  
normaltest
print(normaltest(df['Fault Label']))
alpha =0.5
statistic, pvalue = normaltest(df['Fault Label'])

if pvalue<alpha:
    print('Data is not normally distributed')
else:
    print('Data is not distributed')

print(df['Fault Label'].value_counts())  # Data is not normally distributed.

sns.kdeplot(df)


# In[7]:


#Map numeric labels to descriptive names
label_mapping = {0: 'No Fault', 1: 'Bearing Fault', 2: 'Overheating'}
df['Fault Type'] = df['Fault Label'].map(label_mapping)

#Count each type
counts = df['Fault Type'].value_counts()
print(counts)



# In[8]:


sns.countplot(data=df, x='Fault Type', palette='coolwarm')
plt.title('Count of Each Fault Type')
plt.show()


# In[9]:


feature_cols = ['Vibration (mm/s)', 'Temperature (°C)', 'Pressure (bar)', 'RMS Vibration', 'Mean Temp']
pair_df = df[feature_cols + ['Fault Label']]

Features= ['Vibration (mm/s)', 'Temperature (°C)', 'Pressure (bar)']
pair_df2 = df[Features + ['Fault Label']]


# In[10]:


sns.pairplot(pair_df, hue='Fault Label',diag_kind='kde', height=1.7)

plt.show()  


# ### # According to the Pairplot, Vibration Temporature and Pressure will be taken astraining features because These three show clear separation between fault classes (0, 1, 2).

# In[11]:


sns.pairplot(pair_df2, hue='Fault Label',diag_kind='kde', height=1.7)

plt.show()  


# In[12]:


# T-Test 
# Split the data into two groups
fault0 = df[df["Fault Label"] == 0]["Vibration (mm/s)"]
fault_non0 = df[df["Fault Label"] != 0]["Vibration (mm/s)"]

# Perform independent t-test
t_stat, p_value = ttest_ind(fault0, fault_non0, equal_var=False)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Significant difference detected between groups.")
else:
    print("No significant difference between groups.")
##---------------------------------------------------------------------------------------------
# T-Test 
# Split the data into two groups
fault0 = df[df["Fault Label"] == 0]["Temperature (°C)"]
fault_non0 = df[df["Fault Label"] != 0]["Temperature (°C)"]

# Perform independent t-test
t_stat, p_value = ttest_ind(fault0, fault_non0, equal_var=False)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Significant difference detected between groups.")
else:
    print("No significant difference between groups.")

##---------------------------------------------------------------------------------------------
# T-Test 
# Split the data into two groups
fault0 = df[df["Fault Label"] == 0]["Pressure (bar)"]
fault_non0 = df[df["Fault Label"] != 0]["Pressure (bar)"]

# Perform independent t-test
t_stat, p_value = ttest_ind(fault0, fault_non0, equal_var=False)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Significant difference detected between groups.")
else:
    print("No significant difference between groups.")


# In[13]:


# Mann-Whitney rank-test 
# Groups
g1 = df[df["Fault Label"] == 0]["Pressure (bar)"]
g2 = df[df["Fault Label"] != 0]["Pressure (bar)"]

# Mann-Whitney U-test (non-parametric)
stat, p_value = mannwhitneyu(g1, g2, alternative="two-sided")

print("U-statistic:", stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference between groups.")
else:
    print("No significant difference between groups.")

#-------------------------------------------------------------
# Groups
g1 = df[df["Fault Label"] == 0]["Temperature (°C)"]
g2 = df[df["Fault Label"] != 0]["Temperature (°C)"]

# Mann-Whitney U-test (non-parametric)
stat, p_value = mannwhitneyu(g1, g2, alternative="two-sided")

print("U-statistic:", stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference between groups.")
else:
    print("No significant difference between groups.")
#-------------------------------------------------------------
# Groups
g1 = df[df["Fault Label"] == 0]["Vibration (mm/s)"]
g2 = df[df["Fault Label"] != 0]["Vibration (mm/s)"]

# Mann-Whitney U-test (non-parametric)
stat, p_value = mannwhitneyu(g1, g2, alternative="two-sided")

print("U-statistic:", stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference between groups.")
else:
    print("No significant difference between groups.")


# # **Preprocessing**

# Seperating features and labels

# In[14]:


X = df[Features].values
y = df['Fault Label'].values


# In[15]:


print(X)


# In[16]:


print(y)


# In[17]:


# === Map labels (for interpretability in plots/reports) ===
label_mapping = {0: 'No Fault', 1: 'Bearing Fault', 2: 'Overheating'}
df['Fault Type'] = df['Fault Label'].map(label_mapping)


# **Split first (to prevent data leakage)**

# In[18]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# In[20]:


scaler = StandardScaler()


# In[21]:


# X_train_scaled is the standardized version of the training data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# # **Model Creation**

# In[22]:


# === Compute class weights (optional but explicit) ===
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int(cls): w for cls, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

# === Define candidate models ===
models = {
    'LogisticRegression': Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            multi_class='multinomial',
            class_weight=class_weight_dict,
            max_iter=2000,
            solver='lbfgs',
            random_state=42
        ))
    ]),
    'SVC_RBF': Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='rbf',
            class_weight=class_weight_dict,
            probability=False,  # set True if you need calibrated probabilities later
            C=1.0,
            gamma='scale',
            random_state=42
        ))
    ]),
    'RandomForest': RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight_dict,
        n_jobs=-1,
        random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        random_state=42
        # Note: no native class_weight; included for comparison
    )
}

# === Cross-validation setup ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'f1_macro': 'f1_macro'
}

cv_results = {}

for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    summary = {
        'accuracy_mean': np.mean(scores['test_accuracy']),
        'balanced_accuracy_mean': np.mean(scores['test_balanced_accuracy']),
        'f1_macro_mean': np.mean(scores['test_f1_macro'])
    }
    cv_results[name] = summary
    print(f"\n{name} CV scores: {summary}")

# === Pick best model by macro F1 ===
best_name = max(cv_results, key=lambda k: cv_results[k]['f1_macro_mean'])
best_model = models[best_name]
print(f"\nSelected best model: {best_name} (by macro F1)")

# === Fit best model on train and evaluate on test ===
if isinstance(best_model, Pipeline):
    best_model.fit(X_train, y_train)
else:
    best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("\nClassification report (test set):")
print(classification_report(y_test, y_pred, target_names=[label_mapping[i] for i in sorted(label_mapping.keys())]))

# === Confusion matrix ===
cm = confusion_matrix(y_test, y_pred, labels=sorted(label_mapping.keys()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_mapping[i] for i in sorted(label_mapping.keys())])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.show()

# === Feature importance (for tree or linear) ===
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(7,4))
        sns.barplot(x=fi.values, y=fi.index, palette='viridis')
        plt.title('Feature importance (tree-based)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    elif isinstance(model, Pipeline) and hasattr(model.named_steps['clf'], 'coef_'):
        # For multinomial logistic regression
        coefs = model.named_steps['clf'].coef_  # shape: (n_classes, n_features)
        coef_df = pd.DataFrame(coefs, columns=feature_names)
        coef_df.index = [label_mapping[i] for i in sorted(label_mapping.keys())]
        plt.figure(figsize=(8,5))
        sns.heatmap(coef_df, annot=True, fmt='.3f', cmap='coolwarm', cbar=True)
        plt.title('Class-wise coefficients (Logistic Regression)')
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importance not available for this model.")

# Plot importance if available
if best_name == 'RandomForest':
    plot_feature_importance(best_model, features)
elif best_name == 'LogisticRegression':
    plot_feature_importance(best_model, features)


# In[24]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# -----------------------------
# Load your trained model
# -----------------------------
# Assume you saved the pipeline as 'fault_model.pkl'
joblib.dump(best_model, 'fault_model.pkl')
model = joblib.load('fault_model.pkl')

# Label mapping
label_mapping = {0: 'No Fault', 1: 'Bearing Fault', 2: 'Overheating'}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Industrial Fault Detection System ⚙️")
st.write("Test the trained model with custom sensor inputs.")

# Input widgets
vibration = st.slider("Vibration (mm/s)", 0.0, 1.5, 0.5, 0.01)
temperature = st.slider("Temperature (°C)", 50.0, 130.0, 90.0, 0.5)
pressure = st.slider("Pressure (bar)", 7.0, 10.0, 8.0, 0.1)
rms_vibration = st.slider("RMS Vibration", 0.0, 1.5, 0.6, 0.01)
mean_temp = st.slider("Mean Temp", 50.0, 130.0, 90.0, 0.5)

# Collect inputs
input_data = np.array([[vibration, temperature, pressure]])

# Predict
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

# Display results
st.subheader("Prediction Result")
st.write(f"**Predicted Fault Type:** {label_mapping[prediction]}")

st.subheader("Prediction Probabilities")
prob_df = pd.DataFrame([proba], columns=[label_mapping[i] for i in range(len(proba))])
st.bar_chart(prob_df.T)



# In[ ]:




