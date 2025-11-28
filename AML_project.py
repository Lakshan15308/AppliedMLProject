#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Nish-hub94/AppliedMLProject/blob/main/AML_project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[53]:


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
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


# # **Data loading & Data Reading**

# In[54]:


df = pd.read_csv('industrial_fault_detection_data_1000.csv', index_col=0, skipinitialspace='True')


# # **Data Pre-Processing and Feature Extraction**

# In[55]:


### First, We should get an idea about the Data and their relationship ###

print(df.shape)
print(df.duplicated().any()) # Duplicate value check
print(df.isnull().values.any()) # Null Value Check  # No Null values are available.  
print(df.columns)
print(df.head())
df.describe()


# In[56]:


# Map numeric labels to descriptive names
label_mapping = {0: 'No Fault', 1: 'Bearing Fault', 2: 'Overheating'}
df['Fault Type'] = df['Fault Label'].map(label_mapping)


# In[57]:


feature_cols = ['Vibration (mm/s)', 'Temperature (°C)', 'Pressure (bar)', 'RMS Vibration', 'Mean Temp']
pair_df = df[feature_cols + ['Fault Label']]

Features= ['Vibration (mm/s)', 'Temperature (°C)', 'Pressure (bar)']
pair_df2 = df[Features + ['Fault Label']]

sns.pairplot(pair_df, hue='Fault Label',diag_kind='kde', height=1.7)

plt.show()  

## According to the Pairplot, Vibration Temporature and Pressure will be taken as training features,
## because These three show clear separation between fault classes (0, 1, 2).


# ### According to the Pairplot, Vibration Temporature and Pressure will be taken as training features, because These three show clear separation between fault classes (0, 1, 2).

# In[58]:


# Feature Extraction
sns.pairplot(pair_df2, hue='Fault Label',diag_kind='kde', height=1.7)

plt.show()  

# Graph 1 – Sensor Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['Vibration (mm/s)', 'Temperature (°C)', 'Pressure (bar)']].corr(), annot=True, cmap='coolwarm')
plt.title('Sensor Correlation Heatmap')
plt.show()


# In[59]:


# Check Data imbalance
counts = pair_df2['Fault Label'].value_counts(normalize=True) * 100
print("Data Imbalance =", counts) # This means the data is highly imbalanced (60/30/8).

# Plot Example output 
plt.figure(figsize=(6,4))
sns.countplot(data= df, x='Fault Type', palette='coolwarm')
plt.title('Count of Each Fault Type')
plt.show()

df


# In[60]:


## Quantitative analysis
## Mean
print("Mean \n", df.groupby('Fault Type').mean())
## Standard Deviation
print("Standard Deviation \n" , df.groupby('Fault Type').std())
# the mean and standard deviation values further support the observations to validate relationships between features and fault types.


# In[61]:


# Hypothesis Test was performed to test whether the data were normally distributed or not.
normaltest
print(normaltest(df['Vibration (mm/s)']))
alpha =0.5
statistic, pvalue = normaltest(df['Vibration (mm/s)'])

if pvalue<alpha:
    print('Data is not normally distributed')
else:
    print('Data is not distributed')


# ### Since the Data is not normally distributed. We need to use non-parametric tests and transform the data

# In[62]:


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


# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[Features].values
y = df['Fault Label'].values

# Split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# In[64]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Select Features and Labels
feature_cols = ['Vibration (mm/s)', 'Temperature (°C)', 'Pressure (bar)']
X = df[feature_cols].values
y = df['Fault Label'].values

# 2. Train-Test Split (ALWAYS FIRST)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# 3. Standardization (fit on TRAIN only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. PCA (fit ONLY on training data)
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Graph 2 – PCA Variance Plot
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA Variance Plot')
plt.grid(True)
plt.show()

# Graph 3 – PCA 2D Scatter Plot (Colored by Fault State)
pca_2 = PCA(n_components=2)
X_train_pca2 = pca_2.fit_transform(X_train_scaled)

plt.figure(figsize=(6,5))
for label in np.unique(y_train):
    plt.scatter(
        X_train_pca2[y_train == label, 0],
        X_train_pca2[y_train == label, 1],
        label=f"Fault {label}"
    )

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 2D Scatter Plot by Fault Type')
plt.legend()
plt.show()

# Graph 4 – K-Means Cluster Visualization
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_train_pca2)

plt.figure(figsize=(6,5))
plt.scatter(X_train_pca2[:,0], X_train_pca2[:,1], c=clusters, alpha=0.6)
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    color='red',
    marker='x',
    s=100
)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means Clusters on PCA Components')
plt.show()


# In[65]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot distributions using seaborn (plotData is not defined)
sns.histplot(df['Vibration (mm/s)'], bins=30, kde=True, ax=axes[0], color='C0')
axes[0].set_title('Vibration Distribution')
axes[0].set_xlabel('Vibration (mm/s)')

sns.histplot(df['Temperature (°C)'], bins=30, kde=True, ax=axes[1], color='C1')
axes[1].set_title('Temperature Distribution')
axes[1].set_xlabel('Temperature (°C)')

sns.histplot(df['Pressure (bar)'], bins=30, kde=True, ax=axes[2], color='C2')
axes[2].set_title('Pressure Distribution')
axes[2].set_xlabel('Pressure (bar)')

plt.tight_layout()
plt.show()




# # **Model Creation**

# In[66]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)


# In[67]:


# === Compute class weights (optional but explicit) ===
from xgboost import XGBClassifier

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int(cls): w for cls, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)
xgb_scale_pos_weight = 1
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
    ),
       # === NEW: XGBoost model ===
    'XGBoost': XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_estimators=100,        # sklearn GBDT equivalent
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        colsample_bytree=1.0,
        scale_pos_weight=xgb_scale_pos_weight,
        tree_method="hist",      # faster modern default
        use_label_encoder=False
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

### weer | small decision tree model for feature importance visualization ###
### 


# In[68]:


import nbconvert

get_ipython().system('jupyter nbconvert --to script AML_project.ipynb')

