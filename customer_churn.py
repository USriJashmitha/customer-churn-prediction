# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
# Note: Replace 'telco_churn.csv' with your actual dataset path
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset file not found. Please check the path.")
    exit()

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Data Preprocessing
print("\nPreprocessing the data...")

# Drop customer ID as it's not useful for prediction
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert Churn to binary (1 for 'Yes', 0 for 'No')
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Churn')  # Churn is our target variable

# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Feature scaling for numerical variables
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Model Training
print("\nTraining models...")

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Store results
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print basic metrics
    print(f"{name} Accuracy: {results[name]['accuracy']:.4f}")
    print(f"{name} ROC AUC: {results[name]['roc_auc']:.4f}")

# Hyperparameter tuning for the best model (Random Forest in this case)
print("\nPerforming hyperparameter tuning for Random Forest...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]

# Store best model results
results['Best Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_best),
    'roc_auc': roc_auc_score(y_test, y_prob_best),
    'classification_report': classification_report(y_test, y_pred_best),
    'confusion_matrix': confusion_matrix(y_test, y_pred_best),
    'best_params': grid_search.best_params_
}

# Print best model metrics
print(f"\nBest Random Forest Accuracy: {results['Best Random Forest']['accuracy']:.4f}")
print(f"Best Random Forest ROC AUC: {results['Best Random Forest']['roc_auc']:.4f}")
print("Best Parameters:", results['Best Random Forest']['best_params'])

# Feature Importance
print("\nFeature Importance from Best Random Forest Model:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10))

# Visualization
plt.figure(figsize=(15, 6))

# ROC Curve
plt.subplot(1, 2, 1)
for name in results:
    if name != 'Best Random Forest':  # We'll plot this separately
        fpr, tpr, _ = roc_curve(y_test, models[name].predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.2f})')

# Plot best model ROC
fpr, tpr, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'Best RF (AUC = {results["Best Random Forest"]["roc_auc"]:.2f})', linewidth=2, linestyle='--')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Feature Importance Plot
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()

# Print detailed classification reports
print("\nDetailed Classification Reports:")
for name in results:
    print(f"\n{name} Classification Report:")
    print(results[name]['classification_report'])
    print("Confusion Matrix:")
    print(results[name]['confusion_matrix'])
import sys
# Change this line in your code
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
# Note: Replace 'telco_churn.csv' with your actual dataset path
try:
    df = pd.read_csv('telco_churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset file not found. Please check the path.")
    exit()

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Data Preprocessing
print("\nPreprocessing the data...")

# Drop customer ID as it's not useful for prediction
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert Churn to binary (1 for 'Yes', 0 for 'No')
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Churn')  # Churn is our target variable

# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Feature scaling for numerical variables
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Model Training
print("\nTraining models...")

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Store results
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print basic metrics
    print(f"{name} Accuracy: {results[name]['accuracy']:.4f}")
    print(f"{name} ROC AUC: {results[name]['roc_auc']:.4f}")

# Hyperparameter tuning for the best model (Random Forest in this case)
print("\nPerforming hyperparameter tuning for Random Forest...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]

# Store best model results
results['Best Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_best),
    'roc_auc': roc_auc_score(y_test, y_prob_best),
    'classification_report': classification_report(y_test, y_pred_best),
    'confusion_matrix': confusion_matrix(y_test, y_pred_best),
    'best_params': grid_search.best_params_
}

# Print best model metrics
print(f"\nBest Random Forest Accuracy: {results['Best Random Forest']['accuracy']:.4f}")
print(f"Best Random Forest ROC AUC: {results['Best Random Forest']['roc_auc']:.4f}")
print("Best Parameters:", results['Best Random Forest']['best_params'])

# Feature Importance
print("\nFeature Importance from Best Random Forest Model:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10))

# Visualization
plt.figure(figsize=(15, 6))

# ROC Curve
plt.subplot(1, 2, 1)
for name in results:
    if name != 'Best Random Forest':  # We'll plot this separately
        fpr, tpr, _ = roc_curve(y_test, models[name].predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.2f})')

# Plot best model ROC
fpr, tpr, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'Best RF (AUC = {results["Best Random Forest"]["roc_auc"]:.2f})', linewidth=2, linestyle='--')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Feature Importance Plot
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()

# Print detailed classification reports
print("\nDetailed Classification Reports:")
for name in results:
    print(f"\n{name} Classification Report:")
    print(results[name]['classification_report'])
    print("Confusion Matrix:")
    print(results[name]['confusion_matrix'])
# To something like:
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
# Note: Replace 'telco_churn.csv' with your actual dataset path
try:
    df = pd.read_csv('telco_churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset file not found. Please check the path.")
    exit()

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Data Preprocessing
print("\nPreprocessing the data...")

# Drop customer ID as it's not useful for prediction
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert Churn to binary (1 for 'Yes', 0 for 'No')
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Churn')  # Churn is our target variable

# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Feature scaling for numerical variables
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Model Training
print("\nTraining models...")

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Store results
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print basic metrics
    print(f"{name} Accuracy: {results[name]['accuracy']:.4f}")
    print(f"{name} ROC AUC: {results[name]['roc_auc']:.4f}")

# Hyperparameter tuning for the best model (Random Forest in this case)
print("\nPerforming hyperparameter tuning for Random Forest...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]

# Store best model results
results['Best Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_best),
    'roc_auc': roc_auc_score(y_test, y_prob_best),
    'classification_report': classification_report(y_test, y_pred_best),
    'confusion_matrix': confusion_matrix(y_test, y_pred_best),
    'best_params': grid_search.best_params_
}

# Print best model metrics
print(f"\nBest Random Forest Accuracy: {results['Best Random Forest']['accuracy']:.4f}")
print(f"Best Random Forest ROC AUC: {results['Best Random Forest']['roc_auc']:.4f}")
print("Best Parameters:", results['Best Random Forest']['best_params'])

# Feature Importance
print("\nFeature Importance from Best Random Forest Model:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10))

# Visualization
plt.figure(figsize=(15, 6))

# ROC Curve
plt.subplot(1, 2, 1)
for name in results:
    if name != 'Best Random Forest':  # We'll plot this separately
        fpr, tpr, _ = roc_curve(y_test, models[name].predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.2f})')

# Plot best model ROC
fpr, tpr, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'Best RF (AUC = {results["Best Random Forest"]["roc_auc"]:.2f})', linewidth=2, linestyle='--')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Feature Importance Plot
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()

# Print detailed classification reports
print("\nDetailed Classification Reports:")
for name in results:
    print(f"\n{name} Classification Report:")
    print(results[name]['classification_report'])
    print("Confusion Matrix:")
    print(results[name]['confusion_matrix'])

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
# Note: Replace 'telco_churn.csv' with your actual dataset path
try:
    df = pd.read_csv('telco_churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset file not found. Please check the path.")
    exit()

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Data Preprocessing
print("\nPreprocessing the data...")

# Drop customer ID as it's not useful for prediction
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert Churn to binary (1 for 'Yes', 0 for 'No')
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Churn')  # Churn is our target variable

# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Feature scaling for numerical variables
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Model Training
print("\nTraining models...")

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Store results
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print basic metrics
    print(f"{name} Accuracy: {results[name]['accuracy']:.4f}")
    print(f"{name} ROC AUC: {results[name]['roc_auc']:.4f}")

# Hyperparameter tuning for the best model (Random Forest in this case)
print("\nPerforming hyperparameter tuning for Random Forest...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]

# Store best model results
results['Best Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_best),
    'roc_auc': roc_auc_score(y_test, y_prob_best),
    'classification_report': classification_report(y_test, y_pred_best),
    'confusion_matrix': confusion_matrix(y_test, y_pred_best),
    'best_params': grid_search.best_params_
}

# Print best model metrics
print(f"\nBest Random Forest Accuracy: {results['Best Random Forest']['accuracy']:.4f}")
print(f"Best Random Forest ROC AUC: {results['Best Random Forest']['roc_auc']:.4f}")
print("Best Parameters:", results['Best Random Forest']['best_params'])

# Feature Importance
print("\nFeature Importance from Best Random Forest Model:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10))

# Visualization
plt.figure(figsize=(15, 6))

# ROC Curve
plt.subplot(1, 2, 1)
for name in results:
    if name != 'Best Random Forest':  # We'll plot this separately
        fpr, tpr, _ = roc_curve(y_test, models[name].predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.2f})')

# Plot best model ROC
fpr, tpr, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'Best RF (AUC = {results["Best Random Forest"]["roc_auc"]:.2f})', linewidth=2, linestyle='--')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Feature Importance Plot
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()

# Print detailed classification reports
print("WA_Fn-UseC_-Telco-Customer-Churn.csv")
for name in results:
    print(f"\n{name} Classification Report:")
    print(results[name]['classification_report'])
    print("Confusion Matrix:")
    print(results[name]['confusion_matrix'])
    
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
