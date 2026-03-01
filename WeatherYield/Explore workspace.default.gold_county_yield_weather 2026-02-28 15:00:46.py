# Databricks notebook source
# DBTITLE 1,Cell 1
# Load the dataset as a Spark DataFrame

df = spark.read.table("workspace.default.gold_county_yield_weather")

# Alternative PCA using pandas + scikit-learn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Select relevant numeric columns from the Spark DataFrame
# FIX: Add 'commodity' column to enable downstream filtering

df_features = df.select([
    "prcp_total_mm",
    "tmax_avg_c",
    "tmin_avg_c",
    "heat_days_30c",
    "yield",
    "stations_used",
    "commodity"
])

# Convert to pandas DataFrame
pdf = df_features.toPandas()

# Drop rows with missing values (fix for sklearn PCA)
pdf = pdf.dropna()

# Standardize features
scaler = StandardScaler()
scaled = scaler.fit_transform(pdf[["prcp_total_mm", "tmax_avg_c", "tmin_avg_c", "heat_days_30c", "yield", "stations_used"]])

# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
display(pca_df.head())

# COMMAND ----------

# DBTITLE 1,PCA Scatter Plot
# Visualize the principal components
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Scatter Plot: County Yield & Weather")
plt.grid(True)
plt.show()

# COMMAND ----------

# DBTITLE 1,Color PCA Plot by State
# Color PCA plot: filter for specific Midwest states with more distinct colors

midwest = ["Iowa", "Illinois", "Nebraska", "Minnesota", "Indiana"]

# Re-extract state_name along with features used for PCA
df_state_features = df.select([
    "state_name",
    "prcp_total_mm",
    "tmax_avg_c",
    "tmin_avg_c",
    "heat_days_30c",
    "yield",
    "stations_used"
])

# Convert to pandas DataFrame and drop rows with missing values
pdf_state = df_state_features.toPandas()
pdf_state = pdf_state.dropna()

pdf_state = pdf_state.reset_index(drop=True)
pca_df = pca_df.reset_index(drop=True)

# Merge state information with PCA components
pca_state_df = pd.concat([pdf_state["state_name"], pca_df], axis=1)

# Filter for selected Midwest states
filtered_df = pca_state_df[pca_state_df["state_name"].isin(midwest)]

# Plot with more distinct colors (Set1 palette, max 9 and very colorblind-friendly)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x="PC1", y="PC2", hue="state_name", palette="Set1", alpha=0.7)
plt.title("PCA Scatter Plot: Iowa, Illinois, Nebraska, Minnesota, Indiana (Distinct Colors)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="State", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.show()

# COMMAND ----------

# DBTITLE 1,Feature Preprocessing (pandas/sklearn, weather only)
# Prepare features and target for scikit-learn modeling (weather only)
feature_cols = ["prcp_total_mm", "tmax_avg_c", "tmin_avg_c", "heat_days_30c"]
target_col = "yield"

# Use existing pandas DataFrame from cell 1: pdf

# Drop rows with missing feature or target values (should already be done)
pdf_clean = pdf.dropna(subset=feature_cols + [target_col])

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(pdf_clean[feature_cols])
y = pdf_clean[target_col].values

# COMMAND ----------

# DBTITLE 1,Train/Test Split (pandas/sklearn)
# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# DBTITLE 1,Fit Linear Regression (pandas/sklearn)
# Fit a linear regression model with scikit-learn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test set
y_pred = lr.predict(X_test)

# COMMAND ----------

# DBTITLE 1,Evaluate Model (pandas/sklearn)
# Evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"SKLearn RMSE: {rmse:.2f}")
print(f"SKLearn R^2: {r2:.3f}")

# COMMAND ----------

# DBTITLE 1,Feature Importance (pandas/sklearn)
# Feature importance for linear regression model
import numpy as np
feature_importance = np.abs(lr.coef_)

for name, coef in zip(feature_cols, feature_importance):
    print(f"{name}: {coef:.3f}")

# COMMAND ----------

# DBTITLE 1,Compare Regression Models for Corn and Soybeans (weather only)
# Run yield regression analysis for Corn and Soybeans separately using weather features only
commodity_types = ['Corn', 'Soybeans']
feature_cols = ["prcp_total_mm", "tmax_avg_c", "tmin_avg_c", "heat_days_30c"]
target_col = "yield"

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

results = {}
for crop in commodity_types:
    pdf_crop = pdf[pdf['commodity'] == crop].dropna(subset=feature_cols + [target_col])
    X = StandardScaler().fit_transform(pdf_crop[feature_cols])
    y = pdf_crop[target_col].values
    if len(pdf_crop) < 50:
        results[crop] = {'rmse': None, 'r2': None, 'features': 'Insufficient data'}
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    features = {name: abs(coef) for name, coef in zip(feature_cols, lr.coef_)}
    results[crop] = {'rmse': rmse, 'r2': r2, 'features': features}

# Print results
for crop in commodity_types:
    print(f"=== {crop} ===")
    if results[crop]['rmse'] is None:
        print('Insufficient data')
        continue
    print(f"RMSE: {results[crop]['rmse']:.2f}")
    print(f"R^2: {results[crop]['r2']:.3f}")
    print("Feature Importances:")
    for name, coef in results[crop]['features'].items():
        print(f"  {name}: {coef:.3f}")
    print()

# COMMAND ----------

# DBTITLE 1,Random Forest Regression for Corn and Soybeans (weather only)
# Run random forest regression analysis for Corn and Soybeans separately using weather features only
from sklearn.ensemble import RandomForestRegressor

commodity_types = ['Corn', 'Soybeans']
feature_cols = ["prcp_total_mm", "tmax_avg_c", "tmin_avg_c", "heat_days_30c"]
target_col = "yield"

results_rf = {}
for crop in commodity_types:
    pdf_crop = pdf[pdf['commodity'] == crop].dropna(subset=feature_cols + [target_col])
    X = StandardScaler().fit_transform(pdf_crop[feature_cols])
    y = pdf_crop[target_col].values
    if len(pdf_crop) < 50:
        results_rf[crop] = {'rmse': None, 'r2': None, 'features': 'Insufficient data'}
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    features = {name: importance for name, importance in zip(feature_cols, rf.feature_importances_)}
    results_rf[crop] = {'rmse': rmse, 'r2': r2, 'features': features}

# Print results
for crop in commodity_types:
    print(f"=== {crop} (Random Forest) ===")
    if results_rf[crop]['rmse'] is None:
        print('Insufficient data')
        continue
    print(f"RMSE: {results_rf[crop]['rmse']:.2f}")
    print(f"R^2: {results_rf[crop]['r2']:.3f}")
    print("Feature Importances:")
    for name, importance in results_rf[crop]['features'].items():
        print(f"  {name}: {importance:.3f}")
    print()

# COMMAND ----------

# DBTITLE 1,Feature Importances Visualization (Linear & RF)
import matplotlib.pyplot as plt
import numpy as np

# Linear regression results (from previous cell output)
lr_importance = {
    'Corn': [6.487, 18.605, 39.346, 34.055],
    'Soybeans': [1.862, 2.446, 5.993, 7.138]
}
# Random forest results (from previous cell output)
rf_importance = {
    'Corn': [0.309, 0.065, 0.178, 0.449],
    'Soybeans': [0.225, 0.184, 0.135, 0.457]
}
features = ["prcp_total_mm", "tmax_avg_c", "tmin_avg_c", "heat_days_30c"]

x = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
# Linear regression
ax[0].bar(x - width/2, lr_importance['Corn'], width, label='Corn')
ax[0].bar(x + width/2, lr_importance['Soybeans'], width, label='Soybeans')
ax[0].set_xticks(x)
ax[0].set_xticklabels(features)
ax[0].set_title('Linear Regression Feature Importances')
ax[0].set_ylabel('Absolute Coefficient')
ax[0].legend()
# Random Forest
ax[1].bar(x - width/2, rf_importance['Corn'], width, label='Corn')
ax[1].bar(x + width/2, rf_importance['Soybeans'], width, label='Soybeans')
ax[1].set_xticks(x)
ax[1].set_xticklabels(features)
ax[1].set_title('Random Forest Feature Importances')
ax[1].set_ylabel('Feature Importance')
ax[1].legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Model Performance Visualization
# R^2 and RMSE results from prior outputs
r2_linear = {'Corn': 0.307, 'Soybeans': 0.185}
r2_rf = {'Corn': 0.408, 'Soybeans': 0.276}
rmse_linear = {'Corn': 40.17, 'Soybeans': 11.60}
rmse_rf = {'Corn': 37.12, 'Soybeans': 10.93}

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# R^2 comparison
ax[0].bar(['Corn (Linear)', 'Corn (RF)', 'Soybeans (Linear)', 'Soybeans (RF)'],
          [r2_linear['Corn'], r2_rf['Corn'], r2_linear['Soybeans'], r2_rf['Soybeans']],
          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax[0].set_ylabel('R^2')
ax[0].set_ylim(0, 0.5)
ax[0].set_title('Model R^2 Comparison')

# RMSE comparison
ax[1].bar(['Corn (Linear)', 'Corn (RF)', 'Soybeans (Linear)', 'Soybeans (RF)'],
          [rmse_linear['Corn'], rmse_rf['Corn'], rmse_linear['Soybeans'], rmse_rf['Soybeans']],
          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax[1].set_ylabel('RMSE')
ax[1].set_title('Model RMSE Comparison')
plt.tight_layout()
plt.show()
