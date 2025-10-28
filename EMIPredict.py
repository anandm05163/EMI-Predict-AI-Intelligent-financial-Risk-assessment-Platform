import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import mlflow
import mlflow.sklearn
import streamlit as st
import os


import os

# Define the relative path to the CSV file
file_path = 'emi_prediction_dataset.csv'

# Load the dataset
data = pd.read_csv(file_path, low_memory=False)

# # Load the dataset with low_memory=False to handle mixed types
# file_path = 'C:\\Users\\Jantoin\\Documents\\EMIPredictAI\\emi_prediction_dataset.csv'
# data = pd.read_csv(file_path, low_memory=False)

# Data Cleaning
# 1. Handling missing values
data.ffill(inplace=True)

# 2. Removing duplicates
data.drop_duplicates(inplace=True)

# 3. Handling inconsistencies (example: standardizing text data)
# data['column_name'] = data['column_name'].str.lower()  # Update 'column_name' with actual column name

# Data Quality Assessment and Validation Checks
# Example: Check for any remaining missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Example: Check for data types
# print("Data types:\n", data.dtypes)

# Function to clean numeric columns
def clean_numeric_column(column):
    # Remove any non-numeric characters except the first decimal point
    cleaned_column = column.str.replace(r'[^0-9.]', '', regex=True)
    # Remove any extra decimal points
    cleaned_column = cleaned_column.str.replace(r'\.(?=.*\.)', '', regex=True)
    return pd.to_numeric(cleaned_column, errors='coerce')

# Convert relevant columns to numeric, coercing errors to NaN
numeric_columns = ['existing_loans', 'monthly_salary', 'credit_score', 'monthly_rent',
                   'other_monthly_expenses', 'school_fees', 'college_fees', 'travel_expenses',
                   'groceries_utilities', 'max_monthly_emi', 'years_of_employment', 'age']

for col in numeric_columns:
    data[col] = clean_numeric_column(data[col].astype(str))

# Check the cleaned data
print(data['age'].head())

# Convert all columns to numeric where possible
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Check for columns with non-numeric data
non_numeric_columns = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)


# Convert 'age' column to integer
data['age'] = data['age'].fillna(0).astype(int)

# Check the cleaned data
print(data['age'].head())


# Train-Test-Validation Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, validation_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2


# Output the sizes of each split
print("Training data size:", train_data.shape)
print("Validation data size:", validation_data.shape)
print("Test data size:", test_data.shape)


# Function to clean numeric columns
def clean_numeric_column(column):
    # Remove any non-numeric characters except the first decimal point
    cleaned_column = column.str.replace(r'[^0-9.]', '', regex=True)
    # Remove any extra decimal points
    cleaned_column = cleaned_column.str.replace(r'\.(?=.*\.)', '', regex=True)
    return pd.to_numeric(cleaned_column, errors='coerce')

# Convert relevant columns to numeric, coercing errors to NaN
numeric_columns = ['existing_loans', 'monthly_salary', 'credit_score', 'monthly_rent',
                   'other_monthly_expenses', 'school_fees', 'college_fees', 'travel_expenses',
                   'groceries_utilities', 'max_monthly_emi', 'years_of_employment', 'age']

for col in numeric_columns:
    data[col] = clean_numeric_column(data[col].astype(str))

# 1. Create derived financial ratios
data['debt_to_income'] = data['existing_loans'] / data['monthly_salary']
data['expense_to_income'] = (data['monthly_rent'] + data['school_fees'] + data['college_fees'] +
                             data['travel_expenses'] + data['groceries_utilities'] + data['other_monthly_expenses']) / data['monthly_salary']
data['affordability_ratio'] = data['max_monthly_emi'] / data['monthly_salary']

# 2. Generate risk scoring features
data['employment_stability'] = data['years_of_employment'] / data['age']
data['credit_risk_score'] = data['credit_score'] / 850  # Assuming 850 is the max credit score

# 3. Apply categorical encoding and numerical feature scaling
categorical_features = ['gender', 'marital_status', 'education', 'employment_type', 'company_type', 'house_type']
numerical_features = ['monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size', 'dependents',
                      'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
                      'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
                      'requested_amount', 'requested_tenure', 'debt_to_income', 'expense_to_income', 'affordability_ratio',
                      'employment_stability', 'credit_risk_score']

# Define transformers
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply transformations
data_transformed = pd.DataFrame(preprocessor.fit_transform(data))

# 4. Develop interaction features between numerical variables
data['income_credit_interaction'] = data['monthly_salary'] * data['credit_score']
data['loan_employment_interaction'] = data['existing_loans'] * data['years_of_employment']

# Display the first few rows of the transformed data
print(data.head())


# Assuming train_data, validation_data, and test_data are defined
X_train_reg = train_data.drop('max_monthly_emi', axis=1)
y_train_reg = train_data['max_monthly_emi']
X_val_reg = validation_data.drop('max_monthly_emi', axis=1)
y_val_reg = validation_data['max_monthly_emi']
X_test_reg = test_data.drop('max_monthly_emi', axis=1)  # Assuming test_data is defined

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_reg)
X_val_imputed = imputer.transform(X_val_reg)
X_test_imputed = imputer.transform(X_test_reg)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train XGBoost Regressor
xgb_regressor = XGBRegressor()
xgb_regressor.fit(X_train_scaled, y_train_reg)

# Predict using the validation set
y_pred_xgb = xgb_regressor.predict(X_val_scaled)

# Calculate metrics for XGBoost
mse_xgb = mean_squared_error(y_val_reg, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)  # Manually compute RMSE
mae_xgb = mean_absolute_error(y_val_reg, y_pred_xgb)
r2_xgb = r2_score(y_val_reg, y_pred_xgb)

print("XGBoost Regressor")
print(f"RMSE: {rmse_xgb}")
print(f"MAE: {mae_xgb}")
print(f"R-squared: {r2_xgb}\n")



# Assuming X_train_scaled, X_val_scaled, y_train_reg, y_val_reg are already defined

# Train Linear Regression Model
lr_regressor = LinearRegression()
lr_regressor.fit(X_train_scaled, y_train_reg)

# Predict using the validation set
y_pred_lr = lr_regressor.predict(X_val_scaled)

# Calculate metrics for Linear Regression
mse_lr = mean_squared_error(y_val_reg, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)  # Manually compute RMSE
mae_lr = mean_absolute_error(y_val_reg, y_pred_lr)
r2_lr = r2_score(y_val_reg, y_pred_lr)

print("Linear Regression Model")
print(f"RMSE: {rmse_lr}")
print(f"MAE: {mae_lr}")
print(f"R-squared: {r2_lr}\n")



# XGB Regression ML Flow Integration

# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Ensure MLflow server is running
mlflow.set_experiment("EMI Prediction Experiment")

# Log regression models and metrics
with mlflow.start_run(run_name="Regression Models"):
    for model in [xgb_regressor]:
        model_name = model.__class__.__name__
        mlflow.sklearn.log_model(model, model_name)
        mse = mean_squared_error(y_val_reg, model.predict(X_val_scaled))
        rmse = np.sqrt(mse)  # Manually compute RMSE
        mlflow.log_metric(f"{model_name}_rmse", rmse)
        mlflow.log_metric(f"{model_name}_mae", mean_absolute_error(y_val_reg, model.predict(X_val_scaled)))
        mlflow.log_metric(f"{model_name}_r_squared", r2_score(y_val_reg, model.predict(X_val_scaled)))



# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Ensure MLflow server is running
mlflow.set_experiment("EMI Prediction Experiment")

# Log regression models and metrics
with mlflow.start_run(run_name="Linear Regression Model"):
    model_name = lr_regressor.__class__.__name__
    mlflow.sklearn.log_model(lr_regressor, model_name)
    mse = mean_squared_error(y_val_reg, y_pred_lr)
    rmse = np.sqrt(mse)  # Manually compute RMSE
    mlflow.log_metric(f"{model_name}_rmse", rmse)
    mlflow.log_metric(f"{model_name}_mae", mae_lr)
    mlflow.log_metric(f"{model_name}_r_squared", r2_lr)





# Streamlit UI
st.title("EMI Prediction Model Selection")

# Dropdown for model selection
model_choice = st.selectbox("Select a regression model:", ["XGBoost Regressor", "Linear Regression"])

# Display results based on model choice
if model_choice == "XGBoost Regressor":
    st.write("XGBoost Regressor Results:")
    st.write(f"RMSE: {rmse_xgb}")
    st.write(f"MAE: {mae_xgb}")
    st.write(f"R-squared: {r2_xgb}")
elif model_choice == "Linear Regression":
    st.write("Linear Regression Results:")
    st.write(f"RMSE: {rmse_lr}")
    st.write(f"MAE: {mae_lr}")
    st.write(f"R-squared: {r2_lr}")