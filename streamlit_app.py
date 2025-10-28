import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

st.set_page_config(page_title="EMI Predict - Interactive", layout="wide")

# Helper: clean numeric values similar to notebook
def clean_numeric_column(column: pd.Series) -> pd.Series:
    cleaned_column = column.astype(str).str.replace(r"[^0-9.]", "", regex=True)
    cleaned_column = cleaned_column.str.replace(r"\.(?=.*\.)", "", regex=True)
    return pd.to_numeric(cleaned_column, errors="coerce")

# Derived features function
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # safe division
    def safe_div(a, b):
        return a.div(b).replace([np.inf, -np.inf], np.nan)

    df['debt_to_income'] = safe_div(df.get('existing_loans', pd.Series(np.nan)), df.get('monthly_salary', pd.Series(np.nan)))
    expense_sum = (
        df.get('monthly_rent', 0).fillna(0)
        + df.get('school_fees', 0).fillna(0)
        + df.get('college_fees', 0).fillna(0)
        + df.get('travel_expenses', 0).fillna(0)
        + df.get('groceries_utilities', 0).fillna(0)
        + df.get('other_monthly_expenses', 0).fillna(0)
    )
    df['expense_to_income'] = safe_div(expense_sum, df.get('monthly_salary', pd.Series(np.nan)))
    df['affordability_ratio'] = safe_div(df.get('max_monthly_emi', pd.Series(np.nan)), df.get('monthly_salary', pd.Series(np.nan)))
    df['employment_stability'] = safe_div(df.get('years_of_employment', pd.Series(np.nan)), df.get('age', pd.Series(1)))
    df['credit_risk_score'] = safe_div(df.get('credit_score', pd.Series(np.nan)), 850)
    df['income_credit_interaction'] = df.get('monthly_salary', 0).fillna(0) * df.get('credit_score', 0).fillna(0)
    df['loan_employment_interaction'] = df.get('existing_loans', 0).fillna(0) * df.get('years_of_employment', 0).fillna(0)
    return df

# Default feature lists (same as converted script)
CATEGORICAL_FEATURES = ['gender', 'marital_status', 'education', 'employment_type', 'company_type', 'house_type']
NUMERICAL_FEATURES = [
    'monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size', 'dependents',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure', 'debt_to_income', 'expense_to_income', 'affordability_ratio',
    'employment_stability', 'credit_risk_score'
]

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# OneHotEncoder compatibility helper (scikit-learn changed `sparse` -> `sparse_output`)
def make_onehot_encoder(**kwargs):
    try:
        return OneHotEncoder(**{**kwargs, 'sparse_output': False})
    except TypeError:
        return OneHotEncoder(**{**kwargs, 'sparse': False})

st.title("EMI Predict — Interactive")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("1) Load data")
    uploaded = st.file_uploader("Upload a CSV file with your EMI dataset", type=['csv'])
    use_workspace = st.checkbox("Or use workspace dataset (emi_prediction_dataset.csv)")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, low_memory=False)
            st.success("Loaded uploaded CSV")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            df = None
    elif use_workspace:
        path = os.path.join(os.getcwd(), 'emi_prediction_dataset.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            st.success(f"Loaded workspace file: {path}")
        else:
            st.error(f"Workspace dataset not found at {path}")
            df = None
    else:
        df = None

    if df is not None:
        st.write("Preview:")
        st.dataframe(df.head())

with col2:
    st.header("2) Model options")
    model_choice = st.multiselect("Which models to train", ['LinearRegression', 'RandomForestRegressor'], default=['LinearRegression', 'RandomForestRegressor'])
    random_state = st.number_input("Random seed", value=42, step=1)
    n_estimators = st.number_input("RF n_estimators", value=100, step=10)
    save_models = st.checkbox("Save trained models to ./models", value=True)

st.markdown("---")

st.header("3) Preprocessing & Training")

if df is not None:
    st.write("Cleaning and creating features...")
    # Basic cleaning: forward fill and drop duplicates
    df_clean = df.copy()
    df_clean.ffill(inplace=True)
    df_clean.drop_duplicates(inplace=True)

    # Ensure numeric columns exist and convert
    numeric_candidates = ['existing_loans', 'monthly_salary', 'credit_score', 'monthly_rent',
                          'other_monthly_expenses', 'school_fees', 'college_fees', 'travel_expenses',
                          'groceries_utilities', 'max_monthly_emi', 'years_of_employment', 'age']
    for col in numeric_candidates:
        if col in df_clean.columns:
            df_clean[col] = clean_numeric_column(df_clean[col])

    # Convert age to int if present
    if 'age' in df_clean.columns:
        df_clean['age'] = df_clean['age'].fillna(0).astype(int)

    # Add derived features
    df_features = add_derived_features(df_clean)

    # Target check
    if 'max_monthly_emi' not in df_features.columns:
        st.error("Target column 'max_monthly_emi' not found in the dataset. Cannot train.")
    else:
        # Build feature lists intersecting actual columns
        numerical = [c for c in NUMERICAL_FEATURES if c in df_features.columns]
        categorical = [c for c in CATEGORICAL_FEATURES if c in df_features.columns]

        st.write(f"Using numerical features ({len(numerical)}): {numerical}")
        st.write(f"Using categorical features ({len(categorical)}): {categorical}")

        X = df_features[numerical + categorical].copy()
        y = df_features['max_monthly_emi']

        test_size = st.slider("Test size (portion)", 0.05, 0.5, 0.2)
        val_size = st.slider("Validation size (portion of train)", 0.05, 0.5, 0.25)

        if st.button("Train models"):
            train_df, test_df = train_test_split(df_features, test_size=test_size, random_state=random_state)
            train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)

            X_train = train_df[numerical + categorical]
            y_train = train_df['max_monthly_emi']
            X_val = val_df[numerical + categorical]
            y_val = val_df['max_monthly_emi']

            # Preprocessor
            categorical_transformer = make_onehot_encoder(handle_unknown='ignore')
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical),
                    ('cat', categorical_transformer, categorical)
                ],
                remainder='drop'
            )

            # Fit preprocessor
            # Ensure numerical columns are numeric before imputation
            for col in numerical:
                if col in X_train.columns:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                if col in X_val.columns:
                    X_val[col] = pd.to_numeric(X_val[col], errors='coerce')

            X_train_p = preprocessor.fit_transform(X_train)
            X_val_p = preprocessor.transform(X_val)

            results = {}

            if 'LinearRegression' in model_choice:
                lr = LinearRegression()
                lr.fit(X_train_p, y_train)
                y_pred_lr = lr.predict(X_val_p)
                results['LinearRegression'] = {
                    'rmse': np.sqrt(mean_squared_error(y_val, y_pred_lr)),
                    'mae': mean_absolute_error(y_val, y_pred_lr),
                    'r2': r2_score(y_val, y_pred_lr),
                    'model': lr
                }

            if 'RandomForestRegressor' in model_choice:
                rf = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state))
                rf.fit(X_train_p, y_train)
                y_pred_rf = rf.predict(X_val_p)
                results['RandomForestRegressor'] = {
                    'rmse': np.sqrt(mean_squared_error(y_val, y_pred_rf)),
                    'mae': mean_absolute_error(y_val, y_pred_rf),
                    'r2': r2_score(y_val, y_pred_rf),
                    'model': rf
                }

            # Show metrics
            st.subheader("Validation metrics")
            for name, r in results.items():
                st.write(f"**{name}** — RMSE: {r['rmse']:.4f}, MAE: {r['mae']:.4f}, R2: {r['r2']:.4f}")

            # Save preprocessor and models if requested
            if save_models:
                prep_path = os.path.join(MODEL_DIR, 'preprocessor.joblib')
                joblib.dump(preprocessor, prep_path)
                st.write(f"Saved preprocessor to {prep_path}")
                for name, r in results.items():
                    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
                    joblib.dump(r['model'], model_path)
                    st.write(f"Saved {name} to {model_path}")

            # Make prediction UI for a single record
            st.subheader("Single-record prediction")
            st.write("Enter feature values below and click Predict:")

            with st.form(key='predict_form'):
                input_vals = {}
                for feat in numerical:
                    input_vals[feat] = st.number_input(feat, value=float(df_features[feat].median()) if feat in df_features.columns else 0.0)
                for feat in categorical:
                    # choose unique values if available
                    opts = df_features[feat].dropna().unique().tolist()
                    default = opts[0] if len(opts) > 0 else ''
                    input_vals[feat] = st.selectbox(feat, options=opts, index=0 if len(opts) > 0 else -1)
                submit = st.form_submit_button("Predict")

            if submit:
                rec = pd.DataFrame([input_vals])
                rec = add_derived_features(rec)
                # Ensure columns order
                rec_X = rec.reindex(columns=numerical + categorical, fill_value=0)
                rec_p = preprocessor.transform(rec_X)
                preds = {}
                for name, r in results.items():
                    preds[name] = float(r['model'].predict(rec_p)[0])
                st.write("Predictions:")
                st.json(preds)

else:
    st.info("Upload a CSV or select the workspace dataset to get started.")

st.markdown("---")

st.header("Utilities")
if st.button("List saved models"):
    files = os.listdir(MODEL_DIR)
    st.write(files)

if st.button("Load saved models (if present)"):
    prep_path = os.path.join(MODEL_DIR, 'preprocessor.joblib')
    if os.path.exists(prep_path):
        try:
            preprocessor = joblib.load(prep_path)
            st.success("Loaded preprocessor")
        except Exception as e:
            st.error(f"Failed to load preprocessor: {e}")
    else:
        st.warning("No saved preprocessor found.")

st.caption("Streamlit app: upload dataset, train models, save models, and run single-record predictions.")
