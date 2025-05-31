import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Convert and clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Include gender in features
    features = ['gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
    df = df[features + ['Churn']].copy()
    
    # Encode categoricals
    df['gender'] = df['gender'].map({'Male':0, 'Female':1})
    df['Contract'] = df['Contract'].map({'Month-to-month':0, 'One year':1, 'Two years':2})
    df['InternetService'] = df['InternetService'].map({'DSL':0, 'Fiber optic':1, 'No':2})
    
    return df

@st.cache_data
def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Handle missing values
    if X.isna().any().any():
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Balance classes
    X_res, y_res = SMOTE().fit_resample(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, scaler

# UI Setup
st.title("📞 Customer Churn Prediction")
st.write("Predict if a customer will leave the service")

# Load data and model
try:
    df = load_data()
    model, scaler = train_model(df)
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Customer Details")
        gender = st.radio("Gender", ["Male", "Female"])
        tenure = st.slider("Tenure (months)", 1, 100, 30)
        monthly_charges = st.number_input("Monthly Charges ($)", 10.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2000.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two years"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Prepare input
        input_data = pd.DataFrame({
            'gender': [0 if gender == "Male" else 1],
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [0 if contract == "Month-to-month" else 1 if contract == "One year" else 2],
            'InternetService': [0 if internet == "DSL" else 1 if internet == "Fiber optic" else 2]
        })
        
        # Scale features
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_data[num_cols] = scaler.transform(input_data[num_cols])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display results with gender-specific message
        st.subheader("Prediction Result")
        gender_pronoun = "He" if gender == "Male" else "She"
        
        if prediction == 1:
            st.error(f"🚨 {gender_pronoun} has a HIGH risk of churning ({probability:.1%} probability)")
        else:
            st.success(f"✅ {gender_pronoun} has a LOW risk of churning ({probability:.1%} probability)")
            
        # Show gender insights
        st.write(f"Gender analysis: {'Male' if gender == 'Male' else 'Female'} customers tend to {'churn more' if prediction == 1 else 'stay loyal'}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check your data file and try again.")