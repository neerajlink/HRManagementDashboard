import pandas as pd
import streamlit as st
import os

def load_employee_data():
    """Load employee data from CSV file"""
    try:
        # Get the current working directory
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, "attached_assets", "Employee_data_1752675697484.csv")
        
        # Check if file exists and load it
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.success(f"Successfully loaded {len(df)} employee records")
        else:
            # Try alternative paths
            alt_paths = [
                "./attached_assets/Employee_data_1752675697484.csv",
                "attached_assets/Employee_data_1752675697484.csv",
                "Employee_data_1752675697484.csv"
            ]
            
            df = None
            for path in alt_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    st.success(f"Successfully loaded {len(df)} employee records from {path}")
                    break
            
            if df is None:
                st.error(f"Employee data file not found. Searched locations:")
                st.error(f"- {csv_path}")
                for path in alt_paths:
                    st.error(f"- {path}")
                return pd.DataFrame()
        
        # Data cleaning and preprocessing
        df = clean_employee_data(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading employee data: {str(e)}")
        return pd.DataFrame()

def clean_employee_data(df):
    """Clean and preprocess employee data"""
    try:
        # Ensure proper column names
        df.columns = df.columns.str.strip()
        
        # Handle missing values
        df = df.fillna(method='ffill')
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['Employee_id', 'age', 'tenure', 'monthly_income', 
                         'performance_rating', 'job_satisfaction', 'Password']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate risk categories based on job satisfaction and performance
        df['risk_score'] = calculate_risk_score(df)
        df['risk_category'] = categorize_risk(df['risk_score'])
        
        return df
        
    except Exception as e:
        st.error(f"Error cleaning employee data: {str(e)}")
        return df

def calculate_risk_score(df):
    """Calculate employee risk score based on various factors"""
    try:
        # Normalize job satisfaction and performance rating (lower is riskier)
        job_sat_risk = (6 - df['job_satisfaction']) / 5  # Scale 0-1
        performance_risk = (6 - df['performance_rating']) / 5  # Scale 0-1
        
        # Tenure risk (very low or very high tenure might indicate risk)
        tenure_risk = np.where(df['tenure'] < 2, 0.8, 
                              np.where(df['tenure'] > 15, 0.6, 0.2))
        
        # Age risk (younger employees might be more likely to leave)
        age_risk = np.where(df['age'] < 30, 0.6, 
                           np.where(df['age'] > 55, 0.4, 0.2))
        
        # Calculate weighted risk score
        risk_score = (job_sat_risk * 0.4 + 
                     performance_risk * 0.3 + 
                     tenure_risk * 0.2 + 
                     age_risk * 0.1)
        
        return risk_score
        
    except Exception as e:
        st.error(f"Error calculating risk score: {str(e)}")
        return pd.Series([0.5] * len(df))

def categorize_risk(risk_scores):
    """Categorize employees into risk categories"""
    try:
        return pd.cut(risk_scores, 
                     bins=[0, 0.3, 0.5, 0.7, 1.0], 
                     labels=['Low', 'Medium', 'High', 'Critical'],
                     include_lowest=True)
    except Exception as e:
        st.error(f"Error categorizing risk: {str(e)}")
        return pd.Series(['Medium'] * len(risk_scores))

# Import numpy for risk calculation
import numpy as np
