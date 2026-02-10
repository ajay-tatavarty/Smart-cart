"""Feature engineering utilities."""

import pandas as pd
from datetime import datetime


def calculate_age(df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    """Calculate customer age from birth year.
    
    Args:
        df: Input DataFrame with 'Year_Birth' column
        current_year: Current year for age calculation
        
    Returns:
        DataFrame with 'Age' column added
    """
    df_eng = df.copy()
    df_eng["Age"] = current_year - df_eng["Year_Birth"]
    return df_eng


def calculate_customer_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate customer tenure in days.
    
    Args:
        df: Input DataFrame with 'Dt_Customer' column
        
    Returns:
        DataFrame with 'Customer_Tenure_Days' column added
    """
    df_eng = df.copy()
    df_eng["Dt_Customer"] = pd.to_datetime(df_eng["Dt_Customer"], dayfirst=True)
    
    reference_date = df_eng["Dt_Customer"].max()
    df_eng["Customer_Tenure_Days"] = (reference_date - df_eng["Dt_Customer"]).dt.days
    
    return df_eng


def calculate_total_spending(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total spending across all product categories.
    
    Args:
        df: Input DataFrame with spending columns
        
    Returns:
        DataFrame with 'Total_Spending' column added
    """
    df_eng = df.copy()
    
    spending_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    
    available_cols = [col for col in spending_cols if col in df_eng.columns]
    
    if available_cols:
        df_eng["Total_Spending"] = df_eng[available_cols].sum(axis=1)
    
    return df_eng


def calculate_total_children(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total number of children.
    
    Args:
        df: Input DataFrame with 'Kidhome' and 'Teenhome' columns
        
    Returns:
        DataFrame with 'Total_Children' column added
    """
    df_eng = df.copy()
    
    if "Kidhome" in df_eng.columns and "Teenhome" in df_eng.columns:
        df_eng["Total_Children"] = df_eng["Kidhome"] + df_eng["Teenhome"]
    
    return df_eng


def standardize_education(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize education levels.
    
    Args:
        df: Input DataFrame with 'Education' column
        
    Returns:
        DataFrame with standardized education categories
    """
    df_eng = df.copy()
    
    if "Education" in df_eng.columns:
        education_mapping = {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate"
        }
        df_eng["Education"] = df_eng["Education"].replace(education_mapping)
    
    return df_eng


def standardize_marital_status(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize marital status to living situation.
    
    Args:
        df: Input DataFrame with 'Marital_Status' column
        
    Returns:
        DataFrame with 'Living_With' column added
    """
    df_eng = df.copy()
    
    if "Marital_Status" in df_eng.columns:
        marital_mapping = {
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Alone",
            "Divorced": "Alone",
            "Widow": "Alone",
            "Absurd": "Alone",
            "YOLO": "Alone"
        }
        df_eng["Living_With"] = df_eng["Marital_Status"].replace(marital_mapping)
    
    return df_eng


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with all engineered features
    """
    df_eng = df.copy()
    df_eng = calculate_age(df_eng)
    df_eng = calculate_customer_tenure(df_eng)
    df_eng = calculate_total_spending(df_eng)
    df_eng = calculate_total_children(df_eng)
    df_eng = standardize_education(df_eng)
    df_eng = standardize_marital_status(df_eng)
    
    return df_eng
