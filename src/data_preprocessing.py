"""Data preprocessing and cleaning utilities."""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load customer data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with raw customer data
    """
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    # Fill Income missing values with median
    if "Income" in df_clean.columns:
        df_clean["Income"] = df_clean["Income"].fillna(df_clean["Income"].median())
    
    return df_clean


def remove_outliers(df: pd.DataFrame, age_limit: int = 90, income_limit: int = 600000) -> pd.DataFrame:
    """Remove outliers from the dataset.
    
    Args:
        df: Input DataFrame
        age_limit: Maximum age threshold
        income_limit: Maximum income threshold
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if "Age" in df_clean.columns:
        df_clean = df_clean[df_clean["Age"] < age_limit]
    
    if "Income" in df_clean.columns:
        df_clean = df_clean[df_clean["Income"] < income_limit]
    
    return df_clean


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop specified columns from DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of column names to drop
        
    Returns:
        DataFrame with specified columns removed
    """
    df_clean = df.copy()
    cols_to_drop = [col for col in columns if col in df_clean.columns]
    return df_clean.drop(columns=cols_to_drop)


def get_data_info(df: pd.DataFrame) -> dict:
    """Get basic information about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with dataset information
    """
    return {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.to_dict(),
        "columns": df.columns.tolist()
    }
