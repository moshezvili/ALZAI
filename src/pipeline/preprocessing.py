"""
Data preprocessing utilities for handling missing values, scaling, and validation.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values with different strategies for numeric and categorical features."""
    
    def __init__(self, 
                 numeric_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent',
                 add_indicator: bool = True):
        """
        Initialize missing value handler.
        
        Args:
            numeric_strategy: Strategy for numeric features ('mean', 'median', 'constant')
            categorical_strategy: Strategy for categorical features ('most_frequent', 'constant')
            add_indicator: Whether to add binary indicator for missing values
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.add_indicator = add_indicator
        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.numeric_features_ = []
        self.categorical_features_ = []
        self.missing_indicators_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the missing value handler."""
        start_time = time.time()
        logger.info("Fitting missing value handler...")
        
        # Identify feature types
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit numeric imputer
        if self.numeric_features_:
            self.numeric_imputer_ = SimpleImputer(strategy=self.numeric_strategy)
            self.numeric_imputer_.fit(X[self.numeric_features_])
        
        # Fit categorical imputer
        if self.categorical_features_:
            self.categorical_imputer_ = SimpleImputer(strategy=self.categorical_strategy)
            # Convert None to np.nan for proper sklearn handling during fit
            X_cat_copy = X[self.categorical_features_].copy()
            for col in self.categorical_features_:
                X_cat_copy[col] = X_cat_copy[col].replace({None: np.nan})
            self.categorical_imputer_.fit(X_cat_copy)
        
        # Identify features with missing values for indicators
        if self.add_indicator:
            self.missing_indicators_ = [col for col in X.columns if X[col].isnull().any()]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Fitted missing value handler for {len(self.numeric_features_)} numeric "
                   f"and {len(self.categorical_features_)} categorical features in {elapsed_time:.2f} seconds")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by handling missing values."""
        start_time = time.time()
        logger.info("Transforming missing values...")
        
        X_transformed = X.copy()
        
        # Add missing indicators before imputation
        if self.add_indicator:
            for col in self.missing_indicators_:
                if col in X_transformed.columns:
                    X_transformed[f'{col}_was_missing'] = X_transformed[col].isnull().astype(int)
        
        # Handle numeric features
        if self.numeric_features_ and self.numeric_imputer_:
            available_numeric = [col for col in self.numeric_features_ if col in X_transformed.columns]
            if available_numeric:
                X_transformed[available_numeric] = self.numeric_imputer_.transform(X_transformed[available_numeric])
        
        # Handle categorical features
        if self.categorical_features_ and self.categorical_imputer_:
            available_categorical = [col for col in self.categorical_features_ if col in X_transformed.columns]
            if available_categorical:
                # Convert None to np.nan for proper sklearn handling
                for col in available_categorical:
                    X_transformed[col] = X_transformed[col].replace({None: np.nan})
                X_transformed[available_categorical] = self.categorical_imputer_.transform(X_transformed[available_categorical])
        
        elapsed_time = time.time() - start_time
        logger.info(f"Missing value transformation completed in {elapsed_time:.2f} seconds")
        return X_transformed

class DataValidator:
    """Validate data quality and consistency."""
    
    def __init__(self):
        self.validation_rules = {}
        
    def add_rule(self, feature: str, rule_type: str, **kwargs):
        """Add validation rule for a feature."""
        if feature not in self.validation_rules:
            self.validation_rules[feature] = []
        
        self.validation_rules[feature].append({
            'type': rule_type,
            'params': kwargs
        })
    
    def validate(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate dataframe against rules."""
        violations = {}
        
        for feature, rules in self.validation_rules.items():
            if feature not in df.columns:
                continue
                
            feature_violations = []
            
            for rule in rules:
                if rule['type'] == 'range':
                    min_val = rule['params'].get('min')
                    max_val = rule['params'].get('max')
                    
                    if min_val is not None:
                        violation_count = (df[feature] < min_val).sum()
                        if violation_count > 0:
                            feature_violations.append(f"{violation_count} values below minimum {min_val}")
                    
                    if max_val is not None:
                        violation_count = (df[feature] > max_val).sum()
                        if violation_count > 0:
                            feature_violations.append(f"{violation_count} values above maximum {max_val}")
                
                elif rule['type'] == 'categorical':
                    allowed_values = rule['params'].get('allowed_values', [])
                    invalid_mask = ~df[feature].isin(allowed_values)
                    violation_count = invalid_mask.sum()
                    
                    if violation_count > 0:
                        feature_violations.append(f"{violation_count} invalid categorical values")
                
                elif rule['type'] == 'missing_rate':
                    max_missing_rate = rule['params'].get('max_rate', 0.1)
                    missing_rate = df[feature].isnull().mean()
                    
                    if missing_rate > max_missing_rate:
                        feature_violations.append(f"Missing rate {missing_rate:.2%} exceeds {max_missing_rate:.2%}")
            
            if feature_violations:
                violations[feature] = feature_violations
        
        return violations
    
    def setup_clinical_rules(self):
        """Setup validation rules for clinical data."""
        # Age validation
        self.add_rule('age', 'range', min=0, max=120)
        
        # BMI validation
        self.add_rule('bmi', 'range', min=10, max=60)
        
        # Blood pressure validation
        self.add_rule('systolic_bp', 'range', min=70, max=250)
        self.add_rule('diastolic_bp', 'range', min=40, max=150)
        
        # Lab values
        self.add_rule('glucose', 'range', min=50, max=500)
        self.add_rule('cholesterol', 'range', min=100, max=500)
        
        # Categorical validations
        self.add_rule('gender', 'categorical', allowed_values=['M', 'F'])
        self.add_rule('smoking_status', 'categorical', 
                     allowed_values=['Never', 'Former', 'Current'])
        
        # Missing value thresholds
        for feature in ['age', 'gender', 'year', 'patient_id']:
            self.add_rule(feature, 'missing_rate', max_rate=0.01)

class DataScaler(BaseEstimator, TransformerMixin):
    """Scale numeric features while preserving categorical features."""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize scaler.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scaler_ = None
        self.numeric_features_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the scaler."""
        start_time = time.time()
        logger.info(f"Fitting data scaler with method: {self.method}")
        
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.method == 'minmax':
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        if self.numeric_features_:
            self.scaler_.fit(X[self.numeric_features_])
        
        elapsed_time = time.time() - start_time
        logger.info(f"Fitted scaler for {len(self.numeric_features_)} numeric features in {elapsed_time:.2f} seconds")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by scaling numeric features."""
        start_time = time.time()
        logger.info(f"Scaling numeric features using {self.method} scaling...")
        
        X_transformed = X.copy()
        
        if self.numeric_features_ and self.scaler_:
            available_numeric = [col for col in self.numeric_features_ if col in X_transformed.columns]
            if available_numeric:
                X_transformed[available_numeric] = self.scaler_.transform(X_transformed[available_numeric])
        
        elapsed_time = time.time() - start_time
        logger.info(f"Scaling completed in {elapsed_time:.2f} seconds")
        return X_transformed
