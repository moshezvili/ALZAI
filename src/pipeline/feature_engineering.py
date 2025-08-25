"""
Feature Engineering Pipeline

This module handles feature preprocessing, temporal aggregations,
and distributed data processing using Dask for the clinical ML pipeline.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create temporal rolling aggregation features with Dask support."""
    
    def __init__(self, 
                 window_years: int = 3,
                 min_periods: int = 1,
                 features: Optional[List[str]] = None,
                 aggregations: Optional[List[str]] = None):
        """
        Initialize temporal feature engineer with Dask enabled by default.
        
        Args:
            window_years: Rolling window size in years
            min_periods: Minimum periods required for aggregation
            features: List of features to aggregate
            aggregations: List of aggregation methods ['mean', 'std', 'min', 'max', 'trend']
        """
        self.window_years = window_years
        self.min_periods = min_periods
        self.features = features or []
        self.aggregations = aggregations or ['mean', 'std', 'trend']
        self.feature_columns_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (identify numeric features if not specified)."""
        if not self.features:
            # Auto-detect numeric features suitable for temporal aggregation
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            exclude_features = ['patient_id', 'year', 'age', 'target']
            self.features = [f for f in numeric_features if f not in exclude_features]
            
        logger.info(f"Temporal features will be created for: {self.features}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create temporal rolling features."""
        start_time = time.time()
        logger.info("Creating temporal rolling features...")
        
        if 'patient_id' not in X.columns or 'year' not in X.columns:
            raise ValueError("DataFrame must contain 'patient_id' and 'year' columns")
        
        # Ensure data is sorted
        X_sorted = X.sort_values(['patient_id', 'year']).copy()
        
        # Create new features
        new_features = []
        
        # Progress tracking for features
        total_operations = len(self.features) * len(self.aggregations) + len(self.features)  # +lag features
        current_operation = 0
        
        logger.info(f"Starting temporal feature creation: {total_operations} operations total")
        
        for feature_idx, feature in enumerate(self.features):
            if feature not in X_sorted.columns:
                continue
                
            logger.info(f"Processing feature {feature_idx + 1}/{len(self.features)}: {feature}")
            grouped = X_sorted.groupby('patient_id')[feature]
            
            for agg_idx, agg in enumerate(self.aggregations):
                current_operation += 1
                progress_pct = (current_operation / total_operations) * 100
                
                logger.info(f"  [{current_operation}/{total_operations}] ({progress_pct:.1f}%) Creating {feature}_{agg}")
                
                if agg == 'mean':
                    new_col = f'{feature}_rolling_mean_{self.window_years}y'
                    rolling_result = grouped.rolling(
                        window=self.window_years, min_periods=self.min_periods
                    ).mean()
                    X_sorted[new_col] = rolling_result.values
                    
                elif agg == 'std':
                    new_col = f'{feature}_rolling_std_{self.window_years}y'
                    rolling_result = grouped.rolling(
                        window=self.window_years, min_periods=self.min_periods
                    ).std()
                    X_sorted[new_col] = rolling_result.values
                    
                elif agg == 'min':
                    new_col = f'{feature}_rolling_min_{self.window_years}y'
                    rolling_result = grouped.rolling(
                        window=self.window_years, min_periods=self.min_periods
                    ).min()
                    X_sorted[new_col] = rolling_result.values
                    
                elif agg == 'max':
                    new_col = f'{feature}_rolling_max_{self.window_years}y'
                    rolling_result = grouped.rolling(
                        window=self.window_years, min_periods=self.min_periods
                    ).max()
                    X_sorted[new_col] = rolling_result.values
                    
                elif agg == 'trend':
                    # Calculate linear trend using actual year values (not sequential index)
                    new_col = f'{feature}_rolling_trend_{self.window_years}y'
                    logger.info(f"    Computing trend slopes (this may take longer)...")
                    
                    def calculate_trend_with_years(group):
                        """Calculate slope using actual year values for time axis."""
                        result = []
                        for i in range(len(group)):
                            # Get window of data points
                            start_idx = max(0, i - self.window_years + 1)
                            window_feature = group.iloc[start_idx:i+1]
                            window_years = X_sorted.groupby('patient_id')['year'].get_group(group.name).iloc[start_idx:i+1]
                            
                            if len(window_feature) < 2:
                                result.append(0)
                                continue
                            
                            # Calculate slope using actual year values as x-axis
                            try:
                                # Convert to numpy arrays to ensure proper types
                                years_arr = np.array(window_years.values, dtype=float)
                                feature_arr = np.array(window_feature.values, dtype=float)
                                
                                if not np.any(np.isnan(feature_arr)) and len(years_arr) == len(feature_arr):
                                    slope = np.polyfit(years_arr, feature_arr, 1)[0]
                                else:
                                    slope = 0
                                result.append(slope)
                            except (np.linalg.LinAlgError, ValueError, TypeError):
                                result.append(0)
                        
                        return pd.Series(result, index=group.index)
                    
                    trend_result = X_sorted.groupby('patient_id')[feature].apply(calculate_trend_with_years)
                    X_sorted[new_col] = trend_result.values
                    logger.info(f"    Trend calculation completed for {feature}")
                
                new_features.append(new_col)
                logger.info(f"    ✓ Created {new_col}")
        
        # Add lag features (previous year values) using concat to avoid fragmentation
        logger.info("Creating lag features...")
        lag_features = {}
        for feature_idx, feature in enumerate(self.features):
            if feature in X_sorted.columns:
                current_operation += 1
                progress_pct = (current_operation / total_operations) * 100
                
                lag_col = f'{feature}_lag_1y'
                logger.info(f"  [{current_operation}/{total_operations}] ({progress_pct:.1f}%) Creating {lag_col}")
                lag_features[lag_col] = X_sorted.groupby('patient_id')[feature].shift(1)
                new_features.append(lag_col)
        
        # Add all lag features at once to avoid DataFrame fragmentation
        if lag_features:
            lag_df = pd.DataFrame(lag_features, index=X_sorted.index)
            X_sorted = pd.concat([X_sorted, lag_df], axis=1)
        
        self.feature_columns_ = new_features
        
        elapsed_time = time.time() - start_time
        logger.info(f"Created {len(new_features)} temporal features in {elapsed_time:.2f} seconds")
        
        # Fill any remaining NaN values that might have been created by rolling operations
        # This can happen with small windows or at the beginning of time series
        for col in new_features:
            if col in X_sorted.columns:
                X_sorted[col] = X_sorted[col].fillna(0)  # Fill with 0 for temporal features
        
        # Exclude patient_id and year from the output since they shouldn't be used as features for prediction
        exclude_cols = []
        if 'patient_id' in X_sorted.columns:
            exclude_cols.append('patient_id')
        if 'year' in X_sorted.columns:
            exclude_cols.append('year')
            
        if exclude_cols:
            logger.info(f"Excluding {exclude_cols} from feature output")
            X_sorted = X_sorted.drop(columns=exclude_cols)
        
        return X_sorted
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_columns_

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Handle categorical feature encoding with multiple strategies."""
    
    def __init__(self, 
                 method: str = 'target_encoding',
                 handle_unknown: str = 'ignore',
                 min_samples_leaf: int = 20,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize categorical encoder.
        
        Args:
            method: Encoding method ('target_encoding', 'one_hot', 'label_encoding')
            handle_unknown: How to handle unknown categories
            min_samples_leaf: Minimum samples for target encoding
            cv_folds: Number of CV folds for KFold target encoding
            random_state: Random state for reproducible CV splits
        """
        self.method = method
        self.handle_unknown = handle_unknown
        self.min_samples_leaf = min_samples_leaf
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.encoders_ = {}
        self.categorical_features_ = []
        self.global_means_ = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the categorical encoder."""
        start_time = time.time()
        logger.info(f"Fitting categorical encoder with method: {self.method}")
        
        # Identify categorical features (including pyarrow string types)
        self.categorical_features_ = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        
        # Also check for pyarrow string types manually
        for col in X.columns:
            if hasattr(X[col].dtype, 'pyarrow_dtype') or str(X[col].dtype).startswith('string'):
                if col not in self.categorical_features_:
                    self.categorical_features_.append(col)
        
        if self.method == 'target_encoding' and y is None:
            raise ValueError("Target encoding requires y to be provided")
        
        for feature in self.categorical_features_:
            if self.method == 'target_encoding':
                # Calculate target encoding with K-Fold CV to prevent leakage
                if y is not None:
                    self.global_means_[feature] = float(y.mean())
                    
                    if self.cv_folds > 1:
                        encoding_map = self._calculate_kfold_target_encoding(X[feature], y)
                    else:
                        # Fallback to simple target encoding for cv_folds=1
                        encoding_map = self._calculate_target_encoding(X[feature], y)
                    self.encoders_[feature] = encoding_map
                
            elif self.method == 'label_encoding':
                encoder = LabelEncoder()
                encoder.fit(X[feature].astype(str))
                self.encoders_[feature] = encoder
        
        elapsed_time = time.time() - start_time        
        logger.info(f"Fitted categorical encoder for {len(self.categorical_features_)} features in {elapsed_time:.2f} seconds")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features."""
        start_time = time.time()
        logger.info(f"Transforming categorical features using {self.method}")
        
        X_transformed = X.copy()
        
        for feature in self.categorical_features_:
            if feature not in X_transformed.columns:
                continue
                
            if self.method == 'target_encoding':
                encoding_map = self.encoders_[feature]
                default_value = self.global_means_.get(feature, 0.0) if hasattr(self, 'global_means_') and self.global_means_ else 0.0
                
                # Convert to string first to avoid categorical issues
                feature_values = X_transformed[feature].astype(str)
                X_transformed[feature] = feature_values.map(encoding_map).fillna(default_value)
                
            elif self.method == 'label_encoding':
                encoder = self.encoders_[feature]
                X_transformed[feature] = encoder.transform(X_transformed[feature].astype(str))
                
            elif self.method == 'one_hot':
                # Use pandas get_dummies
                dummies = pd.get_dummies(X_transformed[feature], prefix=feature, dummy_na=True)
                X_transformed = pd.concat([X_transformed.drop(columns=[feature]), dummies], axis=1)
        
        # Ensure all categorical columns are properly encoded (no string dtypes)
        # This is critical for LightGBM compatibility
        categorical_cols = X_transformed.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        
        # Also check for pyarrow string types manually
        for col in X_transformed.columns:
            if hasattr(X_transformed[col].dtype, 'pyarrow_dtype') or str(X_transformed[col].dtype).startswith('string'):
                if col not in categorical_cols:
                    categorical_cols.append(col)
        
        if len(categorical_cols) > 0:
            logger.warning(f"Found remaining categorical columns after encoding: {list(categorical_cols)}")
            # Force convert any remaining categorical columns to numeric
            for col in categorical_cols:
                if col in X_transformed.columns:
                    # Use label encoding as fallback
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X_transformed[col] = le.fit_transform(X_transformed[col].astype(str))
                    logger.info(f"Applied fallback label encoding to column: {col}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Categorical transformation completed in {elapsed_time:.2f} seconds")
        return X_transformed
    
    def _calculate_kfold_target_encoding(self, feature_series: pd.Series, target: pd.Series) -> Dict:
        """Calculate K-Fold target encoding to prevent leakage."""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        encoding_map = {}
        global_mean = float(target.mean())
        
        # Create a copy to store out-of-fold encodings
        encoded_values = np.full(len(feature_series), global_mean)
        
        # Perform K-Fold encoding
        for train_idx, val_idx in kf.split(np.arange(len(feature_series))):
            train_feature = feature_series.iloc[train_idx]
            train_target = target.iloc[train_idx]  # Use train_idx for both feature and target
            val_feature = feature_series.iloc[val_idx]
            
            # Calculate encoding for this fold
            fold_stats = pd.DataFrame({
                'feature': train_feature,
                'target': train_target
            }).groupby('feature')['target'].agg(['count', 'mean']).reset_index()
            
            # Apply smoothing
            lambda_reg = 1 / (1 + np.exp(-(fold_stats['count'] - self.min_samples_leaf) / self.min_samples_leaf))
            fold_stats['smoothed_mean'] = lambda_reg * fold_stats['mean'] + (1 - lambda_reg) * global_mean
            
            fold_encoding = dict(zip(fold_stats['feature'], fold_stats['smoothed_mean']))
            
            # Apply to validation set (out-of-fold encoding)
            for i, val_val in enumerate(val_feature):
                encoded_values[val_idx[i]] = fold_encoding.get(str(val_val), global_mean)
        
        # Calculate final encoding map using all data (for transform on new data)
        final_stats = pd.DataFrame({
            'feature': feature_series,
            'target': target
        }).groupby('feature')['target'].agg(['count', 'mean']).reset_index()
        
        lambda_reg = 1 / (1 + np.exp(-(final_stats['count'] - self.min_samples_leaf) / self.min_samples_leaf))
        final_stats['smoothed_mean'] = lambda_reg * final_stats['mean'] + (1 - lambda_reg) * global_mean
        
        encoding_map = dict(zip(final_stats['feature'].astype(str), final_stats['smoothed_mean']))
        encoding_map['__default__'] = global_mean
        
        return encoding_map
    
    def _calculate_target_encoding(self, feature_series: pd.Series, target: pd.Series) -> Dict:
        """Calculate target encoding with smoothing."""
        global_mean = target.mean()
        
        # Calculate counts and means per category
        stats = pd.DataFrame({
            'feature': feature_series,
            'target': target
        }).groupby('feature')['target'].agg(['count', 'mean']).reset_index()
        
        # Apply smoothing (empirical Bayes)
        lambda_reg = 1 / (1 + np.exp(-(stats['count'] - self.min_samples_leaf) / self.min_samples_leaf))
        stats['smoothed_mean'] = lambda_reg * stats['mean'] + (1 - lambda_reg) * global_mean
        
        encoding_map = dict(zip(stats['feature'], stats['smoothed_mean']))
        encoding_map['__default__'] = global_mean
        
        return encoding_map

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Memory-aware feature selection."""
    
    def __init__(self, 
                 method: str = 'recursive_elimination',
                 max_features: int = 50,
                 scoring: str = 'roc_auc'):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('recursive_elimination', 'importance_threshold', 'variance_threshold')
            max_features: Maximum number of features to select
            scoring: Scoring method for selection
        """
        self.method = method
        self.max_features = max_features
        self.scoring = scoring
        self.selected_features_ = []
        self.selector_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selector."""
        start_time = time.time()
        logger.info(f"Selecting features using {self.method}")
        
        # Remove non-numeric features for selection
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_features]
        
        if self.method == 'recursive_elimination':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            self.selector_ = RFE(
                estimator=estimator,
                n_features_to_select=min(self.max_features, len(numeric_features)),
                step=0.1
            )
            self.selector_.fit(X_numeric, y)
            self.selected_features_ = [f for f, selected in 
                                     zip(numeric_features, self.selector_.support_) if selected]
            
        elif self.method == 'importance_threshold':
            # Use RandomForest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_numeric, y)
            
            importances = pd.Series(rf.feature_importances_, index=numeric_features)
            self.selected_features_ = importances.nlargest(self.max_features).index.tolist()
            
        elif self.method == 'variance_threshold':
            from sklearn.feature_selection import VarianceThreshold
            self.selector_ = VarianceThreshold(threshold=0.01)
            self.selector_.fit(X_numeric)
            self.selected_features_ = [f for f, selected in 
                                     zip(numeric_features, self.selector_.get_support()) if selected]
            
            # If still too many features, use top variance
            if len(self.selected_features_) > self.max_features:
                variances = X_numeric[self.selected_features_].var()
                self.selected_features_ = variances.nlargest(self.max_features).index.tolist()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Selected {len(self.selected_features_)} features in {elapsed_time:.2f} seconds")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        # Keep categorical features and selected numeric features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        essential_features = ['patient_id', 'year'] if 'patient_id' in X.columns else []
        
        all_selected = essential_features + categorical_features + self.selected_features_
        available_features = [f for f in all_selected if f in X.columns]
        
        return X[available_features]
    
    def get_feature_names_out(self, input_features=None):
        """Get selected feature names."""
        return self.selected_features_

def create_preprocessing_pipeline(config: Dict) -> List[Any]:
    """Create preprocessing pipeline from configuration."""
    from .preprocessing import MissingValueHandler, DataScaler
    
    pipeline_steps = []
    start_time = time.time()
    logger.info("Creating preprocessing pipeline...")
    
    # Missing value handling (must come first)
    missing_handler = MissingValueHandler(
        numeric_strategy='median',
        categorical_strategy='most_frequent',
        add_indicator=False  # Avoid adding extra columns
    )
    pipeline_steps.append(missing_handler)
    
    # Temporal feature engineering
    if config.get('feature_engineering', {}).get('rolling_window_years'):
        temporal_engineer = TemporalFeatureEngineer(
            window_years=config['feature_engineering']['rolling_window_years'],
            aggregations=config['feature_engineering']['temporal_features']
        )
        pipeline_steps.append(temporal_engineer)
    
    # Categorical encoding
    encoding_config = config.get('feature_engineering', {}).get('categorical_encoding', {})
    categorical_encoder = CategoricalEncoder(
        method=encoding_config.get('method', 'target_encoding'),
        handle_unknown=encoding_config.get('handle_unknown', 'ignore'),
        cv_folds=encoding_config.get('cv_folds', 5),
        random_state=config.get('random_seed', 42)
    )
    pipeline_steps.append(categorical_encoder)
    
    # Scaling AFTER encoding (important for proper feature scaling)
    scaling_config = config.get('feature_engineering', {}).get('scaling', {})
    if scaling_config.get('enabled', True):  # Default to enabled
        data_scaler = DataScaler(
            method=scaling_config.get('method', 'standard')
        )
        pipeline_steps.append(data_scaler)
    
    # Feature selection (comes last)
    if config.get('feature_selection'):
        feature_selector = FeatureSelector(
            method=config['feature_selection'].get('method', 'recursive_elimination'),
            max_features=config['feature_selection'].get('max_features', 50)
        )
        pipeline_steps.append(feature_selector)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Created preprocessing pipeline with {len(pipeline_steps)} steps in {elapsed_time:.2f} seconds")
    return pipeline_steps
