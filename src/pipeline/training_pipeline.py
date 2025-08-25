"""
Main Training Pipeline
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
# Suppress specific MLflow deprecation warnings from their internal code
warnings.filterwarnings("ignore", message=".*artifact_path.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*artifact_path.*deprecated.*", category=UserWarning)
# Suppress XGBoost categorical data warnings
warnings.filterwarnings("ignore", message=".*DataFrame.dtypes.*must be.*category.*", category=UserWarning)

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml
import logging
import argparse
import joblib
from datetime import datetime

# Dask for distributed processing
import dask
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics.progress import ProgressBar

# ML & metrics
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# HPO + explainability
import optuna
from optuna.pruners import PercentilePruner
import shap
import matplotlib.pyplot as plt

# Local utilities 
from src.pipeline.feature_engineering import create_preprocessing_pipeline
from src.pipeline.preprocessing import DataValidator
from src.utils.experiment_tracking import ExperimentTracker
from src.utils.model_utils import ThresholdOptimizer, ModelEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================
# Stratified Temporal Splitter
# =====================
@dataclass
class StratifiedTemporalSplitter:
    """Time-aware splitter that tries to keep validation prevalence near global prevalence.

    It chooses year cut points (with an optional gap) and builds train/val indices as:
        train:  year <= cut_year - gap
        val:    year >  cut_year

    Options:
      - group_disjoint: if True, ensures no patient appears in both sets (may reduce val size).
    """
    n_splits: int = 5
    test_size: float = 0.2
    gap: int = 1
    year_col: str = "year"
    patient_col: str = "patient_id"
    target_col: str = "target"
    tol: float = 0.02
    group_disjoint: bool = True

    def split(self, X: pd.DataFrame, y: pd.Series):
        df = X[[self.patient_col, self.year_col]].copy()
        df[self.target_col] = y.values

        years = np.sort(df[self.year_col].unique())
        if len(years) < 3:
            # fall back to a simple chronological split by index order
            idx = np.arange(len(df))
            cut_idx = int(len(idx) * (1 - self.test_size))
            return [(idx[:max(0, cut_idx - self.gap)], idx[cut_idx:])]

        global_prev = float(df[self.target_col].mean())
        splits: List[Tuple[np.ndarray, np.ndarray]] = []

        # candidate cut years around quantiles of available years (avoid extremes)
        candidate_years = years[1:-1]
        if len(candidate_years) == 0:
            candidate_years = years
        q_positions = np.linspace(0.2, 0.8, self.n_splits)
        cand = np.unique(np.quantile(candidate_years, q_positions).astype(int))

        def _make_split(cut_year: int) -> Tuple[np.ndarray, np.ndarray]:
            train_mask = df[self.year_col] <= (cut_year - self.gap)
            val_mask = df[self.year_col] > cut_year

            if self.group_disjoint:
                train_patients = set(df.loc[train_mask, self.patient_col].unique())
                # remove any patient from val that appeared in train
                val_mask = val_mask & ~df[self.patient_col].isin(train_patients)

            return df.index[train_mask].to_numpy(), df.index[val_mask].to_numpy()

        for cut_year in cand:
            tr_idx, va_idx = _make_split(cut_year)
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue

            val_prev = float(df.loc[va_idx, self.target_col].mean()) if len(va_idx) else 0.0
            best = (abs(val_prev - global_prev), cut_year, tr_idx, va_idx)

            # try neighbors to improve prevalence closeness
            for delta in (-1, +1, -2, +2):
                alt = cut_year + delta
                if alt in years:
                    tr2, va2 = _make_split(alt)
                    if len(tr2) and len(va2):
                        prev2 = float(df.loc[va2, self.target_col].mean())
                        if abs(prev2 - global_prev) < best[0]:
                            best = (abs(prev2 - global_prev), alt, tr2, va2)

            splits.append((best[2], best[3]))

        # de-duplicate by basic signature (first index & sizes)
        uniq: List[Tuple[np.ndarray, np.ndarray]] = []
        seen = set()
        for tr, va in splits:
            key = (int(tr[0]) if len(tr) else -1, int(va[0]) if len(va) else -1, len(tr), len(va))
            if key not in seen:
                uniq.append((tr, va))
                seen.add(key)

        if not uniq:
            # final fallback to chronological split
            idx = np.arange(len(df))
            cut_idx = int(len(idx) * (1 - self.test_size))
            return [(idx[:max(0, cut_idx - self.gap)], idx[cut_idx:])]

        return uniq[: self.n_splits]


# =====================
# ClinicalMLPipeline
# =====================
class ClinicalMLPipeline:
    """Complete ML pipeline for clinical binary classification (advanced)."""

    def __init__(self, config: Dict):
        self.config = config
        self.model: Optional[ImbPipeline] = None
        self.feature_names: List[str] = []
        self.best_threshold: float = 0.5
        self.preprocessing_pipeline: Optional[List[Any]] = None  # Store preprocessing steps

        # trackers & helpers
        self.experiment_tracker = ExperimentTracker(config.get("mlflow", {}))
        self.threshold_optimizer = ThresholdOptimizer(
            method=self.config.get("threshold", {}).get("method", "f1_optimal")
        )

    # ---------- Data ----------
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load parquet (file/dir) or CSV with Dask for distributed processing."""
        logger.info(f"Loading data from {data_path} using Dask")
        p = Path(data_path)
        
        try:
            logger.info("Using Dask for distributed data loading...")
            # Use Dask readers based on file type
            if p.suffix.lower() == ".csv":
                ddf = dd.read_csv(p, assume_missing=True, dtype_backend="pyarrow")
            else:
                ddf = dd.read_parquet(p)

            logger.info(f"Dask DataFrame partitions: {ddf.npartitions}")

            # Optional downsampling BEFORE compute to avoid OOM on large datasets
            sample_frac = float(self.config.get("processing", {}).get("sample_fraction", 1.0))
            if sample_frac < 1.0:
                logger.info(f"Sampling {sample_frac:.1%} of data for training (pre-compute)")
                ddf = ddf.sample(frac=sample_frac, random_state=42)

            # Ensure year column is numeric for temporal operations (in Dask graph)
            if 'year' in ddf.columns:
                ddf['year'] = dd.to_numeric(ddf['year'], errors='coerce')

            # Convert to pandas for ML training
            with ProgressBar():
                df = ddf.compute()
            logger.info(f"Converted Dask DataFrame to pandas: {df.shape}")
                
        except Exception as e:
            logger.warning(f"Dask loading failed: {e}. Falling back to pandas.")
            # Fallback to pandas
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            else:
                df = pd.read_parquet(p)
            
            # Ensure year column is numeric for temporal operations
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        logger.info(f"Loaded data shape: {df.shape}")
        if "target" in df.columns:
            prev = df["target"].mean()
            logger.info(f"Target prevalence: {prev:.3f}")
        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        logger.info("Validating data quality...")
        validator = DataValidator()
        validator.setup_clinical_rules()
        violations = validator.validate(df)
        if violations:
            logger.warning(f"Found {len(violations)} data quality issues")
        else:
            logger.info("Data validation passed")
        return violations

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        target_col = "target"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        y = df[target_col]
        
        # Keep all columns except target for now (we'll exclude patient_id and year later during model training)
        X = df.drop(columns=[target_col])
        logger.info(f"Prepared features: {len(X.columns)} columns (excluded: {[target_col]})")
        return X, y

    # ---------- Splits ----------
    def create_temporal_splits(self, X: pd.DataFrame, y: pd.Series):
        cv_cfg = self.config.get("cross_validation", {})
        splitter = StratifiedTemporalSplitter(
            n_splits=cv_cfg.get("n_splits", 5),
            test_size=cv_cfg.get("test_size", 0.2),
            gap=cv_cfg.get("gap", 1),
            year_col=cv_cfg.get("year_col", "year"),
            patient_col=cv_cfg.get("group_col", "patient_id"),
            target_col="target",
            tol=cv_cfg.get("prev_tolerance", 0.02),
            group_disjoint=cv_cfg.get("group_disjoint", True),
        )
        return splitter.split(X, y)

    def create_temporal_splits_fallback(self, X: pd.DataFrame, y: pd.Series):
        years = np.sort(X["year"].unique())
        if len(years) < 2:
            idx = np.arange(len(X))
            cut = int(len(idx) * 0.8)
            return [(idx[:cut], idx[cut:])]
        # simple: last 20% of years as validation
        cut_idx = int(len(years) * 0.8)
        cut_year = years[cut_idx]
        gap = self.config.get("cross_validation", {}).get("gap", 1)
        train_idx = X.index[X["year"] <= cut_year - gap]
        val_idx = X.index[X["year"] > cut_year]
        return [(train_idx.to_numpy(), val_idx.to_numpy())]

    # ---------- Preprocess ----------
    def build_preprocessor(self):
        """Return a transformer (no SMOTE here)."""
        pre_steps = create_preprocessing_pipeline(self.config)
        self.preprocessing_pipeline = pre_steps  # Store for SHAP usage
        
        # Return the list of transformers directly, they will be added to the main pipeline
        return pre_steps

    # ---------- HPO ----------
    def hyperparameter_search(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        hpo_cfg = self.config.get("hpo", {})
        if not hpo_cfg.get("enabled", False):
            return {}

        metric_name = hpo_cfg.get("opt_metric", "pr_auc")  # 'roc_auc' or 'pr_auc'
        n_trials = int(hpo_cfg.get("n_trials", 30))
        timeout = hpo_cfg.get("timeout_sec", None)
        algo_space = hpo_cfg.get("algorithms", ["lightgbm", "xgboost", "catboost"])

        cv_splits = self.create_temporal_splits(X, y)
        if not cv_splits:
            cv_splits = self.create_temporal_splits_fallback(X, y)

        pruner = PercentilePruner(50.0, n_startup_trials=5, n_warmup_steps=0)

        def objective(trial: optuna.Trial):
            algorithm = trial.suggest_categorical("algorithm", algo_space)

            if algorithm == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("lgb_n_estimators", 200, 1200, step=100),
                    "num_leaves": trial.suggest_int("lgb_num_leaves", 16, 256),
                    "learning_rate": trial.suggest_float("lgb_learning_rate", 1e-3, 0.2, log=True),
                    "max_depth": trial.suggest_int("lgb_max_depth", 3, 12),
                    "min_child_samples": trial.suggest_int("lgb_min_child_samples", 10, 200),
                    "bagging_fraction": trial.suggest_float("lgb_bagging_fraction", 0.6, 1.0),
                    "feature_fraction": trial.suggest_float("lgb_feature_fraction", 0.6, 1.0),
                    "bagging_freq": 5,
                    "random_state": self.config.get("random_seed", 42),
                    "n_jobs": -1,
                    "verbose": -1,  # Suppress LightGBM warnings
                }
                model = lgb.LGBMClassifier(**params)
                fit_extra = {"model__eval_metric": "auc"}

            elif algorithm == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 1500, step=100),
                    "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("xgb_lr", 1e-3, 0.2, log=True),
                    "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_float("xgb_min_child_weight", 1e-2, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-2, 10.0, log=True),
                    "random_state": self.config.get("random_seed", 42),
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "eval_metric": "auc",
                }
                model = xgb.XGBClassifier(**params)
                fit_extra = {}

            else:  # catboost
                params = {
                    "iterations": trial.suggest_int("cb_iters", 300, 1500, step=100),
                    "depth": trial.suggest_int("cb_depth", 4, 10),
                    "learning_rate": trial.suggest_float("cb_lr", 1e-3, 0.2, log=True),
                    "l2_leaf_reg": trial.suggest_float("cb_l2", 1.0, 10.0),
                    "random_seed": self.config.get("random_seed", 42),
                    "loss_function": "Logloss",
                    "verbose": False,
                }
                model = cb.CatBoostClassifier(**params)
                fit_extra = {}

            preprocessor_steps = self.build_preprocessor()
            
            # Build pipeline steps starting with preprocessing steps
            steps: List[Tuple[str, Any]] = []
            
            # Add each preprocessing step individually
            for i, transformer in enumerate(preprocessor_steps):
                step_name = f"step_{i}_{transformer.__class__.__name__.lower()}"
                steps.append((step_name, transformer))
            imb_cfg = self.config.get("imbalance", {})
            if imb_cfg.get("method") == "smote":
                sp = imb_cfg.get("smote", {})
                # Adjust k_neighbors based on dataset size to prevent SMOTE errors
                default_k = min(3, max(1, len(X) // 50))  # Adaptive k_neighbors for small datasets
                k_neighbors = sp.get("k_neighbors", default_k)
                steps.append(
                    (
                        "smote",
                        SMOTE(
                            sampling_strategy=sp.get("sampling_strategy", 0.1),
                            k_neighbors=k_neighbors,
                            random_state=sp.get("random_state", 42),
                        ),
                    )
                )
            steps.append(("model", model))
            pipe = ImbPipeline(steps)

            fold_scores: List[float] = []
            for fold, (tr, va) in enumerate(cv_splits, 1):
                X_tr, X_va = X.iloc[tr], X.iloc[va]
                y_tr, y_va = y.iloc[tr], y.iloc[va]

                fit_params = fit_extra.copy()
                # Skip eval_set for now to avoid parameter issues
                pipe.fit(X_tr, y_tr, **fit_params)
                p_va = pipe.predict_proba(X_va)[:, 1]
                
                if metric_name == "pr_auc":
                    sc = average_precision_score(y_va, p_va)
                elif metric_name == "roc_auc":
                    sc = roc_auc_score(y_va, p_va)
                elif metric_name in ["f1", "precision", "recall", "accuracy"]:
                    y_pred_va = (p_va >= 0.5).astype(int)
                    if metric_name == "f1":
                        sc = f1_score(y_va, y_pred_va)
                    elif metric_name == "precision":
                        sc = precision_score(y_va, y_pred_va, zero_division=0)
                    elif metric_name == "recall":
                        sc = recall_score(y_va, y_pred_va, zero_division=0)
                    elif metric_name == "accuracy":
                        sc = accuracy_score(y_va, y_pred_va)
                else:
                    # Default to ROC-AUC
                    sc = roc_auc_score(y_va, p_va)
                    
                fold_scores.append(float(sc))
                trial.report(float(np.mean(fold_scores)), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize", pruner=pruner, study_name=hpo_cfg.get("study_name", "clinical_hpo")
        )
        
        # Don't start a new run since we're already inside one
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1, show_progress_bar=False)
        
        # Log best parameters to the current run
        try:
            self.experiment_tracker.log_params(study.best_trial.params)
            self.experiment_tracker.log_metrics({f"best_{metric_name}": study.best_value})
        except Exception as e:
            logger.warning(f"Could not log HPO results: {e}")

        best = study.best_trial.params.copy()
        logger.info(f"[HPO] Best params: {best}")
        return best

    # ---------- Training ----------
    def create_model(self):
        model_cfg = self.config.get("model", {})
        algorithm = model_cfg.get("algorithm", "lightgbm")
        logger.info(f"Creating model: {algorithm}")

        if algorithm == "lightgbm":
            params = {
                **model_cfg.get("lightgbm", {}),
                "random_state": self.config.get("random_seed", 42),
                "n_jobs": -1,
            }
            return lgb.LGBMClassifier(**params)

        if algorithm == "xgboost":
            params = {
                **model_cfg.get("xgboost", {}),
                "random_state": self.config.get("random_seed", 42),
                "n_jobs": -1,
                "tree_method": model_cfg.get("xgboost", {}).get("tree_method", "hist"),
            }
            return xgb.XGBClassifier(**params)

        if algorithm == "catboost":
            params = {**model_cfg.get("catboost", {}), "random_seed": self.config.get("random_seed", 42)}
            return cb.CatBoostClassifier(**params)

        raise ValueError(f"Unknown algorithm: {algorithm}")

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info("Starting model training...")

        # Optional HPO
        best = self.hyperparameter_search(X, y)
        if best:
            algo = best.get("algorithm", self.config.get("model", {}).get("algorithm", "lightgbm"))
            self.config.setdefault("model", {})["algorithm"] = algo
            # Map param prefixes -> model namespaces
            for k, v in best.items():
                if k.startswith("lgb_"):
                    self.config.setdefault("model", {}).setdefault("lightgbm", {})[k.replace("lgb_", "")] = v
                elif k.startswith("xgb_"):
                    self.config.setdefault("model", {}).setdefault("xgboost", {})[k.replace("xgb_", "")] = v
                elif k.startswith("cb_"):
                    param_name = k.replace("cb_", "")
                    # Map cb_iters to iterations for CatBoost
                    if param_name == "iters":
                        param_name = "iterations"
                    # Map cb_lr to learning_rate for CatBoost
                    elif param_name == "lr":
                        param_name = "learning_rate"
                    # Map cb_l2 to l2_leaf_reg for CatBoost  
                    elif param_name == "l2":
                        param_name = "l2_leaf_reg"
                    self.config.setdefault("model", {}).setdefault("catboost", {})[param_name] = v

        model = self.create_model()
        preprocessor_steps = self.build_preprocessor()

        # Build pipeline steps starting with preprocessing steps
        steps: List[Tuple[str, Any]] = []
        
        # Add each preprocessing step individually
        for i, transformer in enumerate(preprocessor_steps):
            step_name = f"step_{i}_{transformer.__class__.__name__.lower()}"
            steps.append((step_name, transformer))
        imb_cfg = self.config.get("imbalance", {})
        if imb_cfg.get("method") == "smote":
            sp = imb_cfg.get("smote", {})
            steps.append(
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=sp.get("sampling_strategy", 0.1),
                        k_neighbors=sp.get("k_neighbors", 5),
                        random_state=sp.get("random_state", 42),
                    ),
                )
            )
        steps.append(("model", model))
        full_pipeline = ImbPipeline(steps)
        self.model = full_pipeline

        cv_splits = self.create_temporal_splits(X, y)
        if not cv_splits:
            cv_splits = self.create_temporal_splits_fallback(X, y)

        cv_scores = {"roc_auc": [], "pr_auc": [], "f1": [], "precision": [], "recall": [], "accuracy": []}
        all_y_val: List[pd.Series] = []
        all_p_val: List[np.ndarray] = []

        algo = self.config.get("model", {}).get("algorithm", "lightgbm")
        fit_extra: Dict[str, Any] = {}
        # Skip early stopping for now to focus on core pipeline functionality

        for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
            logger.info(f"Training fold {fold}/{len(cv_splits)}")
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            fit_params = fit_extra.copy()
            # Skip eval_set for algorithms to avoid parameter issues
            full_pipeline.fit(X_tr, y_tr, **fit_params)

            p_va = full_pipeline.predict_proba(X_va)[:, 1]
            y_pred = (p_va >= 0.5).astype(int)

            cv_scores["roc_auc"].append(roc_auc_score(y_va, p_va))
            cv_scores["pr_auc"].append(average_precision_score(y_va, p_va))
            cv_scores["f1"].append(f1_score(y_va, y_pred))
            cv_scores["precision"].append(precision_score(y_va, y_pred, zero_division=0))
            cv_scores["recall"].append(recall_score(y_va, y_pred, zero_division=0))
            cv_scores["accuracy"].append(accuracy_score(y_va, y_pred))

            all_y_val.append(y_va)
            all_p_val.append(p_va)

        # Threshold from CV validation predictions (avoid overfit)
        y_val_concat = pd.concat(all_y_val)
        p_val_concat = np.concatenate(all_p_val)
        self.best_threshold = float(self.threshold_optimizer.optimize(y_val_concat.to_numpy(), p_val_concat))
        logger.info(f"Optimal threshold (from CV): {self.best_threshold:.3f}")

        # Final fit on all data (no eval_set)
        full_pipeline.fit(X, y)
        self.model = full_pipeline

        avg_scores = {f"cv_{m}": float(np.mean(v)) for m, v in cv_scores.items()}
        avg_scores.update({f"cv_{m}_std": float(np.std(v)) for m, v in cv_scores.items()})
        return avg_scores

    # ---------- Evaluation ----------
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info("Evaluating model performance...")
        if self.model is None:
            raise ValueError("Model not trained yet")

        y_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.best_threshold).astype(int)

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y.to_numpy(), y_pred, y_proba)
        metrics["optimal_threshold"] = float(self.best_threshold)

        # Extract feature names for artifact saving
        pre = self.model.named_steps.get("preprocessing", None)
        if pre is not None and hasattr(pre, "get_feature_names_out"):
            try:
                self.feature_names = list(pre.get_feature_names_out())
                logger.info(f"Extracted {len(self.feature_names)} feature names from preprocessor")
            except Exception as e:
                logger.warning(f"Failed to get feature names from preprocessor: {e}")
                # Fallback: use input column names
                self.feature_names = list(X.columns)
                logger.info(f"Using input column names as feature names: {len(self.feature_names)} features")
        else:
            # Fallback: use input column names
            self.feature_names = list(X.columns)
            logger.info(f"No preprocessor get_feature_names_out, using input columns: {len(self.feature_names)} features")

        return metrics

    # ---------- Explainability (SHAP) ----------
    def compute_and_log_shap(self, X: pd.DataFrame, nsample: int = 2000, out_dir: Optional[Path] = None):
        if self.model is None:
            logger.warning("Model not trained; skipping SHAP.")
            return
        out = Path(out_dir or "./models/explain")
        out.mkdir(parents=True, exist_ok=True)

        try:
            # For SHAP, we need to be more careful about feature preprocessing
            # Use the model's predict method directly which handles all preprocessing
            if self.model is None:
                raise ValueError("Model not trained")
                
            # Prepare clean data for SHAP - convert all columns to proper types
            X_clean = X.copy()
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    # Convert object columns to category for XGBoost compatibility
                    X_clean[col] = X_clean[col].astype('category')
                elif X_clean[col].dtype == 'bool':
                    # Convert boolean to int for XGBoost
                    X_clean[col] = X_clean[col].astype(int)
                
            def model_predict(X_input):
                # Ensure input is DataFrame with proper columns and types
                if not isinstance(X_input, pd.DataFrame):
                    X_input = pd.DataFrame(X_input, columns=X_clean.columns)
                    # Apply same type conversions
                    for col in X_input.columns:
                        if col in X_clean.columns:
                            X_input[col] = X_input[col].astype(X_clean[col].dtype)
                
                if self.model is not None and hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X_input)[:, 1]
                elif self.model is not None:
                    return self.model.predict(X_input)
                else:
                    raise ValueError("Model is None")
                    
            # Use TreeExplainer for tree-based models (much faster than KernelExplainer)
            # Sample data for SHAP computation - use very small sample for speed
            shap_sample = X_clean.sample(min(5, len(X_clean)), random_state=42)
            
            try:
                # Try TreeExplainer first (fastest for tree models)
                if hasattr(self.model.named_steps['model'], 'booster'):  # LightGBM/XGBoost
                    explainer = shap.TreeExplainer(self.model.named_steps['model'])
                    shap_vals = explainer.shap_values(shap_sample)
                    if isinstance(shap_vals, list):  # Multi-class output
                        shap_vals = shap_vals[1]  # Use positive class
                else:
                    # Fallback to KernelExplainer with very small background
                    background_sample = X_clean.sample(min(10, len(X_clean)), random_state=42)
                    explainer = shap.KernelExplainer(model_predict, background_sample)
                    shap_vals = explainer.shap_values(shap_sample)
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}, skipping explanations")
                return
            
            # Use input feature names
            feat_names = list(X_clean.columns)
            self.feature_names = feat_names

            # Generate plots
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, shap_sample, feature_names=self.feature_names, show=False)
            beeswarm_path = out / "shap_beeswarm.png"
            plt.tight_layout(); plt.savefig(beeswarm_path, dpi=150); plt.close()

            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, shap_sample, feature_names=self.feature_names, plot_type="bar", show=False)
            bar_path = out / "shap_bar.png"
            plt.tight_layout(); plt.savefig(bar_path, dpi=150); plt.close()

            # Mean |SHAP| table
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            shap_df = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})\
                .sort_values("mean_abs_shap", ascending=False)
            shap_csv = out / "shap_importance.csv"
            shap_df.to_csv(shap_csv, index=False)

            # Log artifacts to MLflow (best-effort)
            try:
                self.experiment_tracker.log_artifacts(str(out))
            except Exception:
                pass

            logger.info(f"SHAP artifacts saved: {beeswarm_path.name}, {bar_path.name}, {shap_csv.name}")
            
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

    # ---------- Artifact helpers ----------
    def _extract_category_mapping(self) -> Dict[str, Dict[str, List[Any]]]:
        mapping: Dict[str, Dict[str, List[Any]]] = {}
        pre = self.model.named_steps.get("preprocessing", None) if self.model else None
        if pre is None or not hasattr(pre, "transformers_"):
            return mapping
        for name, trans, cols in pre.transformers_:
            if trans is None or name == "remainder":
                continue
            block: Dict[str, List[Any]] = {}
            try:
                if hasattr(trans, "categories_"):
                    cats = trans.categories_
                    for c, col in enumerate(cols):
                        block[str(col)] = list(cats[c])
                elif hasattr(trans, "named_steps"):
                    for tname, step in trans.named_steps.items():
                        if hasattr(step, "categories_"):
                            cats = step.categories_
                            for c, col in enumerate(cols):
                                block[str(col)] = list(cats[c])
            except Exception:
                continue
            if block:
                mapping[name] = block
        return mapping

    def save_artifacts(self, output_dir: str, metrics: Dict[str, float]):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving artifacts to {out}")

        # Save full pipeline (preprocessing + model)
        joblib.dump(self.model, out / "best_model.joblib")

        # Save preprocessor separately
        if hasattr(self.model, "named_steps") and self.model is not None:
            pre = self.model.named_steps.get("preprocessing", None)
            if pre is not None:
                joblib.dump(pre, out / "preprocessor.joblib")
        else:
            pre = None

        # Threshold
        (out / "optimal_threshold.txt").write_text(str(self.best_threshold), encoding="utf-8")

        # Feature names
        if self.feature_names:
            (out / "feature_names.txt").write_text("\n".join(self.feature_names), encoding="utf-8")

        # Metrics & config (convert numpy types to native Python types for YAML)
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for YAML serialization."""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        clean_metrics = convert_numpy_types(metrics)
        clean_config = convert_numpy_types(self.config)
        
        (out / "metrics.yaml").write_text(yaml.dump(clean_metrics), encoding="utf-8")
        (out / "training_config.yaml").write_text(yaml.dump(clean_config), encoding="utf-8")
        
        # Log config to MLflow as well (for centralized experiment tracking)
        try:
            self.experiment_tracker.log_dict(clean_config, "config.yaml")
            self.experiment_tracker.log_dict(clean_metrics, "metrics.yaml")
            
            # Log feature names to MLflow
            if self.feature_names:
                feature_names_dict = {
                    "total_features": len(self.feature_names),
                    "feature_names": self.feature_names
                }
                self.experiment_tracker.log_dict(feature_names_dict, "feature_names.yaml")
            
            # Also log feature importance if available
            if self.model and hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
                base_model = self.model.named_steps['model']
                if hasattr(base_model, 'feature_importances_'):
                    feature_importance = dict(zip(
                        self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(base_model.feature_importances_))],
                        base_model.feature_importances_.tolist()
                    ))
                    self.experiment_tracker.log_dict(feature_importance, "feature_importance.yaml")
                    
            logger.info("Configuration, metrics, feature names, and feature importance logged to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log artifacts to MLflow: {e}")

        # Category mapping (encoders)
        cat_map = self._extract_category_mapping()
        if cat_map:
            (out / "category_mapping.yaml").write_text(yaml.dump(cat_map), encoding="utf-8")

        # Dropped features estimation (best-effort)
        try:
            if hasattr(self.model, "named_steps") and self.model is not None and "preprocessing" in self.model.named_steps:
                input_feats = list(self.model.named_steps["preprocessing"].feature_names_in_)
                used_feats_out = list(self.feature_names) if self.feature_names else []
                base_names = [c.split("__")[0] if "__" in c else c for c in used_feats_out]
                dropped = sorted(set(input_feats) - set(base_names))
                (out / "dropped_features.txt").write_text("\n".join(dropped), encoding="utf-8")
        except Exception:
            pass

        logger.info("Artifacts saved successfully")

    # ---------- Orchestration ----------
    def run_pipeline(self, data_path: str, output_dir: str, skip_shap: bool = True) -> Dict[str, float]:
        logger.info("Starting complete ML pipeline with Dask distributed processing...")
        experiment_name = self.config.get("mlflow", {}).get("experiment_name", "clinical_ml")

        # Initialize Dask client for distributed processing
        client = None
        try:
            # Initialize Dask client with configuration from config
            dask_config = self.config.get("processing", {}).get("dask_config", {})
            
            # Extra conservative settings for free servers
            memory_limit = dask_config.get("memory_limit", "256MB")
            n_workers = dask_config.get("n_workers", 1)
            threads_per_worker = dask_config.get("threads_per_worker", 1)
            
            # Don't initialize Dask if resources are too limited
            if memory_limit in ["256MB", "128MB"] and n_workers == 1:
                logger.info("Resource-constrained environment detected. Skipping Dask initialization for single-threaded processing.")
                client = None
            else:
                client = Client(
                    processes=dask_config.get("processes", False),
                    threads_per_worker=threads_per_worker,
                    n_workers=n_workers,
                    memory_limit=memory_limit
                )
                logger.info(f"Dask client initialized: {client.dashboard_link}")
        except Exception as e:
            logger.warning(f"Failed to initialize Dask client: {e}. Continuing with single-threaded processing.")

        try:
            with self.experiment_tracker.start_run(experiment_name) as run:
                # Log config
                self.experiment_tracker.log_params(self.config)

                # Data (with Dask support)
                df = self.load_data(data_path)
                self.validate_data(df)
                X, y = self.prepare_features(df)

                # Train (feature engineering will use Dask if available)
                cv_metrics = self.train_model(X, y)

                # Evaluate on all data (for report only; real deployment should use holdout)
                final_metrics = self.evaluate_model(X, y)
                all_metrics = {**cv_metrics, **final_metrics}
                self.experiment_tracker.log_metrics(all_metrics)

                # Save artifacts (model, preprocessor, mappings, metrics, config)
                self.save_artifacts(output_dir, all_metrics)

                # SHAP (sampled) - only if explicitly enabled
                if not skip_shap:
                    try:
                        Xsample = X.sample(min(100, len(X)), random_state=42)  # Much smaller sample
                        self.compute_and_log_shap(Xsample, out_dir=Path(output_dir) / "explain")
                    except Exception as e:
                        logger.warning(f"SHAP failed: {e}")
                else:
                    logger.info("Skipping SHAP computation (default behavior for faster training)")

                # Log directory to MLflow
                try:
                    self.experiment_tracker.log_artifacts(output_dir)
                    # Create a simple input example for signature inference, avoiding categorical conversion issues
                    try:
                        input_example = X.sample(min(5, len(X)), random_state=42)
                        # Convert all categorical columns to strings to avoid schema enforcement issues
                        for col in input_example.columns:
                            if input_example[col].dtype == 'object' or input_example[col].dtype.name == 'category':
                                input_example[col] = input_example[col].astype(str)
                        self.experiment_tracker.log_model(self.model, "model", input_example=input_example)
                    except Exception as e:
                        logger.warning(f"Input example failed: {e}, logging model without example")
                        self.experiment_tracker.log_model(self.model, "model")
                except Exception:
                    pass

                logger.info("Pipeline completed successfully!")
                roc = final_metrics.get("roc_auc")
                return all_metrics
        
        finally:
            # Clean up Dask client
            if client:
                client.close()
                logger.info("Dask client closed")
            pr = final_metrics.get("pr_auc")
            f1 = final_metrics.get("f1_score") or final_metrics.get("f1")
            if roc is not None:
                logger.info(f"Final ROC-AUC: {roc:.4f}")
            if pr is not None:
                logger.info(f"Final PR-AUC: {pr:.4f}")
            if f1 is not None:
                logger.info(f"Final F1-Score: {f1:.4f}")

            return all_metrics


# =====================
# CLI entrypoint
# =====================

def main():
    parser = argparse.ArgumentParser(description="Train clinical ML model (advanced)")
    parser.add_argument("--config", type=str, required=True, help="Path to training configuration file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (parquet dir/file or CSV)")
    parser.add_argument("--output", type=str, default="./models", help="Output directory for artifacts")
    parser.add_argument("--enable-shap", action="store_true", help="Enable SHAP computation (disabled by default for faster training)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Seed control (recommended for reproducibility)
    seed = config.get("random_seed", 42)
    np.random.seed(seed)

    pipeline = ClinicalMLPipeline(config)
    # Skip SHAP by default, only enable if flag is provided
    pipeline.run_pipeline(args.data, args.output, skip_shap=not args.enable_shap)

    print("Training completed! Artifacts in:", args.output)


if __name__ == "__main__":
    main()
