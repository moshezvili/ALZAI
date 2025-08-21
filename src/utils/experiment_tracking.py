"""
Experiment tracking utilities using MLflow.
"""

import mlflow
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """MLflow experiment tracking wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize experiment tracker."""
        self.config = config
        mlflow_config = config.get('mlflow', {})
        self.tracking_uri = mlflow_config.get('tracking_uri', 'file:./mlruns')
        self.experiment_name = mlflow_config.get('experiment_name', 'default')
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment, handling deleted experiments
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        except Exception:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment and experiment.lifecycle_stage != "deleted":
                experiment_id = experiment.experiment_id
            else:
                # If experiment is deleted or doesn't exist, create a new one with timestamp
                import time
                new_name = f"{self.experiment_name}_{int(time.time())}"
                try:
                    experiment_id = mlflow.create_experiment(new_name)
                    self.experiment_name = new_name
                except Exception:
                    # Fall back to default experiment
                    experiment_id = "0"
        
        if experiment_id and experiment_id != "0":
            mlflow.set_experiment(experiment_id=experiment_id)
        else:
            # Use default experiment
            mlflow.set_experiment("Default")
    
    def start_run(self, run_name: Optional[str] = None):
        """Start MLflow run."""
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any], prefix: str = ""):
        """Log parameters to MLflow."""
        flat_params = self._flatten_dict(params, prefix)
        for key, value in flat_params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")
    
    def log_artifacts(self, artifact_path: str):
        """Log artifacts to MLflow."""
        try:
            mlflow.log_artifacts(artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")
    
    def log_dict(self, dictionary: Union[Dict[str, Any], Any], artifact_file: str):
        """Log dictionary as YAML artifact to MLflow."""
        try:
            mlflow.log_dict(dictionary, artifact_file)
            logger.info(f"Dictionary logged as {artifact_file}")
        except Exception as e:
            logger.warning(f"Failed to log dictionary to MLflow: {e}")
    
    def log_model(self, model, model_name: str, input_example=None, **kwargs):
        """Log model to MLflow with proper signature and input example."""
        try:
            # Clean up any deprecated parameters that might be passed in
            kwargs.pop('artifact_path', None)
            
            # Add input example if provided and not already in kwargs
            if input_example is not None and 'input_example' not in kwargs:
                kwargs['input_example'] = input_example
            
            # Detect model type and use appropriate logging function
            model_type = type(model).__name__
            
            # Use getattr to access MLflow modules (avoids type checker issues)
            if 'lightgbm' in model_type.lower() or 'lgb' in model_type.lower():
                lightgbm_module = getattr(mlflow, 'lightgbm')
                lightgbm_module.log_model(model, artifact_path=model_name, **kwargs)
            elif 'xgb' in model_type.lower():
                xgboost_module = getattr(mlflow, 'xgboost')
                xgboost_module.log_model(model, artifact_path=model_name, **kwargs)
            else:
                sklearn_module = getattr(mlflow, 'sklearn')
                sklearn_module.log_model(model, artifact_path=model_name, **kwargs)
                
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
            # Fallback to sklearn logging
            try:
                clean_kwargs = {'input_example': input_example} if input_example is not None else {}
                sklearn_module = getattr(mlflow, 'sklearn')
                sklearn_module.log_model(model, artifact_path=model_name, **clean_kwargs)
            except Exception as e2:
                logger.error(f"Failed to log model with sklearn fallback: {e2}")
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """Flatten nested dictionary for parameter logging."""
        items = []
        
        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key).items())
            else:
                # Convert to string for MLflow
                items.append((new_key, str(value)))
        
        return dict(items)

class WandBTracker:
    """Weights & Biases tracking wrapper (optional)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize W&B tracker."""
        try:
            import wandb
            self.wandb = wandb
            self.config = config
            
            project_name = config.get('project', 'clinical-ml')
            self.run = self.wandb.init(
                project=project_name,
                config=config,
                name=config.get('run_name'),
                tags=config.get('tags', [])
            )
            
        except ImportError:
            logger.warning("wandb not installed, skipping W&B tracking")
            self.wandb = None
            self.run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.run:
            self.run.log(metrics, step=step)
    
    def log_artifacts(self, artifact_path: str):
        """Log artifacts to W&B."""
        if self.run:
            self.run.save(artifact_path)
    
    def finish(self):
        """Finish W&B run."""
        if self.run:
            self.run.finish()

def setup_experiment_tracking(config: Dict[str, Any]) -> Optional[Union[ExperimentTracker, WandBTracker]]:
    """Setup experiment tracking based on configuration."""
    tracking_config = config.get('experiment_tracking', {})
    
    if tracking_config.get('backend', 'mlflow') == 'mlflow':
        return ExperimentTracker(tracking_config.get('mlflow', {}))
    elif tracking_config.get('backend') == 'wandb':
        return WandBTracker(tracking_config.get('wandb', {}))
    else:
        logger.warning("No experiment tracking configured")
        return None
