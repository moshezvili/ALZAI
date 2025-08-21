"""
Synthetic Clinical Data Generator with Dask

This module generates synthetic patient-year clinical data for ML training using Dask
for efficient parallel computation and memory management.
Simulates realistic clinical patterns while ensuring privacy.
"""

import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
from dask.delayed import delayed
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from faker import Faker
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalDataGenerator:
    """Generate synthetic clinical data in patient-year format."""
    
    def __init__(self, seed: int = 42, missing_value_rates: Optional[Dict[str, float]] = None):
        """Initialize the generator with a random seed and missing value configuration.
        
        Args:
            seed: Random seed for reproducibility
            missing_value_rates: Dict specifying missing value rates for different fields.
                Default: {'cholesterol': 0.05, 'bmi': 0.10, 'glucose': 0.03, 'blood_pressure': 0.02}
        """
        self.seed = seed
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
        
        # Configure missing value rates
        self.missing_value_rates = missing_value_rates or {
            'cholesterol': 0.05,  # 5% missing cholesterol (common lab test)
            'bmi': 0.10,          # 10% missing BMI (failed measurements)
            'glucose': 0.03,      # 3% missing glucose
            'blood_pressure': 0.02  # 2% missing blood pressure
        }
        
        # Clinical parameters for realistic simulation
        self.icd_codes = [
            'I10', 'E11', 'I25', 'J44', 'N18', 'F17', 'Z51', 'I48', 
            'E78', 'K21', 'M79', 'I50', 'G47', 'E03', 'M25'
        ]
        
        self.medications = [
            'Metformin', 'Lisinopril', 'Atorvastatin', 'Amlodipine',
            'Omeprazole', 'Metoprolol', 'Losartan', 'Hydrochlorothiazide',
            'Gabapentin', 'Sertraline', 'Furosemide', 'Prednisone'
        ]
        
    def _generate_patient_demographics(self, patient_id: str) -> Dict:
        """Generate stable demographics for a patient (helper method for vectorized generation)."""
        # Use patient_id as seed for consistent demographics
        local_random = np.random.RandomState(hash(patient_id) % 2**32)
        
        gender = local_random.choice(['M', 'F'], p=[0.48, 0.52])
        birth_year = local_random.randint(1940, 2000)
        
        # Correlated characteristics
        if gender == 'M':
            baseline_bmi = local_random.normal(27.5, 4.5)
            smoking_prob = 0.25
        else:
            baseline_bmi = local_random.normal(26.8, 5.2)
            smoking_prob = 0.18
            
        smoking_status = local_random.choice(
            ['Never', 'Former', 'Current'], 
            p=[1-smoking_prob, smoking_prob*0.6, smoking_prob*0.4]
        )
        
        return {
            'gender': gender,
            'birth_year': birth_year,
            'baseline_bmi': max(16, min(50, baseline_bmi)),  # Realistic bounds
            'smoking_status': smoking_status
        }
    
    def generate_patient_demographics(self, patient_id: str) -> Dict:
        """Public method for generating patient demographics (for test compatibility)."""
        return self._generate_patient_demographics(patient_id)
    
    def generate_year_data(self, patient_id: str, year: int, demographics: Dict) -> Dict:
        """Generate data for a specific patient-year combination (for test compatibility)."""
        local_random = np.random.RandomState(hash(f"{patient_id}_{year}") % 2**32)
        
        age = year - demographics['birth_year']
        gender = demographics['gender']
        baseline_bmi = demographics['baseline_bmi']
        smoking_status = demographics['smoking_status']
        
        # Generate year-specific data
        bmi = max(16, min(50, local_random.normal(baseline_bmi + (age - 40) * 0.1, 2.5)))
        
        systolic_bp = local_random.normal(120 + (age - 40) * 0.5, 15)
        diastolic_bp = local_random.normal(80 + (age - 40) * 0.2, 10)
        systolic_bp = max(90, min(200, systolic_bp))
        diastolic_bp = max(60, min(120, diastolic_bp))
        
        cholesterol = local_random.normal(200 + (age - 40) * 1.2, 30)
        cholesterol = max(120, min(400, cholesterol))
        
        glucose = local_random.normal(95 + (age - 40) * 0.8, 15)
        glucose = max(70, min(300, glucose))
        
        num_visits = max(1, local_random.poisson(max(1, 3 + (age - 40) * 0.05)))
        medications_count = max(0, local_random.poisson(max(0.1, 1 + (age - 60) * 0.1)))
        
        lab_abnormal_flag = local_random.random() < (0.1 + (age - 40) * 0.005)
        
        # Primary diagnosis
        diagnosis_weights = np.ones(len(self.icd_codes))
        if age > 65:
            diagnosis_weights[[0, 2, 3, 5, 7]] *= 2
        if bmi > 30:
            diagnosis_weights[1] *= 3
        if smoking_status == 'Current':
            diagnosis_weights[[3, 5]] *= 2
            
        diagnosis_probs = diagnosis_weights / diagnosis_weights.sum()
        primary_diagnosis = local_random.choice(self.icd_codes, p=diagnosis_probs)
        
        return {
            'patient_id': patient_id,
            'year': year,
            'age': age,
            'gender': gender,
            'bmi': round(bmi, 1),
            'systolic_bp': round(systolic_bp, 1),
            'diastolic_bp': round(diastolic_bp, 1),
            'cholesterol': round(cholesterol, 1),
            'glucose': round(glucose, 1),
            'smoking_status': smoking_status,
            'num_visits': num_visits,
            'medications_count': medications_count,
            'lab_abnormal_flag': lab_abnormal_flag,
            'primary_diagnosis': primary_diagnosis
        }
    
    def generate_target_variable(self, data: pd.DataFrame, prevalence: float = 0.07) -> pd.Series:
        import time
        start_all = time.time()
        logger.info(f"Generating target with {prevalence:.1%} prevalence (vectorized)")

        # --- build risk_factors vectorized ---
        start = time.time()
        age_risk = (data['age'] - 40).clip(lower=0) / 60
        bmi_risk = (data['bmi'] - 25).clip(lower=0) / 15
        bp_risk = (((data['systolic_bp'] > 140) | (data['diastolic_bp'] > 90)).astype(float) * 0.5)
        glucose_risk = (data['glucose'] > 126).astype(float) * 0.7
        chol_risk = (data['cholesterol'] > 240).astype(float) * 0.3
        smoking_map = {'Never': 0.0, 'Former': 0.3, 'Current': 0.6}
        smoking_risk = data['smoking_status'].map(smoking_map).fillna(0.0)
        utilization_risk = (data['num_visits'] > 5).astype(float) * 0.2
        lab_risk = data['lab_abnormal_flag'].astype(float) * 0.4
        diagnosis_risk = data['primary_diagnosis'].isin(['I10', 'E11', 'I25', 'J44', 'I50']).astype(float) * 0.5

        # stack into DataFrame in consistent order
        risk_df = pd.DataFrame({
            'age_risk': age_risk,
            'bmi_risk': bmi_risk,
            'bp_risk': bp_risk,
            'glucose_risk': glucose_risk,
            'chol_risk': chol_risk,
            'smoking_risk': smoking_risk,
            'utilization_risk': utilization_risk,
            'lab_risk': lab_risk,
            'diagnosis_risk': diagnosis_risk
        }, index=data.index)
        logger.info(f"Built risk factors in {time.time()-start:.2f}s")

        # --- combine with weights (vectorized dot product) ---
        start = time.time()
        weights = np.array([0.15, 0.12, 0.18, 0.20, 0.08, 0.10, 0.05, 0.07, 0.05])
        # ensure columns aligned
        total_risk = risk_df.values.dot(weights)
        logger.info(f"Combined weighted risk in {time.time()-start:.2f}s")

        # --- temporal factor: vectorized using groupby().cumcount() ---
        start = time.time()
        # sort by patient_id and year to compute proper order
        order_idx = data.sort_values(['patient_id', 'year']).index
        # compute year-index (0,1,2...) per patient; cast to numpy array (float) to satisfy type checkers
        year_idx = (
            data.loc[order_idx]
                .groupby('patient_id')
                .cumcount()
                .to_numpy(dtype=float)
        )
        temporal_factor_sorted = 0.1 * year_idx
        # map back to original order
        temporal_factor = np.empty(len(data), dtype=float)
        temporal_factor[np.argsort(order_idx)] = temporal_factor_sorted  # aligns sorted->original
        logger.info(f"Computed temporal factor in {time.time()-start:.2f}s")

        # --- noise (use a local RNG, do not reset global seed) ---
        start = time.time()
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, 0.1, size=len(data))
        logger.info(f"Generated noise in {time.time()-start:.2f}s")

        # --- final risk and probability mapping ---
        start = time.time()
        final_risk = total_risk + noise + temporal_factor

        # compute threshold to get approx desired prevalence
        threshold = float(np.percentile(final_risk, (1 - prevalence) * 100))
        shifted = final_risk - threshold
        probabilities = 1 / (1 + np.exp(-shifted))

        # pick exact top-N to guarantee prevalence
        n_positive = int(len(data) * prevalence)
        if n_positive <= 0:
            logger.warning("prevalence too small for dataset size, returning all zeros")
            return pd.Series(0, index=data.index)
        chosen_idx_sorted = np.argsort(probabilities)[-n_positive:]  # indices relative to data order
        target = pd.Series(0, index=data.index)
        target.iloc[chosen_idx_sorted] = 1
        logger.info(f"Assigned {target.sum()} positives in {time.time()-start:.2f}s")
        logger.info(f"Total generate_target_variable time: {time.time()-start_all:.2f}s")

        return target

    @delayed
    def _generate_patient_batch_vectorized(self, patient_batch: List[str], years_per_patient: int, start_year: int) -> List[Dict]:
        """Generate all records for a batch of patients using vectorized operations."""
        all_records = []
        
        # Pre-generate all random numbers for the entire batch
        batch_size = len(patient_batch)
        total_records = batch_size * years_per_patient
        
        # Create a single large random generator for this batch
        batch_seed = hash(str(patient_batch[:5])) % 2**32  # Use first 5 patient IDs as seed
        batch_rng = np.random.RandomState(batch_seed)
        
        # Pre-generate all demographics for the batch
        patient_demographics = {}
        for patient_id in patient_batch:
            patient_demographics[patient_id] = self._generate_patient_demographics(patient_id)
        
        # Vectorized generation for all patient-years
        patient_ids_expanded = []
        years_expanded = []
        ages_expanded = []
        genders_expanded = []
        baseline_bmis = []
        smoking_statuses = []
        birth_years = []
        
        for patient_id in patient_batch:
            demographics = patient_demographics[patient_id]
            for year_offset in range(years_per_patient):
                year = start_year + year_offset
                age = year - demographics['birth_year']
                
                # Skip if patient would be too young or too old
                if age < 18 or age > 90:
                    continue
                    
                patient_ids_expanded.append(patient_id)
                years_expanded.append(year)
                ages_expanded.append(age)
                genders_expanded.append(demographics['gender'])
                baseline_bmis.append(demographics['baseline_bmi'])
                smoking_statuses.append(demographics['smoking_status'])
                birth_years.append(demographics['birth_year'])
        
        if not patient_ids_expanded:
            return []
        
        # Convert to numpy arrays for vectorized operations
        n_records = len(patient_ids_expanded)
        ages = np.array(ages_expanded)
        genders = np.array(genders_expanded)
        baseline_bmis = np.array(baseline_bmis)
        smoking_statuses = np.array(smoking_statuses)
        years = np.array(years_expanded)
        
        # Vectorized BMI calculation with drift
        # Generate all BMI drift values at once
        bmi_drift_noise = batch_rng.normal(0.1, 0.8, size=n_records)
        
        # Calculate BMI with temporal progression
        bmis = np.zeros(n_records)
        patient_year_counts = {}
        
        for i, patient_id in enumerate(patient_ids_expanded):
            if patient_id not in patient_year_counts:
                patient_year_counts[patient_id] = 0
                bmis[i] = baseline_bmis[i]  # First year uses baseline
            else:
                # Subsequent years use drift
                prev_idx = i - 1
                while prev_idx >= 0 and patient_ids_expanded[prev_idx] != patient_id:
                    prev_idx -= 1
                bmis[i] = bmis[prev_idx] + bmi_drift_noise[i]
            patient_year_counts[patient_id] += 1
        
        # Apply BMI bounds vectorized
        bmis = np.clip(bmis, 16, 50)
        
        # Vectorized blood pressure calculation
        age_factors = np.maximum(0, (ages - 40) * 0.3)
        bmi_factors = np.maximum(0, (bmis - 25) * 0.8)
        
        systolic_bp = batch_rng.normal(
            120 + age_factors + bmi_factors, 15, size=n_records
        )
        diastolic_bp = batch_rng.normal(
            80 + age_factors * 0.5 + bmi_factors * 0.5, 10, size=n_records
        )
        systolic_bp = np.clip(systolic_bp, 80, 220)
        diastolic_bp = np.clip(diastolic_bp, 50, 120)
        
        # Vectorized cholesterol calculation
        chol_base = np.where(genders == 'M', 200, 195)
        cholesterol = batch_rng.normal(
            chol_base + ages * 0.5, 35, size=n_records
        )
        cholesterol = np.clip(cholesterol, 120, 400)
        
        # Vectorized glucose calculation
        glucose_base = 95 + (bmis - 25) * 1.2 + (ages - 50) * 0.3
        glucose = batch_rng.normal(glucose_base, 15, size=n_records)
        glucose = np.clip(glucose, 70, 300)
        
        # Vectorized healthcare utilization
        visit_lambda = 3 + (ages - 50) * 0.02
        num_visits = batch_rng.poisson(visit_lambda, size=n_records)
        num_visits = np.maximum(0, num_visits)
        
        med_lambda = np.maximum(0, (ages - 60) * 0.1)
        medications_count = batch_rng.poisson(med_lambda, size=n_records)
        medications_count = np.maximum(0, medications_count)
        
        # Vectorized lab abnormal flag
        risk_scores = (
            (systolic_bp > 140) * 0.3 +
            (cholesterol > 240) * 0.2 +
            (glucose > 126) * 0.4 +
            (bmis > 30) * 0.1
        )
        lab_abnormal_flags = batch_rng.random(size=n_records) < risk_scores
        
        # Vectorized primary diagnosis selection
        primary_diagnoses = []
        additional_diagnoses_list = []
        
        for i in range(n_records):
            # Primary diagnosis (still need individual logic for weighted selection)
            diagnosis_weights = np.ones(len(self.icd_codes))
            if ages[i] > 65:
                diagnosis_weights[[0, 2, 3, 5, 7]] *= 2
            if bmis[i] > 30:
                diagnosis_weights[1] *= 3
            if smoking_statuses[i] == 'Current':
                diagnosis_weights[[3, 5]] *= 2
                
            diagnosis_probs = diagnosis_weights / diagnosis_weights.sum()
            primary_diagnosis = batch_rng.choice(self.icd_codes, p=diagnosis_probs)
            primary_diagnoses.append(primary_diagnosis)
            
            # Additional diagnoses
            num_additional = batch_rng.poisson(max(0, (ages[i] - 60) * 0.05))
            if num_additional > 0:
                additional_diagnoses = batch_rng.choice(
                    self.icd_codes, size=min(num_additional, 3), replace=False
                ).tolist()
            else:
                additional_diagnoses = []
            additional_diagnoses_list.append(','.join(additional_diagnoses))
        
        # Simulate missing values (realistic clinical data quality issues)
        missing_cholesterol_mask = batch_rng.random(n_records) < self.missing_value_rates['cholesterol']
        missing_bmi_mask = batch_rng.random(n_records) < self.missing_value_rates['bmi']
        missing_glucose_mask = batch_rng.random(n_records) < self.missing_value_rates['glucose']
        missing_bp_mask = batch_rng.random(n_records) < self.missing_value_rates['blood_pressure']
        
        # Apply missing values
        cholesterol_final = np.where(missing_cholesterol_mask, np.nan, cholesterol)
        bmis_final = np.where(missing_bmi_mask, np.nan, bmis)
        glucose_final = np.where(missing_glucose_mask, np.nan, glucose)
        systolic_bp_final = np.where(missing_bp_mask, np.nan, systolic_bp)
        diastolic_bp_final = np.where(missing_bp_mask, np.nan, diastolic_bp)
        
        # Convert back to list of dictionaries
        for i in range(n_records):
            record = {
                'patient_id': patient_ids_expanded[i],
                'year': int(years[i]),
                'age': int(ages[i]),
                'gender': genders[i],
                'bmi': round(float(bmis_final[i]), 1) if not np.isnan(bmis_final[i]) else None,
                'systolic_bp': round(float(systolic_bp_final[i]), 1) if not np.isnan(systolic_bp_final[i]) else None,
                'diastolic_bp': round(float(diastolic_bp_final[i]), 1) if not np.isnan(diastolic_bp_final[i]) else None,
                'cholesterol': round(float(cholesterol_final[i]), 1) if not np.isnan(cholesterol_final[i]) else None,
                'glucose': round(float(glucose_final[i]), 1) if not np.isnan(glucose_final[i]) else None,
                'smoking_status': smoking_statuses[i],
                'num_visits': int(num_visits[i]),
                'medications_count': int(medications_count[i]),
                'lab_abnormal_flag': bool(lab_abnormal_flags[i]),
                'primary_diagnosis': primary_diagnoses[i],
                'additional_diagnoses': additional_diagnoses_list[i],
                'birth_year': birth_years[i]
            }
            all_records.append(record)
        
        return all_records

    def generate_dask_dataset(self, num_patients: int, years_per_patient: int,
                             start_year: int = 2015, target_prevalence: float = 0.07,
                             batch_size: int = 1000) -> dd.DataFrame:
        """Generate complete synthetic clinical dataset as Dask DataFrame using vectorized operations."""
        logger.info(f"Generating Dask dataset for {num_patients:,} patients over {years_per_patient} years (vectorized)")
        
        # Generate patient IDs
        patient_ids = [f"P{i:06d}" for i in range(num_patients)]
        
        # Create delayed tasks for patient record generation in batches
        delayed_tasks = []
        
        # Use vectorized batch processing
        for i in range(0, len(patient_ids), batch_size):
            batch_ids = patient_ids[i:i + batch_size]
            batch_task = self._generate_patient_batch_vectorized(batch_ids, years_per_patient, start_year)
            delayed_tasks.append(batch_task)
        
        logger.info(f"Created {len(delayed_tasks)} delayed tasks")
        
        # Compute delayed tasks and flatten results
        @delayed
        def flatten_and_create_dataframe(batch_records: List[Dict]) -> pd.DataFrame:
            """Create DataFrame from batch records."""
            if not batch_records:
                return pd.DataFrame()
            
            df = pd.DataFrame(batch_records)
            
            # Add derived features
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100], 
                                    labels=['18-30', '31-45', '46-60', '61-75', '75+'])
            df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                                       labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            
            return df
        
        # Create Dask DataFrame directly from delayed tasks
        batch_dfs = [flatten_and_create_dataframe(task) for task in delayed_tasks]
        
        # Convert to Dask DataFrame
        logger.info("Converting to Dask DataFrame...")
        dask_df = dd.from_delayed(batch_dfs)
        
        # Add noise features
        logger.info("Adding noise features...")
        np.random.seed(self.seed)
        for i in range(10):
            # Add noise features as delayed computations
            dask_df[f'noise_feature_{i}'] = dask_df.apply(
                lambda x: np.random.normal(0, 1), axis=1, meta=('noise', 'f8')
            )
        
        return dask_df

    def add_target_variable_dask(self, dask_df: dd.DataFrame, target_prevalence: float = 0.07) -> dd.DataFrame:
        """Add target variable to Dask DataFrame using delayed computation."""
        logger.info(f"Adding target variable with {target_prevalence:.1%} prevalence")
        
        def _compute_target_for_partition(partition_df: pd.DataFrame) -> pd.DataFrame:
            """Compute target variable for a single partition."""
            if len(partition_df) == 0:
                return partition_df
            
            target_series = self.generate_target_variable(partition_df, target_prevalence)
            partition_df = partition_df.copy()
            partition_df['target'] = target_series
            return partition_df
        
        # Apply target generation to each partition
        result_df = dask_df.map_partitions(_compute_target_for_partition, meta=dask_df._meta.assign(target=0))
        
        return result_df

    def generate_and_save_to_parquet(self, num_patients: int, years_per_patient: int,
                                   output_path: str, start_year: int = 2015,
                                   target_prevalence: float = 0.07,
                                   batch_size: int = 1000) -> None:
        """Generate dataset and save directly to partitioned Parquet without loading into memory."""
        logger.info(f"Generating and saving dataset to {output_path}")
        
        # Generate Dask DataFrame
        dask_df = self.generate_dask_dataset(
            num_patients=num_patients,
            years_per_patient=years_per_patient,
            start_year=start_year,
            target_prevalence=target_prevalence,
            batch_size=batch_size
        )
        
        # Add target variable
        dask_df = self.add_target_variable_dask(dask_df, target_prevalence)
        
        # Save to partitioned Parquet
        logger.info("Saving to partitioned Parquet...")
        dask_df.to_parquet(
            output_path,
            partition_on=['year'],
            engine='pyarrow',
            write_index=False,
            compression='snappy'
        )
        
        logger.info(f"Dataset saved to {output_path}")

    def get_computed_dataframe(self, dask_df: dd.DataFrame) -> pd.DataFrame:
        """Compute and return the Dask DataFrame as a Pandas DataFrame."""
        logger.info("Computing Dask DataFrame to Pandas DataFrame...")
        return dask_df.compute()

    def generate_dataset(self, num_patients: int, years_per_patient: int,
                        start_year: int = 2015, target_prevalence: float = 0.07) -> pd.DataFrame:
        """Generate complete synthetic clinical dataset (backward compatibility - returns computed DataFrame)."""
        logger.info("Using vectorized Dask implementation for data generation")
        
        # Use Dask implementation and compute result
        dask_df = self.generate_dask_dataset(
            num_patients=num_patients,
            years_per_patient=years_per_patient,
            start_year=start_year,
            target_prevalence=target_prevalence
        )
        
        # Add target variable
        dask_df = self.add_target_variable_dask(dask_df, target_prevalence)
        
        # Compute and return
        return self.get_computed_dataframe(dask_df)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate synthetic clinical data with Dask")
    parser.add_argument("--num_patients", type=int, default=50000,
                       help="Number of patients to generate")
    parser.add_argument("--years_per_patient", type=int, default=5,
                       help="Number of years per patient")
    parser.add_argument("--output_dir", type=str, default="./data/raw",
                       help="Output directory for generated data")
    parser.add_argument("--prevalence", type=float, default=0.07,
                       help="Target prevalence for positive class")
    parser.add_argument("--config", type=str, default="./config/training_config.yaml",
                       help="Configuration file")
    parser.add_argument("--use_dask", action="store_true", default=True,
                       help="Use Dask for efficient processing (default: True)")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="Batch size for Dask processing")
    
    args = parser.parse_args()
    
    # Load config if provided
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Override with config values
        data_config = config.get('data_generation', {})
        num_patients = data_config.get('num_patients', args.num_patients)
        years_per_patient = data_config.get('years_per_patient', args.years_per_patient)
        prevalence = data_config.get('target_prevalence', args.prevalence)
        batch_size = data_config.get('batch_size', args.batch_size)
        
        # Get missing value configuration
        missing_config = data_config.get('missing_values', {})
        missing_enabled = missing_config.get('enabled', True)
        missing_rates = missing_config.get('rates') if missing_enabled else None
        
    else:
        num_patients = args.num_patients
        years_per_patient = args.years_per_patient
        prevalence = args.prevalence
        batch_size = args.batch_size
        missing_rates = None
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clinical_data.parquet"
    
    # Generate data with configuration
    generator = ClinicalDataGenerator(seed=42, missing_value_rates=missing_rates)
    
    if args.use_dask:
        # Use Dask for efficient processing and direct parquet writing
        logger.info("Using Dask for efficient data generation and saving...")
        generator.generate_and_save_to_parquet(
            num_patients=num_patients,
            years_per_patient=years_per_patient,
            output_path=str(output_path),
            target_prevalence=prevalence,
            batch_size=batch_size
        )
        
        # Read back a sample for summary statistics
        logger.info("Reading sample for summary statistics...")
        sample_df = dd.read_parquet(str(output_path)).head(1000)
        
        # Create summary with available information
        summary = {
            'total_records': 'computed_lazily',
            'unique_patients': num_patients,
            'years_covered': list(range(2015, 2015 + years_per_patient)),
            'target_prevalence': prevalence,
            'features': list(sample_df.columns),
            'batch_size_used': batch_size,
            'processing_method': 'dask',
            'missing_value_configuration': {
                'enabled': missing_rates is not None,
                'rates': missing_rates if missing_rates else 'default',
                'sample_missing_counts': sample_df.isnull().sum().to_dict()
            },
            'data_generation_config': {
                'vectorized': True,
                'dask_enabled': True,
                'seed': generator.seed
            }
        }
    else:
        # Use traditional method for backward compatibility
        logger.info("Using traditional method...")
        df = generator.generate_dataset(
            num_patients=num_patients,
            years_per_patient=years_per_patient,
            target_prevalence=prevalence
        )
        
        # Save to parquet with partitioning
        df.to_parquet(output_path, index=False, engine='pyarrow',
                      partition_cols=['year'] if len(df['year'].unique()) > 1 else None)
        
        logger.info(f"Data saved to {output_path}")
        
        # Save summary statistics
        summary = {
            'total_records': len(df),
            'unique_patients': df['patient_id'].nunique(),
            'years_covered': sorted(df['year'].unique()),
            'target_prevalence': df['target'].mean(),
            'features': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'processing_method': 'pandas',
            'missing_value_configuration': {
                'enabled': missing_rates is not None,
                'rates': missing_rates if missing_rates else 'default'
            },
            'data_generation_config': {
                'vectorized': True,
                'dask_enabled': False,
                'seed': generator.seed
            }
        }
    
    summary_path = output_dir / "data_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    logger.info(f"Summary saved to {summary_path}")


def demo_dask_usage():
    """Demonstrate Dask usage for generating clinical data."""
    logger.info("=== Dask ClinicalDataGenerator Demo ===")
    
    # Initialize generator
    generator = ClinicalDataGenerator(seed=42)
    
    # Example 1: Generate Dask DataFrame (lazy evaluation)
    logger.info("\n1. Creating Dask DataFrame (lazy evaluation)...")
    dask_df = generator.generate_dask_dataset(
        num_patients=1000,
        years_per_patient=3,
        start_year=2015,
        target_prevalence=0.07,
        batch_size=250
    )
    
    # Add target variable
    dask_df = generator.add_target_variable_dask(dask_df, target_prevalence=0.07)
    
    logger.info(f"Dask DataFrame created: {dask_df}")
    logger.info(f"Columns: {list(dask_df.columns)}")
    
    # Example 2: Preview data without full computation
    logger.info("\n2. Preview first 10 rows...")
    preview = dask_df.head(10)
    print(preview[['patient_id', 'year', 'age', 'bmi', 'target']])
    
    # Example 3: Compute specific statistics
    logger.info("\n3. Computing basic statistics...")
    total_records = dask_df.shape[0].compute()
    unique_patients = dask_df['patient_id'].nunique().compute()
    target_prevalence = dask_df['target'].mean().compute()
    
    logger.info(f"Total records: {total_records}")
    logger.info(f"Unique patients: {unique_patients}")
    logger.info(f"Target prevalence: {target_prevalence:.1%}")
    
    # Example 4: Save to partitioned Parquet (efficient for large datasets)
    logger.info("\n4. Saving to partitioned Parquet...")
    output_path = "./data/raw/demo_clinical_data.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    dask_df.to_parquet(
        output_path,
        partition_on=['year'],
        engine='pyarrow',
        write_index=False,
        compression='snappy'
    )
    logger.info(f"Data saved to {output_path}")
    
    # Example 5: Direct generation and saving for large datasets
    logger.info("\n5. Direct generation and saving for large dataset (50K patients)...")
    generator.generate_and_save_to_parquet(
        num_patients=50000,
        years_per_patient=5,
        output_path="./data/raw/large_clinical_data.parquet",
        target_prevalence=0.07,
        batch_size=2000
    )
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_dask_usage()
    else:
        main()
