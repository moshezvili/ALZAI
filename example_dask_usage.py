#!/usr/bin/env python3
"""
Example usage of the Dask-enabled ClinicalDataGenerator

This script demonstrates how to use the improved ClinicalDataGenerator
with Dask for efficient parallel computation and memory management.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_generation.generate_clinical_data import ClinicalDataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_basic_dask_usage():
    """Example 1: Basic Dask DataFrame generation and preview."""
    logger.info("=== Example 1: Basic Dask DataFrame generation ===")
    
    # Initialize generator
    generator = ClinicalDataGenerator(seed=42)
    
    # Generate Dask DataFrame (lazy evaluation - no computation yet)
    dask_df = generator.generate_dask_dataset(
        num_patients=5000,
        years_per_patient=3,
        start_year=2015,
        target_prevalence=0.07,
        batch_size=500
    )
    
    # Add target variable (still lazy)
    dask_df = generator.add_target_variable_dask(dask_df, target_prevalence=0.07)
    
    logger.info(f"Dask DataFrame structure: {dask_df}")
    logger.info(f"Columns: {list(dask_df.columns)}")
    
    # Preview first few rows (triggers partial computation)
    logger.info("\nPreview of first 5 rows:")
    preview = dask_df.head(5)
    print(preview[['patient_id', 'year', 'age', 'gender', 'bmi', 'target']])
    
    # Compute some basic statistics
    logger.info("\nComputing basic statistics...")
    total_records = dask_df.shape[0].compute()
    unique_patients = dask_df['patient_id'].nunique().compute()
    avg_age = dask_df['age'].mean().compute()
    target_prevalence = dask_df['target'].mean().compute()
    
    logger.info(f"Total records: {total_records:,}")
    logger.info(f"Unique patients: {unique_patients:,}")
    logger.info(f"Average age: {avg_age:.1f}")
    logger.info(f"Target prevalence: {target_prevalence:.1%}")
    

def example_2_efficient_parquet_saving():
    """Example 2: Efficient generation and saving to partitioned Parquet."""
    logger.info("\n=== Example 2: Efficient Parquet saving ===")
    
    # Initialize generator
    generator = ClinicalDataGenerator(seed=123)
    
    # Define output path
    output_path = "./data/raw/example_clinical_data.parquet"
    
    # Generate and save directly to Parquet (memory efficient)
    logger.info("Generating 10K patients × 5 years and saving to Parquet...")
    generator.generate_and_save_to_parquet(
        num_patients=10_000,
        years_per_patient=5,
        output_path=output_path,
        target_prevalence=0.07,
        batch_size=1000
    )
    
    logger.info(f"Data saved to {output_path}")
    
    # Read back and verify the data
    import dask.dataframe as dd
    
    logger.info("Reading back the saved data...")
    saved_df = dd.read_parquet(output_path)
    
    logger.info(f"Saved data structure: {saved_df}")
    logger.info(f"Data partitions: {saved_df.npartitions}")
    
    # Check partition structure
    logger.info("\nPartition information:")
    partition_info = saved_df.get_partition(0).head(1)
    logger.info(f"Sample from first partition: {list(partition_info.columns)}")


def example_3_comparing_performance():
    """Example 3: Compare traditional vs Dask approach."""
    logger.info("\n=== Example 3: Performance comparison ===")
    
    generator = ClinicalDataGenerator(seed=456)
    
    import time
    
    # Small dataset for comparison
    num_patients = 2000
    years_per_patient = 3
    
    # Traditional approach (loads everything into memory)
    logger.info("Testing traditional approach...")
    start_time = time.time()
    traditional_df = generator.generate_dataset(
        num_patients=num_patients,
        years_per_patient=years_per_patient,
        target_prevalence=0.07
    )
    traditional_time = time.time() - start_time
    
    logger.info(f"Traditional approach: {traditional_time:.2f}s, shape: {traditional_df.shape}")
    
    # Dask approach (lazy evaluation)
    logger.info("Testing Dask approach...")
    start_time = time.time()
    dask_df = generator.generate_dask_dataset(
        num_patients=num_patients,
        years_per_patient=years_per_patient,
        target_prevalence=0.07,
        batch_size=500
    )
    dask_df = generator.add_target_variable_dask(dask_df, target_prevalence=0.07)
    
    # Only compute when needed
    computed_df = generator.get_computed_dataframe(dask_df)
    dask_time = time.time() - start_time
    
    logger.info(f"Dask approach: {dask_time:.2f}s, shape: {computed_df.shape}")
    logger.info(f"Speedup: {traditional_time/dask_time:.1f}x")


def example_4_large_dataset_demo():
    """Example 4: Generate the requested 50K patients × 5 years dataset."""
    logger.info("\n=== Example 4: Large dataset generation (50K patients × 5 years) ===")
    
    generator = ClinicalDataGenerator(seed=789)
    
    # Generate the large dataset as requested
    output_path = "./data/raw/clinical_data.parquet"
    
    logger.info("Generating 50K patients × 5 years dataset...")
    logger.info("This demonstrates efficient memory usage with Dask...")
    
    import time
    start_time = time.time()
    
    generator.generate_and_save_to_parquet(
        num_patients=50000,
        years_per_patient=5,
        output_path=output_path,
        target_prevalence=0.07,
        batch_size=2000  # Process in batches of 2000 patients
    )
    
    generation_time = time.time() - start_time
    
    logger.info(f"Large dataset generation completed in {generation_time:.1f}s")
    logger.info(f"Data saved to {output_path}")
    
    # Verify the data
    import dask.dataframe as dd
    
    saved_df = dd.read_parquet(output_path)
    logger.info(f"Verification - Total partitions: {saved_df.npartitions}")
    
    # Sample some statistics without loading all data
    sample_stats = {
        'total_records': saved_df.shape[0].compute(),
        'unique_patients': saved_df['patient_id'].nunique().compute(),
        'years_covered': sorted(saved_df['year'].unique().compute()),
        'avg_age': saved_df['age'].mean().compute(),
        'target_prevalence': saved_df['target'].mean().compute()
    }
    
    logger.info("Dataset statistics:")
    for key, value in sample_stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        elif isinstance(value, list):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value:,}")


def main():
    """Run all examples."""
    logger.info("Starting Dask ClinicalDataGenerator examples...")
    
    # Create output directory
    Path("./data/raw").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    try:
        example_1_basic_dask_usage()
        example_2_efficient_parquet_saving()
        example_3_comparing_performance()
        example_4_large_dataset_demo()
        
        logger.info("\n=== All examples completed successfully! ===")
        
        logger.info("\nKey benefits of the Dask implementation:")
        logger.info("1. ✅ Lazy evaluation - computations happen only when needed")
        logger.info("2. ✅ Memory efficient - processes data in chunks")
        logger.info("3. ✅ Parallel processing - utilizes multiple CPU cores")
        logger.info("4. ✅ Direct Parquet writing - no intermediate memory loading")
        logger.info("5. ✅ Partitioned output - optimized for analytics workloads")
        logger.info("6. ✅ Scalable - can handle datasets larger than RAM")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
