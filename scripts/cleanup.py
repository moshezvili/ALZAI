#!/usr/bin/env python3
"""
Cleanup script for the ML project workspace.
Removes temporary files, test models, and other artifacts.
"""

import os
import shutil
import glob
from pathlib import Path


def clean_workspace():
    """Clean up temporary files and directories."""
    
    # Directories to remove
    temp_dirs = [
        "models_*test*",
        "catboost_info",
        "logs",
        "data/processed",
        "data/test",
        "mlruns/.trash"
    ]
    
    # Files to remove
    temp_files = [
        "*.log",
        "*.tmp",
        "*.temp",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    removed_items = []
    
    # Remove temporary directories
    for pattern in temp_dirs:
        for path in glob.glob(pattern):
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    removed_items.append(f"Directory: {path}")
    
    # Remove temporary files
    for pattern in temp_files:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                os.remove(path)
                removed_items.append(f"File: {path}")
    
    # Clean __pycache__ directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                shutil.rmtree(pycache_path)
                removed_items.append(f"Cache: {pycache_path}")
    
    return removed_items


if __name__ == "__main__":
    print("🧹 Cleaning workspace...")
    
    removed = clean_workspace()
    
    if removed:
        print(f"✅ Removed {len(removed)} items:")
        for item in removed:
            print(f"  - {item}")
    else:
        print("✨ Workspace is already clean!")
    
    print("\n📁 Current workspace structure:")
    for item in sorted(os.listdir(".")):
        if not item.startswith("."):
            if os.path.isdir(item):
                print(f"  📂 {item}/")
            else:
                print(f"  📄 {item}")
    
    print("\n🚀 Key files:")
    print("  📄 src/pipeline/training_pipeline.py - Main ML training pipeline")
    print("  📄 src/serving/api.py - FastAPI serving endpoint")
    print("  📄 README.md - Complete documentation")
