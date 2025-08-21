#!/usr/bin/env python3
"""
Test script to validate Docker configuration.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str) -> bool:
    """Run shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {cmd}")
            return True
        else:
            print(f"❌ {cmd}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {cmd}")
        print(f"Exception: {e}")
        return False

def check_docker_files():
    """Check Docker file syntax and structure."""
    docker_dir = Path(__file__).parent
    root_dir = docker_dir.parent
    
    print("🔍 Checking Docker files...")
    
    # Check if Docker is available
    if not run_command("docker --version"):
        print("Docker is not available. Skipping Docker tests.")
        return False
    
    # Check Dockerfile syntax
    serving_dockerfile = docker_dir / "Dockerfile.serving"
    training_dockerfile = docker_dir / "Dockerfile.training"
    compose_file = docker_dir / "docker-compose.yml"
    
    print(f"\n📁 Checking files exist:")
    files_ok = True
    for file_path in [serving_dockerfile, training_dockerfile, compose_file]:
        if file_path.exists():
            print(f"✅ {file_path.name}")
        else:
            print(f"❌ {file_path.name} not found")
            files_ok = False
    
    if not files_ok:
        return False
    
    # Check required files in parent directory
    print(f"\n📁 Checking required project files:")
    required_files = [
        root_dir / "requirements.txt",
        root_dir / "src" / "serving" / "api.py",
        root_dir / "src" / "pipeline" / "training_pipeline.py",
        root_dir / "config" / "training_config.yaml",
        root_dir / "config" / "serving_config.yaml"
    ]
    
    project_files_ok = True
    for file_path in required_files:
        if file_path.exists():
            print(f"✅ {file_path.relative_to(root_dir)}")
        else:
            print(f"❌ {file_path.relative_to(root_dir)} not found")
            project_files_ok = False
    
    # Check docker-compose syntax
    print(f"\n🐳 Validating docker-compose syntax:")
    compose_cmd = f"cd {docker_dir} && docker-compose config"
    compose_ok = run_command(compose_cmd)
    
    return files_ok and project_files_ok and compose_ok

def main():
    """Main function."""
    print("🧪 Docker Configuration Test")
    print("=" * 40)
    
    success = check_docker_files()
    
    if success:
        print("\n🎉 All Docker configuration checks passed!")
        print("\nTo test the containers:")
        print("1. cd docker")
        print("2. docker-compose up --build")
        return 0
    else:
        print("\n💥 Some Docker configuration checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
