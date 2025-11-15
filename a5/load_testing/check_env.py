#!/usr/bin/env python3
"""
Environment checker for Locust load testing setup
Verifies all required dependencies and configuration
"""

import os
import sys
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("⚠ python-dotenv not installed, skipping .env file loading")
    print("  Install with: pip install python-dotenv")

def check_file(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} NOT installed")
        return False

def check_env_var(var_name, required=True, example=None):
    """Check if an environment variable is set"""
    value = os.getenv(var_name)
    if value:
        # Mask sensitive values
        if "SECRET" in var_name or "KEY" in var_name:
            display_value = value[:10] + "..." if len(value) > 10 else "***"
        else:
            display_value = value
        print(f"✓ {var_name}: {display_value}")
        return True
    else:
        if required:
            print(f"✗ {var_name}: NOT SET")
        else:
            print(f"⚠ {var_name}: NOT SET")
        if example:
            print(f"  Example: {example}")
        return not required

def main():
    print("=" * 80)
    print("ENVIRONMENT CHECK FOR LOCUST LOAD TESTING")
    print("=" * 80)
    
    all_checks_passed = True
    
    # Check required files
    print("\n📁 Checking Required Files...")
    all_checks_passed &= check_file("locustfile.py", "Locust test file")
    all_checks_passed &= check_file(".env.example", "Environment example file")
    all_checks_passed &= check_file(".env", "Environment file")
    
    # Check Python dependencies
    print("\n📦 Checking Python Dependencies...")
    all_checks_passed &= check_package("locust")
    all_checks_passed &= check_package("psutil")
    all_checks_passed &= check_package("requests")
    check_package("dotenv")  # Optional, don't fail if missing
    
    # Check Modal credentials (optional for load testing only)
    print("\n🚀 Checking Modal Configuration (Optional)...")
    print("  ℹ️  AWS credentials are stored in Modal Secrets, not needed here!")
    print("  ℹ️  Modal tokens only needed if you're deploying from this machine")
    check_env_var("MODAL_TOKEN_ID", required=False, example="ak-xxxxx")
    check_env_var("MODAL_TOKEN_SECRET", required=False, example="as-xxxxx")
    check_env_var("MODAL_ENV", required=False, example="dev")
    
    # Check Modal endpoint (required)
    print("\n🌐 Checking Modal Endpoint...")
    all_checks_passed &= check_env_var("MODAL_ENDPOINT_URL", required=True, 
                                       example="https://your-username--app-name.modal.run")
    
    # Check video URLs
    print("\n🎥 Checking Test Video URLs...")
    all_checks_passed &= check_env_var("LOW_FRAME_VIDEO_URL", required=True)
    all_checks_passed &= check_env_var("HIGH_FRAME_VIDEO_URL", required=True)
    
    # Check load test configuration (optional)
    print("\n⚙️  Checking Load Test Configuration (Optional)...")
    check_env_var("LOCUST_USERS", required=False, example="5")
    check_env_var("LOCUST_SPAWN_RATE", required=False, example="1")
    check_env_var("LOCUST_RUN_TIME", required=False, example="10m")
    
    # Summary
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("✅ ALL REQUIRED CHECKS PASSED!")
        print("\nYou can now run Locust:")
        print("  locust -f locustfile.py --host=$MODAL_ENDPOINT_URL")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running load tests.")
        print("See README.md for detailed setup instructions.")
    print("=" * 80)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
