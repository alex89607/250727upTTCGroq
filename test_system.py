#!/usr/bin/env python3
"""
System validation script for TTC benchmarking
"""

import sys
import os
from pathlib import Path
import asyncio
import json

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import yaml
        print("✓ YAML parser available")
    except ImportError as e:
        print(f"✗ YAML import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas available")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        from models.ttc_policies import POLICY_REGISTRY, DEFAULT_POLICY_CONFIGS
        print(f"✓ TTC policies loaded: {list(POLICY_REGISTRY.keys())}")
    except ImportError as e:
        print(f"✗ TTC policies import failed: {e}")
        return False
    
    try:
        from evaluation.metrics import calculate_quality_metrics
        print("✓ Evaluation metrics available")
    except ImportError as e:
        print(f"✗ Evaluation metrics import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("✗ config.yaml not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['model_name', 'benchmarks', 'policies']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required config key: {key}")
                return False
        
        print(f"✓ Configuration valid with {len(config['policies'])} policies")
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_benchmark_data():
    """Test benchmark data availability"""
    print("\nTesting benchmark data...")
    
    jobs_dir = Path("jobs")
    if not jobs_dir.exists():
        print("✗ jobs directory not found")
        return False
    
    required_files = ["crmarena_jobs.jsonl", "worfbench_jobs.jsonl"]
    for filename in required_files:
        filepath = jobs_dir / filename
        if not filepath.exists():
            print(f"✗ {filename} not found")
            return False
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    print(f"✗ {filename} is empty")
                    return False
                
                # Test first line is valid JSON
                json.loads(lines[0])
                print(f"✓ {filename} valid with {len(lines)} samples")
                
        except Exception as e:
            print(f"✗ Error reading {filename}: {e}")
            return False
    
    return True

def test_metrics():
    """Test metrics calculation"""
    print("\nTesting metrics calculation...")
    
    try:
        from evaluation.metrics import calculate_quality_metrics
        
        # Test CRM metrics
        prediction = "I apologize for the delay. Let me check your order status and provide an update."
        reference = "apologize for delay, provide tracking information, offer compensation"
        
        crm_metrics = calculate_quality_metrics(prediction, reference, 'crmarena')
        
        expected_keys = ['exact_match', 'f1', 'rouge_l', 'precision', 'recall']
        for key in expected_keys:
            if key not in crm_metrics:
                print(f"✗ Missing metric: {key}")
                return False
        
        print(f"✓ CRM metrics calculated: F1={crm_metrics['f1']:.3f}")
        
        # Test WorfBench metrics
        prediction2 = "For this architecture decision, I recommend considering scalability, maintainability, and team expertise."
        reference2 = "Consider scalability, team size, and technical complexity when choosing architecture."
        
        worf_metrics = calculate_quality_metrics(prediction2, reference2, 'worfbench')
        print(f"✓ WorfBench metrics calculated: F1={worf_metrics['f1']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        return False

async def test_policy_instantiation():
    """Test policy instantiation (without API calls)"""
    print("\nTesting policy instantiation...")
    
    try:
        from models.ttc_policies import POLICY_REGISTRY, TTCConfig
        from groq import AsyncGroq
        
        # Mock client for testing
        client = None  # We won't make actual API calls
        
        for policy_name, policy_class in POLICY_REGISTRY.items():
            try:
                config = TTCConfig(policy_name=policy_name)
                # Don't actually instantiate with client to avoid API dependency
                print(f"✓ Policy {policy_name} class available")
            except Exception as e:
                print(f"✗ Policy {policy_name} instantiation failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Policy testing failed: {e}")
        return False

def test_directory_structure():
    """Test required directory structure"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "models", "evaluation", "jobs", "dbt", "streamlit", 
        "results", ".github/workflows"
    ]
    
    for dirname in required_dirs:
        dirpath = Path(dirname)
        if not dirpath.exists():
            print(f"✗ Missing directory: {dirname}")
            return False
        print(f"✓ Directory exists: {dirname}")
    
    return True

def test_environment():
    """Test environment variables"""
    print("\nTesting environment...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file found")
        # Don't actually load API key for security
    else:
        print("⚠ .env file not found (use .env.example as template)")
    
    example_env = Path(".env.example")
    if example_env.exists():
        print("✓ .env.example template available")
    else:
        print("✗ .env.example template missing")
        return False
    
    return True

async def main():
    """Run all tests"""
    print("🚀 TTC Benchmarking System Validation\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Benchmark Data", test_benchmark_data),
        ("Metrics Calculation", test_metrics),
        ("Policy Classes", test_policy_instantiation),
        ("Directory Structure", test_directory_structure),
        ("Environment", test_environment),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! System is ready for benchmarking.")
        print("\nNext steps:")
        print("1. Add your GROQ_API_KEY to .env file")
        print("2. Run: python runner.py --policy speculative_decoding --verbose")
        print("3. Launch dashboard: cd streamlit && streamlit run dash.py")
        return True
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
        return False

if __name__ == "__main__":
    import yaml
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
