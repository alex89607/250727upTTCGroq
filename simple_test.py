#!/usr/bin/env python3
"""
Simple test to demonstrate TTC system functionality
"""

import json
import time
import pandas as pd
from pathlib import Path
from evaluation.metrics import calculate_quality_metrics

def test_data_loading():
    """Test loading benchmark data"""
    print("🔍 Testing data loading...")
    
    # Test CRMArena data
    crm_file = Path("jobs/crmarena_jobs.jsonl")
    if crm_file.exists():
        with open(crm_file, 'r') as f:
            crm_jobs = [json.loads(line) for line in f.readlines()[:3]]
        print(f"✓ Loaded {len(crm_jobs)} CRMArena samples")
        
        # Show sample
        sample = crm_jobs[0]
        print(f"  Sample ID: {sample['sample_id']}")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Ground truth: {sample['ground_truth'][:100]}...")
    else:
        print("❌ CRMArena data not found")
    
    # Test WorfBench data
    worf_file = Path("jobs/worfbench_jobs.jsonl")
    if worf_file.exists():
        with open(worf_file, 'r') as f:
            worf_jobs = [json.loads(line) for line in f.readlines()[:3]]
        print(f"✓ Loaded {len(worf_jobs)} WorfBench samples")
        
        # Show sample
        sample = worf_jobs[0]
        print(f"  Sample ID: {sample['sample_id']}")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Ground truth: {sample['ground_truth'][:100]}...")
    else:
        print("❌ WorfBench data not found")
    
    return crm_jobs, worf_jobs

def test_metrics_calculation():
    """Test metrics calculation"""
    print("\n📊 Testing metrics calculation...")
    
    # Mock response and ground truth
    response = "Thank you for contacting customer service. I understand your concern about the billing issue and will help resolve it promptly."
    ground_truth = "I appreciate you reaching out regarding your billing concern. Let me assist you in resolving this matter quickly."
    
    # Calculate metrics for CRMArena
    crm_metrics = calculate_quality_metrics(response, ground_truth, "crmarena")
    print(f"✓ CRMArena metrics calculated:")
    print(f"  F1 Score: {crm_metrics['f1']:.3f}")
    print(f"  ROUGE-L: {crm_metrics['rouge_l']:.3f}")
    print(f"  Exact Match: {crm_metrics['exact_match']}")
    print(f"  Semantic Similarity: {crm_metrics['semantic_similarity']:.3f}")
    
    # Calculate metrics for WorfBench
    worf_response = "The system architecture should consider scalability, maintainability, and performance requirements."
    worf_ground_truth = "A well-designed system architecture must prioritize scalability, maintainability, and optimal performance."
    
    worf_metrics = calculate_quality_metrics(worf_response, worf_ground_truth, "worfbench")
    print(f"✓ WorfBench metrics calculated:")
    print(f"  F1 Score: {worf_metrics['f1']:.3f}")
    print(f"  ROUGE-L: {worf_metrics['rouge_l']:.3f}")
    print(f"  Exact Match: {worf_metrics['exact_match']}")
    print(f"  Reasoning Depth: {worf_metrics['reasoning_depth']:.3f}")
    
    return crm_metrics, worf_metrics

def test_results_storage():
    """Test results storage"""
    print("\n💾 Testing results storage...")
    
    # Create mock results
    results = []
    for i in range(5):
        result = {
            'sample_id': f'test_{i}',
            'policy_name': 'speculative_decoding',
            'benchmark': 'crmarena',
            'prompt': f'Test prompt {i}',
            'response': f'Test response {i}',
            'ground_truth': f'Test ground truth {i}',
            'timestamp': time.time(),
            'first_token_latency': 0.1 + i * 0.02,
            'avg_token_latency': 0.05 + i * 0.01,
            'throughput': 100 - i * 5,
            'total_tokens': 50 + i * 10,
            'f1': 0.8 + i * 0.02,
            'rouge_l': 0.75 + i * 0.03,
            'exact_match': i % 2 == 0,
            'semantic_similarity': 0.85 + i * 0.01
        }
        results.append(result)
    
    # Save to parquet
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(results)
    output_file = results_dir / "test_results_demo.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"✓ Saved {len(results)} test results to {output_file}")
    
    # Show summary statistics
    print(f"📈 Summary statistics:")
    print(f"  Avg first token latency: {df['first_token_latency'].mean():.3f}s")
    print(f"  Avg throughput: {df['throughput'].mean():.1f} tokens/s")
    print(f"  Avg F1 score: {df['f1'].mean():.3f}")
    print(f"  Avg ROUGE-L: {df['rouge_l'].mean():.3f}")
    
    return df

def test_policy_registry():
    """Test policy registry"""
    print("\n🔧 Testing policy registry...")
    
    from models.ttc_policies import POLICY_REGISTRY
    
    print(f"✓ Available policies: {len(POLICY_REGISTRY)}")
    for policy_name in POLICY_REGISTRY.keys():
        print(f"  - {policy_name}")
    
    return list(POLICY_REGISTRY.keys())

def main():
    """Run all tests"""
    print("🧪 Simple TTC System Test")
    print("=" * 50)
    
    try:
        # Test 1: Data loading
        crm_jobs, worf_jobs = test_data_loading()
        
        # Test 2: Metrics calculation
        crm_metrics, worf_metrics = test_metrics_calculation()
        
        # Test 3: Results storage
        results_df = test_results_storage()
        
        # Test 4: Policy registry
        policies = test_policy_registry()
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 System Status:")
        print(f"  - Benchmark data: ✓ Loaded")
        print(f"  - Metrics calculation: ✓ Working")
        print(f"  - Results storage: ✓ Working")
        print(f"  - Policy registry: ✓ {len(policies)} policies available")
        
        print("\n🚀 Next steps:")
        print("1. Add real GROQ_API_KEY to .env file")
        print("2. Run: python runner.py --policy speculative_decoding")
        print("3. Launch dashboard: cd streamlit && streamlit run dash.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
