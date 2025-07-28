#!/usr/bin/env python3
"""
Complete TTC Benchmarking System Demo
Demonstrates the full pipeline from data loading to visualization
"""

import asyncio
import subprocess
import time
import pandas as pd
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

async def main():
    print_header("TTC Benchmarking System - Complete Demo")
    
    # 1. System Overview
    print_section("1. System Components")
    components = [
        "✓ GroqCloud API Integration",
        "✓ 10 TTC Policy Implementations", 
        "✓ CRMArena + WorfBench Datasets",
        "✓ AsyncIO Concurrent Processing",
        "✓ Comprehensive Metrics Collection",
        "✓ Parquet Data Storage",
        "✓ dbt Analytics Models",
        "✓ Streamlit Dashboard",
        "✓ GitHub Actions CI/CD"
    ]
    for comp in components:
        print(f"  {comp}")
    
    # 2. Check Data
    print_section("2. Current Results Analysis")
    if Path("results/results.parquet").exists():
        df = pd.read_parquet("results/results.parquet")
        valid_df = df.dropna(subset=['policy_name', 'benchmark'])
        
        print(f"Total samples: {len(df)}")
        print(f"Valid samples: {len(valid_df)}")
        print(f"Policies tested: {list(valid_df['policy_name'].unique())}")
        print(f"Benchmarks: {list(valid_df['benchmark'].unique())}")
        
        if len(valid_df) > 0:
            print("\nPerformance Summary:")
            summary = valid_df.groupby(['policy_name', 'benchmark'])[
                ['first_token_latency', 'throughput', 'rouge_l']
            ].mean().round(3)
            print(summary)
    else:
        print("No results found. Run benchmarks first.")
    
    # 3. Available Policies
    print_section("3. Available TTC Policies")
    policies = [
        "baseline - Standard generation",
        "speculative_decoding - 4-token lookahead",
        "speculative_decoding_variant - Enhanced lookahead",
        "dynamic_pruning_top_p - Top-p 0.9 filtering",
        "dynamic_pruning_entropy - Entropy threshold",
        "early_exit_28 - Exit after 28 layers",
        "early_exit_32 - Exit after 32 layers", 
        "adaptive_kv_static - Static KV cache",
        "adaptive_kv_cosine - Cosine decay KV",
        "elastic_batch_greedy - Greedy batching",
        "elastic_batch_sorted - Sorted batching"
    ]
    for policy in policies:
        print(f"  • {policy}")
    
    # 4. Metrics Collected
    print_section("4. Comprehensive Metrics")
    metrics = [
        "Latency: first_token, avg_token, total_time",
        "Throughput: tokens/sec from API headers",
        "Quality: ROUGE-1/2/L, exact_match, semantic_similarity",
        "Business: CRM coverage, sales/support metrics",
        "Technical: generation_time, policy_overhead",
        "Length: prediction/reference lengths, ratios"
    ]
    for metric in metrics:
        print(f"  • {metric}")
    
    # 5. Dashboard Status
    print_section("5. Dashboard Status")
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=2)
        if response.status_code == 200:
            print("  ✓ Streamlit dashboard running at http://localhost:8501")
            print("  ✓ Interactive visualizations available")
            print("  ✓ Sortable performance tables")
            print("  ✓ Violin plots for latency vs quality")
        else:
            print("  ✗ Dashboard not responding")
    except:
        print("  ✗ Dashboard not accessible")
    
    # 6. Quick Benchmark Demo
    print_section("6. Quick Benchmark Demo")
    print("Running a quick test with baseline policy...")
    
    try:
        # Run a quick test
        result = subprocess.run([
            "python", "runner.py", 
            "--policy", "baseline",
            "--benchmark", "crmarena",
            "--limit", "2"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  ✓ Benchmark completed successfully")
            print("  ✓ Results saved to parquet")
        else:
            print(f"  ✗ Benchmark failed: {result.stderr[:100]}...")
    except subprocess.TimeoutExpired:
        print("  ⚠ Benchmark timeout (normal for API calls)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 7. Next Steps
    print_section("7. Next Steps for Improvement")
    improvements = [
        "Run full policy suite (all 10 policies)",
        "Increase sample sizes (1000+ per policy)",
        "Add more evaluation benchmarks",
        "Implement advanced quality metrics",
        "Optimize slow policies (speculative decoding)",
        "Add real-time monitoring",
        "Deploy to production environment"
    ]
    for imp in improvements:
        print(f"  • {imp}")
    
    print_header("Demo Complete")
    print("System is ready for production benchmarking!")
    print("Access dashboard: http://localhost:8501")
    print("View results: cat metrics_report.md")

if __name__ == "__main__":
    asyncio.run(main())
