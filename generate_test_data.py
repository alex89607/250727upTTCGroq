#!/usr/bin/env python3
"""
Generate test data for TTC benchmarking system
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_test_data():
    """Generate realistic test data for the TTC system"""
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Define policies and benchmarks
    policies = [
        'early_exit_28', 'early_exit_32',
        'speculative_decoding', 'speculative_decoding_4token',
        'dynamic_pruning_top_p', 'dynamic_pruning_entropy',
        'adaptive_kv_static', 'adaptive_kv_cosine',
        'elastic_batch_greedy', 'elastic_batch_sorted'
    ]
    
    benchmarks = ['crmarena', 'worfbench']
    
    # Generate data
    data = []
    sample_id = 1
    
    for policy in policies:
        for benchmark in benchmarks:
            # Generate 25 samples per policy-benchmark combination
            for i in range(25):
                # Simulate realistic performance characteristics
                if 'early_exit' in policy:
                    # Early exit: fast but lower quality
                    first_token_latency = np.random.normal(0.25, 0.05)
                    throughput = np.random.normal(300, 30)
                    rouge_l = np.random.normal(0.45, 0.1)
                elif 'speculative' in policy:
                    # Speculative: slower but better quality
                    first_token_latency = np.random.normal(0.8, 0.15)
                    throughput = np.random.normal(180, 25)
                    rouge_l = np.random.normal(0.65, 0.08)
                elif 'dynamic_pruning' in policy:
                    # Dynamic pruning: balanced
                    first_token_latency = np.random.normal(0.5, 0.1)
                    throughput = np.random.normal(220, 20)
                    rouge_l = np.random.normal(0.55, 0.09)
                elif 'adaptive_kv' in policy:
                    # Adaptive KV: memory efficient
                    first_token_latency = np.random.normal(0.6, 0.12)
                    throughput = np.random.normal(200, 30)
                    rouge_l = np.random.normal(0.58, 0.07)
                else:  # elastic_batch
                    # Elastic batch: throughput optimized
                    first_token_latency = np.random.normal(0.4, 0.08)
                    throughput = np.random.normal(250, 25)
                    rouge_l = np.random.normal(0.52, 0.08)
                
                # Ensure positive values
                first_token_latency = max(0.1, first_token_latency)
                throughput = max(50, throughput)
                rouge_l = max(0.0, min(1.0, rouge_l))
                
                # Calculate derived metrics
                avg_token_latency = first_token_latency * np.random.uniform(0.8, 1.2)
                generation_time = first_token_latency + avg_token_latency * np.random.randint(10, 100)
                policy_overhead = np.random.uniform(0.01, 0.05)
                
                # Quality metrics
                exact_match = 1.0 if rouge_l > 0.8 else 0.0
                precision = rouge_l * np.random.uniform(0.9, 1.1)
                recall = rouge_l * np.random.uniform(0.8, 1.0)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                rouge_1 = rouge_l * np.random.uniform(1.1, 1.3)
                rouge_2 = rouge_l * np.random.uniform(0.7, 0.9)
                
                semantic_similarity = rouge_l * np.random.uniform(0.85, 1.15)
                
                # Length metrics
                prediction_length = np.random.randint(50, 200)
                reference_length = np.random.randint(40, 180)
                length_ratio = prediction_length / reference_length if reference_length > 0 else 1.0
                length_difference = abs(prediction_length - reference_length)
                
                # Business metrics
                customer_service_coverage = np.random.uniform(0.3, 0.9)
                sales_coverage = np.random.uniform(0.2, 0.8)
                support_coverage = np.random.uniform(0.4, 0.9)
                retention_coverage = np.random.uniform(0.3, 0.7)
                overall_keyword_coverage = np.mean([customer_service_coverage, sales_coverage, support_coverage])
                analytical_coverage = np.random.uniform(0.4, 0.8)
                structure_coverage = np.random.uniform(0.5, 0.9)
                reasoning_depth = np.random.uniform(0.3, 0.8)
                
                # Create sample
                sample = {
                    'sample_id': f"{policy}_{benchmark}_{sample_id:04d}",
                    'policy_name': policy,
                    'benchmark': benchmark,
                    'prompt': f"Sample prompt for {benchmark} benchmark",
                    'response': f"Generated response using {policy} policy",
                    'ground_truth': f"Expected response for {benchmark}",
                    'timestamp': datetime.now().timestamp(),
                    
                    # Latency metrics
                    'first_token_latency': first_token_latency,
                    'avg_token_latency': avg_token_latency,
                    'generation_time': generation_time,
                    'policy_overhead': policy_overhead,
                    
                    # Throughput metrics
                    'throughput': throughput,
                    'total_tokens': int(throughput * generation_time),
                    
                    # Quality metrics
                    'exact_match': exact_match,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'rouge_1': rouge_1,
                    'rouge_2': rouge_2,
                    'rouge_l': rouge_l,
                    'semantic_similarity': semantic_similarity,
                    
                    # Length metrics
                    'prediction_length': prediction_length,
                    'reference_length': reference_length,
                    'length_ratio': length_ratio,
                    'length_difference': length_difference,
                    
                    # Business metrics
                    'customer_service_coverage': customer_service_coverage,
                    'sales_coverage': sales_coverage,
                    'support_coverage': support_coverage,
                    'retention_coverage': retention_coverage,
                    'overall_keyword_coverage': overall_keyword_coverage,
                    'analytical_coverage': analytical_coverage,
                    'structure_coverage': structure_coverage,
                    'reasoning_depth': reasoning_depth,
                }
                
                data.append(sample)
                sample_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to parquet
    df.to_parquet('results/results.parquet', index=False)
    
    print(f"Generated {len(df)} samples")
    print(f"Policies: {df['policy_name'].unique()}")
    print(f"Benchmarks: {df['benchmark'].unique()}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show summary statistics
    print("\nPerformance Summary:")
    summary = df.groupby(['policy_name', 'benchmark'])[
        ['first_token_latency', 'throughput', 'rouge_l']
    ].mean().round(3)
    print(summary)
    
    return df

if __name__ == "__main__":
    generate_test_data()
