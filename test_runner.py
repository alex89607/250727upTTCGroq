#!/usr/bin/env python3
"""
Test runner for TTC benchmarking - simulates API calls for testing
"""

import asyncio
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from dataclasses import asdict

from models.ttc_policies import POLICY_REGISTRY, TTCConfig, GenerationMetrics
from evaluation.metrics import calculate_quality_metrics

class MockGroqClient:
    """Mock Groq client for testing without API calls"""
    
    def __init__(self):
        self.model = "llama-3.3-70b-versatile"
        self.chat = MockChat()

class MockChat:
    """Mock chat completions"""
    
    def __init__(self):
        self.completions = MockCompletions()

class MockCompletions:
    """Mock completions"""
    
    async def create(self, **kwargs):
        """Simulate API response"""
        await asyncio.sleep(random.uniform(0.1, 0.3))  # Simulate network delay
        
        messages = kwargs.get('messages', [])
        prompt = messages[-1].get('content', '') if messages else ''
        
        # Generate mock response based on prompt type
        if 'customer' in prompt.lower() or 'service' in prompt.lower():
            response = "Thank you for contacting us. I understand your concern and will help resolve this issue promptly. Let me check your account details and provide you with the best solution."
        elif 'architecture' in prompt.lower() or 'system' in prompt.lower():
            response = "For this system architecture decision, I recommend considering scalability, maintainability, and team expertise. A microservices approach would provide flexibility while ensuring proper separation of concerns."
        else:
            response = "Based on the requirements provided, I suggest implementing a comprehensive solution that addresses the key challenges while maintaining optimal performance and user experience."
        
        # Mock response object
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
                self.usage = MockUsage()
        
        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)
        
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        class MockUsage:
            def __init__(self):
                self.total_tokens = random.randint(50, 200)
                self.prompt_tokens = random.randint(20, 100)
                self.completion_tokens = self.total_tokens - self.prompt_tokens
        
        return MockResponse(response)

async def run_test_benchmark(policy_name: str, benchmark: str = None, num_samples: int = 5):
    """Run test benchmark with mock data"""
    print(f"🚀 Running test benchmark for policy: {policy_name}")
    
    # Load benchmark data
    if benchmark:
        benchmarks = [benchmark]
    else:
        benchmarks = ['crmarena', 'worfbench']
    
    results = []
    
    for bench_name in benchmarks:
        print(f"\n📊 Testing {bench_name} benchmark...")
        
        # Load sample data
        jobs_file = Path(f"jobs/{bench_name}_jobs.jsonl")
        if not jobs_file.exists():
            print(f"❌ Benchmark file not found: {jobs_file}")
            continue
        
        with open(jobs_file, 'r') as f:
            jobs = [json.loads(line) for line in f.readlines()[:num_samples]]
        
        # Initialize policy
        config = TTCConfig(policy_name=policy_name)
        policy_class = POLICY_REGISTRY[policy_name]
        mock_client = MockGroqClient()
        policy = policy_class(mock_client, config)
        
        print(f"✓ Initialized {policy_name} policy")
        
        # Process samples
        for i, job in enumerate(jobs):
            print(f"  Processing sample {i+1}/{len(jobs)}...", end=" ")
            
            start_time = time.monotonic()
            
            try:
                # Run policy
                response, metrics = await policy.forward(
                    job['prompt'], 
                    {}  # Empty params for testing
                )
                
                # Calculate quality metrics
                quality_metrics = calculate_quality_metrics(
                    response, 
                    job['ground_truth'], 
                    bench_name
                )
                
                # Combine results
                result = {
                    'sample_id': job['sample_id'],
                    'policy_name': policy_name,
                    'benchmark': bench_name,
                    'prompt': job['prompt'],
                    'response': response,
                    'ground_truth': job['ground_truth'],
                    'timestamp': time.time(),
                    **asdict(metrics),
                    **quality_metrics
                }
                
                results.append(result)
                print("✓")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    # Save results
    if results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        df = pd.DataFrame(results)
        output_file = results_dir / f"test_results_{policy_name}.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"\n✅ Saved {len(results)} results to {output_file}")
        
        # Print summary
        print(f"\n📈 Summary for {policy_name}:")
        print(f"  Samples processed: {len(results)}")
        print(f"  Avg first token latency: {df['first_token_latency'].mean():.3f}s")
        print(f"  Avg throughput: {df['throughput'].mean():.1f} tokens/s")
        print(f"  Avg F1 score: {df['f1'].mean():.3f}")
        print(f"  Avg ROUGE-L: {df['rouge_l'].mean():.3f}")
        
        return df
    else:
        print("❌ No results generated")
        return None

async def run_multiple_policies(policies: List[str], num_samples: int = 3):
    """Run multiple policies for comparison"""
    print(f"🔄 Running comparison test with {len(policies)} policies")
    
    all_results = []
    
    for policy in policies:
        print(f"\n{'='*50}")
        df = await run_test_benchmark(policy, num_samples=num_samples)
        if df is not None:
            all_results.append(df)
    
    if all_results:
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        results_dir = Path("results")
        combined_file = results_dir / "test_results_combined.parquet"
        combined_df.to_parquet(combined_file, index=False)
        
        print(f"\n🎉 Combined results saved to {combined_file}")
        
        # Generate comparison summary
        print(f"\n📊 COMPARISON SUMMARY:")
        print("="*60)
        
        summary = combined_df.groupby('policy_name').agg({
            'first_token_latency': 'mean',
            'throughput': 'mean',
            'f1': 'mean',
            'rouge_l': 'mean'
        }).round(3)
        
        print(summary)
        
        # Find best performers
        best_latency = summary['first_token_latency'].idxmin()
        best_throughput = summary['throughput'].idxmax()
        best_quality = summary['f1'].idxmax()
        
        print(f"\n🏆 Best Performers:")
        print(f"  Fastest (latency): {best_latency}")
        print(f"  Highest throughput: {best_throughput}")
        print(f"  Best quality (F1): {best_quality}")
        
        return combined_df
    
    return None

async def main():
    """Main test function"""
    print("🧪 TTC Benchmarking Test Suite")
    print("="*50)
    
    # Test single policy first
    print("\n1️⃣ Testing single policy...")
    await run_test_benchmark("speculative_decoding", num_samples=3)
    
    # Test multiple policies
    print("\n2️⃣ Testing multiple policies...")
    test_policies = [
        "speculative_decoding",
        "dynamic_pruning_top_p", 
        "early_exit_28"
    ]
    
    await run_multiple_policies(test_policies, num_samples=2)
    
    print("\n✅ Test suite completed!")
    print("\nNext steps:")
    print("1. Check results in results/ directory")
    print("2. Run: cd streamlit && streamlit run dash.py")
    print("3. Add real GROQ_API_KEY to .env for production testing")

if __name__ == "__main__":
    asyncio.run(main())
