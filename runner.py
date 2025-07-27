#!/usr/bin/env python3
"""
TTC Benchmarking Runner - Main harness for test-time compute evaluation
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import click
import pandas as pd
from dataclasses import asdict
from dotenv import load_dotenv
from groq import AsyncGroq
from rich.console import Console
from rich.progress import Progress, TaskID
from asyncio_throttle import Throttler

from models.ttc_policies import (
    POLICY_REGISTRY, 
    DEFAULT_POLICY_CONFIGS, 
    TTCConfig, 
    GenerationMetrics
)
from evaluation.evaluator import BenchmarkEvaluator
from evaluation.metrics import calculate_quality_metrics

# Load environment variables
load_dotenv()

console = Console()


class TTCBenchmarkRunner:
    """Main runner for TTC benchmarking experiments"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.client = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))
        self.throttler = Throttler(rate_limit=int(os.getenv('MAX_CONCURRENT_REQUESTS', 50)))
        self.evaluator = BenchmarkEvaluator()
        self.results = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def run_benchmark(self, policy_name: str, benchmark_name: str) -> List[Dict[str, Any]]:
        """Run benchmark for a specific policy"""
        console.print(f"[bold blue]Running {policy_name} on {benchmark_name}[/bold blue]")
        
        # Load benchmark data
        benchmark_data = self._load_benchmark_data(benchmark_name)
        
        # Initialize policy
        policy_config = self.config['policies'].get(policy_name, {})
        ttc_config = TTCConfig(
            policy_name=policy_name,
            model_name=self.config.get('model_name', 'llama-3.3-70b-versatile'),
            max_tokens=policy_config.get('max_tokens', 1024),
            temperature=policy_config.get('temperature', 0.7),
            top_p=policy_config.get('top_p', 1.0),
            params=policy_config.get('params', {})
        )
        
        policy_class = POLICY_REGISTRY[policy_name]
        policy = policy_class(self.client, ttc_config)
        
        # Get policy-specific configuration
        policy_cfg = {**DEFAULT_POLICY_CONFIGS.get(policy_name, {}), **policy_config.get('params', {})}
        
        # Run evaluation with progress tracking
        results = []
        with Progress() as progress:
            task = progress.add_task(f"[green]Processing {benchmark_name}...", total=len(benchmark_data))
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(int(os.getenv('MAX_CONCURRENT_REQUESTS', 50)))
            
            # Process samples concurrently
            tasks = [
                self._process_sample(semaphore, policy, policy_cfg, sample, benchmark_name, progress, task)
                for sample in benchmark_data
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        # Filter out exceptions and return valid results
        valid_results = [r for r in results if not isinstance(r, Exception)]
        console.print(f"[green]Completed {len(valid_results)}/{len(benchmark_data)} samples[/green]")
        
        return valid_results
    
    async def _process_sample(
        self, 
        semaphore: asyncio.Semaphore,
        policy: Any,
        policy_cfg: Dict[str, Any],
        sample: Dict[str, Any],
        benchmark_name: str,
        progress: Progress,
        task_id: TaskID
    ) -> Dict[str, Any]:
        """Process a single benchmark sample"""
        async with semaphore:
            async with self.throttler:
                try:
                    # Generate response using TTC policy
                    response, metrics = await policy.forward(sample['prompt'], policy_cfg)
                    
                    # Calculate quality metrics
                    quality_metrics = calculate_quality_metrics(
                        response, 
                        sample['ground_truth'], 
                        benchmark_name
                    )
                    
                    # Combine all metrics
                    result = {
                        'sample_id': sample['sample_id'],
                        'policy_name': policy.config.policy_name,
                        'benchmark': benchmark_name,
                        'prompt': sample['prompt'],
                        'response': response,
                        'ground_truth': sample['ground_truth'],
                        'timestamp': time.time(),
                        **asdict(metrics),
                        **quality_metrics
                    }
                    
                    progress.advance(task_id)
                    return result
                    
                except Exception as e:
                    console.print(f"[red]Error processing sample {sample.get('sample_id', 'unknown')}: {e}[/red]")
                    progress.advance(task_id)
                    return None
    
    def _load_benchmark_data(self, benchmark_name: str) -> List[Dict[str, Any]]:
        """Load benchmark data from JSONL file"""
        data_path = Path(f"jobs/{benchmark_name}_jobs.jsonl")
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark data not found: {data_path}")
        
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        return data
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = "results/results.parquet"):
        """Save results to parquet file, appending to existing data"""
        if not results:
            console.print("[yellow]No results to save[/yellow]")
            return
        
        # Create results directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert new results to DataFrame
        new_df = pd.DataFrame(results)
        
        # Check if file exists and append if it does
        if Path(output_path).exists():
            try:
                existing_df = pd.read_parquet(output_path)
                # Combine existing and new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_parquet(output_path, index=False)
                console.print(f"[green]Results appended to {output_path} (total: {len(combined_df)} samples)[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not append to existing file, overwriting: {e}[/yellow]")
                new_df.to_parquet(output_path, index=False)
                console.print(f"[green]Results saved to {output_path}[/green]")
        else:
            new_df.to_parquet(output_path, index=False)
            console.print(f"[green]Results saved to {output_path}[/green]")
    
    async def run_all_benchmarks(self, policy_name: str) -> List[Dict[str, Any]]:
        """Run all configured benchmarks for a policy"""
        all_results = []
        
        for benchmark_name in self.config.get('benchmarks', ['crmarena', 'worfbench']):
            try:
                results = await self.run_benchmark(policy_name, benchmark_name)
                all_results.extend(results)
            except Exception as e:
                console.print(f"[red]Error running {benchmark_name}: {e}[/red]")
        
        return all_results


async def run_main(config: str, policy: str, benchmark: Optional[str], output: str, verbose: bool):
    """Async main function"""
    
    if verbose:
        console.print("[bold]TTC Benchmarking Runner[/bold]")
        console.print(f"Config: {config}")
        console.print(f"Policy: {policy}")
        console.print(f"Output: {output}")
    
    # Validate policy exists
    if policy not in POLICY_REGISTRY:
        console.print(f"[red]Unknown policy: {policy}[/red]")
        console.print(f"Available policies: {list(POLICY_REGISTRY.keys())}")
        return
    
    # Initialize runner
    runner = TTCBenchmarkRunner(config)
    
    try:
        if benchmark:
            # Run specific benchmark
            results = await runner.run_benchmark(policy, benchmark)
        else:
            # Run all benchmarks
            results = await runner.run_all_benchmarks(policy)
        
        # Save results
        runner.save_results(results, output)
        
        # Print summary
        if results:
            df = pd.DataFrame(results)
            console.print(f"\n[bold green]Summary for {policy}:[/bold green]")
            console.print(f"Total samples: {len(results)}")
            console.print(f"Average latency: {df['first_token_latency'].mean():.3f}s")
            console.print(f"Average throughput: {df['throughput'].mean():.2f} tokens/s")
            if 'exact_match' in df.columns:
                console.print(f"Average exact match: {df['exact_match'].mean():.3f}")
            if 'rouge_l' in df.columns:
                console.print(f"Average ROUGE-L: {df['rouge_l'].mean():.3f}")
        
    except Exception as e:
        console.print(f"[red]Error running benchmarks: {e}[/red]")
        raise


if __name__ == "__main__":
    # Wrapper to handle async main
    def sync_main():
        import sys
        args = sys.argv[1:]
        
        # Parse arguments manually for async execution
        config = 'config.yaml'
        policy = None
        benchmark = None
        output = 'results/results.parquet'
        verbose = False
        
        i = 0
        while i < len(args):
            if args[i] in ['-c', '--config']:
                config = args[i + 1]
                i += 2
            elif args[i] in ['-p', '--policy']:
                policy = args[i + 1]
                i += 2
            elif args[i] in ['-b', '--benchmark']:
                benchmark = args[i + 1]
                i += 2
            elif args[i] in ['-o', '--output']:
                output = args[i + 1]
                i += 2
            elif args[i] in ['-v', '--verbose']:
                verbose = True
                i += 1
            else:
                i += 1
        
        if not policy:
            console.print("[red]Error: Policy is required. Use -p or --policy[/red]")
            return
        
        asyncio.run(run_main(config, policy, benchmark, output, verbose))
    
    sync_main()
