#!/usr/bin/env python3
"""
TTC Benchmarking System Demo
Demonstrates the complete pipeline with sample data
"""

import asyncio
import os
import time
from pathlib import Path
import subprocess
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

class TTCDemo:
    """Demo runner for TTC benchmarking system"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "results"
        
    def print_banner(self):
        """Print demo banner"""
        banner = """
🚀 TTC Benchmarking System Demo
═══════════════════════════════════════════════════════════════

A comprehensive Test-Time Compute evaluation framework
Built for GroqCloud A100x8 + llama-3.3-70b-versatile

Features:
• 5 core TTC algorithms with variants
• AsyncIO concurrency (100 RPS)
• Real-time Streamlit dashboard
• dbt analytics pipeline
• GitHub Actions CI/CD
• Comprehensive metrics collection
        """
        console.print(Panel(banner, style="bold blue"))
    
    def check_prerequisites(self):
        """Check system prerequisites"""
        console.print("\n[bold yellow]Checking Prerequisites...[/bold yellow]")
        
        checks = [
            ("Python 3.11+", sys.version_info >= (3, 11)),
            ("Groq API Key", bool(os.getenv('GROQ_API_KEY'))),
            ("Required packages", self.check_packages()),
            ("Benchmark data", self.check_data()),
        ]
        
        table = Table(title="System Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        all_good = True
        for name, status in checks:
            status_text = "✅ OK" if status else "❌ MISSING"
            table.add_row(name, status_text)
            if not status:
                all_good = False
        
        console.print(table)
        
        if not all_good:
            console.print("\n[red]Please fix missing prerequisites before continuing[/red]")
            return False
        
        console.print("\n[green]All prerequisites satisfied![/green]")
        return True
    
    def check_packages(self):
        """Check if required packages are installed"""
        try:
            import groq
            import pandas as pd
            import streamlit
            import asyncio
            return True
        except ImportError:
            return False
    
    def check_data(self):
        """Check if benchmark data exists"""
        crm_data = self.base_dir / "jobs" / "crmarena_jobs.jsonl"
        worf_data = self.base_dir / "jobs" / "worfbench_jobs.jsonl"
        return crm_data.exists() and worf_data.exists()
    
    async def run_sample_benchmarks(self):
        """Run sample benchmarks to demonstrate the system"""
        console.print("\n[bold yellow]Running Sample Benchmarks...[/bold yellow]")
        
        # Test policies to demonstrate
        test_cases = [
            ("early_exit_28", "crmarena"),
            ("speculative_decoding", "worfbench"),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for policy, benchmark in test_cases:
                task = progress.add_task(f"Running {policy} on {benchmark}...", total=None)
                
                # Run benchmark
                cmd = [
                    sys.executable, "runner.py",
                    "--policy", policy,
                    "--benchmark", benchmark,
                    "--verbose"
                ]
                
                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=self.base_dir,
                        timeout=300  # 5 minute timeout
                    )
                    
                    if result.returncode == 0:
                        progress.update(task, description=f"✅ {policy} on {benchmark} completed")
                    else:
                        progress.update(task, description=f"❌ {policy} on {benchmark} failed")
                        console.print(f"[red]Error: {result.stderr}[/red]")
                        
                except subprocess.TimeoutExpired:
                    progress.update(task, description=f"⏰ {policy} on {benchmark} timed out")
                except Exception as e:
                    progress.update(task, description=f"❌ {policy} on {benchmark} error: {e}")
                
                # Small delay between runs
                await asyncio.sleep(1)
    
    def show_results(self):
        """Display benchmark results"""
        console.print("\n[bold yellow]Benchmark Results Summary[/bold yellow]")
        
        results_file = self.results_dir / "results.parquet"
        if not results_file.exists():
            console.print("[red]No results file found. Run benchmarks first.[/red]")
            return
        
        try:
            import pandas as pd
            df = pd.read_parquet(results_file)
            
            # Create summary table
            table = Table(title="Performance Summary")
            table.add_column("Policy", style="cyan")
            table.add_column("Benchmark", style="magenta")
            table.add_column("Samples", justify="right")
            table.add_column("Avg Latency (s)", justify="right", style="green")
            table.add_column("Throughput (tok/s)", justify="right", style="blue")
            table.add_column("ROUGE-L", justify="right", style="yellow")
            
            # Group by policy and benchmark
            summary = df.groupby(['policy_name', 'benchmark']).agg({
                'sample_id': 'count',
                'first_token_latency': 'mean',
                'throughput': 'mean',
                'rouge_l': 'mean'
            }).round(3)
            
            for (policy, benchmark), row in summary.iterrows():
                table.add_row(
                    policy,
                    benchmark,
                    str(int(row['sample_id'])),
                    f"{row['first_token_latency']:.3f}",
                    f"{row['throughput']:.2f}",
                    f"{row['rouge_l']:.3f}"
                )
            
            console.print(table)
            
            # Show key insights
            console.print("\n[bold green]Key Insights:[/bold green]")
            best_latency = summary['first_token_latency'].min()
            best_throughput = summary['throughput'].max()
            best_quality = summary['rouge_l'].max()
            
            console.print(f"• Best Latency: {best_latency:.3f}s")
            console.print(f"• Best Throughput: {best_throughput:.2f} tokens/s")
            console.print(f"• Best Quality: {best_quality:.3f} ROUGE-L")
            
        except Exception as e:
            console.print(f"[red]Error reading results: {e}[/red]")
    
    def show_dashboard_info(self):
        """Show dashboard information"""
        console.print("\n[bold yellow]Interactive Dashboard[/bold yellow]")
        
        dashboard_info = """
🎛️ Streamlit Dashboard Features:

• Real-time performance monitoring
• Interactive policy comparison
• Latency vs quality trade-offs
• Historical trend analysis
• Filterable results table

To launch the dashboard:
    streamlit run streamlit/dash.py --server.port 8501

Then visit: http://localhost:8501
        """
        
        console.print(Panel(dashboard_info, style="blue"))
    
    def show_cicd_info(self):
        """Show CI/CD pipeline information"""
        console.print("\n[bold yellow]GitHub Actions CI/CD[/bold yellow]")
        
        cicd_info = """
🔄 Automated Benchmarking Pipeline:

• Matrix execution across all policies
• 68GB model weight caching
• Automated HTML report generation
• PR comment integration
• Daily scheduled runs (2 AM UTC)

Trigger options:
    # Manual with specific policy
    gh workflow run ttc-benchmark.yml -f policy=early_exit_28
    
    # Automatic on push/PR
    git push origin main
        """
        
        console.print(Panel(cicd_info, style="green"))
    
    def show_next_steps(self):
        """Show next steps and recommendations"""
        console.print("\n[bold yellow]Next Steps & Recommendations[/bold yellow]")
        
        next_steps = """
🚀 Recommended Actions:

1. Scale Testing:
   • Run all 10+ policy variants
   • Expand to larger datasets
   • Test with different model sizes

2. Production Deployment:
   • Set up monitoring stack
   • Implement auto-scaling
   • Add caching layer

3. Advanced Features:
   • Custom policy development
   • Multi-modal support
   • A/B testing framework

4. Performance Optimization:
   • Batch size tuning
   • Pipeline parallelism
   • Request routing optimization
        """
        
        console.print(Panel(next_steps, style="cyan"))
    
    async def run_demo(self):
        """Run complete demo"""
        self.print_banner()
        
        # Check prerequisites
        if not self.check_prerequisites():
            return
        
        # Run sample benchmarks
        await self.run_sample_benchmarks()
        
        # Show results
        self.show_results()
        
        # Show additional information
        self.show_dashboard_info()
        self.show_cicd_info()
        self.show_next_steps()
        
        # Final message
        console.print("\n[bold green]Demo completed successfully! 🎉[/bold green]")
        console.print("\nFor detailed documentation, see: README.md")
        console.print("For performance analysis, see: PERFORMANCE_REPORT.md")


async def main():
    """Main demo function"""
    demo = TTCDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
