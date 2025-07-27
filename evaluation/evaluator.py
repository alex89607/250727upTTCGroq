"""
Benchmark Evaluator for CRMArena and WorfBench
"""

import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from dataclasses import dataclass


@dataclass
class BenchmarkSample:
    """Standard format for benchmark samples"""
    sample_id: str
    prompt: str
    ground_truth: str
    metadata: Dict[str, Any] = None


class BenchmarkEvaluator:
    """Evaluator for different benchmark suites"""
    
    def __init__(self):
        self.benchmark_configs = {
            'crmarena': {
                'metric_type': 'exact_match',
                'preprocessing': self._preprocess_crm_response
            },
            'worfbench': {
                'metric_type': 'rouge_l',
                'preprocessing': self._preprocess_worf_response
            }
        }
    
    def load_benchmark_data(self, benchmark_name: str, data_path: Optional[str] = None) -> List[BenchmarkSample]:
        """Load benchmark data from file"""
        if data_path is None:
            data_path = f"jobs/{benchmark_name}_jobs.jsonl"
        
        samples = []
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                sample = BenchmarkSample(
                    sample_id=data['sample_id'],
                    prompt=data['prompt'],
                    ground_truth=data['ground_truth'],
                    metadata=data.get('metadata', {})
                )
                samples.append(sample)
        
        return samples
    
    def _preprocess_crm_response(self, response: str) -> str:
        """Preprocess CRM Arena responses for evaluation"""
        # Remove common prefixes and clean up
        response = response.strip()
        
        # Extract the main answer (often after "Answer:" or similar)
        patterns = [
            r"(?:Answer|Response|Solution):\s*(.+)",
            r"(?:The answer is|The result is)\s*(.+)",
            r"^(.+?)(?:\n|$)"  # First line if no clear pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                response = match.group(1).strip()
                break
        
        # Clean up common artifacts
        response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
        response = response.rstrip('.')  # Remove trailing periods
        
        return response.lower()
    
    def _preprocess_worf_response(self, response: str) -> str:
        """Preprocess WorfBench responses for evaluation"""
        # WorfBench typically requires longer-form responses
        response = response.strip()
        
        # Remove markdown formatting
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Bold
        response = re.sub(r'\*(.*?)\*', r'\1', response)      # Italic
        response = re.sub(r'`(.*?)`', r'\1', response)        # Code
        
        # Normalize whitespace
        response = re.sub(r'\s+', ' ', response)
        
        return response
    
    def create_sample_jobs(self, benchmark_name: str, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Create sample job files for benchmarks"""
        if benchmark_name == 'crmarena':
            return self._create_crm_samples(num_samples)
        elif benchmark_name == 'worfbench':
            return self._create_worf_samples(num_samples)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    def _create_crm_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Create CRM Arena sample data"""
        samples = []
        
        # Sample CRM-style questions (customer service, sales, support)
        crm_templates = [
            {
                "prompt": "A customer is complaining about a delayed shipment. They ordered on {date} and expected delivery by {expected_date}. How should I respond?",
                "ground_truth": "apologize for delay, provide tracking information, offer compensation",
                "category": "customer_service"
            },
            {
                "prompt": "A lead is interested in our premium package but concerned about the price. They mentioned budget constraints. What's the best approach?",
                "ground_truth": "acknowledge concerns, highlight value proposition, offer flexible payment options",
                "category": "sales"
            },
            {
                "prompt": "Customer reports login issues with our platform. They've tried resetting password twice. Next steps?",
                "ground_truth": "escalate to technical support, check account status, provide alternative access method",
                "category": "technical_support"
            },
            {
                "prompt": "A client wants to upgrade their subscription but is confused about the different tiers. How do I help them choose?",
                "ground_truth": "assess current usage, explain tier differences, recommend based on needs",
                "category": "account_management"
            },
            {
                "prompt": "Customer received damaged product and wants immediate replacement. They're threatening to cancel their subscription.",
                "ground_truth": "immediate replacement, expedited shipping, retention offer, follow-up",
                "category": "retention"
            }
        ]
        
        import random
        from datetime import datetime, timedelta
        
        for i in range(num_samples):
            template = random.choice(crm_templates)
            
            # Fill in template variables
            prompt = template["prompt"]
            if "{date}" in prompt:
                order_date = datetime.now() - timedelta(days=random.randint(3, 14))
                expected_date = order_date + timedelta(days=random.randint(2, 7))
                prompt = prompt.format(
                    date=order_date.strftime("%Y-%m-%d"),
                    expected_date=expected_date.strftime("%Y-%m-%d")
                )
            
            sample = {
                "sample_id": f"crm_{i:04d}",
                "prompt": prompt,
                "ground_truth": template["ground_truth"],
                "metadata": {
                    "category": template["category"],
                    "difficulty": random.choice(["easy", "medium", "hard"]),
                    "priority": random.choice(["low", "medium", "high"])
                }
            }
            samples.append(sample)
        
        return samples
    
    def _create_worf_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Create WorfBench sample data"""
        samples = []
        
        # Sample WorfBench-style questions (reasoning, analysis, complex tasks)
        worf_templates = [
            {
                "prompt": "Analyze the following business scenario and provide strategic recommendations: A SaaS company has 10,000 users, 15% churn rate, and $50 ARPU. They want to expand to enterprise market. What should they consider?",
                "ground_truth": "The company should focus on reducing churn rate first, as 15% is high for SaaS. For enterprise expansion, they need dedicated sales team, enterprise features like SSO and compliance, longer contract terms, and higher price points. Current metrics suggest solid foundation but retention needs improvement before scaling.",
                "category": "business_analysis"
            },
            {
                "prompt": "Explain the trade-offs between microservices and monolithic architecture for a startup with 5 engineers building a social media platform.",
                "ground_truth": "For a 5-person startup, monolithic architecture is typically better initially. Benefits: faster development, easier debugging, simpler deployment, less operational overhead. Microservices add complexity that small teams can't handle effectively. Consider microservices later when team grows beyond 15-20 engineers and clear service boundaries emerge.",
                "category": "technical_architecture"
            },
            {
                "prompt": "A machine learning model shows 95% accuracy on training data but 70% on validation data. The training set has 10,000 samples, validation has 1,000. What's happening and how to fix it?",
                "ground_truth": "This indicates overfitting - the model memorized training data rather than learning generalizable patterns. Solutions: 1) Regularization (L1/L2, dropout), 2) More training data, 3) Simpler model architecture, 4) Cross-validation, 5) Early stopping, 6) Data augmentation. The large gap between train/validation accuracy is the key indicator.",
                "category": "machine_learning"
            },
            {
                "prompt": "Design a database schema for an e-commerce platform that needs to handle products, orders, customers, inventory, and reviews. Consider scalability and performance.",
                "ground_truth": "Core tables: Users, Products, Categories, Orders, OrderItems, Reviews, Inventory. Key considerations: 1) Separate order items from orders for flexibility, 2) Index on frequently queried fields (user_id, product_id, created_at), 3) Consider partitioning orders by date, 4) Use separate inventory table for stock tracking, 5) Implement soft deletes for data integrity, 6) Consider read replicas for product catalog queries.",
                "category": "system_design"
            }
        ]
        
        import random
        
        for i in range(num_samples):
            template = random.choice(worf_templates)
            
            sample = {
                "sample_id": f"worf_{i:04d}",
                "prompt": template["prompt"],
                "ground_truth": template["ground_truth"],
                "metadata": {
                    "category": template["category"],
                    "complexity": random.choice(["low", "medium", "high"]),
                    "domain": template["category"].split("_")[0]
                }
            }
            samples.append(sample)
        
        return samples
    
    def save_benchmark_jobs(self, benchmark_name: str, samples: List[Dict[str, Any]], output_path: Optional[str] = None):
        """Save benchmark samples to JSONL file"""
        if output_path is None:
            output_path = f"jobs/{benchmark_name}_jobs.jsonl"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    # Generate sample benchmark data
    evaluator = BenchmarkEvaluator()
    
    # Create CRM Arena samples
    crm_samples = evaluator.create_sample_jobs('crmarena', 50)
    evaluator.save_benchmark_jobs('crmarena', crm_samples)
    
    # Create WorfBench samples
    worf_samples = evaluator.create_sample_jobs('worfbench', 50)
    evaluator.save_benchmark_jobs('worfbench', worf_samples)
    
    print("Sample benchmark data generated!")
