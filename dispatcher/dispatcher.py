"""
Main TTC Policy Dispatcher
Integrates feature extraction, policy selection, and performance tracking
"""

import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

from .feature_extractor import FeatureExtractor
from .policy_selector import PolicySelector
from .data_collector import DataCollector


class TTCDispatcher:
    """Main dispatcher for TTC policy selection"""
    
    def __init__(self, model_path: str = "dispatcher/policy_selector_model.pkl"):
        self.model_path = model_path
        self.policy_selector = PolicySelector()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.request_count = 0
        self.total_dispatch_time = 0.0
        self.policy_usage_stats = {}
        self.feedback_buffer = []
        
        # Load trained model if available
        if Path(model_path).exists():
            try:
                self.policy_selector.load_model(model_path)
                self.logger.info("Loaded pre-trained policy selector model")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self.logger.info("Will need to train model before use")
        else:
            self.logger.info("No pre-trained model found. Will need to train before use")
    
    def is_ready(self) -> bool:
        """Check if dispatcher is ready to make predictions"""
        return self.policy_selector.is_trained
    
    def train_model(self, results_path: str = "results/results.parquet", quality_threshold: float = 0.9):
        """Train the policy selector model"""
        self.logger.info("Training policy selector model...")
        
        # Collect and prepare training data
        collector = DataCollector(results_path)
        dataset = collector.create_training_dataset(quality_threshold)
        
        # Train model
        training_results = self.policy_selector.train(dataset)
        
        # Save trained model
        self.policy_selector.save_model(self.model_path)
        
        self.logger.info("Model training completed and saved")
        return training_results
    
    def dispatch(self, prompt: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        Main dispatch method - selects optimal TTC policy for a prompt
        
        Returns:
            Dict containing policy selection and metadata
        """
        if not self.is_ready():
            raise ValueError("Dispatcher not ready. Please train model first.")
        
        start_time = time.time()
        
        # Get policy recommendation
        policy_name, confidence, policy_probabilities = self.policy_selector.predict(
            prompt, user_id, session_id
        )
        
        dispatch_time = time.time() - start_time
        
        # Update statistics
        self.request_count += 1
        self.total_dispatch_time += dispatch_time
        
        if policy_name not in self.policy_usage_stats:
            self.policy_usage_stats[policy_name] = 0
        self.policy_usage_stats[policy_name] += 1
        
        # Prepare response
        response = {
            'policy_name': policy_name,
            'confidence': confidence,
            'policy_probabilities': policy_probabilities,
            'dispatch_time_ms': dispatch_time * 1000,
            'request_id': f"req_{self.request_count}_{int(time.time() * 1000)}",
            'timestamp': time.time(),
            'user_id': user_id,
            'session_id': session_id
        }
        
        self.logger.debug(f"Dispatched policy '{policy_name}' with confidence {confidence:.3f} in {dispatch_time*1000:.2f}ms")
        
        return response
    
    def add_feedback(self, request_id: str, policy_name: str, actual_metrics: Dict[str, float]):
        """Add performance feedback for online learning"""
        feedback = {
            'request_id': request_id,
            'policy_name': policy_name,
            'actual_metrics': actual_metrics,
            'timestamp': time.time()
        }
        
        self.feedback_buffer.append(feedback)
        
        # Also add to policy selector for online learning
        # Note: We'd need the original prompt for this, which would require storing request history
        # For now, just buffer the feedback
        
        self.logger.debug(f"Added feedback for request {request_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher performance statistics"""
        avg_dispatch_time = self.total_dispatch_time / self.request_count if self.request_count > 0 else 0
        
        stats = {
            'total_requests': self.request_count,
            'avg_dispatch_time_ms': avg_dispatch_time * 1000,
            'policy_usage_stats': self.policy_usage_stats.copy(),
            'feedback_count': len(self.feedback_buffer),
            'model_stats': self.policy_selector.get_performance_stats()
        }
        
        return stats
    
    def retrain_if_needed(self, min_feedback_count: int = 1000) -> bool:
        """Check if retraining is needed and perform it"""
        if len(self.feedback_buffer) >= min_feedback_count:
            self.logger.info(f"Retraining with {len(self.feedback_buffer)} feedback samples")
            
            # In a full implementation, you'd:
            # 1. Convert feedback to training data
            # 2. Retrain the model
            # 3. Validate performance
            # 4. Deploy new model if better
            
            # For now, just clear the buffer and log
            self.feedback_buffer.clear()
            return True
        
        return False
    
    def benchmark_performance(self, n_samples: int = 1000) -> Dict[str, Any]:
        """Benchmark dispatcher performance"""
        if not self.is_ready():
            raise ValueError("Dispatcher not ready for benchmarking")
        
        # Generate test prompts
        test_prompts = [
            f"Test prompt {i} for performance benchmarking with various content types"
            for i in range(n_samples)
        ]
        
        start_time = time.time()
        results = []
        
        for prompt in test_prompts:
            result = self.dispatch(prompt)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        dispatch_times = [r['dispatch_time_ms'] for r in results]
        policy_counts = {}
        
        for result in results:
            policy = result['policy_name']
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
        
        benchmark_results = {
            'total_time_seconds': total_time,
            'avg_dispatch_time_ms': sum(dispatch_times) / len(dispatch_times),
            'min_dispatch_time_ms': min(dispatch_times),
            'max_dispatch_time_ms': max(dispatch_times),
            'requests_per_second': n_samples / total_time,
            'policy_distribution': policy_counts,
            'n_samples': n_samples
        }
        
        return benchmark_results


class TTCPolicyRunner:
    """
    Integration with existing TTC policies
    This would integrate with your existing runner.py and models/ttc_policies.py
    """
    
    def __init__(self, dispatcher: TTCDispatcher, policies_config: Dict[str, Any] = None):
        self.dispatcher = dispatcher
        self.policies_config = policies_config or {}
        self.logger = logging.getLogger(__name__)
        
        # This would normally import and initialize your TTC policies
        # from models.ttc_policies import TTCPolicies
        # self.ttc_policies = TTCPolicies(config)
        
    def run_with_dispatch(self, prompt: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        Run inference with intelligent policy dispatch
        
        This replaces the manual policy selection in your current runner
        """
        # Get policy recommendation from dispatcher
        dispatch_result = self.dispatcher.dispatch(prompt, user_id, session_id)
        policy_name = dispatch_result['policy_name']
        
        self.logger.info(f"Dispatcher selected policy: {policy_name}")
        
        # Run inference with selected policy
        # This would integrate with your existing TTC policy execution
        start_time = time.time()
        
        # Placeholder for actual policy execution
        # response = self.ttc_policies.run(prompt, policy_name)
        response = {
            'text': f"Response generated using {policy_name} policy",
            'policy_used': policy_name,
            'generation_time': time.time() - start_time
        }
        
        # Combine dispatch info with response
        full_response = {
            **response,
            'dispatch_info': dispatch_result,
            'total_time': time.time() - start_time
        }
        
        return full_response
    
    def add_performance_feedback(self, request_id: str, policy_name: str, 
                                actual_latency: float, quality_score: float):
        """Add performance feedback to the dispatcher"""
        metrics = {
            'latency': actual_latency,
            'quality_score': quality_score,
            'timestamp': time.time()
        }
        
        self.dispatcher.add_feedback(request_id, policy_name, metrics)


if __name__ == "__main__":
    # Test the dispatcher
    logging.basicConfig(level=logging.INFO)
    
    dispatcher = TTCDispatcher()
    
    # Check if model needs training
    if not dispatcher.is_ready():
        print("Training model...")
        try:
            training_results = dispatcher.train_model()
            print("Training completed:")
            for key, value in training_results.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Training failed: {e}")
            exit(1)
    
    # Test dispatching
    test_prompts = [
        "Write a Python function to calculate prime numbers",
        "Analyze the market trends for Q4 2024",
        "Debug this JavaScript error in my web application",
        "Create a creative story about time travel",
        "Explain quantum computing in simple terms"
    ]
    
    print("\nTesting policy dispatch:")
    for prompt in test_prompts:
        result = dispatcher.dispatch(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Policy: {result['policy_name']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Dispatch time: {result['dispatch_time_ms']:.2f}ms")
    
    # Show statistics
    stats = dispatcher.get_stats()
    print(f"\nDispatcher Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Benchmark performance
    print(f"\nBenchmarking performance...")
    benchmark_results = dispatcher.benchmark_performance(100)
    print(f"Benchmark Results:")
    for key, value in benchmark_results.items():
        print(f"  {key}: {value}")
