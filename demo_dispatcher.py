"""
TTC Policy Dispatcher Demo
Complete demonstration of the intelligent policy selection system
"""

import logging
import time
import json
from pathlib import Path
import pandas as pd

from dispatcher import TTCDispatcher, FeatureExtractor, DataCollector, PolicySelector


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demo_feature_extraction():
    """Demonstrate feature extraction capabilities"""
    print("=" * 60)
    print("FEATURE EXTRACTION DEMO")
    print("=" * 60)
    
    extractor = FeatureExtractor()
    
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers efficiently",
        "Analyze the quarterly sales performance and identify key trends",
        "Debug this API connection error: 500 internal server error",
        "Create a creative story about a robot discovering emotions",
        "What is the capital of France?",
        "Explain quantum computing concepts in simple terms for beginners"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: {prompt}")
        print("-" * 50)
        
        features = extractor.extract_all_features(prompt)
        
        # Show key features
        key_features = [
            'prompt_length', 'word_count', 'is_question', 
            'domain_technical', 'domain_analytical', 'domain_creative',
            'format_code', 'format_paragraph', 'instruction_count'
        ]
        
        for feature in key_features:
            if feature in features:
                print(f"  {feature}: {features[feature]}")
    
    print(f"\nTotal features extracted: {len(features)}")


def demo_data_collection():
    """Demonstrate data collection and processing"""
    print("\n" + "=" * 60)
    print("DATA COLLECTION DEMO")
    print("=" * 60)
    
    collector = DataCollector()
    
    # Load results
    df = collector.load_results()
    if df.empty:
        print("No benchmark results found. Please run benchmarks first.")
        return None
    
    print(f"Loaded {len(df)} benchmark results")
    print(f"Policies: {sorted(df['policy_name'].unique())}")
    print(f"Benchmarks: {sorted(df['benchmark'].unique())}")
    
    # Analyze policy performance
    analysis = collector.analyze_policy_performance(df)
    
    print("\nPolicy Performance Analysis:")
    for policy, metrics in analysis.items():
        print(f"\n{policy}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Create training dataset
    try:
        dataset = collector.create_training_dataset(quality_threshold=0.9)
        print(f"\nTraining Dataset Created:")
        print(f"  Samples: {dataset['n_samples']}")
        print(f"  Features: {dataset['n_features']}")
        print(f"  Policies: {dataset['n_policies']}")
        print(f"  Policy distribution: {dataset['policy_distribution']}")
        
        return dataset
    except Exception as e:
        print(f"Failed to create training dataset: {e}")
        return None


def demo_policy_selection(dataset):
    """Demonstrate policy selection training and inference"""
    print("\n" + "=" * 60)
    print("POLICY SELECTION DEMO")
    print("=" * 60)
    
    if dataset is None:
        print("No dataset available for training")
        return None
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_selector = PolicySelector(model_type="random_forest")
    rf_results = rf_selector.train(dataset)
    
    print("Random Forest Training Results:")
    for key, value in rf_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Train Gradient Boosting model
    print("\nTraining Gradient Boosting model...")
    gb_selector = PolicySelector(model_type="gradient_boosting")
    gb_results = gb_selector.train(dataset)
    
    print("Gradient Boosting Training Results:")
    for key, value in gb_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Choose best model (by F1 score)
    best_selector = rf_selector if rf_results['f1_score'] >= gb_results['f1_score'] else gb_selector
    best_model_type = "Random Forest" if best_selector == rf_selector else "Gradient Boosting"
    
    print(f"\nBest model: {best_model_type}")
    
    # Test predictions
    test_prompts = [
        "Write a Python function to sort a list using quicksort algorithm",
        "Analyze customer churn data and provide actionable insights",
        "Debug this JavaScript error in my React application",
        "Create a poem about artificial intelligence and humanity",
        "What are the benefits of renewable energy sources?",
        "Explain machine learning algorithms for classification tasks"
    ]
    
    print(f"\nTest Predictions using {best_model_type}:")
    for prompt in test_prompts:
        policy, confidence, probabilities = best_selector.predict(prompt)
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"  → Policy: {policy}")
        print(f"  → Confidence: {confidence:.3f}")
        
        # Show top 3 policy probabilities
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  → Top policies: {', '.join([f'{p}({prob:.3f})' for p, prob in sorted_probs])}")
    
    # Feature importance
    importance = best_selector.get_feature_importance()
    print(f"\nTop 10 Most Important Features:")
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1:2d}. {feature:<25} {score:.4f}")
    
    # Speed benchmark
    print(f"\nSpeed Benchmark:")
    speed_results = best_selector.benchmark_speed(1000)
    for key, value in speed_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return best_selector


def demo_full_dispatcher(selector):
    """Demonstrate the complete dispatcher system"""
    print("\n" + "=" * 60)
    print("FULL DISPATCHER DEMO")
    print("=" * 60)
    
    if selector is None:
        print("No trained selector available")
        return
    
    # Save the trained model
    selector.save_model()
    
    # Create dispatcher
    dispatcher = TTCDispatcher()
    
    if not dispatcher.is_ready():
        print("Training dispatcher model...")
        try:
            training_results = dispatcher.train_model()
            print("Dispatcher training completed:")
            for key, value in training_results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Dispatcher training failed: {e}")
            return
    
    # Test dispatcher
    test_scenarios = [
        {
            "prompt": "Implement a binary search algorithm in Python with error handling",
            "user_id": "user_123",
            "session_id": "session_456"
        },
        {
            "prompt": "Analyze the impact of COVID-19 on global supply chains",
            "user_id": "user_789",
            "session_id": "session_101"
        },
        {
            "prompt": "Write a creative short story about time travel paradoxes",
            "user_id": "user_123",
            "session_id": "session_456"
        },
        {
            "prompt": "How do I fix a memory leak in my C++ application?",
            "user_id": "user_456",
            "session_id": "session_789"
        }
    ]
    
    print(f"\nDispatcher Test Results:")
    for i, scenario in enumerate(test_scenarios, 1):
        result = dispatcher.dispatch(**scenario)
        
        print(f"\n{i}. Prompt: {scenario['prompt'][:50]}...")
        print(f"   → Policy: {result['policy_name']}")
        print(f"   → Confidence: {result['confidence']:.3f}")
        print(f"   → Dispatch time: {result['dispatch_time_ms']:.2f}ms")
        print(f"   → Request ID: {result['request_id']}")
        
        # Simulate adding feedback
        dispatcher.add_feedback(
            request_id=result['request_id'],
            policy_name=result['policy_name'],
            actual_metrics={
                'latency': 0.1 + (i * 0.05),  # Simulated latency
                'quality_score': 0.8 + (i * 0.02)  # Simulated quality
            }
        )
    
    # Show dispatcher statistics
    stats = dispatcher.get_stats()
    print(f"\nDispatcher Statistics:")
    for key, value in stats.items():
        if key == 'policy_usage_stats':
            print(f"  {key}:")
            for policy, count in value.items():
                print(f"    {policy}: {count}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Benchmark dispatcher performance
    print(f"\nDispatcher Performance Benchmark:")
    benchmark_results = dispatcher.benchmark_performance(500)
    for key, value in benchmark_results.items():
        if key == 'policy_distribution':
            print(f"  {key}:")
            for policy, count in value.items():
                print(f"    {policy}: {count}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def demo_api_integration():
    """Demonstrate API integration concepts"""
    print("\n" + "=" * 60)
    print("API INTEGRATION DEMO")
    print("=" * 60)
    
    print("The dispatcher can be deployed as a microservice using FastAPI.")
    print("Key endpoints:")
    print("  POST /dispatch     - Select policy for a prompt")
    print("  POST /feedback     - Add performance feedback")
    print("  GET  /stats        - Get performance statistics")
    print("  GET  /health       - Health check")
    print("  POST /benchmark    - Performance benchmarking")
    print("  POST /retrain      - Trigger model retraining")
    
    print("\nExample integration with existing runner:")
    print("""
# Before (manual policy selection):
policy_name = "speculative_decoding"  # Fixed choice
response = ttc_policies.run(prompt, policy_name)

# After (intelligent dispatch):
import requests
dispatch_response = requests.post("http://localhost:8000/dispatch", 
                                 json={"prompt": prompt})
policy_name = dispatch_response.json()["policy_name"]
response = ttc_policies.run(prompt, policy_name)

# Add feedback for online learning
requests.post("http://localhost:8000/feedback", json={
    "request_id": dispatch_response.json()["request_id"],
    "policy_name": policy_name,
    "actual_latency": response["latency"],
    "quality_score": calculate_quality(response["text"])
})
""")
    
    print("\nTo start the API server:")
    print("  python -m dispatcher.api")
    print("  or")
    print("  uvicorn dispatcher.api:app --host 0.0.0.0 --port 8000")


def main():
    """Run the complete demo"""
    setup_logging()
    
    print("TTC POLICY DISPATCHER - COMPLETE DEMO")
    print("=" * 60)
    print("This demo showcases the intelligent policy selection system")
    print("that automatically chooses the optimal TTC policy for each request.")
    print("=" * 60)
    
    # Run all demo components
    demo_feature_extraction()
    dataset = demo_data_collection()
    selector = demo_policy_selection(dataset)
    demo_full_dispatcher(selector)
    demo_api_integration()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    print("The TTC Policy Dispatcher is now ready for production use!")
    print("\nNext steps:")
    print("1. Integrate with your existing TTC policies")
    print("2. Deploy the API microservice")
    print("3. Update your runner to use intelligent dispatch")
    print("4. Monitor performance and collect feedback")
    print("5. Enable automatic retraining for continuous improvement")


if __name__ == "__main__":
    main()
