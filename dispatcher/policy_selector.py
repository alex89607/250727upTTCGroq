"""
Lightweight Policy Selector for TTC Optimization
Fast classifier to select optimal TTC policy based on prompt features
"""

import numpy as np
import pickle
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

from .feature_extractor import FeatureExtractor
from .data_collector import DataCollector


class PolicySelector:
    """Lightweight classifier for TTC policy selection"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.feature_names = []
        self.policies = []
        self.policy_to_idx = {}
        self.idx_to_policy = {}
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.prediction_times = []
        self.online_feedback = []
        
    def _create_model(self) -> Any:
        """Create the appropriate model based on model_type"""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=50,  # Keep small for speed
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, dataset: Dict[str, Any], test_size: float = 0.2) -> Dict[str, Any]:
        """Train the policy selector model"""
        X = dataset['X']
        y = dataset['y']
        self.feature_names = dataset['feature_names']
        self.policies = dataset['policies']
        self.policy_to_idx = dataset['policy_to_idx']
        self.idx_to_policy = dataset['idx_to_policy']
        
        self.logger.info(f"Training {self.model_type} with {len(X)} samples")
        
        # Split data - use stratify only if all classes have at least 2 samples
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = np.min(counts)
        
        if min_class_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # Don't use stratify if some classes have only 1 sample
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        
        # Training results
        results = {
            'training_time': training_time,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': len(self.feature_names),
            'n_policies': len(self.policies)
        }
        
        self.logger.info(f"Training completed:")
        self.logger.info(f"  Training accuracy: {train_score:.4f}")
        self.logger.info(f"  Test accuracy: {test_score:.4f}")
        self.logger.info(f"  F1 score: {f1:.4f}")
        self.logger.info(f"  CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def predict(self, prompt: str, user_id: str = None, session_id: str = None) -> Tuple[str, float, Dict[str, float]]:
        """Predict the best policy for a given prompt"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        start_time = time.time()
        
        # Extract features
        features = self.feature_extractor.extract_all_features(prompt, user_id, session_id)
        feature_array = np.array([features.get(name, 0.0) for name in self.feature_names]).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction_idx = self.model.predict(feature_array_scaled)[0]
        prediction_proba = self.model.predict_proba(feature_array_scaled)[0]
        
        # Get policy name and confidence
        policy_name = self.idx_to_policy[prediction_idx]
        confidence = prediction_proba[prediction_idx]
        
        # Get probabilities for all policies
        policy_probabilities = {
            self.idx_to_policy[idx]: prob 
            for idx, prob in enumerate(prediction_proba)
        }
        
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        return policy_name, confidence, policy_probabilities
    
    def predict_batch(self, prompts: List[str]) -> List[Tuple[str, float, Dict[str, float]]]:
        """Predict policies for a batch of prompts"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        results = []
        for prompt in prompts:
            result = self.predict(prompt)
            results.append(result)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = {
            name: importance 
            for name, importance in zip(self.feature_names, self.model.feature_importances_)
        }
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def add_online_feedback(self, prompt: str, predicted_policy: str, actual_metrics: Dict[str, float]):
        """Add online feedback for model improvement"""
        feedback = {
            'prompt': prompt,
            'predicted_policy': predicted_policy,
            'actual_metrics': actual_metrics,
            'timestamp': time.time()
        }
        
        self.online_feedback.append(feedback)
        
        # Keep only recent feedback (last 10000 samples)
        if len(self.online_feedback) > 10000:
            self.online_feedback = self.online_feedback[-10000:]
    
    def retrain_with_feedback(self, min_feedback_samples: int = 1000) -> bool:
        """Retrain model with online feedback"""
        if len(self.online_feedback) < min_feedback_samples:
            self.logger.info(f"Not enough feedback samples ({len(self.online_feedback)} < {min_feedback_samples})")
            return False
        
        # This is a simplified version - in production, you'd implement proper online learning
        self.logger.info(f"Online retraining with {len(self.online_feedback)} feedback samples")
        
        # For now, just log that retraining would happen
        # In production, you'd:
        # 1. Convert feedback to training data
        # 2. Combine with original training data
        # 3. Retrain model
        # 4. Validate performance
        
        return True
    
    def save_model(self, model_path: str = "dispatcher/policy_selector_model.pkl"):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'policies': self.policies,
            'policy_to_idx': self.policy_to_idx,
            'idx_to_policy': self.idx_to_policy,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "dispatcher/policy_selector_model.pkl"):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.policies = model_data['policies']
        self.policy_to_idx = model_data['policy_to_idx']
        self.idx_to_policy = model_data['idx_to_policy']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.prediction_times:
            return {}
        
        return {
            'avg_prediction_time_ms': np.mean(self.prediction_times) * 1000,
            'max_prediction_time_ms': np.max(self.prediction_times) * 1000,
            'min_prediction_time_ms': np.min(self.prediction_times) * 1000,
            'total_predictions': len(self.prediction_times),
            'feedback_samples': len(self.online_feedback)
        }
    
    def benchmark_speed(self, n_samples: int = 1000) -> Dict[str, float]:
        """Benchmark prediction speed"""
        if not self.is_trained:
            raise ValueError("Model must be trained before benchmarking")
        
        # Generate test prompts
        test_prompts = [
            f"Test prompt {i} with some content to analyze" 
            for i in range(n_samples)
        ]
        
        start_time = time.time()
        
        for prompt in test_prompts:
            self.predict(prompt)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / n_samples) * 1000
        
        return {
            'total_time_seconds': total_time,
            'avg_prediction_time_ms': avg_time_ms,
            'predictions_per_second': n_samples / total_time,
            'n_samples': n_samples
        }


if __name__ == "__main__":
    # Test the policy selector
    logging.basicConfig(level=logging.INFO)
    
    # Create data collector and generate training data
    collector = DataCollector()
    
    try:
        # Load or create training dataset
        try:
            dataset = collector.load_dataset()
        except FileNotFoundError:
            print("Creating new training dataset...")
            dataset = collector.create_training_dataset()
            collector.save_dataset(dataset)
        
        # Train model
        selector = PolicySelector(model_type="random_forest")
        results = selector.train(dataset)
        
        print(f"\nTraining Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        # Test predictions
        test_prompts = [
            "Write a Python function to sort a list",
            "Analyze the quarterly sales performance",
            "Debug this API connection error",
            "Create a creative story about space travel"
        ]
        
        print(f"\nTest Predictions:")
        for prompt in test_prompts:
            policy, confidence, probabilities = selector.predict(prompt)
            print(f"\nPrompt: {prompt[:50]}...")
            print(f"  Predicted policy: {policy}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  All probabilities: {probabilities}")
        
        # Feature importance
        importance = selector.get_feature_importance()
        print(f"\nTop 10 Most Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        # Speed benchmark
        speed_results = selector.benchmark_speed(100)
        print(f"\nSpeed Benchmark:")
        for key, value in speed_results.items():
            print(f"  {key}: {value}")
        
        # Save model
        selector.save_model()
        print(f"\nModel saved successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
