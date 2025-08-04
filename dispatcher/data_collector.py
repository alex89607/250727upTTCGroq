"""
Data Collection and Processing for TTC Policy Selection
Processes existing benchmark results to create training data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import logging
from .feature_extractor import FeatureExtractor


class DataCollector:
    """Collect and process data from TTC benchmark results"""
    
    def __init__(self, results_path: str = "results/results.parquet"):
        self.results_path = results_path
        self.feature_extractor = FeatureExtractor()
        self.logger = logging.getLogger(__name__)
        
    def load_results(self) -> pd.DataFrame:
        """Load benchmark results from parquet file"""
        try:
            df = pd.read_parquet(self.results_path)
            self.logger.info(f"Loaded {len(df)} results from {self.results_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            return pd.DataFrame()
    
    def calculate_quality_score(self, row: pd.Series, quality_threshold: float = 0.9) -> float:
        """Calculate composite quality score for a result"""
        # Weighted combination of quality metrics
        weights = {
            'f1': 0.25,
            'rouge_l': 0.25,
            'semantic_similarity': 0.20,
            'exact_match': 0.15,
            'precision': 0.10,
            'recall': 0.05
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in row and pd.notna(row[metric]):
                score += row[metric] * weight
                total_weight += weight
        
        # Normalize by actual weights used
        if total_weight > 0:
            score = score / total_weight
        
        return score
    
    def identify_best_policies(self, df: pd.DataFrame, quality_threshold: float = 0.9) -> pd.DataFrame:
        """Identify the best policy for each prompt based on quality and latency"""
        results = []
        
        # Group by sample_id to compare policies for the same prompt
        for sample_id, group in df.groupby('sample_id'):
            if len(group) < 2:  # Need at least 2 policies to compare
                continue
            
            # Calculate quality scores
            group = group.copy()
            group['quality_score'] = group.apply(self.calculate_quality_score, axis=1)
            
            # Find policies that meet quality threshold
            best_quality = group['quality_score'].max()
            quality_candidates = group[group['quality_score'] >= quality_threshold * best_quality]
            
            if len(quality_candidates) == 0:
                # If no policy meets threshold, take the best quality
                quality_candidates = group[group['quality_score'] == best_quality]
            
            # Among quality candidates, select the one with minimum latency
            best_policy = quality_candidates.loc[quality_candidates['first_token_latency'].idxmin()]
            
            # Mark this as the best policy for this prompt
            best_policy_data = best_policy.to_dict()
            best_policy_data['is_best'] = True
            best_policy_data['quality_score'] = best_policy['quality_score']
            
            results.append(best_policy_data)
        
        return pd.DataFrame(results)
    
    def extract_features_for_training(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features and labels for training"""
        features_list = []
        labels = []
        
        # Get all unique policies
        policies = sorted(df['policy_name'].unique())
        policy_to_idx = {policy: idx for idx, policy in enumerate(policies)}
        
        self.logger.info(f"Found {len(policies)} unique policies: {policies}")
        
        for _, row in df.iterrows():
            # Extract features from prompt
            prompt_features = self.feature_extractor.extract_all_features(row['prompt'])
            
            # Add performance context features
            prompt_features['benchmark_crmarena'] = 1.0 if row['benchmark'] == 'crmarena' else 0.0
            prompt_features['benchmark_worfbench'] = 1.0 if row['benchmark'] == 'worfbench' else 0.0
            
            features_list.append(prompt_features)
            labels.append(policy_to_idx[row['policy_name']])
        
        # Convert to arrays
        feature_names = list(features_list[0].keys()) if features_list else []
        X = np.array([[features.get(name, 0.0) for name in feature_names] for features in features_list])
        y = np.array(labels)
        
        self.logger.info(f"Extracted {X.shape[0]} samples with {X.shape[1]} features")
        
        return X, y, feature_names, policies
    
    def create_training_dataset(self, quality_threshold: float = 0.9) -> Dict[str, Any]:
        """Create complete training dataset from benchmark results"""
        # Load results
        df = self.load_results()
        if df.empty:
            raise ValueError("No results data available")
        
        # Identify best policies for each prompt
        best_policies_df = self.identify_best_policies(df, quality_threshold)
        
        if best_policies_df.empty:
            raise ValueError("No best policies identified")
        
        # Extract features and labels
        X, y, feature_names, policies = self.extract_features_for_training(best_policies_df)
        
        # Create dataset
        dataset = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'policies': policies,
            'policy_to_idx': {policy: idx for idx, policy in enumerate(policies)},
            'idx_to_policy': {idx: policy for idx, policy in enumerate(policies)},
            'quality_threshold': quality_threshold,
            'n_samples': len(X),
            'n_features': len(feature_names),
            'n_policies': len(policies)
        }
        
        # Add statistics
        dataset['policy_distribution'] = {
            policy: int(np.sum(y == idx)) for policy, idx in dataset['policy_to_idx'].items()
        }
        
        self.logger.info(f"Created training dataset:")
        self.logger.info(f"  Samples: {dataset['n_samples']}")
        self.logger.info(f"  Features: {dataset['n_features']}")
        self.logger.info(f"  Policies: {dataset['n_policies']}")
        self.logger.info(f"  Policy distribution: {dataset['policy_distribution']}")
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], output_path: str = "dispatcher/training_data.json"):
        """Save training dataset to file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_dataset = dataset.copy()
        serializable_dataset['X'] = dataset['X'].tolist()
        serializable_dataset['y'] = dataset['y'].tolist()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_dataset, f, indent=2)
        
        self.logger.info(f"Saved training dataset to {output_path}")
    
    def load_dataset(self, input_path: str = "dispatcher/training_data.json") -> Dict[str, Any]:
        """Load training dataset from file"""
        with open(input_path, 'r') as f:
            dataset = json.load(f)
        
        # Convert lists back to numpy arrays
        dataset['X'] = np.array(dataset['X'])
        dataset['y'] = np.array(dataset['y'])
        
        self.logger.info(f"Loaded training dataset from {input_path}")
        return dataset
    
    def analyze_policy_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze performance characteristics of each policy"""
        analysis = {}
        
        for policy in df['policy_name'].unique():
            policy_data = df[df['policy_name'] == policy]
            
            analysis[policy] = {
                'avg_first_token_latency': policy_data['first_token_latency'].mean(),
                'avg_throughput': policy_data['throughput'].mean(),
                'avg_quality_score': policy_data.apply(self.calculate_quality_score, axis=1).mean(),
                'avg_f1': policy_data['f1'].mean(),
                'avg_rouge_l': policy_data['rouge_l'].mean(),
                'sample_count': len(policy_data)
            }
        
        return analysis


if __name__ == "__main__":
    # Test the data collector
    logging.basicConfig(level=logging.INFO)
    
    collector = DataCollector()
    
    try:
        # Create training dataset
        dataset = collector.create_training_dataset(quality_threshold=0.9)
        
        # Save dataset
        collector.save_dataset(dataset)
        
        # Analyze policy performance
        df = collector.load_results()
        analysis = collector.analyze_policy_performance(df)
        
        print("\nPolicy Performance Analysis:")
        for policy, metrics in analysis.items():
            print(f"\n{policy}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
                
    except Exception as e:
        print(f"Error: {e}")
