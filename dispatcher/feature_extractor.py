"""
Feature Extraction for TTC Policy Selection
Fast, lightweight features that don't kill first-token latency
"""

import re
import string
from typing import Dict, List, Any
import numpy as np
from collections import Counter


class FeatureExtractor:
    """Extract computational features from prompts for policy selection"""
    
    def __init__(self):
        # Domain keywords for classification
        self.domain_keywords = {
            'technical': ['code', 'debug', 'error', 'api', 'database', 'server', 'algorithm', 'function'],
            'business': ['revenue', 'profit', 'market', 'customer', 'sales', 'strategy', 'analysis'],
            'creative': ['story', 'creative', 'imagine', 'design', 'art', 'write', 'poem', 'narrative'],
            'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'calculate', 'determine', 'reason'],
            'conversational': ['hello', 'help', 'please', 'thank', 'sorry', 'question', 'chat']
        }
        
        # Response format indicators
        self.format_patterns = {
            'list': [r'\d+\.', r'\d+\)', r'[-*•]', 'list', 'steps', 'points'],
            'json': ['json', '{', '}', 'object', 'key', 'value'],
            'code': ['```', 'code', 'function', 'class', 'def', 'import'],
            'table': ['table', '|', 'column', 'row', 'data'],
            'paragraph': ['explain', 'describe', 'essay', 'paragraph', 'detailed']
        }
        
        # Language patterns
        self.language_patterns = {
            'english': re.compile(r'[a-zA-Z]'),
            'numbers': re.compile(r'\d'),
            'special_chars': re.compile(r'[^\w\s]')
        }
    
    def extract_basic_features(self, prompt: str) -> Dict[str, float]:
        """Extract basic prompt characteristics"""
        features = {}
        
        # Length features
        features['prompt_length'] = len(prompt)
        features['word_count'] = len(prompt.split())
        features['char_count'] = len(prompt)
        features['avg_word_length'] = np.mean([len(word) for word in prompt.split()]) if prompt.split() else 0
        
        # Character composition
        total_chars = len(prompt)
        if total_chars > 0:
            features['alpha_ratio'] = len(re.findall(r'[a-zA-Z]', prompt)) / total_chars
            features['digit_ratio'] = len(re.findall(r'\d', prompt)) / total_chars
            features['punct_ratio'] = len(re.findall(r'[^\w\s]', prompt)) / total_chars
            features['space_ratio'] = prompt.count(' ') / total_chars
        else:
            features['alpha_ratio'] = features['digit_ratio'] = features['punct_ratio'] = features['space_ratio'] = 0
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', prompt)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # Question indicators
        features['question_marks'] = prompt.count('?')
        features['is_question'] = 1.0 if '?' in prompt else 0.0
        
        return features
    
    def extract_domain_features(self, prompt: str) -> Dict[str, float]:
        """Extract domain-specific features"""
        features = {}
        prompt_lower = prompt.lower()
        
        # Domain keyword matching
        for domain, keywords in self.domain_keywords.items():
            count = sum(1 for keyword in keywords if keyword in prompt_lower)
            features[f'domain_{domain}'] = count
            features[f'domain_{domain}_binary'] = 1.0 if count > 0 else 0.0
        
        # Calculate dominant domain
        domain_scores = {domain: features[f'domain_{domain}'] for domain in self.domain_keywords.keys()}
        max_domain = max(domain_scores, key=domain_scores.get) if any(domain_scores.values()) else 'unknown'
        
        # One-hot encode dominant domain
        for domain in self.domain_keywords.keys():
            features[f'dominant_{domain}'] = 1.0 if domain == max_domain else 0.0
        
        return features
    
    def extract_format_features(self, prompt: str) -> Dict[str, float]:
        """Extract expected response format features"""
        features = {}
        prompt_lower = prompt.lower()
        
        for format_type, patterns in self.format_patterns.items():
            score = 0
            for pattern in patterns:
                if isinstance(pattern, str):
                    score += prompt_lower.count(pattern)
                else:  # regex pattern
                    score += len(re.findall(pattern, prompt_lower))
            
            features[f'format_{format_type}'] = score
            features[f'format_{format_type}_binary'] = 1.0 if score > 0 else 0.0
        
        return features
    
    def extract_complexity_features(self, prompt: str) -> Dict[str, float]:
        """Extract prompt complexity indicators"""
        features = {}
        
        # Vocabulary complexity
        words = prompt.lower().split()
        if words:
            word_lengths = [len(word.strip(string.punctuation)) for word in words]
            features['max_word_length'] = max(word_lengths)
            features['long_words_ratio'] = sum(1 for length in word_lengths if length > 6) / len(word_lengths)
        else:
            features['max_word_length'] = 0
            features['long_words_ratio'] = 0
        
        # Syntactic complexity
        features['comma_count'] = prompt.count(',')
        features['semicolon_count'] = prompt.count(';')
        features['colon_count'] = prompt.count(':')
        features['parentheses_count'] = prompt.count('(') + prompt.count(')')
        features['bracket_count'] = prompt.count('[') + prompt.count(']')
        
        # Instruction complexity
        instruction_words = ['analyze', 'compare', 'evaluate', 'explain', 'describe', 'summarize', 'create', 'generate']
        features['instruction_count'] = sum(1 for word in instruction_words if word in prompt.lower())
        
        # Multi-step indicators
        step_patterns = [r'\d+\.', r'\d+\)', 'first', 'second', 'third', 'then', 'next', 'finally']
        features['step_indicators'] = sum(len(re.findall(pattern, prompt.lower())) for pattern in step_patterns)
        
        return features
    
    def extract_context_features(self, prompt: str, user_id: str = None, session_id: str = None) -> Dict[str, float]:
        """Extract contextual features (user history, session info)"""
        features = {}
        
        # For now, simple placeholders - in production, these would come from user/session data
        features['user_id_hash'] = hash(user_id) % 1000 if user_id else 0
        features['session_id_hash'] = hash(session_id) % 1000 if session_id else 0
        
        # Time-based features (would be populated with actual time data)
        features['hour_of_day'] = 12  # placeholder
        features['day_of_week'] = 3   # placeholder
        features['is_weekend'] = 0.0  # placeholder
        
        return features
    
    def extract_all_features(self, prompt: str, user_id: str = None, session_id: str = None) -> Dict[str, float]:
        """Extract all features for a prompt"""
        features = {}
        
        # Combine all feature types
        features.update(self.extract_basic_features(prompt))
        features.update(self.extract_domain_features(prompt))
        features.update(self.extract_format_features(prompt))
        features.update(self.extract_complexity_features(prompt))
        features.update(self.extract_context_features(prompt, user_id, session_id))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names"""
        # Generate a dummy prompt to get all feature names
        dummy_features = self.extract_all_features("dummy prompt for feature extraction")
        return list(dummy_features.keys())
    
    def features_to_array(self, features: Dict[str, float], feature_names: List[str] = None) -> np.ndarray:
        """Convert feature dict to numpy array"""
        if feature_names is None:
            feature_names = self.get_feature_names()
        
        return np.array([features.get(name, 0.0) for name in feature_names])


if __name__ == "__main__":
    # Test the feature extractor
    extractor = FeatureExtractor()
    
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Analyze the quarterly sales data and provide insights",
        "Tell me a creative story about a robot",
        "What is 2 + 2?",
        "Help me debug this API error: 500 internal server error"
    ]
    
    for prompt in test_prompts:
        features = extractor.extract_all_features(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Features extracted: {len(features)}")
        
        # Show some key features
        key_features = ['prompt_length', 'word_count', 'is_question', 'domain_technical', 'domain_analytical']
        for key in key_features:
            if key in features:
                print(f"  {key}: {features[key]}")
