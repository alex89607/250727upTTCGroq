"""
Quality Metrics Calculation for TTC Benchmarking
"""

import re
import string
from typing import Dict, Any, List
from collections import Counter
import evaluate
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def calculate_exact_match(prediction: str, reference: str) -> float:
    """Calculate exact match score"""
    pred_normalized = normalize_text(prediction)
    ref_normalized = normalize_text(reference)
    
    return 1.0 if pred_normalized == ref_normalized else 0.0


def calculate_token_overlap(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate token-level overlap metrics"""
    pred_tokens = set(word_tokenize(normalize_text(prediction)))
    ref_tokens = set(word_tokenize(normalize_text(reference)))
    
    if not ref_tokens:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    intersection = pred_tokens.intersection(ref_tokens)
    
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    return {
        'rouge_1': scores['rouge1'].fmeasure,
        'rouge_2': scores['rouge2'].fmeasure,
        'rouge_l': scores['rougeL'].fmeasure
    }


def calculate_semantic_similarity(prediction: str, reference: str) -> float:
    """Calculate semantic similarity using simple word overlap"""
    # This is a simplified version - in production, you'd use embeddings
    pred_words = Counter(word_tokenize(normalize_text(prediction)))
    ref_words = Counter(word_tokenize(normalize_text(reference)))
    
    # Calculate cosine similarity based on word counts
    intersection = sum((pred_words & ref_words).values())
    magnitude_pred = sum(pred_words.values())
    magnitude_ref = sum(ref_words.values())
    
    if magnitude_pred == 0 or magnitude_ref == 0:
        return 0.0
    
    return intersection / (magnitude_pred * magnitude_ref) ** 0.5


def calculate_length_metrics(prediction: str, reference: str) -> Dict[str, Any]:
    """Calculate length-based metrics"""
    pred_len = len(word_tokenize(prediction))
    ref_len = len(word_tokenize(reference))
    
    length_ratio = pred_len / ref_len if ref_len > 0 else 0.0
    length_diff = abs(pred_len - ref_len)
    
    return {
        'prediction_length': pred_len,
        'reference_length': ref_len,
        'length_ratio': length_ratio,
        'length_difference': length_diff
    }


def calculate_crm_specific_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate CRM Arena specific metrics"""
    # Check for key CRM concepts
    crm_keywords = {
        'customer_service': ['apologize', 'sorry', 'understand', 'help', 'assist', 'resolve'],
        'sales': ['value', 'benefit', 'offer', 'discount', 'price', 'cost', 'investment'],
        'support': ['technical', 'issue', 'problem', 'solution', 'fix', 'troubleshoot'],
        'retention': ['keep', 'stay', 'continue', 'loyalty', 'relationship', 'satisfaction']
    }
    
    pred_lower = prediction.lower()
    ref_lower = reference.lower()
    
    keyword_scores = {}
    for category, keywords in crm_keywords.items():
        pred_count = sum(1 for kw in keywords if kw in pred_lower)
        ref_count = sum(1 for kw in keywords if kw in ref_lower)
        
        if ref_count > 0:
            keyword_scores[f'{category}_coverage'] = min(pred_count / ref_count, 1.0)
        else:
            keyword_scores[f'{category}_coverage'] = 0.0
    
    # Overall keyword coverage
    all_keywords = [kw for kws in crm_keywords.values() for kw in kws]
    pred_total = sum(1 for kw in all_keywords if kw in pred_lower)
    ref_total = sum(1 for kw in all_keywords if kw in ref_lower)
    
    keyword_scores['overall_keyword_coverage'] = min(pred_total / ref_total, 1.0) if ref_total > 0 else 0.0
    
    return keyword_scores


def calculate_worf_specific_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate WorfBench specific metrics"""
    # Check for analytical thinking indicators
    analysis_indicators = [
        'because', 'therefore', 'however', 'although', 'consider', 'analyze',
        'recommend', 'suggest', 'approach', 'strategy', 'solution', 'trade-off',
        'advantage', 'disadvantage', 'benefit', 'risk', 'factor', 'aspect'
    ]
    
    pred_lower = prediction.lower()
    ref_lower = reference.lower()
    
    # Count analytical indicators
    pred_indicators = sum(1 for indicator in analysis_indicators if indicator in pred_lower)
    ref_indicators = sum(1 for indicator in analysis_indicators if indicator in ref_lower)
    
    analytical_coverage = min(pred_indicators / ref_indicators, 1.0) if ref_indicators > 0 else 0.0
    
    # Check for structured thinking (numbered lists, bullet points)
    structure_patterns = [
        r'\d+\)',  # 1) 2) 3)
        r'\d+\.',  # 1. 2. 3.
        r'[•\-\*]',  # bullet points
        r'first|second|third|finally',  # sequence words
    ]
    
    pred_structure = sum(1 for pattern in structure_patterns if re.search(pattern, pred_lower))
    ref_structure = sum(1 for pattern in structure_patterns if re.search(pattern, ref_lower))
    
    structure_coverage = min(pred_structure / ref_structure, 1.0) if ref_structure > 0 else 0.0
    
    return {
        'analytical_coverage': analytical_coverage,
        'structure_coverage': structure_coverage,
        'reasoning_depth': (analytical_coverage + structure_coverage) / 2
    }


def calculate_quality_metrics(prediction: str, reference: str, benchmark_name: str) -> Dict[str, Any]:
    """Calculate comprehensive quality metrics for a prediction"""
    metrics = {}
    
    # Basic metrics
    metrics['exact_match'] = calculate_exact_match(prediction, reference)
    
    # Token overlap metrics
    token_metrics = calculate_token_overlap(prediction, reference)
    metrics.update(token_metrics)
    
    # ROUGE scores
    rouge_metrics = calculate_rouge_scores(prediction, reference)
    metrics.update(rouge_metrics)
    
    # Semantic similarity
    metrics['semantic_similarity'] = calculate_semantic_similarity(prediction, reference)
    
    # Length metrics
    length_metrics = calculate_length_metrics(prediction, reference)
    metrics.update(length_metrics)
    
    # Benchmark-specific metrics
    if benchmark_name == 'crmarena':
        crm_metrics = calculate_crm_specific_metrics(prediction, reference)
        metrics.update(crm_metrics)
    elif benchmark_name == 'worfbench':
        worf_metrics = calculate_worf_specific_metrics(prediction, reference)
        metrics.update(worf_metrics)
    
    return metrics


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple results"""
    if not results:
        return {}
    
    # Get all metric keys
    metric_keys = set()
    for result in results:
        metric_keys.update(result.keys())
    
    # Remove non-numeric keys
    non_numeric_keys = {'sample_id', 'policy_name', 'benchmark', 'prompt', 'response', 'ground_truth', 'timestamp'}
    metric_keys = metric_keys - non_numeric_keys
    
    aggregated = {}
    
    for key in metric_keys:
        values = [result.get(key, 0) for result in results if isinstance(result.get(key), (int, float))]
        
        if values:
            aggregated[key] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5,
                'count': len(values)
            }
    
    return aggregated


def calculate_percentiles(values: List[float], percentiles: List[float] = [50, 90, 95, 99]) -> Dict[str, float]:
    """Calculate percentiles for a list of values"""
    if not values:
        return {}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    percentile_dict = {}
    for p in percentiles:
        index = int((p / 100) * (n - 1))
        percentile_dict[f'p{int(p)}'] = sorted_values[index]
    
    return percentile_dict


if __name__ == "__main__":
    # Test the metrics
    prediction = "I apologize for the delay in your shipment. Let me check the tracking information and provide you with an update. We can also offer expedited shipping for your next order."
    reference = "apologize for delay, provide tracking information, offer compensation"
    
    metrics = calculate_quality_metrics(prediction, reference, 'crmarena')
    
    print("Sample CRM Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    prediction2 = "For a startup with 5 engineers, I recommend starting with a monolithic architecture. This approach offers faster development cycles, easier debugging, and simpler deployment processes. The team can focus on building features rather than managing distributed systems complexity."
    reference2 = "For a 5-person startup, monolithic architecture is typically better initially. Benefits: faster development, easier debugging, simpler deployment, less operational overhead."
    
    metrics2 = calculate_quality_metrics(prediction2, reference2, 'worfbench')
    
    print("Sample WorfBench Metrics:")
    for key, value in metrics2.items():
        print(f"{key}: {value}")
