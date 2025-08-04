"""
TTC Policy Dispatcher Package
Intelligent policy selection for Time-to-Content optimization
"""

from .feature_extractor import FeatureExtractor
from .data_collector import DataCollector
from .policy_selector import PolicySelector
from .dispatcher import TTCDispatcher, TTCPolicyRunner

__version__ = "1.0.0"
__all__ = [
    "FeatureExtractor",
    "DataCollector", 
    "PolicySelector",
    "TTCDispatcher",
    "TTCPolicyRunner"
]
