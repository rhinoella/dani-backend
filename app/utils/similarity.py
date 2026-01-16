"""
Utility functions for vector similarity calculations.
"""

import numpy as np
from typing import List


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical direction).
    
    Args:
        vec_a: First vector as list of floats
        vec_b: Second vector as list of floats
        
    Returns:
        Cosine similarity score between -1.0 and 1.0
        Returns 0.0 if either vector has zero magnitude
        
    Examples:
        >>> cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        1.0
        
        >>> cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        0.0
        
        >>> cosine_similarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0])
        -1.0
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))
