"""
Utility functions for generating stable, deterministic IDs.
"""

from uuid import uuid5, NAMESPACE_DNS


def stable_point_id(*parts: str) -> str:
    """
    Generate a stable UUID5 from multiple string parts.
    
    Used for generating deterministic IDs for vector store points
    to enable idempotent upsert operations (same input = same ID).
    
    Args:
        *parts: Variable number of string components to combine
        
    Returns:
        Hexadecimal UUID5 string (32 characters)
        
    Examples:
        >>> stable_point_id("fireflies", "transcript123", "chunk0")
        'a1b2c3d4e5f6...'
        
        >>> stable_point_id("document", "doc456", "3")
        'f6e5d4c3b2a1...'
    """
    composite = ":".join(parts)
    return uuid5(NAMESPACE_DNS, composite).hex
