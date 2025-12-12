"""
Neural Network Verification Module

Provides Marabou-based verification functionality for neural networks.
"""

from .marabou_verifier import MarabouVerifier, verify_local_robustness

__all__ = ['MarabouVerifier', 'verify_local_robustness']
