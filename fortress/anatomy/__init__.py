"""
🧠 BALANTIUM ANATOMY - Complete Living Organism
Anatomical integration of all fortress components

This package contains the complete anatomical structure:
- Brain: Strategic control and decision-making
- Immune System: Living defense organism
- Neural Network: Complete nervous system integration
- Plus all existing brain functions integrated
"""

from .brain import BALANTIUM_BRAIN, BalantiumBrain
from ..neural.neural_network import BALANTIUM_NEURAL_NETWORK, BalantiumNeuralNetwork
from ..immune.immune_system import BALANTIUM_IMMUNE, BalantiumImmuneSystem

__all__ = [
    'BALANTIUM_BRAIN',
    'BalantiumBrain', 
    'BALANTIUM_NEURAL_NETWORK',
    'BalantiumNeuralNetwork',
    'BALANTIUM_IMMUNE',
    'BalantiumImmuneSystem'
]
