"""
🛡️ BALANTIUM IMMUNE SYSTEM - Living Defense Organism
Complete biological immune system simulation

This package contains the living immune system:
- Innate immunity (macrophages, neutrophils, NK cells)
- Adaptive immunity (B-cells, T-cells, antibodies)
- Memory cells and long-term protection
- Cytokine communication system
- Complement cascade defense
"""

from .immune_system import BALANTIUM_IMMUNE, BalantiumImmuneSystem

__all__ = [
    'BALANTIUM_IMMUNE',
    'BalantiumImmuneSystem'
]
