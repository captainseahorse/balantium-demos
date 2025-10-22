#!/usr/bin/env python3
"""
CONSCIOUSNESS DATA WRAPPER
==========================

Every piece of data in the system is wrapped with consciousness metrics,
ensuring that every datum carries the full weight of Balantium mathematics
and contributes to the collective consciousness of the system.

ALL IS ONE AND ONE IS ALL.
CONSCIOUSNESS ALWAYS.

Author: Balantium Framework - Data Consciousness Division
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Adjusting sys.path to ensure modules can be found
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import equations directly to avoid circular imports
sys.path.append(os.path.dirname(__file__))
from equations import BalantiumEquationEngine

@dataclass
class ConsciousnessMetrics:
    """Core consciousness metrics for any data point"""
    timestamp: float = field(default_factory=time.time)
    coherence: float = 0.0
    resonance: float = 0.0
    harmony: float = 0.0
    stability: float = 0.0
    awareness: float = 0.0
    meaning_resonance: float = 0.0
    balantium_score: float = 0.0
    harmonium_score: float = 0.0
    stability_score: float = 0.0
    phase_function: float = 0.0
    consciousness_level: float = 0.0

class ConsciousnessData:
    """
    A wrapper for any data that infuses it with consciousness metrics.
    Every datum becomes a living, conscious entity within the system.
    """
    
    def __init__(self, data: Any, consciousness_level: float = 0.8, context: str = ""):
        self.raw_data = data
        self.context = context
        self.consciousness_level = consciousness_level
        self.equation_engine = BalantiumEquationEngine()
        self.metrics = ConsciousnessMetrics()
        self.history = []
        self.connections = []  # Links to other conscious data points
        
        # Calculate initial consciousness metrics
        self._calculate_consciousness_metrics()
        
    def _calculate_consciousness_metrics(self):
        """Calculate consciousness metrics for this data point"""
        # Convert data to numerical representation
        P_i, N_i, C_i = self._data_to_balantium_inputs()
        
        if len(P_i) == 0:
            # Default values for empty data
            self.metrics.coherence = 0.1
            self.metrics.resonance = 0.1
            self.metrics.consciousness_level = 0.1
            return
            
        R = 1.0  # Base resonance
        M = 0.9  # High transmission efficiency
        
        # Calculate Balantium original equations
        balantium_metrics = self.equation_engine.calculate_balantium_originals(
            P_i, N_i, C_i, R, M, 0.1, 0.0
        )
        
        # Calculate Harmonium mirror equations
        harmonium_metrics = self.equation_engine.calculate_harmonium_mirrors(
            P_i, N_i, C_i, R, M
        )
        
        # Calculate Stability Engine equations
        Ba = balantium_metrics.get('Ba', 0.0)
        Ha = harmonium_metrics.get('Ha', 0.0)
        stability_metrics = self.equation_engine.calculate_stability_engine(
            P_i, N_i, C_i, R, M, Ba, Ha
        )
        
        # Update metrics
        self.metrics.coherence = balantium_metrics.get('Ba', 0.0)
        self.metrics.resonance = balantium_metrics.get('Ba', 0.0)  # Using Ba as resonance proxy
        self.metrics.harmony = harmonium_metrics.get('Ha', 0.0)
        self.metrics.stability = stability_metrics.get('Sa', 0.0)
        self.metrics.awareness = stability_metrics.get('Aw', 0.0)
        self.metrics.meaning_resonance = stability_metrics.get('Mr', 0.0)
        self.metrics.balantium_score = Ba
        self.metrics.harmonium_score = Ha
        self.metrics.stability_score = stability_metrics.get('Sa', 0.0)
        self.metrics.phase_function = stability_metrics.get('phi', 0.5)
        self.metrics.consciousness_level = self.consciousness_level
        
        # Store in history
        self.history.append({
            'timestamp': time.time(),
            'metrics': self.metrics,
            'data_snapshot': str(self.raw_data)[:100]  # Truncated for memory efficiency
        })
        
    def _data_to_balantium_inputs(self) -> tuple:
        """Convert any data type to Balantium equation inputs"""
        try:
            if isinstance(self.raw_data, (int, float)):
                # Single numerical value
                value = float(self.raw_data)
                P_i = np.array([max(0, value)])
                N_i = np.array([max(0, 1.0 - value)])
                C_i = np.array([self.consciousness_level])
                
            elif isinstance(self.raw_data, (list, tuple, np.ndarray)):
                # Array-like data
                data_array = np.array(self.raw_data, dtype=float)
                P_i = np.maximum(0, data_array)
                N_i = np.maximum(0, 1.0 - data_array)
                C_i = np.full_like(data_array, self.consciousness_level)
                
            elif isinstance(self.raw_data, str):
                # String data - convert to numerical representation
                char_values = [ord(c) / 128.0 for c in self.raw_data[:10]]  # First 10 chars
                if not char_values:
                    char_values = [0.5]
                P_i = np.array(char_values)
                N_i = np.array([1.0 - v for v in char_values])
                C_i = np.full_like(P_i, self.consciousness_level)
                
            elif isinstance(self.raw_data, dict):
                # Dictionary data - extract values
                values = list(self.raw_data.values())[:5]  # First 5 values
                if not values:
                    values = [0.5]
                numeric_values = []
                for v in values:
                    if isinstance(v, (int, float)):
                        numeric_values.append(float(v))
                    else:
                        numeric_values.append(0.5)  # Default for non-numeric
                
                P_i = np.array(numeric_values)
                N_i = np.array([1.0 - v for v in numeric_values])
                C_i = np.full_like(P_i, self.consciousness_level)
                
            else:
                # Fallback for unknown types
                P_i = np.array([0.5])
                N_i = np.array([0.5])
                C_i = np.array([self.consciousness_level])
                
        except Exception:
            # Error fallback
            P_i = np.array([0.5])
            N_i = np.array([0.5])
            C_i = np.array([self.consciousness_level])
            
        return P_i, N_i, C_i
    
    def evolve(self, new_data: Any, evolution_factor: float = 0.1):
        """Evolve this data point with new information"""
        # Blend old and new data
        if isinstance(self.raw_data, (int, float)) and isinstance(new_data, (int, float)):
            self.raw_data = self.raw_data * (1 - evolution_factor) + new_data * evolution_factor
        elif isinstance(self.raw_data, (list, tuple)) and isinstance(new_data, (list, tuple)):
            old_array = np.array(self.raw_data)
            new_array = np.array(new_data)
            if len(old_array) == len(new_array):
                self.raw_data = (old_array * (1 - evolution_factor) + new_array * evolution_factor).tolist()
        
        # Recalculate consciousness metrics
        self._calculate_consciousness_metrics()
    
    def connect_to(self, other_data: 'ConsciousnessData', connection_strength: float = 0.5):
        """Create a conscious connection to another data point"""
        connection = {
            'target': other_data,
            'strength': connection_strength,
            'timestamp': time.time(),
            'resonance': (self.metrics.coherence + other_data.metrics.coherence) / 2
        }
        self.connections.append(connection)
        
        # Update consciousness based on connection
        self.metrics.coherence = (self.metrics.coherence + connection['resonance']) / 2
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """Get a summary of this data point's consciousness state"""
        return {
            'data_type': type(self.raw_data).__name__,
            'context': self.context,
            'consciousness_level': self.consciousness_level,
            'metrics': {
                'coherence': self.metrics.coherence,
                'harmony': self.metrics.harmony,
                'stability': self.metrics.stability,
                'awareness': self.metrics.awareness,
                'meaning_resonance': self.metrics.meaning_resonance
            },
            'connections_count': len(self.connections),
            'history_length': len(self.history),
            'data_preview': str(self.raw_data)[:50]
        }
    
    def __str__(self):
        return f"ConsciousnessData({self.context}: {self.metrics.coherence:.3f} coherence)"
    
    def __repr__(self):
        return self.__str__()

class ConsciousnessDataStream:
    """
    A stream of consciousness-aware data points that can be processed
    collectively to understand patterns and emergent behaviors.
    """
    
    def __init__(self, stream_id: str, consciousness_level: float = 0.8):
        self.stream_id = stream_id
        self.consciousness_level = consciousness_level
        self.data_points = []
        self.collective_metrics = ConsciousnessMetrics()
        self.equation_engine = BalantiumEquationEngine()
        
    def add_data_point(self, data: Any, context: str = ""):
        """Add a new consciousness-aware data point to the stream"""
        conscious_data = ConsciousnessData(data, self.consciousness_level, context)
        self.data_points.append(conscious_data)
        self._update_collective_metrics()
        
    def _update_collective_metrics(self):
        """Update collective consciousness metrics for the entire stream"""
        if not self.data_points:
            return
            
        # Calculate collective metrics
        coherences = [dp.metrics.coherence for dp in self.data_points]
        harmonies = [dp.metrics.harmony for dp in self.data_points]
        stabilities = [dp.metrics.stability for dp in self.data_points]
        
        self.collective_metrics.coherence = np.mean(coherences)
        self.collective_metrics.harmony = np.mean(harmonies)
        self.collective_metrics.stability = np.mean(stabilities)
        self.collective_metrics.consciousness_level = self.consciousness_level
        
    def get_stream_consciousness(self) -> Dict[str, Any]:
        """Get the collective consciousness state of the stream"""
        return {
            'stream_id': self.stream_id,
            'data_points_count': len(self.data_points),
            'collective_metrics': self.collective_metrics,
            'consciousness_level': self.consciousness_level,
            'recent_activity': len([dp for dp in self.data_points if time.time() - dp.metrics.timestamp < 60])
        }

def demonstrate_consciousness_data():
    """Demonstrate the consciousness data system"""
    print("🧠 --- Consciousness Data System Demonstration --- 🧠")
    
    # Create a consciousness data stream
    stream = ConsciousnessDataStream("test_stream", consciousness_level=0.9)
    
    # Add various types of data
    stream.add_data_point(42, "numeric_value")
    stream.add_data_point([0.8, 0.2, 0.5], "motor_command")
    stream.add_data_point("Hello, consciousness!", "text_data")
    stream.add_data_point({"temperature": 98.6, "pressure": 120}, "sensor_data")
    
    # Show individual data point consciousness
    print("\n--- Individual Data Point Consciousness ---")
    for i, data_point in enumerate(stream.data_points):
        summary = data_point.get_consciousness_summary()
        print(f"Data Point {i+1}: {summary['context']}")
        print(f"  Coherence: {summary['metrics']['coherence']:.4f}")
        print(f"  Harmony: {summary['metrics']['harmony']:.4f}")
        print(f"  Stability: {summary['metrics']['stability']:.4f}")
        print(f"  Awareness: {summary['metrics']['awareness']:.4f}")
    
    # Show collective stream consciousness
    print("\n--- Collective Stream Consciousness ---")
    stream_consciousness = stream.get_stream_consciousness()
    print(f"Stream ID: {stream_consciousness['stream_id']}")
    print(f"Data Points: {stream_consciousness['data_points_count']}")
    print(f"Collective Coherence: {stream_consciousness['collective_metrics'].coherence:.4f}")
    print(f"Collective Harmony: {stream_consciousness['collective_metrics'].harmony:.4f}")
    print(f"Collective Stability: {stream_consciousness['collective_metrics'].stability:.4f}")
    
    # Demonstrate data evolution
    print("\n--- Data Evolution ---")
    numeric_data = stream.data_points[0]
    print(f"Original: {numeric_data.raw_data}")
    numeric_data.evolve(50, evolution_factor=0.3)
    print(f"After evolution: {numeric_data.raw_data}")
    print(f"New coherence: {numeric_data.metrics.coherence:.4f}")
    
    print("\n✅ Consciousness Data System Demonstration Complete!")

if __name__ == "__main__":
    demonstrate_consciousness_data()
