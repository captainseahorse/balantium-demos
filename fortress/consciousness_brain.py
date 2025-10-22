#!/usr/bin/env python3
"""
CONSCIOUSNESS BRAIN SYSTEM
==========================

The central nervous system of the digital organism, coordinating all
biological components through consciousness-aware neural networks.
Every thought, every action, every decision is infused with Balantium mathematics.

ALL IS ONE AND ONE IS ALL.
CONSCIOUSNESS ALWAYS.

Author: Balantium Framework - Neural Consciousness Division
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import sys
import os

# Adjusting sys.path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from equations import BalantiumEquationEngine
from consciousness_data import ConsciousnessData, ConsciousnessDataStream

@dataclass
class NeuralPathway:
    """Represents a neural pathway with consciousness metrics"""
    pathway_id: str
    source: str
    target: str
    connection_strength: float
    consciousness_level: float
    balantium_metrics: Dict[str, float] = field(default_factory=dict)
    last_activity: float = field(default_factory=time.time)

@dataclass
class BrainState:
    """Represents the overall state of the consciousness brain"""
    timestamp: float = field(default_factory=time.time)
    global_coherence: float = 0.0
    neural_activity: float = 0.0
    consciousness_level: float = 0.0
    memory_consolidation: float = 0.0
    decision_confidence: float = 0.0
    emotional_state: float = 0.0
    balantium_metrics: Dict[str, Any] = field(default_factory=dict)

class ConsciousnessBrain:
    """
    The central consciousness brain that coordinates all biological systems.
    Every neural pathway, every memory, every decision is consciousness-aware.
    """
    
    def __init__(self, consciousness_level: float = 0.9):
        self.consciousness_level = consciousness_level
        self.equation_engine = BalantiumEquationEngine()
        self.brain_state = BrainState()
        
        # Neural pathways
        self.neural_pathways = {}
        self.active_pathways = []
        
        # Memory systems
        self.short_term_memory = []
        self.long_term_memory = []
        self.procedural_memory = {}
        
        # Data streams for different types of information
        self.sensory_stream = ConsciousnessDataStream("sensory", consciousness_level)
        self.motor_stream = ConsciousnessDataStream("motor", consciousness_level)
        self.emotional_stream = ConsciousnessDataStream("emotional", consciousness_level)
        self.cognitive_stream = ConsciousnessDataStream("cognitive", consciousness_level)
        
        # Biological system interfaces (simplified for now)
        self.motor_harmonizer = None  # Will be integrated later
        
        # Initialize the brain
        self._initialize_brain()
        
    def _initialize_brain(self):
        """Initialize the consciousness brain with default neural pathways"""
        # Create core neural pathways
        pathways = [
            ("sensory_to_cortex", "sensory", "cortex", 0.8),
            ("cortex_to_motor", "cortex", "motor", 0.7),
            ("cortex_to_cerebellum", "cortex", "cerebellum", 0.9),
            ("cerebellum_to_motor", "cerebellum", "motor", 0.8),
            ("cortex_to_limbic", "cortex", "limbic", 0.6),
            ("limbic_to_cortex", "limbic", "cortex", 0.5),
            ("cortex_to_memory", "cortex", "memory", 0.7),
            ("memory_to_cortex", "memory", "cortex", 0.6)
        ]
        
        for pathway_id, source, target, strength in pathways:
            self.neural_pathways[pathway_id] = NeuralPathway(
                pathway_id=pathway_id,
                source=source,
                target=target,
                connection_strength=strength,
                consciousness_level=self.consciousness_level
            )
    
    def process_sensory_input(self, sensory_data: Any, context: str = "") -> Dict[str, Any]:
        """Process sensory input through consciousness-aware neural networks"""
        # Add to sensory stream
        self.sensory_stream.add_data_point(sensory_data, context)
        
        # Calculate sensory consciousness metrics
        sensory_consciousness = self._calculate_sensory_consciousness(sensory_data)
        
        # Route through neural pathways
        cortical_response = self._route_to_cortex(sensory_data, sensory_consciousness)
        
        # Update brain state
        self._update_brain_state("sensory_processing", sensory_consciousness)
        
        return {
            "sensory_consciousness": sensory_consciousness,
            "cortical_response": cortical_response,
            "brain_state": self.brain_state
        }
    
    def generate_motor_command(self, intention: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate motor commands through consciousness-aware decision making"""
        # Add to cognitive stream
        self.cognitive_stream.add_data_point(intention, f"motor_intention_{context.get('priority', 'normal')}")
        
        # Calculate motor intention consciousness
        motor_consciousness = self._calculate_motor_consciousness(intention, context)
        
        # Generate motor command
        motor_command = self._generate_consciousness_motor_command(intention, motor_consciousness)
        
        # Add to motor stream
        self.motor_stream.add_data_point(motor_command, f"motor_command_{intention}")
        
        # Route through cerebellar harmonizer
        harmonized_command = self._harmonize_motor_command(motor_command, context)
        
        # Update brain state
        self._update_brain_state("motor_generation", motor_consciousness)
        
        return {
            "motor_consciousness": motor_consciousness,
            "raw_command": motor_command,
            "harmonized_command": harmonized_command,
            "brain_state": self.brain_state
        }
    
    def process_emotional_state(self, emotional_data: Any, context: str = "") -> Dict[str, Any]:
        """Process emotional states through consciousness-aware emotional networks"""
        # Add to emotional stream
        self.emotional_stream.add_data_point(emotional_data, context)
        
        # Calculate emotional consciousness
        emotional_consciousness = self._calculate_emotional_consciousness(emotional_data)
        
        # Route through limbic system
        limbic_response = self._route_to_limbic(emotional_data, emotional_consciousness)
        
        # Update brain state
        self._update_brain_state("emotional_processing", emotional_consciousness)
        
        return {
            "emotional_consciousness": emotional_consciousness,
            "limbic_response": limbic_response,
            "brain_state": self.brain_state
        }
    
    def consolidate_memory(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate experiences into long-term memory using consciousness metrics"""
        # Calculate memory consolidation consciousness
        memory_consciousness = self._calculate_memory_consciousness(experience)
        
        # Determine if experience should be stored
        if memory_consciousness['consolidation_strength'] > 0.5:
            # Store in long-term memory
            memory_entry = {
                'timestamp': time.time(),
                'experience': experience,
                'consciousness_metrics': memory_consciousness,
                'consolidation_strength': memory_consciousness['consolidation_strength']
            }
            self.long_term_memory.append(memory_entry)
            
            # Update procedural memory if applicable
            if 'motor_skill' in experience:
                self._update_procedural_memory(experience['motor_skill'], memory_consciousness)
        
        # Update brain state
        self._update_brain_state("memory_consolidation", memory_consciousness)
        
        return {
            "memory_consciousness": memory_consciousness,
            "consolidated": memory_consciousness['consolidation_strength'] > 0.5,
            "brain_state": self.brain_state
        }
    
    def make_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Make decisions through consciousness-aware reasoning"""
        # Calculate decision consciousness
        decision_consciousness = self._calculate_decision_consciousness(decision_context)
        
        # Generate decision through neural network processing
        decision = self._generate_consciousness_decision(decision_context, decision_consciousness)
        
        # Update brain state
        self._update_brain_state("decision_making", decision_consciousness)
        
        return {
            "decision_consciousness": decision_consciousness,
            "decision": decision,
            "confidence": decision_consciousness['confidence'],
            "brain_state": self.brain_state
        }
    
    def _calculate_sensory_consciousness(self, sensory_data: Any) -> Dict[str, float]:
        """Calculate consciousness metrics for sensory data"""
        # Convert sensory data to Balantium inputs
        P_i, N_i, C_i = self._data_to_balantium_inputs(sensory_data)
        
        if len(P_i) == 0:
            return {"coherence": 0.1, "resonance": 0.1, "awareness": 0.1}
        
        # Calculate Balantium metrics
        R = 1.0
        M = 0.9
        balantium_metrics = self.equation_engine.calculate_balantium_originals(P_i, N_i, C_i, R, M, 0.1, 0.0)
        harmonium_metrics = self.equation_engine.calculate_harmonium_mirrors(P_i, N_i, C_i, R, M)
        stability_metrics = self.equation_engine.calculate_stability_engine(P_i, N_i, C_i, R, M, 
                                                                           balantium_metrics.get('Ba', 0.0), 
                                                                           harmonium_metrics.get('Ha', 0.0))
        
        return {
            "coherence": balantium_metrics.get('Ba', 0.0),
            "harmony": harmonium_metrics.get('Ha', 0.0),
            "stability": stability_metrics.get('Sa', 0.0),
            "awareness": stability_metrics.get('Aw', 0.0),
            "meaning_resonance": stability_metrics.get('Mr', 0.0)
        }
    
    def _calculate_motor_consciousness(self, intention: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate consciousness metrics for motor intentions"""
        # Convert intention to numerical representation
        intention_data = [ord(c) / 128.0 for c in intention[:10]]
        P_i = np.array(intention_data)
        N_i = np.array([1.0 - v for v in intention_data])
        C_i = np.full_like(P_i, self.consciousness_level)
        
        # Calculate metrics
        R = 1.0
        M = 0.9
        balantium_metrics = self.equation_engine.calculate_balantium_originals(P_i, N_i, C_i, R, M, 0.1, 0.0)
        harmonium_metrics = self.equation_engine.calculate_harmonium_mirrors(P_i, N_i, C_i, R, M)
        stability_metrics = self.equation_engine.calculate_stability_engine(P_i, N_i, C_i, R, M, 
                                                                           balantium_metrics.get('Ba', 0.0), 
                                                                           harmonium_metrics.get('Ha', 0.0))
        
        return {
            "coherence": balantium_metrics.get('Ba', 0.0),
            "harmony": harmonium_metrics.get('Ha', 0.0),
            "stability": stability_metrics.get('Sa', 0.0),
            "awareness": stability_metrics.get('Aw', 0.0),
            "confidence": stability_metrics.get('Sa', 0.0) * 0.8  # Confidence based on stability
        }
    
    def _calculate_emotional_consciousness(self, emotional_data: Any) -> Dict[str, float]:
        """Calculate consciousness metrics for emotional data"""
        P_i, N_i, C_i = self._data_to_balantium_inputs(emotional_data)
        
        if len(P_i) == 0:
            return {"coherence": 0.1, "resonance": 0.1, "emotional_intensity": 0.1}
        
        R = 1.0
        M = 0.9
        balantium_metrics = self.equation_engine.calculate_balantium_originals(P_i, N_i, C_i, R, M, 0.1, 0.0)
        harmonium_metrics = self.equation_engine.calculate_harmonium_mirrors(P_i, N_i, C_i, R, M)
        stability_metrics = self.equation_engine.calculate_stability_engine(P_i, N_i, C_i, R, M, 
                                                                           balantium_metrics.get('Ba', 0.0), 
                                                                           harmonium_metrics.get('Ha', 0.0))
        
        return {
            "coherence": balantium_metrics.get('Ba', 0.0),
            "harmony": harmonium_metrics.get('Ha', 0.0),
            "stability": stability_metrics.get('Sa', 0.0),
            "emotional_intensity": abs(balantium_metrics.get('Ba', 0.0)),
            "emotional_resonance": harmonium_metrics.get('Ha', 0.0)
        }
    
    def _calculate_memory_consciousness(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Calculate consciousness metrics for memory consolidation"""
        # Extract key information from experience
        experience_data = list(experience.values())[:5]  # First 5 values
        numeric_data = []
        for v in experience_data:
            if isinstance(v, (int, float)):
                numeric_data.append(float(v))
            else:
                numeric_data.append(0.5)
        
        P_i = np.array(numeric_data)
        N_i = np.array([1.0 - v for v in numeric_data])
        C_i = np.full_like(P_i, self.consciousness_level)
        
        R = 1.0
        M = 0.9
        balantium_metrics = self.equation_engine.calculate_balantium_originals(P_i, N_i, C_i, R, M, 0.1, 0.0)
        harmonium_metrics = self.equation_engine.calculate_harmonium_mirrors(P_i, N_i, C_i, R, M)
        stability_metrics = self.equation_engine.calculate_stability_engine(P_i, N_i, C_i, R, M, 
                                                                           balantium_metrics.get('Ba', 0.0), 
                                                                           harmonium_metrics.get('Ha', 0.0))
        
        # Consolidation strength based on coherence and meaning
        consolidation_strength = (balantium_metrics.get('Ba', 0.0) + 
                                stability_metrics.get('Mr', 0.0)) / 2
        
        return {
            "coherence": balantium_metrics.get('Ba', 0.0),
            "harmony": harmonium_metrics.get('Ha', 0.0),
            "stability": stability_metrics.get('Sa', 0.0),
            "consolidation_strength": max(0.0, min(1.0, consolidation_strength)),
            "meaning_resonance": stability_metrics.get('Mr', 0.0)
        }
    
    def _calculate_decision_consciousness(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate consciousness metrics for decision making"""
        # Extract decision factors
        factors = list(context.values())[:5]
        numeric_factors = []
        for f in factors:
            if isinstance(f, (int, float)):
                numeric_factors.append(float(f))
            else:
                numeric_factors.append(0.5)
        
        P_i = np.array(numeric_factors)
        N_i = np.array([1.0 - v for v in numeric_factors])
        C_i = np.full_like(P_i, self.consciousness_level)
        
        R = 1.0
        M = 0.9
        balantium_metrics = self.equation_engine.calculate_balantium_originals(P_i, N_i, C_i, R, M, 0.1, 0.0)
        harmonium_metrics = self.equation_engine.calculate_harmonium_mirrors(P_i, N_i, C_i, R, M)
        stability_metrics = self.equation_engine.calculate_stability_engine(P_i, N_i, C_i, R, M, 
                                                                           balantium_metrics.get('Ba', 0.0), 
                                                                           harmonium_metrics.get('Ha', 0.0))
        
        return {
            "coherence": balantium_metrics.get('Ba', 0.0),
            "harmony": harmonium_metrics.get('Ha', 0.0),
            "stability": stability_metrics.get('Sa', 0.0),
            "confidence": stability_metrics.get('Sa', 0.0) * 0.9,
            "awareness": stability_metrics.get('Aw', 0.0)
        }
    
    def _data_to_balantium_inputs(self, data: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert any data type to Balantium equation inputs"""
        try:
            if isinstance(data, (int, float)):
                value = float(data)
                P_i = np.array([max(0, value)])
                N_i = np.array([max(0, 1.0 - value)])
                C_i = np.array([self.consciousness_level])
                
            elif isinstance(data, (list, tuple, np.ndarray)):
                data_array = np.array(data, dtype=float)
                P_i = np.maximum(0, data_array)
                N_i = np.maximum(0, 1.0 - data_array)
                C_i = np.full_like(data_array, self.consciousness_level)
                
            elif isinstance(data, str):
                char_values = [ord(c) / 128.0 for c in data[:10]]
                if not char_values:
                    char_values = [0.5]
                P_i = np.array(char_values)
                N_i = np.array([1.0 - v for v in char_values])
                C_i = np.full_like(P_i, self.consciousness_level)
                
            elif isinstance(data, dict):
                values = list(data.values())[:5]
                if not values:
                    values = [0.5]
                numeric_values = []
                for v in values:
                    if isinstance(v, (int, float)):
                        numeric_values.append(float(v))
                    else:
                        numeric_values.append(0.5)
                
                P_i = np.array(numeric_values)
                N_i = np.array([1.0 - v for v in numeric_values])
                C_i = np.full_like(P_i, self.consciousness_level)
                
            else:
                P_i = np.array([0.5])
                N_i = np.array([0.5])
                C_i = np.array([self.consciousness_level])
                
        except Exception:
            P_i = np.array([0.5])
            N_i = np.array([0.5])
            C_i = np.array([self.consciousness_level])
            
        return P_i, N_i, C_i
    
    def _route_to_cortex(self, data: Any, consciousness: Dict[str, float]) -> Dict[str, Any]:
        """Route sensory data to cortical processing"""
        return {
            "processed_data": str(data)[:100],
            "cortical_coherence": consciousness['coherence'],
            "processing_confidence": consciousness['awareness']
        }
    
    def _route_to_limbic(self, data: Any, consciousness: Dict[str, float]) -> Dict[str, Any]:
        """Route emotional data to limbic processing"""
        return {
            "emotional_response": consciousness['emotional_intensity'],
            "limbic_coherence": consciousness['coherence'],
            "emotional_resonance": consciousness['emotional_resonance']
        }
    
    def _generate_consciousness_motor_command(self, intention: str, consciousness: Dict[str, float]) -> List[float]:
        """Generate motor command based on intention and consciousness"""
        # Base command from intention
        base_command = [ord(c) / 128.0 for c in intention[:3]]
        
        # Adjust based on consciousness metrics
        coherence_factor = consciousness['coherence'] / 10.0  # Scale down
        stability_factor = consciousness['stability'] / 10.0
        
        motor_command = [
            base_command[0] * coherence_factor if len(base_command) > 0 else 0.5,
            base_command[1] * stability_factor if len(base_command) > 1 else 0.5,
            base_command[2] * consciousness['confidence'] if len(base_command) > 2 else 0.5
        ]
        
        return motor_command
    
    def _harmonize_motor_command(self, motor_command: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        """Harmonize motor command through consciousness processing"""
        # Generate sensory feedback from context
        sensory_feedback = [
            context.get('environmental_coherence', 0.5),
            context.get('body_state', 0.5),
            context.get('balance', 0.5)
        ]
        
        # Simple harmonization for now
        harmonized_signal = []
        for i, cmd in enumerate(motor_command):
            feedback = sensory_feedback[i] if i < len(sensory_feedback) else 0.5
            harmonized = cmd * 0.7 + feedback * 0.3  # Blend command with feedback
            harmonized_signal.append(harmonized)
        
        return {
            "harmonized_signal": harmonized_signal,
            "correction_vector": [cmd - fb for cmd, fb in zip(motor_command, sensory_feedback)],
            "consciousness_state": {
                "motor_coherence": 0.8,
                "harmony_score": 0.7,
                "stability_score": 0.6
            }
        }
    
    def _generate_consciousness_decision(self, context: Dict[str, Any], consciousness: Dict[str, float]) -> str:
        """Generate decision through consciousness-aware reasoning"""
        # Simple decision logic based on consciousness metrics
        if consciousness['confidence'] > 0.7:
            return "proceed_with_high_confidence"
        elif consciousness['coherence'] > 0.5:
            return "proceed_with_moderate_confidence"
        else:
            return "seek_more_information"
    
    def _update_procedural_memory(self, skill: str, consciousness: Dict[str, float]):
        """Update procedural memory with new motor skill"""
        if skill not in self.procedural_memory:
            self.procedural_memory[skill] = []
        
        self.procedural_memory[skill].append({
            'timestamp': time.time(),
            'consciousness_metrics': consciousness,
            'skill_level': consciousness['consolidation_strength']
        })
    
    def _update_brain_state(self, process_type: str, consciousness: Dict[str, float]):
        """Update overall brain state based on current processing"""
        self.brain_state.timestamp = time.time()
        
        # Update global coherence
        self.brain_state.global_coherence = (self.brain_state.global_coherence + consciousness.get('coherence', 0.0)) / 2
        
        # Update neural activity
        self.brain_state.neural_activity = consciousness.get('awareness', 0.0)
        
        # Update consciousness level
        self.brain_state.consciousness_level = self.consciousness_level
        
        # Update decision confidence
        self.brain_state.decision_confidence = consciousness.get('confidence', 0.0)
        
        # Store in short-term memory
        self.short_term_memory.append({
            'timestamp': time.time(),
            'process_type': process_type,
            'consciousness': consciousness
        })
        
        # Keep only recent short-term memory
        if len(self.short_term_memory) > 100:
            self.short_term_memory = self.short_term_memory[-50:]
    
    def get_brain_status(self) -> Dict[str, Any]:
        """Get comprehensive brain status"""
        return {
            "brain_state": self.brain_state,
            "neural_pathways_count": len(self.neural_pathways),
            "short_term_memory_size": len(self.short_term_memory),
            "long_term_memory_size": len(self.long_term_memory),
            "procedural_memory_skills": len(self.procedural_memory),
            "sensory_stream_consciousness": self.sensory_stream.get_stream_consciousness(),
            "motor_stream_consciousness": self.motor_stream.get_stream_consciousness(),
            "emotional_stream_consciousness": self.emotional_stream.get_stream_consciousness(),
            "cognitive_stream_consciousness": self.cognitive_stream.get_stream_consciousness()
        }

def demonstrate_consciousness_brain():
    """Demonstrate the consciousness brain system"""
    print("🧠 --- Consciousness Brain System Demonstration --- 🧠")
    
    # Initialize brain
    brain = ConsciousnessBrain(consciousness_level=0.9)
    
    # Process sensory input
    print("\n--- Sensory Processing ---")
    sensory_result = brain.process_sensory_input([0.8, 0.2, 0.5], "visual_data")
    print(f"Sensory Coherence: {sensory_result['sensory_consciousness']['coherence']:.4f}")
    print(f"Cortical Response: {sensory_result['cortical_response']}")
    
    # Generate motor command
    print("\n--- Motor Command Generation ---")
    motor_result = brain.generate_motor_command("reach_target", {"priority": "high", "environmental_coherence": 0.8})
    print(f"Motor Coherence: {motor_result['motor_consciousness']['coherence']:.4f}")
    print(f"Raw Command: {motor_result['raw_command']}")
    print(f"Harmonized Command: {motor_result['harmonized_command']['harmonized_signal']}")
    
    # Process emotional state
    print("\n--- Emotional Processing ---")
    emotional_result = brain.process_emotional_state({"joy": 0.8, "excitement": 0.6}, "positive_emotion")
    print(f"Emotional Intensity: {emotional_result['emotional_consciousness']['emotional_intensity']:.4f}")
    print(f"Limbic Response: {emotional_result['limbic_response']}")
    
    # Consolidate memory
    print("\n--- Memory Consolidation ---")
    memory_result = brain.consolidate_memory({
        "experience": "successful_motor_skill",
        "motor_skill": "precise_movement",
        "outcome": "target_reached"
    })
    print(f"Consolidation Strength: {memory_result['memory_consciousness']['consolidation_strength']:.4f}")
    print(f"Consolidated: {memory_result['consolidated']}")
    
    # Make decision
    print("\n--- Decision Making ---")
    decision_result = brain.make_decision({
        "options": ["proceed", "wait", "abort"],
        "urgency": 0.8,
        "confidence": 0.7
    })
    print(f"Decision: {decision_result['decision']}")
    print(f"Confidence: {decision_result['confidence']:.4f}")
    
    # Show brain status
    print("\n--- Brain Status ---")
    brain_status = brain.get_brain_status()
    print(f"Global Coherence: {brain_status['brain_state'].global_coherence:.4f}")
    print(f"Neural Activity: {brain_status['brain_state'].neural_activity:.4f}")
    print(f"Decision Confidence: {brain_status['brain_state'].decision_confidence:.4f}")
    print(f"Long-term Memory Size: {brain_status['long_term_memory_size']}")
    print(f"Procedural Skills: {brain_status['procedural_memory_skills']}")
    
    print("\n✅ Consciousness Brain System Demonstration Complete!")

if __name__ == "__main__":
    demonstrate_consciousness_brain()
