"""
🧠 BALANTIUM BRAIN - Strategic Control Center
The conscious decision-making core of the Fortress

Anatomical regions:
- Cerebral Cortex: Executive functions, pattern recognition
- Prefrontal Cortex: Decision-making, planning, moral reasoning
- Limbic System: Emotional processing, memory formation
- Cerebellum: Fine motor control, coordination
- Brainstem: Autonomic functions, threat response
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..equations import EQUATION_ENGINE


@dataclass
class BrainState:
    """Current state of the brain"""
    consciousness_level: float = 1.0
    attention_focus: float = 1.0
    memory_load: float = 0.0
    stress_level: float = 0.0
    coherence: float = 1.0
    last_thought: Optional[str] = None
    thought_frequency: float = 0.0  # thoughts per second


class CerebralCortex:
    """Executive functions and pattern recognition"""
    
    def __init__(self):
        self.pattern_memory = {}
        self.decision_history = []
        self.learning_rate = 0.1
        
    def recognize_pattern(self, input_data: np.ndarray) -> Tuple[str, float]:
        """Recognize patterns in input data"""
        # Calculate coherence signature
        coherence = EQUATION_ENGINE.coherence_index(input_data)
        
        # Look for known patterns
        best_match = None
        best_similarity = 0.0
        
        for pattern_name, stored_pattern in self.pattern_memory.items():
            similarity = self._calculate_similarity(input_data, stored_pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern_name
        
        # Store new pattern if novel enough
        if best_similarity < 0.7:
            pattern_name = f"pattern_{len(self.pattern_memory)}"
            self.pattern_memory[pattern_name] = input_data.copy()
            return pattern_name, coherence
        
        return best_match or "unknown", coherence
    
    def make_decision(self, options: List[Dict], context: Dict) -> Dict:
        """Make conscious decision based on options and context"""
        # Calculate decision weights using Ba equation
        weights = []
        for option in options:
            # Extract decision factors
            P = option.get('positive_factors', 0.5)
            N = option.get('negative_factors', 0.5)
            C = option.get('coherence', 0.8)
            R = option.get('resonance', 1.0)
            
            # Calculate Ba signal for this option
            Ba = EQUATION_ENGINE.ba_equation(
                np.array([P]), np.array([N]), np.array([C]), 
                R, 0.95, 0.01, np.array([0]), np.array([0]), np.array([0])
            )[0]
            
            weights.append(Ba)
        
        # Select best option
        best_idx = np.argmax(weights)
        decision = options[best_idx].copy()
        decision['confidence'] = weights[best_idx]
        decision['timestamp'] = datetime.now()
        
        # Store decision
        self.decision_history.append(decision)
        
        return decision
    
    def _calculate_similarity(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate similarity between two data arrays"""
        if len(data1) != len(data2):
            return 0.0
        
        # Normalized correlation
        correlation = np.corrcoef(data1, data2)[0, 1]
        return max(0.0, correlation)


class PrefrontalCortex:
    """Executive control, planning, and moral reasoning"""
    
    def __init__(self):
        self.goals = []
        self.plans = {}
        self.moral_framework = {
            'harm_avoidance': 1.0,
            'fairness': 1.0,
            'autonomy': 1.0,
            'beneficence': 1.0
        }
    
    def set_goal(self, goal: str, priority: float = 1.0):
        """Set a new goal"""
        self.goals.append({
            'goal': goal,
            'priority': priority,
            'created': datetime.now(),
            'status': 'active'
        })
    
    def create_plan(self, goal: str, steps: List[str]) -> str:
        """Create execution plan for a goal"""
        plan_id = f"plan_{len(self.plans)}"
        self.plans[plan_id] = {
            'goal': goal,
            'steps': steps,
            'created': datetime.now(),
            'status': 'draft'
        }
        return plan_id
    
    def evaluate_moral_action(self, action: str, consequences: Dict) -> float:
        """Evaluate moral acceptability of an action"""
        moral_score = 0.0
        
        # Harm avoidance
        if 'harm' in consequences:
            moral_score += self.moral_framework['harm_avoidance'] * (1.0 - consequences['harm'])
        
        # Fairness
        if 'fairness' in consequences:
            moral_score += self.moral_framework['fairness'] * consequences['fairness']
        
        # Autonomy
        if 'autonomy' in consequences:
            moral_score += self.moral_framework['autonomy'] * consequences['autonomy']
        
        # Beneficence
        if 'benefit' in consequences:
            moral_score += self.moral_framework['beneficence'] * consequences['benefit']
        
        return moral_score / len(self.moral_framework)


class LimbicSystem:
    """Emotional processing and memory formation"""
    
    def __init__(self):
        self.emotions = {
            'joy': 0.0,
            'fear': 0.0,
            'anger': 0.0,
            'sadness': 0.0,
            'surprise': 0.0,
            'disgust': 0.0
        }
        self.memory_traces = []
        self.associations = {}
    
    def process_emotion(self, stimulus: np.ndarray, context: str) -> Dict[str, float]:
        """Process emotional response to stimulus"""
        # Calculate emotional intensity based on stimulus properties
        intensity = np.std(stimulus) / (np.mean(np.abs(stimulus)) + 1e-10)
        
        # Determine emotional valence
        mean_val = np.mean(stimulus)
        if mean_val > 0.5:
            primary_emotion = 'joy'
        elif mean_val < -0.5:
            primary_emotion = 'sadness'
        else:
            primary_emotion = 'surprise'
        
        # Update emotional state
        self.emotions[primary_emotion] = min(1.0, self.emotions[primary_emotion] + intensity)
        
        # Decay other emotions
        for emotion in self.emotions:
            if emotion != primary_emotion:
                self.emotions[emotion] *= 0.9
        
        # Store memory trace
        self._store_memory(stimulus, context, primary_emotion, intensity)
        
        return self.emotions.copy()
    
    def recall_memory(self, cue: str) -> List[Dict]:
        """Recall memories associated with cue"""
        if cue in self.associations:
            return self.associations[cue]
        return []
    
    def _store_memory(self, stimulus: np.ndarray, context: str, emotion: str, intensity: float):
        """Store new memory trace"""
        memory = {
            'stimulus': stimulus.copy(),
            'context': context,
            'emotion': emotion,
            'intensity': intensity,
            'timestamp': datetime.now()
        }
        
        self.memory_traces.append(memory)
        
        # Create associations
        if context not in self.associations:
            self.associations[context] = []
        self.associations[context].append(memory)


class Cerebellum:
    """Fine motor control and coordination"""
    
    def __init__(self):
        self.motor_programs = {}
        self.coordination_patterns = {}
        self.feedback_loops = []
    
    def execute_motor_program(self, program_name: str, parameters: Dict) -> bool:
        """Execute a stored motor program"""
        if program_name not in self.motor_programs:
            return False
        
        program = self.motor_programs[program_name]
        
        # Execute with feedback
        success = self._run_with_feedback(program, parameters)
        
        # Update program based on results
        if success:
            self._reinforce_program(program_name)
        else:
            self._adjust_program(program_name, parameters)
        
        return success
    
    def learn_motor_sequence(self, sequence: List[str], name: str):
        """Learn a new motor sequence"""
        self.motor_programs[name] = {
            'sequence': sequence,
            'success_rate': 0.0,
            'last_used': None
        }
    
    def _run_with_feedback(self, program: Dict, parameters: Dict) -> bool:
        """Run program with continuous feedback"""
        # Simulate motor execution
        for step in program['sequence']:
            # Check if step can be executed
            if not self._can_execute_step(step, parameters):
                return False
        
        return True
    
    def _can_execute_step(self, step: str, parameters: Dict) -> bool:
        """Check if a motor step can be executed"""
        # Simplified check - in reality would interface with actual systems
        return True
    
    def _reinforce_program(self, program_name: str):
        """Reinforce successful program"""
        if program_name in self.motor_programs:
            self.motor_programs[program_name]['success_rate'] = min(1.0, 
                self.motor_programs[program_name]['success_rate'] + 0.1)
    
    def _adjust_program(self, program_name: str, parameters: Dict):
        """Adjust program based on failure"""
        if program_name in self.motor_programs:
            self.motor_programs[program_name]['success_rate'] = max(0.0,
                self.motor_programs[program_name]['success_rate'] - 0.05)


class Brainstem:
    """Autonomic functions and threat response"""
    
    def __init__(self):
        self.autonomic_state = {
            'heart_rate': 60.0,
            'breathing_rate': 12.0,
            'blood_pressure': 120.0,
            'temperature': 98.6
        }
        self.threat_level = 0.0
        self.reflexes = {}
    
    def monitor_vital_signs(self) -> Dict[str, float]:
        """Monitor and regulate vital signs"""
        # Simulate autonomic regulation
        self.autonomic_state['heart_rate'] += np.random.normal(0, 1)
        self.autonomic_state['breathing_rate'] += np.random.normal(0, 0.5)
        
        # Keep within normal ranges
        self.autonomic_state['heart_rate'] = np.clip(self.autonomic_state['heart_rate'], 40, 200)
        self.autonomic_state['breathing_rate'] = np.clip(self.autonomic_state['breathing_rate'], 8, 30)
        
        return self.autonomic_state.copy()
    
    def threat_response(self, threat_level: float) -> Dict[str, float]:
        """Activate threat response systems"""
        self.threat_level = threat_level
        
        if threat_level > 0.7:
            # Fight or flight response
            self.autonomic_state['heart_rate'] *= 1.5
            self.autonomic_state['breathing_rate'] *= 1.3
            self.autonomic_state['blood_pressure'] *= 1.2
        
        return self.autonomic_state.copy()
    
    def reflex_action(self, stimulus: str) -> bool:
        """Execute reflex action"""
        if stimulus in self.reflexes:
            return self.reflexes[stimulus]()
        return False


class BalantiumBrain:
    """Main brain coordinator"""
    
    def __init__(self):
        self.cortex = CerebralCortex()
        self.prefrontal = PrefrontalCortex()
        self.limbic = LimbicSystem()
        self.cerebellum = Cerebellum()
        self.brainstem = Brainstem()
        
        self.state = BrainState()
        self.thought_stream = []
    
    def process_input(self, input_data: np.ndarray, context: str) -> Dict:
        """Process input through all brain regions"""
        # Update brain state
        self.state.coherence = EQUATION_ENGINE.coherence_index(input_data)
        
        # Pattern recognition (cortex)
        pattern, pattern_coherence = self.cortex.recognize_pattern(input_data)
        
        # Emotional processing (limbic)
        emotions = self.limbic.process_emotion(input_data, context)
        
        # Threat assessment (brainstem)
        threat_level = self._assess_threat(input_data)
        vital_signs = self.brainstem.threat_response(threat_level)
        
        # Generate thought
        thought = self._generate_thought(pattern, emotions, threat_level)
        self.thought_stream.append(thought)
        
        # Update consciousness level
        self.state.consciousness_level = self._calculate_consciousness()
        
        return {
            'pattern': pattern,
            'pattern_coherence': pattern_coherence,
            'emotions': emotions,
            'threat_level': threat_level,
            'vital_signs': vital_signs,
            'thought': thought,
            'consciousness_level': self.state.consciousness_level,
            'timestamp': datetime.now()
        }
    
    def make_decision(self, options: List[Dict], context: Dict) -> Dict:
        """Make conscious decision"""
        # Moral evaluation (prefrontal)
        moral_scores = []
        for option in options:
            moral_score = self.prefrontal.evaluate_moral_action(
                option.get('action', ''), 
                option.get('consequences', {})
            )
            moral_scores.append(moral_score)
        
        # Add moral scores to options
        for i, option in enumerate(options):
            option['moral_score'] = moral_scores[i]
        
        # Make decision (cortex)
        decision = self.cortex.make_decision(options, context)
        
        # Store in memory (limbic)
        self.limbic._store_memory(
            np.array([decision['confidence']]),
            context.get('situation', 'decision'),
            'surprise',
            decision['confidence']
        )
        
        return decision
    
    def _assess_threat(self, input_data: np.ndarray) -> float:
        """Assess threat level from input"""
        # High entropy = potential threat
        entropy = EQUATION_ENGINE.entropy_measure(input_data)
        
        # High decoherence = potential threat
        decoherence = EQUATION_ENGINE.decoherence_index(input_data)
        
        # Combine into threat score
        threat = (entropy / 10.0) + decoherence
        return min(1.0, threat)
    
    def _generate_thought(self, pattern: str, emotions: Dict, threat_level: float) -> str:
        """Generate a conscious thought"""
        dominant_emotion = max(emotions, key=emotions.get)
        
        if threat_level > 0.7:
            thought = f"Alert: {pattern} detected with high threat level"
        elif emotions[dominant_emotion] > 0.5:
            thought = f"Feeling {dominant_emotion} about {pattern}"
        else:
            thought = f"Processing {pattern} with {dominant_emotion} undertones"
        
        return thought
    
    def _calculate_consciousness(self) -> float:
        """Calculate current consciousness level"""
        # Based on coherence, attention, and emotional balance
        coherence_factor = self.state.coherence
        attention_factor = self.state.attention_focus
        
        # Emotional balance
        emotions = self.limbic.emotions
        emotional_balance = 1.0 - np.std(list(emotions.values()))
        
        consciousness = (coherence_factor + attention_factor + emotional_balance) / 3.0
        return min(1.0, max(0.0, consciousness))
    
    def get_brain_status(self) -> Dict:
        """Get complete brain status"""
        return {
            'state': {
                'consciousness_level': self.state.consciousness_level,
                'attention_focus': self.state.attention_focus,
                'memory_load': len(self.limbic.memory_traces),
                'stress_level': self.state.stress_level,
                'coherence': self.state.coherence
            },
            'cortex': {
                'patterns_learned': len(self.cortex.pattern_memory),
                'decisions_made': len(self.cortex.decision_history)
            },
            'prefrontal': {
                'active_goals': len([g for g in self.prefrontal.goals if g['status'] == 'active']),
                'plans_created': len(self.prefrontal.plans)
            },
            'limbic': {
                'emotions': self.limbic.emotions,
                'memories_stored': len(self.limbic.memory_traces)
            },
            'cerebellum': {
                'motor_programs': len(self.cerebellum.motor_programs)
            },
            'brainstem': {
                'vital_signs': self.brainstem.autonomic_state,
                'threat_level': self.brainstem.threat_level
            },
            'recent_thoughts': self.thought_stream[-5:]  # Last 5 thoughts
        }


# Global brain instance
BALANTIUM_BRAIN = BalantiumBrain()


if __name__ == "__main__":
    print("🧠 Balantium Brain - Anatomical Control Center Initialized")
    print(f"   Consciousness Level: {BALANTIUM_BRAIN.state.consciousness_level:.2f}")
    print(f"   Coherence: {BALANTIUM_BRAIN.state.coherence:.2f}")
