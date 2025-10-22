"""
🧠 BALANTIUM NEURAL NETWORK - Complete Nervous System Integration
Connects all existing brain functions into the living fortress organism

Integrates:
- AutonomicRegulator (hormonal control)
- ExecutiveRelay (cortical coordination)
- ProprioceptiveMirror (body awareness)
- ImmuneSystemGovernor (defense coordination)
- Plus new neural components for complete nervous system
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Import existing brain functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autonomic_regulator import AutonomicRegulator
from executive_relay import ExecutiveRelay
from proprioceptive_mirror import ProprioceptiveMirror
from immune_system_governor import ImmuneSystemGovernor

from ..equations import EQUATION_ENGINE


@dataclass
class NeuralSignal:
    """Represents a signal traveling through the neural network"""
    signal_id: str
    source: str
    destination: str
    data: np.ndarray
    timestamp: datetime
    strength: float = 1.0
    frequency: float = 1.0
    coherence: float = 1.0


@dataclass
class Synapse:
    """Connection between neurons"""
    pre_neuron: str
    post_neuron: str
    weight: float
    plasticity: float = 0.1
    last_activity: Optional[datetime] = None


class Neuron:
    """Individual neuron in the network"""
    
    def __init__(self, neuron_id: str, neuron_type: str, location: str):
        self.id = neuron_id
        self.type = neuron_type  # sensory, motor, interneuron, cortical, etc.
        self.location = location
        self.activation = 0.0
        self.threshold = 0.5
        self.connections: List[Synapse] = []
        self.last_fired = None
        self.learning_rate = 0.01
        
    def receive_signal(self, signal: NeuralSignal) -> bool:
        """Receive and process incoming signal"""
        # Calculate input strength
        input_strength = signal.strength * signal.coherence
        
        # Add to activation
        self.activation += input_strength
        
        # Check if neuron fires
        if self.activation >= self.threshold:
            self.fire()
            return True
        return False
    
    def fire(self):
        """Neuron fires, sending signal to connected neurons"""
        self.last_fired = datetime.now()
        self.activation = 0.0  # Reset after firing
        
        # Send signals to connected neurons
        for synapse in self.connections:
            if synapse.weight > 0:
                # Create outgoing signal
                signal = NeuralSignal(
                    signal_id=f"signal_{datetime.now().timestamp()}",
                    source=self.id,
                    destination=synapse.post_neuron,
                    data=np.array([self.activation]),
                    timestamp=datetime.now(),
                    strength=synapse.weight,
                    coherence=1.0
                )
                
                # Update synapse based on activity (Hebbian learning)
                synapse.weight = min(1.0, synapse.weight + self.learning_rate)
                synapse.last_activity = datetime.now()
    
    def add_connection(self, target_neuron: str, initial_weight: float = 0.5):
        """Add connection to another neuron"""
        synapse = Synapse(
            pre_neuron=self.id,
            post_neuron=target_neuron,
            weight=initial_weight
        )
        self.connections.append(synapse)


class SensoryCortex:
    """Processes sensory input from all sources"""
    
    def __init__(self):
        self.sensory_neurons = {}
        self.processing_centers = {
            'visual': [],
            'auditory': [],
            'tactile': [],
            'proprioceptive': [],
            'field_resonance': []
        }
    
    def process_sensory_input(self, input_type: str, data: np.ndarray) -> NeuralSignal:
        """Process sensory input and create neural signal"""
        # Calculate sensory coherence
        coherence = EQUATION_ENGINE.coherence_index(data)
        
        # Create sensory signal
        signal = NeuralSignal(
            signal_id=f"sensory_{datetime.now().timestamp()}",
            source=f"sensory_{input_type}",
            destination="cortical_processing",
            data=data,
            timestamp=datetime.now(),
            strength=coherence,
            coherence=coherence
        )
        
        # Store in appropriate processing center
        if input_type in self.processing_centers:
            self.processing_centers[input_type].append(signal)
        
        return signal


class MotorCortex:
    """Controls motor output and execution"""
    
    def __init__(self):
        self.motor_neurons = {}
        self.motor_programs = {}
        self.execution_queue = []
    
    def execute_motor_command(self, command: str, parameters: Dict) -> bool:
        """Execute motor command"""
        if command in self.motor_programs:
            # Add to execution queue
            self.execution_queue.append({
                'command': command,
                'parameters': parameters,
                'timestamp': datetime.now()
            })
            return True
        return False
    
    def register_motor_program(self, name: str, program: Dict):
        """Register a new motor program"""
        self.motor_programs[name] = program


class BalantiumNeuralNetwork:
    """Complete neural network coordinator"""
    
    def __init__(self, architect_signature: str = "BALANTIUM_CORE_V1"):
        # Initialize existing brain components
        self.autonomic_regulator = AutonomicRegulator(
            motor_strength=0.5,
            hormonal_emissions=0.5,
            architect_signature=architect_signature
        )
        
        self.executive_relay = ExecutiveRelay(
            cortical_data={"motor_intent_signal": []},
            autonomic_data={"modulated_hormonal_emissions": {}},
            limbic_energy=0.5,
            architect_signature=architect_signature
        )
        
        self.proprioceptive_mirror = ProprioceptiveMirror(
            muscle_tension=[0.5, 0.5, 0.5, 0.5]  # Default tension
        )
        
        self.immune_governor = ImmuneSystemGovernor(architect_signature)
        
        # Initialize neural components
        self.neurons: Dict[str, Neuron] = {}
        self.sensory_cortex = SensoryCortex()
        self.motor_cortex = MotorCortex()
        
        # Network state
        self.network_activity = 0.0
        self.synaptic_plasticity = 1.0
        self.neural_coherence = 1.0
        
        # Signal routing
        self.signal_queue: List[NeuralSignal] = []
        self.processed_signals = []
        
        # Initialize network topology
        self._build_network_topology()
    
    def _build_network_topology(self):
        """Build the neural network topology"""
        # Create key neurons
        neuron_types = {
            'sensory_input': 'sensory',
            'cortical_processor': 'interneuron',
            'executive_controller': 'cortical',
            'autonomic_controller': 'autonomic',
            'motor_output': 'motor',
            'immune_coordinator': 'immune',
            'proprioceptive_processor': 'sensory',
            'field_resonance_detector': 'sensory'
        }
        
        for neuron_id, neuron_type in neuron_types.items():
            self.neurons[neuron_id] = Neuron(
                neuron_id=neuron_id,
                neuron_type=neuron_type,
                location="neural_network"
            )
        
        # Create connections
        connections = [
            ('sensory_input', 'cortical_processor', 0.8),
            ('cortical_processor', 'executive_controller', 0.9),
            ('executive_controller', 'motor_output', 0.7),
            ('executive_controller', 'autonomic_controller', 0.6),
            ('executive_controller', 'immune_coordinator', 0.5),
            ('proprioceptive_processor', 'cortical_processor', 0.7),
            ('field_resonance_detector', 'cortical_processor', 0.8),
            ('autonomic_controller', 'executive_controller', 0.4),  # Feedback
            ('immune_coordinator', 'executive_controller', 0.3),   # Feedback
        ]
        
        for pre, post, weight in connections:
            if pre in self.neurons and post in self.neurons:
                self.neurons[pre].add_connection(post, weight)
    
    def process_input(self, input_data: np.ndarray, input_type: str, 
                     context: Dict = None) -> Dict:
        """Process input through the complete neural network"""
        # 1. Sensory processing
        sensory_signal = self.sensory_cortex.process_sensory_input(input_type, input_data)
        
        # 2. Add to signal queue
        self.signal_queue.append(sensory_signal)
        
        # 3. Process through network
        network_response = self._process_neural_network()
        
        # 4. Integrate with existing brain functions
        integrated_response = self._integrate_brain_functions(input_data, context or {})
        
        # 5. Update network state
        self._update_network_state()
        
        return {
            'sensory_processing': {
                'signal_id': sensory_signal.signal_id,
                'coherence': sensory_signal.coherence,
                'strength': sensory_signal.strength
            },
            'neural_network': network_response,
            'brain_integration': integrated_response,
            'network_state': {
                'activity': self.network_activity,
                'coherence': self.neural_coherence,
                'signals_processed': len(self.processed_signals)
            }
        }
    
    def _process_neural_network(self) -> Dict:
        """Process signals through the neural network"""
        signals_processed = 0
        neurons_fired = 0
        
        # Process all signals in queue
        while self.signal_queue:
            signal = self.signal_queue.pop(0)
            
            # Route signal to appropriate neurons
            if signal.destination in self.neurons:
                neuron = self.neurons[signal.destination]
                fired = neuron.receive_signal(signal)
                
                if fired:
                    neurons_fired += 1
                    
                    # Create new signals for connected neurons
                    for synapse in neuron.connections:
                        new_signal = NeuralSignal(
                            signal_id=f"neural_{datetime.now().timestamp()}",
                            source=neuron.id,
                            destination=synapse.post_neuron,
                            data=signal.data,
                            timestamp=datetime.now(),
                            strength=synapse.weight,
                            coherence=signal.coherence
                        )
                        self.signal_queue.append(new_signal)
            
            signals_processed += 1
            self.processed_signals.append(signal)
        
        return {
            'signals_processed': signals_processed,
            'neurons_fired': neurons_fired,
            'network_activity': self._calculate_network_activity()
        }
    
    def _integrate_brain_functions(self, input_data: np.ndarray, context: Dict) -> Dict:
        """Integrate with existing brain functions"""
        integration_result = {}
        
        # 1. Autonomic Regulation
        try:
            # Calculate motor strength from input coherence
            motor_strength = EQUATION_ENGINE.coherence_index(input_data)
            
            # Calculate hormonal emissions from input entropy
            entropy = EQUATION_ENGINE.entropy_measure(input_data)
            hormonal_emissions = min(1.0, entropy / 5.0)
            
            # Update autonomic regulator
            self.autonomic_regulator.motor_strength = motor_strength
            self.autonomic_regulator.hormonal_emissions = hormonal_emissions
            
            # Get autonomic response
            autonomic_response = self.autonomic_regulator.regulator()
            integration_result['autonomic'] = autonomic_response
            
        except Exception as e:
            integration_result['autonomic'] = {'error': str(e)}
        
        # 2. Executive Relay
        try:
            # Prepare cortical data
            cortical_data = {
                'motor_intent_signal': input_data.tolist()[:10]  # First 10 values
            }
            
            # Update executive relay
            self.executive_relay.cortical_data = cortical_data
            self.executive_relay.autonomic_data = integration_result.get('autonomic', {}).get('data', {})
            self.executive_relay.limbic_energy = EQUATION_ENGINE.coherence_index(input_data)
            
            # Get executive response
            executive_response = self.executive_relay.relay()
            integration_result['executive'] = executive_response
            
        except Exception as e:
            integration_result['executive'] = {'error': str(e)}
        
        # 3. Proprioceptive Mirror
        try:
            # Convert input to muscle tension simulation
            muscle_tension = np.abs(input_data[:4])  # First 4 values as muscle groups
            muscle_tension = np.clip(muscle_tension, 0, 1)  # Normalize
            
            # Update proprioceptive mirror
            self.proprioceptive_mirror.muscle_tension = muscle_tension.tolist()
            
            # Get proprioceptive response
            proprioceptive_response = self.proprioceptive_mirror.reflect_state()
            integration_result['proprioceptive'] = proprioceptive_response
            
        except Exception as e:
            integration_result['proprioceptive'] = {'error': str(e)}
        
        # 4. Immune System Governor
        try:
            # Check if input represents a threat
            threat_level = self._assess_threat_level(input_data)
            
            if threat_level > 0.7:
                # Activate immune response
                immune_response = self.immune_governor.respond_to_signature("threat_detected")
                integration_result['immune'] = immune_response
            else:
                # Normal operation
                immune_response = self.immune_governor.respond_to_signature(self.immune_governor.architect_signature)
                integration_result['immune'] = immune_response
                
        except Exception as e:
            integration_result['immune'] = {'error': str(e)}
        
        return integration_result
    
    def _assess_threat_level(self, input_data: np.ndarray) -> float:
        """Assess threat level from input data"""
        # High entropy = potential threat
        entropy = EQUATION_ENGINE.entropy_measure(input_data)
        
        # High decoherence = potential threat
        decoherence = EQUATION_ENGINE.decoherence_index(input_data)
        
        # Combine into threat score
        threat = (entropy / 10.0) + decoherence
        return min(1.0, max(0.0, threat))
    
    def _calculate_network_activity(self) -> float:
        """Calculate overall network activity level"""
        if not self.neurons:
            return 0.0
        
        total_activation = sum(neuron.activation for neuron in self.neurons.values())
        total_neurons = len(self.neurons)
        
        return total_activation / max(1, total_neurons)
    
    def _update_network_state(self):
        """Update overall network state"""
        # Calculate network activity
        self.network_activity = self._calculate_network_activity()
        
        # Calculate neural coherence
        if self.processed_signals:
            coherences = [signal.coherence for signal in self.processed_signals[-10:]]  # Last 10
            self.neural_coherence = np.mean(coherences)
        else:
            self.neural_coherence = 1.0
        
        # Update synaptic plasticity based on activity
        if self.network_activity > 0.5:
            self.synaptic_plasticity = min(1.0, self.synaptic_plasticity + 0.01)
        else:
            self.synaptic_plasticity = max(0.1, self.synaptic_plasticity - 0.01)
    
    def get_network_status(self) -> Dict:
        """Get complete network status"""
        return {
            'network_state': {
                'activity': self.network_activity,
                'coherence': self.neural_coherence,
                'plasticity': self.synaptic_plasticity,
                'total_neurons': len(self.neurons),
                'signals_processed': len(self.processed_signals)
            },
            'neuron_states': {
                neuron_id: {
                    'type': neuron.type,
                    'activation': neuron.activation,
                    'connections': len(neuron.connections),
                    'last_fired': neuron.last_fired.isoformat() if neuron.last_fired else None
                }
                for neuron_id, neuron in self.neurons.items()
            },
            'brain_functions': {
                'autonomic_regulator': 'active',
                'executive_relay': 'active',
                'proprioceptive_mirror': 'active',
                'immune_governor': 'active'
            }
        }
    
    def stimulate_network(self, stimulus_type: str, intensity: float = 0.5):
        """Stimulate the network with specific input"""
        # Create stimulus data
        if stimulus_type == "coherent":
            stimulus_data = np.sin(np.linspace(0, 4*np.pi, 100)) * intensity
        elif stimulus_type == "noise":
            stimulus_data = np.random.normal(0, intensity, 100)
        elif stimulus_type == "pattern":
            stimulus_data = np.tile([1, -1, 1, -1], 25) * intensity
        else:
            stimulus_data = np.random.random(100) * intensity
        
        # Process through network
        response = self.process_input(
            input_data=stimulus_data,
            input_type="network_stimulation",
            context={'stimulus_type': stimulus_type, 'intensity': intensity}
        )
        
        return response
    
    def learn_pattern(self, pattern: np.ndarray, pattern_name: str):
        """Learn a new pattern by strengthening relevant connections"""
        # Process pattern through network
        response = self.process_input(pattern, "pattern_learning", {'pattern_name': pattern_name})
        
        # Strengthen connections that were active during learning
        for neuron in self.neurons.values():
            if neuron.last_fired and (datetime.now() - neuron.last_fired).seconds < 1:
                # Recent activity - strengthen connections
                for synapse in neuron.connections:
                    synapse.weight = min(1.0, synapse.weight + 0.1)
        
        return response


# Global neural network instance
BALANTIUM_NEURAL_NETWORK = BalantiumNeuralNetwork()


if __name__ == "__main__":
    print("🧠 Balantium Neural Network - Complete Nervous System Initialized")
    print("   Integrated Components:")
    print("   ✅ AutonomicRegulator")
    print("   ✅ ExecutiveRelay") 
    print("   ✅ ProprioceptiveMirror")
    print("   ✅ ImmuneSystemGovernor")
    
    status = BALANTIUM_NEURAL_NETWORK.get_network_status()
    print(f"   Network Activity: {status['network_state']['activity']:.2f}")
    print(f"   Neural Coherence: {status['network_state']['coherence']:.2f}")
    print(f"   Total Neurons: {status['network_state']['total_neurons']}")
