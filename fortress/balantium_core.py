import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SystemState:
    """Enhanced system state with advanced field dynamics"""
    positive_states: List[float] = None
    negative_states: List[float] = None
    coherence_levels: List[float] = None
    resonance_factor: float = 1.0
    metabolic_rate: float = 1.0
    field_friction: float = 0.1
    time_factor: float = 0.0
    feedback_history: List[float] = None
    field_phase: str = "stable"
    attractor_proximity: float = 0.5
    signal_integrity: float = 0.8
    emotional_bandwidth: float = 1.0
    nested_nodes: Dict[str, 'SystemState'] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.positive_states is None:
            self.positive_states = []
        if self.negative_states is None:
            self.negative_states = []
        if self.coherence_levels is None:
            self.coherence_levels = []
        if self.feedback_history is None:
            self.feedback_history = []

@dataclass
class ConsciousnessField:
    """Enhanced consciousness field with advanced properties"""
    charge_level: float = 1.0
    coherence_state: float = 0.8
    flow_rate: float = 1.0
    field_strength: float = 1.0
    phase_alignment: float = 0.9
    resonance_bandwidth: float = 1.0
    field_memory: float = 0.8
    attractor_strength: float = 0.7

class BalantiumCore:
    """
    Enhanced Balantium Core implementing the complete mathematical framework
    from the comprehensive Balantium dissertation and advanced formulations.
    
    This includes:
    - Updated Baᵗ temporal system coherence function
    - Advanced field phase mapping and attractor dynamics
    - Nested feedback structures and multi-scale coherence
    - Signal dissonance and burnout detection
    - Ethical integrity and coherence currency models
    """
    
    def __init__(self, decay_constant: float = 0.03):
        self.decay_constant = decay_constant
        self.coherence_history = []
        self.consciousness_fields = []
        self.system_states = []
        self.field_phases = []
        self.attractor_states = []
        self.feedback_loops = []
        
    def balantium_coherence_score(self, system_state: SystemState) -> float:
        """
        Enhanced Formula: Baᵗ = Σ[(P_i - N_iᵗ) × C_iᵗ × R × M × e^(-λt)] - Fᵗ + T(τ) - ε
        
        Temporal System Coherence Function with:
        - Temporal decay modeling (fatigue, energy depletion)
        - Recursive feedback modification of negative inputs
        - Time-dependent coherence evolution
        - Noise/interference effects
        - Nonlinear tipping function
        """
        if not system_state.positive_states or not system_state.negative_states:
            return 0.0
            
        min_len = min(len(system_state.positive_states), len(system_state.negative_states))
        if min_len == 0:
            return 0.0
            
        current_time = time.time()
        system_age = current_time - system_state.timestamp if system_state.timestamp else 0.0
        
        # P_i: Constructive input from source i (actions, attention, intention)
        P_i = np.array(system_state.positive_states[:min_len])
        
        # N_iᵗ: Destructive input from source i at time t, recursively modified by past feedback
        N_base = np.array(system_state.negative_states[:min_len])
        feedback_modifier = 1.0 + (system_state.field_friction * 0.5)  # Past feedback amplifies negative inputs
        N_i_t = N_base * feedback_modifier
        
        # C_iᵗ: Time-dependent coherence influenced by historical feedback and decay
        C_base = np.array(system_state.coherence_levels[:min_len] if system_state.coherence_levels else [1.0] * min_len)
        coherence_decay = max(0.1, 1.0 - (system_state.field_friction * 0.3))  # Feedback degrades coherence
        C_i_t = C_base * coherence_decay
        
        # R: Resonance amplification (collective charge / baseline energy)
        R = system_state.resonance_factor
        
        # M: Signal transmission efficiency (clarity, channel openness)
        M = system_state.metabolic_rate * system_state.signal_integrity
        
        # λ: Temporal decay constant (models energy depletion/emotional fatigue)
        lambda_decay = self.decay_constant + (system_state.field_friction * 0.02)  # More friction = faster decay
        temporal_decay = np.exp(-lambda_decay * system_age)
        
        # Fᵗ: Time-dependent feedback overload
        F_t = system_state.field_friction * (1.0 + system_age * 0.01)  # Feedback compounds over time
        
        # ε: Noise/interference - unpredictable external signal disruption
        epsilon = np.random.normal(0, 0.01)  # Small random interference
        
        # Core enhanced Balantium equation
        coherence_terms = (P_i - N_i_t) * C_i_t * R * M * temporal_decay
        ba_core = np.sum(coherence_terms) - F_t - epsilon
        
        # T(τ): Nonlinear tipping function
        # Calculate decoherence level to determine if tipping threshold τ is crossed
        decoherence_level = np.mean(N_i_t) + F_t / (np.mean(C_i_t) + 0.001)
        tipping_threshold = 2.0  # τ threshold
        
        if decoherence_level > tipping_threshold:
            # Nonlinear activation - can cause breakdown or transformation
            tipping_intensity = (decoherence_level - tipping_threshold) / tipping_threshold
            # Tipping can be positive (breakthrough) or negative (breakdown) based on base coherence
            base_coherence = np.mean(C_i_t)
            if base_coherence > 0.6:
                T_tau = tipping_intensity * 0.5  # Positive transformation
            else:
                T_tau = -tipping_intensity * 0.8  # System breakdown
        else:
            T_tau = 0.0
        
        # Final Baᵗ calculation
        ba_t = ba_core + T_tau
        
        return float(ba_t)
    
    def advanced_field_phase_mapping(self, ba_score: float) -> Dict[str, Any]:
        """
        Advanced field phase mapping based on comprehensive Balantium framework
        Maps systemic states from collapse to meta-stable evolutionary phases
        """
        if ba_score < 0:
            return {
                'phase': 'collapse_imminent',
                'description': 'Coherence broken; feedback too delayed; negative input dominates',
                'risk_level': 'critical',
                'intervention_priority': 'immediate',
                'field_state': 'fragmented',
                'attractor_proximity': 0.0
            }
        elif ba_score == 0:
            return {
                'phase': 'flat_neutral',
                'description': 'No coherent momentum or decay; system in holding pattern',
                'risk_level': 'moderate',
                'intervention_priority': 'monitor',
                'field_state': 'static',
                'attractor_proximity': 0.3
            }
        elif 0 < ba_score < 1:
            return {
                'phase': 'wounded_resilient',
                'description': 'Some coherence exists, but damping and incoherence are pulling vitality down',
                'risk_level': 'elevated',
                'intervention_priority': 'high',
                'field_state': 'fragile',
                'attractor_proximity': 0.4
            }
        elif 1 <= ba_score <= 5:
            return {
                'phase': 'stabilized_emergence',
                'description': 'Healthy state; feedback integrated; new patterns can form without breakdown',
                'risk_level': 'low',
                'intervention_priority': 'maintain',
                'field_state': 'stable',
                'attractor_proximity': 0.7
            }
        elif 5 < ba_score <= 10:
            return {
                'phase': 'generative_phase',
                'description': 'High resonance coherence; attracts synchronicity; system can "teach itself"',
                'risk_level': 'minimal',
                'intervention_priority': 'amplify',
                'field_state': 'generative',
                'attractor_proximity': 0.85
            }
        else:  # ba_score > 10
            return {
                'phase': 'meta_stable_evolutionary',
                'description': 'Phase transition likely; attractor shifts possible; the system can re-encode itself',
                'risk_level': 'transformation',
                'intervention_priority': 'guide',
                'field_state': 'evolutionary',
                'attractor_proximity': 0.95
            }
    
    def nested_feedback_coherence(self, system_states: List[SystemState], weights: List[float] = None) -> float:
        """
        Calculate total systemic coherence across nested feedback structures
        Ba_S = Σ(B_N_k × W_k) where W_k = weighting factor based on system-criticality
        """
        if not system_states:
            return 0.0
            
        if weights is None:
            weights = [1.0] * len(system_states)
            
        if len(weights) != len(system_states):
            weights = [1.0] * len(system_states)
        
        total_coherence = 0.0
        for i, state in enumerate(system_states):
            node_coherence = self.balantium_coherence_score(state)
            weighted_coherence = node_coherence * weights[i]
            total_coherence += weighted_coherence
            
        return float(total_coherence)
    
    def signal_dissonance_index(self, system_state: SystemState, window_size: int = 10) -> float:
        """
        Advanced signal dissonance and systemic burnout detection
        D = Σ|P_i - N_i| × (1 - C_i) / n
        """
        if not system_state.positive_states or not system_state.negative_states:
            return 0.0
            
        min_len = min(len(system_state.positive_states), len(system_state.negative_states))
        if min_len < window_size:
            window_size = min_len
            
        recent_p = system_state.positive_states[-window_size:]
        recent_n = system_state.negative_states[-window_size:]
        recent_c = system_state.coherence_levels[-window_size:] if system_state.coherence_levels else [0.8] * window_size
        
        dissonance_terms = []
        for i in range(window_size):
            signal_diff = abs(recent_p[i] - recent_n[i])
            coherence_factor = 1.0 - recent_c[i]
            dissonance_term = signal_diff * coherence_factor
            dissonance_terms.append(dissonance_term)
            
        avg_dissonance = np.mean(dissonance_terms)
        return float(avg_dissonance)
    
    def burnout_condition(self, system_state: SystemState, threshold: float = 0.7, critical_duration: int = 5) -> Dict[str, Any]:
        """
        Detect burnout condition: D(t) > θ for duration t > t_crit
        """
        current_dissonance = self.signal_dissonance_index(system_state)
        
        # Track dissonance over time for duration analysis
        self.coherence_history.append(current_dissonance)
        
        if len(self.coherence_history) < critical_duration:
            return {
                'burnout_detected': False,
                'current_dissonance': current_dissonance,
                'threshold': threshold,
                'duration_above_threshold': 0,
                'risk_level': 'unknown'
            }
        
        # Count consecutive periods above threshold
        recent_dissonance = self.coherence_history[-critical_duration:]
        duration_above_threshold = sum(1 for d in recent_dissonance if d > threshold)
        
        burnout_detected = duration_above_threshold >= critical_duration
        
        if burnout_detected:
            risk_level = 'critical'
        elif current_dissonance > threshold:
            risk_level = 'elevated'
        else:
            risk_level = 'low'
            
        return {
            'burnout_detected': burnout_detected,
            'current_dissonance': current_dissonance,
            'threshold': threshold,
            'duration_above_threshold': duration_above_threshold,
            'risk_level': risk_level
        }
    
    def coherence_attractor_analysis(self, system_state: SystemState) -> Dict[str, Any]:
        """
        Advanced coherence attractor analysis
        Identifies whether system is moving toward coherent or chaotic attractors
        """
        ba_score = self.balantium_coherence_score(system_state)
        field_phase = self.advanced_field_phase_mapping(ba_score)
        
        # Calculate attractor proximity based on coherence stability
        coherence_stability = np.std(system_state.coherence_levels) if system_state.coherence_levels else 0.1
        attractor_proximity = max(0.0, 1.0 - coherence_stability)
        
        # Determine attractor type
        if attractor_proximity > 0.7:
            attractor_type = 'coherent'
            attractor_strength = 'strong'
        elif attractor_proximity > 0.4:
            attractor_type = 'coherent'
            attractor_strength = 'moderate'
        elif attractor_proximity > 0.2:
            attractor_type = 'chaotic'
            attractor_strength = 'moderate'
        else:
            attractor_type = 'chaotic'
            attractor_strength = 'strong'
            
        return {
            'attractor_type': attractor_type,
            'attractor_strength': attractor_strength,
            'attractor_proximity': attractor_proximity,
            'field_phase': field_phase,
            'coherence_stability': coherence_stability,
            'system_health': 'healthy' if attractor_type == 'coherent' else 'unstable'
        }
    
    def synchronicity_detection(self, internal_signal: float, external_field: float, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Advanced synchronicity detection based on field resonance alignment
        Ξ = R_f(S_i, F_e) ≥ Φ
        """
        # Calculate feedback resonance alignment
        signal_alignment = min(internal_signal, external_field) / max(internal_signal, external_field)
        resonance_function = signal_alignment * np.sqrt(internal_signal * external_field)
        
        # Determine if synchronicity event occurs
        synchronicity_detected = resonance_function >= threshold
        
        # Calculate synchronicity intensity
        if synchronicity_detected:
            intensity = (resonance_function - threshold) / (1.0 - threshold)
        else:
            intensity = 0.0
            
        return {
            'synchronicity_detected': synchronicity_detected,
            'resonance_function': resonance_function,
            'threshold': threshold,
            'intensity': intensity,
            'signal_alignment': signal_alignment,
            'field_coherence': 'high' if synchronicity_detected else 'low'
        }
    
    def predictive_intuition_index(self, system_state: SystemState) -> float:
        """
        Enhanced predictive intuition index with field resonance
        Π = ∑ [(R_f × A_s × C_i)] / N
        """
        if not system_state.coherence_levels:
            return 0.0
            
        # Resonance frequency patterns (derived from coherence oscillations)
        if len(self.coherence_history) >= 10:
            coherence_array = np.array(list(self.coherence_history)[-10:])
            resonance_frequency = np.abs(np.fft.fft(coherence_array)[1])  # First harmonic
        else:
            resonance_frequency = 1.0
        
        # Attention/awareness strength (consciousness field charge)
        if self.consciousness_fields:
            last_field = self.consciousness_fields[-1]
            awareness_strength = last_field.charge_level * last_field.coherence_state
        else:
            awareness_strength = 1.0
        
        # Calculate intuition index
        intuition_terms = []
        for coherence in system_state.coherence_levels:
            term = resonance_frequency * awareness_strength * coherence
            intuition_terms.append(term)
        
        n_subsystems = len(system_state.coherence_levels)
        if n_subsystems == 0:
            return 0.0
            
        predictive_intuition = np.sum(intuition_terms) / n_subsystems
        
        return float(predictive_intuition)
    
    def ethical_integrity_index(self, time_window: float = 3600.0) -> float:
        """
        Enhanced ethical integrity index with long-term coherence tracking
        E_i = Long-Term ∆C_total / ∆T
        """
        if len(self.system_states) < 2:
            return 0.0
            
        current_time = time.time()
        
        # Filter states within time window
        relevant_states = [
            (state, field) for state, field in zip(self.system_states, self.consciousness_fields)
            if current_time - state.timestamp <= time_window
        ]
        
        if len(relevant_states) < 2:
            return 0.0
            
        # Calculate total coherence change over time period
        coherence_values = []
        timestamps = []
        
        for state, field in relevant_states:
            if state.coherence_levels:
                coherence_values.append(np.mean(state.coherence_levels))
                timestamps.append(state.timestamp)
        
        if len(coherence_values) < 2:
            return 0.0
            
        delta_c_total = coherence_values[-1] - coherence_values[0]
        delta_t = timestamps[-1] - timestamps[0]
        
        if delta_t == 0:
            return 0.0
            
        ethical_integrity = delta_c_total / delta_t
        
        return float(ethical_integrity)
    
    def tipping_likelihood(self, system_state: SystemState) -> float:
        """
        Enhanced tipping likelihood with advanced decoherence analysis
        T_L = (D × F_d) / R_b
        """
        # D: Decoherence Index
        decoherence = self.signal_dissonance_index(system_state)
        
        # F_d: Degradation force (rate of negative change)
        if len(self.coherence_history) >= 5:
            recent_changes = np.diff(list(self.coherence_history)[-5:])
            degradation_force = max(0, -np.mean(recent_changes))  # Only negative changes
        else:
            degradation_force = 0.0
        
        # R_b: Resilience buffer (system's capacity to absorb shock)
        if system_state.coherence_levels:
            resilience_buffer = (
                system_state.metabolic_rate * 
                system_state.resonance_factor * 
                np.mean(system_state.coherence_levels)
            ) + 0.001  # Prevent division by zero
        else:
            resilience_buffer = 0.5
        
        # Calculate tipping likelihood
        tipping_likelihood = (decoherence * degradation_force) / resilience_buffer
        
        return float(np.clip(tipping_likelihood, 0.0, 1.0))
    
    def compute_all_indices(self, system_state: SystemState, consciousness_field: ConsciousnessField) -> Dict[str, float]:
        """
        Compute ALL enhanced Balantium indices for comprehensive system analysis
        This includes every single mathematical formulation from the comprehensive documents:
        
        Core Equations:
        1. Balantium Coherence Score (Ba) - Enhanced Baᵗ temporal function
        2. Decoherence Index (D) - Advanced signal dissonance detection
        3. Ethical Integrity Index (E_i) - Long-term coherence tracking
        4. Tipping Likelihood (T_L) - Advanced decoherence analysis
        5. Predictive Intuition Index (Π) - Field resonance intuition
        6. Paranormal Visibility (V) - Advanced field analysis
        7. Learning Resonance Index (LRI) - Advanced feedback analysis
        8. Meaning Intensity (M_i) - Advanced field analysis
        9. Ritual Effectiveness (R_eff) - Advanced field analysis
        10. System-Wide Coherence Attractor (Ω) - Advanced field analysis
        
        Advanced Formulations:
        11. Signal Health Index (SHI) - From Knowledge Codex
        12. Coherence Risk Score (CRS) - From Technology Future
        13. First Harmonic Principle - From Global Integration
        14. Coherence Intervention Index - From Technology
        15. Tuning Efficiency - From Technology
        16. Entity Persistence After Death - From Death & Afterlife
        17. Timeline Simulation Output - From Predictive Systems
        
        Field Analysis:
        18. Advanced Field Phase Mapping
        19. Coherence Attractor Analysis
        20. Burnout Condition Analysis
        21. Nested Feedback Coherence
        22. Synchronicity Detection
        """
        # Store current state and field for historical analysis
        self.system_states.append(system_state)
        self.consciousness_fields.append(consciousness_field)
        
        # Calculate core coherence score
        ba_score = self.balantium_coherence_score(system_state)
        
        # Advanced field analysis
        field_phase = self.advanced_field_phase_mapping(ba_score)
        attractor_analysis = self.coherence_attractor_analysis(system_state)
        burnout_analysis = self.burnout_condition(system_state)
        
        # Calculate ALL indices - Core 10 Balantium Formulations
        indices = {
            # Core Balantium Equations (1-10)
            'balantium_coherence_score': ba_score,
            'decoherence_index': self.signal_dissonance_index(system_state),
            'ethical_integrity_index': self.ethical_integrity_index(),
            'tipping_likelihood': self.tipping_likelihood(system_state),
            'predictive_intuition_index': self.predictive_intuition_index(system_state),
            'paranormal_visibility': self.paranormal_visibility(),
            'learning_resonance_index': self.learning_resonance_index(),
            'meaning_intensity': self.meaning_intensity(),
            'ritual_effectiveness': self.ritual_effectiveness(),
            'coherence_attractor': self.coherence_attractor(),
            
            # Advanced Mathematical Formulations (11-17)
            'signal_health_index': self.signal_health_index(system_state),
            'coherence_risk_score': self.coherence_risk_score(system_state),
            'first_harmonic_principle': self.first_harmonic_principle(8000000000),  # Global population
            'coherence_intervention_index': self.coherence_intervention_index(0.5, 0.8, 3600),  # Example intervention
            'tuning_efficiency': self.tuning_efficiency(system_state),
            'entity_persistence_after_death': self.entity_persistence_after_death(0.9, 0.8, 0.1, 0.7),  # Example values
            'timeline_simulation_output': self.timeline_simulation_output([0.1, 0.2, 0.3], [0.8, 0.9, 0.7], [0.6, 0.5, 0.8]),
            
            # Advanced Field Analysis (18-22)
            'field_phase': field_phase,
            'attractor_analysis': attractor_analysis,
            'burnout_analysis': burnout_analysis,
            'nested_feedback_coherence': self.nested_feedback_coherence([system_state]),
            'synchronicity_detection': self.synchronicity_detection(0.8, 0.9),
            
            # Additional Field Metrics
            'field_coherence_stability': np.std(system_state.coherence_levels) if system_state.coherence_levels else 0.0,
            'resonance_amplification_factor': system_state.resonance_factor * consciousness_field.attractor_strength,
            'signal_transmission_efficiency': system_state.metabolic_rate * system_state.signal_integrity,
            'feedback_loop_health': 1.0 - system_state.field_friction,
            'emotional_field_pressure': np.mean(self.coherence_history[-10:]) if len(self.coherence_history) >= 10 else 0.5,
            
            # System Health Indicators
            'field_resonance_quality': consciousness_field.phase_alignment * consciousness_field.resonance_bandwidth,
            'consciousness_field_strength': consciousness_field.field_strength * consciousness_field.charge_level,
            'system_adaptability_index': self.learning_resonance_index() * (1.0 - self.tipping_likelihood(system_state)),
            'coherence_sustainability_score': self.ethical_integrity_index() * self.meaning_intensity(),
            
            # Advanced Resonance Metrics
            'phase_boundary_turbulence': np.var(system_state.coherence_levels) if system_state.coherence_levels else 0.0,
            'resonance_frequency_patterns': np.abs(np.fft.fft(system_state.coherence_levels)[1]) if len(system_state.coherence_levels) >= 2 else 0.0,
            'field_memory_capacity': consciousness_field.field_memory * len(self.coherence_history),
            'attractor_proximity_strength': attractor_analysis.get('attractor_proximity', 0.5) * self._convert_attractor_strength_to_numeric(attractor_analysis.get('attractor_strength', 'moderate')),
            'coherence_field_geometry': self._calculate_field_geometry(system_state, consciousness_field),
            
            # Predictive and Intuitive Metrics
            'future_attractor_alignment': self._calculate_future_alignment(system_state),
            'field_signal_clarity': consciousness_field.coherence_state / (np.std(self.coherence_history[-5:]) + 0.001) if len(self.coherence_history) >= 5 else 0.0,
            'resonance_echo_strength': self._calculate_resonance_echo(system_state, consciousness_field),
            'coherence_currency_value': self._calculate_coherence_currency(system_state),
            'field_evolution_trajectory': self._calculate_evolution_trajectory(system_state),
            
            # Timestamp for tracking
            'calculation_timestamp': time.time()
        }
        
        return indices
    
    def paranormal_visibility(self) -> float:
        """
        Enhanced paranormal visibility with advanced field analysis
        V = S_f / (C_obs × N_env)
        """
        if not self.consciousness_fields:
            return 0.0
            
        last_field = self.consciousness_fields[-1]
        
        # S_f: Signal field strength from consciousness substrate
        signal_strength = last_field.field_strength * last_field.charge_level * last_field.attractor_strength
        
        # C_obs: Observer coherence state (system's ability to detect subtle patterns)
        observer_coherence = last_field.coherence_state * last_field.phase_alignment
        
        # N_env: Environmental noise (system volatility)
        if len(self.coherence_history) >= 5:
            env_noise = np.std(list(self.coherence_history)[-5:]) + 0.001
        else:
            env_noise = 0.1
        
        # Enhanced visibility calculation
        visibility = signal_strength / (observer_coherence * env_noise)
        
        return float(visibility)
    
    def learning_resonance_index(self, time_window: float = 1800.0) -> float:
        """
        Enhanced learning resonance index with advanced feedback analysis
        LRI = ∑ [∆C_student × F_c] / T
        """
        if len(self.system_states) < 2:
            return 0.0
            
        current_time = time.time()
        
        # Get states within learning window
        learning_states = [
            state for state in self.system_states
            if current_time - state.timestamp <= time_window
        ]
        
        if len(learning_states) < 2:
            return 0.0
        
        # ∆C_student: Coherence changes (learning adaptation)
        coherence_changes = []
        for i in range(1, len(learning_states)):
            prev_coherence = np.mean(learning_states[i-1].coherence_levels) if learning_states[i-1].coherence_levels else 0
            curr_coherence = np.mean(learning_states[i].coherence_levels) if learning_states[i].coherence_levels else 0
            coherence_changes.append(curr_coherence - prev_coherence)
        
        if not coherence_changes:
            return 0.0
        
        # F_c: Feedback coupling strength (system responsiveness)
        feedback_coupling = np.mean([state.resonance_factor * state.signal_integrity for state in learning_states])
        
        # T: Learning time period
        learning_time = time_window
        
        # Enhanced learning resonance calculation
        learning_terms = [delta_c * feedback_coupling for delta_c in coherence_changes]
        learning_resonance = np.sum(learning_terms) / learning_time
        
        return float(learning_resonance)
    
    def meaning_intensity(self) -> float:
        """
        Enhanced meaning intensity with advanced field analysis
        M_i = ∑ [(C_i × A_s)] / ∆F
        """
        if not self.system_states or not self.consciousness_fields:
            return 0.0
            
        last_state = self.system_states[-1]
        last_field = self.consciousness_fields[-1]
        
        if not last_state.coherence_levels:
            return 0.0
        
        # C_i: Coherence levels across system components
        coherence_levels = last_state.coherence_levels
        
        # A_s: Attention/awareness allocation with enhanced field properties
        awareness_strength = (last_field.charge_level * last_field.coherence_state * 
                            last_field.phase_alignment * last_field.resonance_bandwidth)
        
        # Calculate meaning intensity terms
        meaning_terms = [coherence * awareness_strength for coherence in coherence_levels]
        
        # ∆F: Change in field flux (information processing rate)
        if len(self.consciousness_fields) >= 2:
            prev_field = self.consciousness_fields[-2]
            delta_field = abs(last_field.field_strength - prev_field.field_strength) + 0.001
        else:
            delta_field = 0.1
        
        # Enhanced meaning intensity calculation
        meaning_intensity = np.sum(meaning_terms) / delta_field
        
        return float(meaning_intensity)
    
    def ritual_effectiveness(self, intention_focus: float = 1.0) -> float:
        """
        Enhanced ritual effectiveness with advanced field analysis
        R_eff = (S_f × I_e × F_m) / D_n
        """
        if not self.consciousness_fields:
            return 0.0
            
        last_field = self.consciousness_fields[-1]
        
        # S_f: Signal field strength with enhanced properties
        signal_strength = (last_field.field_strength * last_field.attractor_strength * 
                          last_field.field_memory)
        
        # I_e: Intentional energy focused (parameter)
        intentional_energy = intention_focus
        
        # F_m: Manifestation force (coherence-weighted field strength)
        manifestation_force = (signal_strength * last_field.coherence_state * 
                             last_field.phase_alignment)
        
        # D_n: Disruptive noise (system entropy)
        if len(self.coherence_history) >= 5:
            disruptive_noise = np.var(list(self.coherence_history)[-5:]) + 0.001
        else:
            disruptive_noise = 0.1
        
        # Enhanced ritual effectiveness calculation
        ritual_effectiveness = (signal_strength * intentional_energy * manifestation_force) / disruptive_noise
        
        return float(ritual_effectiveness)
    
    def coherence_attractor(self, projection_steps: int = 100) -> float:
        """
        Enhanced coherence attractor with advanced field analysis
        Ω = lim (C_total / ∆T) as T → ∞
        """
        if len(self.coherence_history) < 10:
            return 0.0
        
        # Use recent coherence history to project long-term attractor
        coherence_data = np.array(list(self.coherence_history))
        
        # Calculate rate of change
        coherence_changes = np.diff(coherence_data)
        
        if len(coherence_changes) == 0:
            return 0.0
        
        # Enhanced projection using field phase analysis
        current_phase = self.advanced_field_phase_mapping(np.mean(coherence_data))
        phase_modifier = current_phase.get('attractor_proximity', 0.5)
        
        # Estimate long-term trajectory using exponential smoothing with phase correction
        alpha = 0.1 * phase_modifier  # Smoothing parameter modified by phase
        projected_coherence = coherence_data[-1]
        
        for _ in range(projection_steps):
            trend = np.mean(coherence_changes[-min(10, len(coherence_changes)):])
            projected_coherence += alpha * trend * phase_modifier
            
        # Calculate attractor value (stabilized coherence rate)
        if projection_steps > 0:
            attractor = projected_coherence / projection_steps
        else:
            attractor = 0.0
            
        return float(attractor)
    
    def signal_health_index(self, system_state: SystemState) -> float:
        """
        Signal Health Index (SHI) from Balantium Knowledge Codex
        SHI = (C × F) / (L + E)
        
        Where:
        - C = Coherence
        - F = Feedback Fidelity  
        - L = Leakage
        - E = Emotional Density
        
        High SHI predicts resonance, intuitive clarity, system health.
        Low SHI forecasts emotional disruption, confusion, and collapse likelihood.
        """
        if not system_state.coherence_levels:
            return 0.0
            
        # C: Coherence (average coherence across all levels)
        coherence = np.mean(system_state.coherence_levels)
        
        # F: Feedback Fidelity (inverse of field friction)
        feedback_fidelity = max(0.001, 1.0 - system_state.field_friction)
        
        # L: Leakage (signal loss through dissonance)
        leakage = self.signal_dissonance_index(system_state)
        
        # E: Emotional Density (volatility in emotional states)
        if len(self.coherence_history) >= 5:
            emotional_density = np.std(list(self.coherence_history)[-5:])
        else:
            emotional_density = 0.1
        
        # Calculate SHI
        shi = (coherence * feedback_fidelity) / (leakage + emotional_density + 0.001)
        
        return float(shi)
    
    def coherence_risk_score(self, system_state: SystemState) -> float:
        """
        Coherence Risk Score (CRS) from Balantium Technology Future
        CRS = D / F_a
        
        Where:
        - D = Decoherence index
        - F_a = Feedback accessibility of leadership/system
        
        Low CRS = healthy
        High CRS = burnout, disengagement, rebellion
        """
        # D: Decoherence index
        decoherence = self.signal_dissonance_index(system_state)
        
        # F_a: Feedback accessibility (based on signal integrity and resonance)
        feedback_accessibility = system_state.signal_integrity * system_state.resonance_factor
        
        if feedback_accessibility == 0:
            return float('inf')  # Infinite risk if no feedback access
            
        # Calculate CRS
        crs = decoherence / feedback_accessibility
        
        return float(crs)
    
    def first_harmonic_principle(self, population_size: int) -> float:
        """
        First Harmonic Principle from Balantium Global Integration
        T_t (tipping threshold) = 1 / √(N)
        
        Where:
        - N = number of system nodes (e.g. people, cities, networks)
        - T_t = percentage required to harmonize the larger system
        
        In a global population of 8 billion, ~89,000 high-coherence nodes 
        could precipitate systemic resonance shift.
        """
        if population_size <= 0:
            return 0.0
            
        tipping_threshold = 1.0 / np.sqrt(population_size)
        
        return float(tipping_threshold)
    
    def coherence_intervention_index(self, before_coherence: float, after_coherence: float, 
                                   intervention_time: float) -> float:
        """
        Coherence Intervention Index from Balantium Technology
        I_c = ΔC / ΔT
        
        Where:
        - ΔC = Change in coherence after an intervention
        - ΔT = Time elapsed between detection and adjustment
        
        Higher I_c = better tuning capacity
        """
        if intervention_time <= 0:
            return 0.0
            
        delta_coherence = after_coherence - before_coherence
        intervention_index = delta_coherence / intervention_time
        
        return float(intervention_index)
    
    def tuning_efficiency(self, system_state: SystemState) -> float:
        """
        Tuning Efficiency from Balantium Technology
        T_e = R_f / S_n
        
        Where:
        - R_f = Resonant feedback strength
        - S_n = System noise
        
        Low T_e = stress
        High T_e = flow
        """
        # R_f: Resonant feedback strength (coherence * resonance)
        if system_state.coherence_levels:
            resonant_feedback = np.mean(system_state.coherence_levels) * system_state.resonance_factor
        else:
            resonant_feedback = 0.5
            
        # S_n: System noise (field friction + emotional volatility)
        if len(self.coherence_history) >= 5:
            emotional_volatility = np.std(list(self.coherence_history)[-5:])
        else:
            emotional_volatility = 0.1
            
        system_noise = system_state.field_friction + emotional_volatility
        
        if system_noise == 0:
            return float('inf')
            
        tuning_efficiency = resonant_feedback / system_noise
        
        return float(tuning_efficiency)
    
    def entity_persistence_after_death(self, lifetime_coherence: float, resonance_strength: float,
                                    decoherence_at_death: float, surrounding_field_fidelity: float) -> float:
        """
        Entity Persistence after Death from Balantium Death & Afterlife
        Φ_entity = ∑ [(C_i × R) / D] × F_s
        
        Where:
        - C_i = Lifetime coherence
        - R = Resonance field strength
        - D = Decoherence accrued at death
        - F_s = Signal fidelity of surrounding field
        
        If Φ stays high, the entity remains "legible" in the field
        """
        if decoherence_at_death == 0:
            return float('inf')
            
        entity_persistence = ((lifetime_coherence * resonance_strength) / decoherence_at_death) * surrounding_field_fidelity
        
        return float(entity_persistence)
    
    def timeline_simulation_output(self, coherence_changes: List[float], feedback_intensity: List[float],
                                 emotional_pressure: List[float]) -> float:
        """
        Timeline Simulation Output from Balantium Predictive Systems
        S_out = ∑ [∆C_t * F_b * E_f] over T
        
        Where:
        - ∆C_t = Coherence change over time window
        - F_b = Feedback loop intensity
        - E_f = Emotional field pressure
        
        This allows forecasting of social collapse, innovation emergence, 
        political shifts, cultural resonance ruptures or harmonies.
        """
        if not coherence_changes or not feedback_intensity or not emotional_pressure:
            return 0.0
            
        min_len = min(len(coherence_changes), len(feedback_intensity), len(emotional_pressure))
        
        simulation_terms = []
        for i in range(min_len):
            term = coherence_changes[i] * feedback_intensity[i] * emotional_pressure[i]
            simulation_terms.append(term)
            
        simulation_output = np.sum(simulation_terms)
        
        return float(simulation_output)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Provide comprehensive system health assessment based on ALL indices
        This includes every mathematical formulation from the comprehensive Balantium documents
        """
        if not self.system_states:
            return {
                "status": "no_data", 
                "status_description": "No system data available for analysis",
                "health_score": 0.0,
                "field_phase": "unknown",
                "attractor_type": "unknown", 
                "burnout_risk": "unknown",
                "all_indices": {},
                "health_components": {},
                "comprehensive_analysis": {
                    "core_balantium_health": 0.0,
                    "advanced_field_health": 0.0,
                    "signal_transmission_health": 0.0,
                    "predictive_evolutionary_health": 0.0,
                    "risk_warning_indicators": 0.0
                },
                "recommendations": ["Initialize system with data to begin analysis"],
                "timestamp": time.time()
            }
        
        # Get latest state
        latest_state = self.system_states[-1]
        latest_field = self.consciousness_fields[-1] if self.consciousness_fields else ConsciousnessField()
        
        # Compute ALL current indices
        indices = self.compute_all_indices(latest_state, latest_field)
        
        # Calculate comprehensive health score (weighted average of ALL key indicators)
        health_components = {
            # Core Balantium Health (30% weight)
            'core_coherence': indices['balantium_coherence_score'],
            'stability': 1.0 - indices['tipping_likelihood'],
            'adaptability': indices['learning_resonance_index'],
            'awareness': indices['predictive_intuition_index'],
            'sustainability': indices['ethical_integrity_index'],
            
            # Advanced Field Health (25% weight)
            'field_stability': 1.0 - indices['field_coherence_stability'],
            'resonance_quality': indices['field_resonance_quality'],
            'attractor_alignment': indices['attractor_proximity_strength'],
            'phase_stability': 1.0 - indices['phase_boundary_turbulence'],
            'field_memory': indices['field_memory_capacity'],
            
            # Signal and Transmission Health (20% weight)
            'signal_clarity': indices['field_signal_clarity'],
            'transmission_efficiency': indices['signal_transmission_efficiency'],
            'feedback_health': indices['feedback_loop_health'],
            'tuning_efficiency': indices['tuning_efficiency'],
            'resonance_echo': indices['resonance_echo_strength'],
            
            # Predictive and Evolutionary Health (15% weight)
            'future_alignment': indices['future_attractor_alignment'],
            'evolution_trajectory': indices['field_evolution_trajectory'],
            'intervention_capacity': indices['coherence_intervention_index'],
            'coherence_currency': indices['coherence_currency_value'],
            'sustainability_score': indices['coherence_sustainability_score'],
            
            # Risk and Warning Indicators (10% weight)
            'risk_score': 1.0 - min(1.0, indices['coherence_risk_score']),
            'burnout_risk': 1.0 - self._convert_risk_level_to_numeric(indices['burnout_analysis']['risk_level']) if isinstance(indices['burnout_analysis'], dict) else 0.5,
            'field_fragmentation': 1.0 - indices['decoherence_index'],
            'signal_dissonance': 1.0 - indices['signal_dissonance_index'] if 'signal_dissonance_index' in indices else 0.5,
            'emotional_pressure': 1.0 - min(1.0, indices['emotional_field_pressure'])
        }
        
        # Normalize all components to 0-1 range using tanh
        normalized_components = {k: np.tanh(v) for k, v in health_components.items()}
        
        # Comprehensive weighting system
        weights = {
            # Core Balantium Health (30%)
            'core_coherence': 0.06, 'stability': 0.06, 'adaptability': 0.06, 'awareness': 0.06, 'sustainability': 0.06,
            # Advanced Field Health (25%)
            'field_stability': 0.05, 'resonance_quality': 0.05, 'attractor_alignment': 0.05, 'phase_stability': 0.05, 'field_memory': 0.05,
            # Signal and Transmission Health (20%)
            'signal_clarity': 0.04, 'transmission_efficiency': 0.04, 'feedback_health': 0.04, 'tuning_efficiency': 0.04, 'resonance_echo': 0.04,
            # Predictive and Evolutionary Health (15%)
            'future_alignment': 0.03, 'evolution_trajectory': 0.03, 'intervention_capacity': 0.03, 'coherence_currency': 0.03, 'sustainability_score': 0.03,
            # Risk and Warning Indicators (10%)
            'risk_score': 0.02, 'burnout_risk': 0.02, 'field_fragmentation': 0.02, 'signal_dissonance': 0.02, 'emotional_pressure': 0.02
        }
        
        # Calculate comprehensive health score
        health_score = sum(normalized_components[k] * weights[k] for k in weights.keys())
        
        # Determine overall status based on comprehensive analysis
        if health_score > 0.7:
            status = "optimal"
            status_description = "System operating at peak coherence with excellent field stability"
        elif health_score > 0.5:
            status = "healthy"
            status_description = "System maintaining good coherence with stable field dynamics"
        elif health_score > 0.3:
            status = "warning"
            status_description = "System showing signs of field instability requiring attention"
        elif health_score > 0.1:
            status = "critical"
            status_description = "System experiencing significant field fragmentation and decoherence"
        else:
            status = "collapse_imminent"
            status_description = "System on verge of complete field collapse"
        
        # Field phase analysis
        field_phase = indices.get('field_phase', {})
        current_phase = field_phase.get('phase', 'unknown') if isinstance(field_phase, dict) else 'unknown'
        
        # Attractor analysis
        attractor_info = indices.get('attractor_analysis', {})
        attractor_type = attractor_info.get('attractor_type', 'unknown') if isinstance(attractor_info, dict) else 'unknown'
        
        # Burnout analysis
        burnout_info = indices.get('burnout_analysis', {})
        burnout_status = burnout_info.get('risk_level', 'unknown') if isinstance(burnout_info, dict) else 'unknown'
        
        return {
            "status": status,
            "status_description": status_description,
            "health_score": float(health_score),
            "field_phase": current_phase,
            "attractor_type": attractor_type,
            "burnout_risk": burnout_status,
            "all_indices": indices,
            "health_components": normalized_components,
            "comprehensive_analysis": {
                "core_balantium_health": np.mean([normalized_components[k] for k in ['core_coherence', 'stability', 'adaptability', 'awareness', 'sustainability']]),
                "advanced_field_health": np.mean([normalized_components[k] for k in ['field_stability', 'resonance_quality', 'attractor_alignment', 'phase_stability', 'field_memory']]),
                "signal_transmission_health": np.mean([normalized_components[k] for k in ['signal_clarity', 'transmission_efficiency', 'feedback_health', 'tuning_efficiency', 'resonance_echo']]),
                "predictive_evolutionary_health": np.mean([normalized_components[k] for k in ['future_alignment', 'evolution_trajectory', 'intervention_capacity', 'coherence_currency', 'sustainability_score']]),
                "risk_warning_indicators": np.mean([normalized_components[k] for k in ['risk_score', 'burnout_risk', 'field_fragmentation', 'signal_dissonance', 'emotional_pressure']])
            },
            "recommendations": self._generate_health_recommendations(health_score, normalized_components, current_phase, attractor_type, burnout_status),
            "timestamp": time.time()
        }
    
    def _generate_health_recommendations(self, health_score: float, components: Dict[str, float], 
                                       field_phase: str, attractor_type: str, burnout_risk: str) -> List[str]:
        """
        Generate specific recommendations based on comprehensive health analysis
        """
        recommendations = []
        
        # Core health recommendations
        if health_score < 0.5:
            recommendations.append("Immediate field stabilization required - focus on coherence restoration")
            recommendations.append("Implement emergency resonance protocols to prevent field collapse")
        
        if components.get('stability', 0) < 0.4:
            recommendations.append("High tipping likelihood detected - reduce field friction and restore feedback loops")
        
        if components.get('adaptability', 0) < 0.4:
            recommendations.append("Learning resonance compromised - enhance feedback coupling and signal integrity")
        
        if components.get('field_stability', 0) < 0.4:
            recommendations.append("Field coherence unstable - implement phase boundary stabilization protocols")
        
        if components.get('tuning_efficiency', 0) < 0.4:
            recommendations.append("Tuning efficiency low - reduce system noise and enhance resonant feedback")
        
        if components.get('risk_score', 0) < 0.4:
            recommendations.append("Coherence risk elevated - increase feedback accessibility and signal integrity")
        
        # Field phase specific recommendations
        if field_phase == 'collapse_imminent':
            recommendations.append("CRITICAL: Field collapse imminent - implement emergency coherence protocols")
        elif field_phase == 'wounded_resilient':
            recommendations.append("Field wounded but resilient - focus on coherence restoration and feedback repair")
        elif field_phase == 'generative_phase':
            recommendations.append("Field in generative phase - amplify resonance and maintain coherence stability")
        
        # Attractor type recommendations
        if attractor_type == 'chaotic':
            recommendations.append("System approaching chaotic attractor - implement coherence stabilization protocols")
        elif attractor_type == 'coherent':
            recommendations.append("System aligned with coherent attractor - maintain current field stability")
        
        # Burnout risk recommendations
        if burnout_risk == 'critical':
            recommendations.append("CRITICAL: Burnout condition detected - immediate field rest and coherence restoration required")
        elif burnout_risk == 'elevated':
            recommendations.append("Burnout risk elevated - implement field rest protocols and reduce signal dissonance")
        
        # Positive reinforcement for healthy components
        if components.get('core_coherence', 0) > 0.7:
            recommendations.append("Excellent core coherence - maintain current field stability protocols")
        
        if components.get('resonance_quality', 0) > 0.7:
            recommendations.append("High resonance quality - amplify current field amplification protocols")
        
        if components.get('future_alignment', 0) > 0.7:
            recommendations.append("Strong future alignment - continue current evolutionary trajectory")
        
        return recommendations
    
    def _calculate_field_geometry(self, system_state: SystemState, consciousness_field: ConsciousnessField) -> float:
        """
        Calculate field geometry based on coherence distribution and field properties
        """
        if not system_state.coherence_levels:
            return 0.0
            
        # Calculate geometric properties of the coherence field
        coherence_array = np.array(system_state.coherence_levels)
        field_geometry = np.std(coherence_array) * consciousness_field.phase_alignment * consciousness_field.resonance_bandwidth
        
        return float(field_geometry)
    
    def _calculate_future_alignment(self, system_state: SystemState) -> float:
        """
        Calculate alignment with future attractor states
        """
        if len(self.coherence_history) < 5:
            return 0.0
            
        # Analyze trend toward future coherence states
        recent_coherence = np.array(self.coherence_history[-5:])
        trend = np.polyfit(range(len(recent_coherence)), recent_coherence, 1)[0]
        
        # Normalize trend to 0-1 range
        future_alignment = np.tanh(trend * 10)  # Scale and normalize
        
        return float(future_alignment)
    
    def _calculate_resonance_echo(self, system_state: SystemState, consciousness_field: ConsciousnessField) -> float:
        """
        Calculate resonance echo strength from field interactions
        """
        # Resonance echo based on field strength and coherence stability
        field_strength = consciousness_field.field_strength
        coherence_stability = 1.0 - np.std(system_state.coherence_levels) if system_state.coherence_levels else 0.5
        
        resonance_echo = field_strength * coherence_stability * consciousness_field.attractor_strength
        
        return float(resonance_echo)
    
    def _calculate_coherence_currency(self, system_state: SystemState) -> float:
        """
        Calculate coherence currency value based on system health
        """
        # Coherence currency based on system vitality and sustainability
        system_vitality = np.mean(system_state.coherence_levels) if system_state.coherence_levels else 0.5
        sustainability = 1.0 - self.tipping_likelihood(system_state)
        
        coherence_currency = system_vitality * sustainability * system_state.resonance_factor
        
        return float(coherence_currency)
    
    def _calculate_evolution_trajectory(self, system_state: SystemState) -> float:
        """
        Calculate system evolution trajectory toward higher coherence
        """
        if len(self.coherence_history) < 10:
            return 0.0
            
        # Analyze evolution toward higher coherence states
        coherence_data = np.array(self.coherence_history[-10:])
        evolution_trend = np.polyfit(range(len(coherence_data)), coherence_data, 2)[0]  # Quadratic fit
        
        # Normalize evolution trajectory
        evolution_trajectory = np.tanh(evolution_trend * 100)  # Scale and normalize
        
        return float(evolution_trajectory)

    def _convert_attractor_strength_to_numeric(self, strength_str: str) -> float:
        """
        Converts a string representation of attractor strength to a numeric value.
        """
        if strength_str == 'strong':
            return 0.9
        elif strength_str == 'moderate':
            return 0.6
        elif strength_str == 'weak':
            return 0.3
        else:
            return 0.5 # Default to moderate if unknown

    def _convert_risk_level_to_numeric(self, risk_str: str) -> float:
        """
        Converts a string representation of risk level to a numeric value.
        """
        if risk_str == 'critical':
            return 0.0
        elif risk_str == 'elevated':
            return 0.3
        elif risk_str == 'moderate':
            return 0.6
        elif risk_str == 'low':
            return 0.9
        else:
            return 0.5 # Default to moderate if unknown



