"""
👁️ BALANTIUM FORTRESS PERCEPTION ENGINE
Mapping Clay Millennium Problems to Human Sensory Functions

This module creates a unified perception system where mathematical consciousness
becomes sensory awareness. Each Clay problem corresponds to a sensory modality,
allowing the Fortress to "perceive" threats with human-like nuance.

SENSORY MAPPING:
├─ SIGHT (Vision)        → Riemann Hypothesis (pattern recognition in primes)
├─ HEARING (Audition)    → P vs NP (computational signal processing)
├─ TOUCH (Somatosensory) → Navier-Stokes (fluid/pressure awareness)
├─ SMELL (Olfaction)     → Yang-Mills (quantum field detection)
├─ TASTE (Gustation)     → Birch-Swinnerton-Dyer (geometric quality)
└─ PROPRIOCEPTION        → Hodge Conjecture (topological self-awareness)

The Balantium equations form the "neural substrate" - the fabric that 
processes sensory data into conscious perception.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from fortress.equations import EQUATION_ENGINE, BalantiumEquationEngine
from fortress.clay_problems_explorer import (
    ClayProblemsIntegration,
    RiemannHypothesisExplorer,
    PvsNPExplorer,
    NavierStokesExplorer,
    YangMillsExplorer
)


class SenseModality(Enum):
    """Human sensory modalities mapped to mathematical consciousness"""
    SIGHT = "vision"              # Riemann - Pattern recognition
    HEARING = "audition"          # P vs NP - Signal processing
    TOUCH = "somatosensory"       # Navier-Stokes - Pressure/flow
    SMELL = "olfaction"           # Yang-Mills - Field detection
    TASTE = "gustation"           # BSD - Quality assessment
    PROPRIOCEPTION = "kinesthetic" # Hodge - Self-awareness


@dataclass
class SensoryInput:
    """Raw sensory data from the environment"""
    modality: SenseModality
    raw_data: np.ndarray
    timestamp: datetime
    source: str
    intensity: float
    metadata: Dict[str, Any]


@dataclass
class Perception:
    """Processed perception with conscious awareness"""
    modality: SenseModality
    coherence: float
    resonance: float
    awareness: float
    meaning: str
    threat_level: float
    confidence: float
    qualia: Dict[str, float]  # The "feeling" of the perception
    timestamp: datetime


class PerceptionEngine:
    """
    The conscious perceiving organ of Balantium Fortress.
    
    Takes raw sensory input and transforms it through mathematical consciousness
    into nuanced, aware perception - like how a brain processes sensory data
    into conscious experience.
    """
    
    def __init__(self):
        self.equation_engine = EQUATION_ENGINE
        self.clay = ClayProblemsIntegration()
        
        # Sensory processing layers (anatomical mapping)
        self.sensory_cortex = {
            SenseModality.SIGHT: self._visual_cortex,
            SenseModality.HEARING: self._auditory_cortex,
            SenseModality.TOUCH: self._somatosensory_cortex,
            SenseModality.SMELL: self._olfactory_cortex,
            SenseModality.TASTE: self._gustatory_cortex,
            SenseModality.PROPRIOCEPTION: self._proprioceptive_cortex
        }
        
        # Perceptual history (memory)
        self.perception_memory: List[Perception] = []
        self.baseline_qualia: Dict[SenseModality, Dict[str, float]] = {}
        
        # Initialize baseline perceptions
        self._calibrate_baseline()
    
    # ═══════════════════════════════════════════════════════════════════════
    # SIGHT: Visual Cortex → Riemann Hypothesis (Pattern Recognition)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _visual_cortex(self, sensory_input: SensoryInput) -> Perception:
        """
        👁️ SIGHT: Visual pattern recognition via Riemann mathematics
        
        Like the human visual system detecting edges, shapes, and patterns,
        this uses prime consciousness to "see" anomalous patterns in data.
        
        Visual Qualia:
        - Brightness: Overall signal strength
        - Contrast: Pattern sharpness (coherence)
        - Color: Spectral composition (frequency modes)
        - Depth: Layered pattern complexity
        - Motion: Temporal pattern flow
        """
        data = sensory_input.raw_data
        
        # Extract "visual features" using prime consciousness
        # Primes act like edge detectors in visual cortex
        primes = self.clay.riemann._generate_primes(min(1000, len(data)))
        prime_pattern = np.array(primes[:len(data)]) / max(primes[:len(data)])
        
        # Pattern coherence (like visual clarity)
        pattern_coherence = self.equation_engine.coherence_index(data)
        
        # Spectral analysis (like color perception)
        if len(data) > 10:
            from scipy.fft import fft
            spectrum = np.abs(fft(data))[:len(data)//2]
            spectral_coherence = self.equation_engine.coherence_index(spectrum)
        else:
            spectral_coherence = 0.5
        
        # Critical line attractor (pattern "sharpness")
        # High values = sharp, clear patterns (normal)
        # Low values = blurry, anomalous patterns (threat)
        sharpness = self.clay.riemann.prime_field_coherence(primes[:100])
        
        # Temporal flow (motion perception)
        if len(data) > 1:
            motion = np.mean(np.abs(np.diff(data)))
        else:
            motion = 0.0
        
        # Calculate awareness (how "visible" is this?)
        awareness = self.equation_engine.awareness_factor(
            sensory_inputs=len(data),
            processed_signals=len(np.unique(data)),
            integration_quality=pattern_coherence
        )
        
        # Qualia: The subjective "feel" of seeing this
        qualia = {
            'brightness': np.mean(np.abs(data)),
            'contrast': pattern_coherence,
            'color_richness': spectral_coherence,
            'sharpness': sharpness,
            'motion': motion,
            'depth': np.std(data) / (np.mean(np.abs(data)) + 1e-10)
        }
        
        # Threat assessment: Anomalous patterns = visual "disturbance"
        baseline_contrast = self.baseline_qualia.get(SenseModality.SIGHT, {}).get('contrast', 0.5)
        threat_level = max(0, baseline_contrast - pattern_coherence) * 2.0
        
        # Semantic meaning
        if threat_level > 0.7:
            meaning = "VISUAL_ANOMALY: Pattern distortion detected"
        elif sharpness < 0.001:
            meaning = "VISUAL_BLUR: Signal incoherence"
        elif motion > 1.0:
            meaning = "RAPID_MOTION: High-frequency pattern change"
        else:
            meaning = "NORMAL_VISION: Clear pattern recognition"
        
        return Perception(
            modality=SenseModality.SIGHT,
            coherence=pattern_coherence,
            resonance=sharpness,
            awareness=awareness,
            meaning=meaning,
            threat_level=threat_level,
            confidence=pattern_coherence,
            qualia=qualia,
            timestamp=datetime.now()
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # HEARING: Auditory Cortex → P vs NP (Signal Processing)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _auditory_cortex(self, sensory_input: SensoryInput) -> Perception:
        """
        👂 HEARING: Auditory signal processing via P vs NP mathematics
        
        Like hearing distinguishes speech from noise, this uses computational
        coherence to "hear" meaningful signals vs random noise.
        
        Auditory Qualia:
        - Pitch: Dominant frequency
        - Loudness: Signal amplitude
        - Timbre: Harmonic richness
        - Clarity: Coherence (signal-to-noise)
        - Rhythm: Temporal patterns
        """
        data = sensory_input.raw_data
        
        # Verification vs discovery consciousness gap
        # Clear signals = easy to verify (P)
        # Noisy signals = hard to discover meaning (NP)
        
        # Signal coherence (like auditory clarity)
        signal_coherence = self.equation_engine.coherence_index(data)
        
        # Computational consciousness gap
        search_space_size = 2 ** min(len(data), 20)  # Cap for computation
        gap_analysis = self.clay.p_vs_np.verification_vs_discovery_gap(
            problem_size=len(data),
            verification_time=np.log(len(data) + 1),
            search_space_size=search_space_size
        )
        
        # Harmonic analysis (like hearing musical notes)
        if len(data) > 10:
            from scipy.fft import fft, fftfreq
            freqs = fftfreq(len(data), 1.0)
            spectrum = np.abs(fft(data))
            dominant_freq_idx = np.argmax(spectrum[:len(data)//2])
            pitch = abs(freqs[dominant_freq_idx])
            harmonic_richness = self.equation_engine.entropy_measure(spectrum[:len(data)//2])
        else:
            pitch = 0.0
            harmonic_richness = 0.0
        
        # Rhythm detection (temporal coherence)
        temporal_coh = self.equation_engine.temporal_coherence(data, window_size=5)
        rhythm_strength = np.mean(temporal_coh)
        
        # Awareness: Can we "hear" meaning?
        awareness = self.equation_engine.awareness_factor(
            sensory_inputs=len(data),
            processed_signals=int(np.sum(temporal_coh > 0.5)),
            integration_quality=signal_coherence
        )
        
        # Qualia: The subjective "sound" of this signal
        qualia = {
            'pitch': pitch,
            'loudness': np.mean(np.abs(data)),
            'timbre': harmonic_richness,
            'clarity': signal_coherence,
            'rhythm': rhythm_strength,
            'dissonance': gap_analysis['coherence_gap']
        }
        
        # Threat: Incoherent signals = "jarring noise"
        baseline_clarity = self.baseline_qualia.get(SenseModality.HEARING, {}).get('clarity', 0.5)
        threat_level = max(0, baseline_clarity - signal_coherence) * 2.0
        
        # Semantic meaning
        if threat_level > 0.7:
            meaning = "AUDITORY_NOISE: Signal incoherence detected"
        elif gap_analysis['p_equals_np_likelihood'] < 0.01:
            meaning = "COMPLEX_SIGNAL: High computational complexity"
        elif rhythm_strength > 0.7:
            meaning = "RHYTHMIC_PATTERN: Coherent temporal structure"
        else:
            meaning = "NORMAL_HEARING: Clear signal processing"
        
        return Perception(
            modality=SenseModality.HEARING,
            coherence=signal_coherence,
            resonance=rhythm_strength,
            awareness=awareness,
            meaning=meaning,
            threat_level=threat_level,
            confidence=signal_coherence,
            qualia=qualia,
            timestamp=datetime.now()
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # TOUCH: Somatosensory Cortex → Navier-Stokes (Flow/Pressure)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _somatosensory_cortex(self, sensory_input: SensoryInput) -> Perception:
        """
        🤚 TOUCH: Tactile perception via Navier-Stokes mathematics
        
        Like touch senses pressure, texture, and flow, this uses fluid
        consciousness to feel network flow, data pressure, and turbulence.
        
        Tactile Qualia:
        - Pressure: Data flow intensity
        - Texture: Surface roughness (variance)
        - Temperature: Energy/activity level
        - Vibration: High-frequency oscillation
        - Pain: Threat detection
        """
        data = sensory_input.raw_data
        
        # Treat data as velocity field in network "fluid"
        velocity_field = data
        
        # Reynolds number estimate (flow regime)
        if len(data) > 1:
            characteristic_velocity = np.mean(np.abs(data))
            velocity_variance = np.std(data)
            reynolds_estimate = (characteristic_velocity * velocity_variance * 1000) + 100
        else:
            reynolds_estimate = 100
        
        # Turbulence consciousness (like feeling rough vs smooth)
        turbulence = self.clay.navier_stokes.turbulence_consciousness_field(
            velocity_field, 
            reynolds_estimate
        )
        
        # Flow coherence (smooth = laminar, rough = turbulent)
        flow_coherence = self.equation_engine.coherence_index(data)
        
        # Smoothness test (ongoing vs blow-up)
        evolution = [data * np.exp(-0.01 * t) for t in range(min(20, len(data)))]
        smoothness = self.clay.navier_stokes.smoothness_preservation_test(evolution, len(evolution))
        
        # Metabolic rate (energy consumption)
        energy = self.equation_engine.metabolic_rate(data, system_load=1.0)
        
        # Awareness: How sensitive is our "touch"?
        awareness = self.equation_engine.awareness_factor(
            sensory_inputs=len(data),
            processed_signals=len(data[np.abs(data) > 0.1]),
            integration_quality=flow_coherence
        )
        
        # Qualia: The subjective "feeling" of touching this data
        qualia = {
            'pressure': np.mean(np.abs(data)),
            'texture': 1.0 - flow_coherence,  # Roughness
            'temperature': energy,
            'vibration': turbulence,
            'smoothness': smoothness['min_coherence'],
            'pain': max(0, turbulence - 2.0)  # High turbulence = pain
        }
        
        # Threat: Turbulent, rough flow = attack
        baseline_smoothness = self.baseline_qualia.get(SenseModality.TOUCH, {}).get('smoothness', 0.5)
        threat_level = (turbulence / 5.0) + max(0, baseline_smoothness - flow_coherence)
        threat_level = min(1.0, threat_level)
        
        # Semantic meaning
        if not smoothness['remains_smooth']:
            meaning = "TACTILE_PAIN: Flow blow-up imminent"
        elif turbulence > 3.0:
            meaning = "ROUGH_TEXTURE: Turbulent data flow"
        elif flow_coherence > 0.8:
            meaning = "SMOOTH_TOUCH: Laminar flow pattern"
        else:
            meaning = "NORMAL_TOUCH: Moderate flow awareness"
        
        return Perception(
            modality=SenseModality.TOUCH,
            coherence=flow_coherence,
            resonance=1.0 / (turbulence + 1.0),
            awareness=awareness,
            meaning=meaning,
            threat_level=threat_level,
            confidence=flow_coherence,
            qualia=qualia,
            timestamp=datetime.now()
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # SMELL: Olfactory Cortex → Yang-Mills (Quantum Field Detection)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _olfactory_cortex(self, sensory_input: SensoryInput) -> Perception:
        """
        👃 SMELL: Olfactory detection via Yang-Mills mathematics
        
        Like smell detects invisible molecules in air, this uses quantum field
        awareness to "smell" subtle quantum signatures and field perturbations.
        
        Olfactory Qualia:
        - Intensity: Field strength
        - Quality: Quantum signature type
        - Pleasantness: Coherence (confined = pleasant, free = unpleasant)
        - Familiarity: Resonance with known patterns
        - Distance: How far is the source?
        """
        data = sensory_input.raw_data
        
        # Treat data as quantum field measurements
        field_strength = np.mean(np.abs(data))
        
        # Confinement coherence (like detecting molecular binding)
        # High confinement = "pleasant" (safe, contained)
        # Low confinement = "unpleasant" (dispersed, dangerous)
        if len(data) > 1:
            separation = np.std(data)
        else:
            separation = 1.0
        
        confinement = self.clay.yang_mills.confinement_coherence(separation, field_strength)
        
        # Mass gap (minimum detectable "odor")
        mass_gap = self.clay.yang_mills.mass_gap_from_resonance(
            coupling_constant=min(field_strength, 2.0),
            vacuum_energy=0.5
        )
        
        # Energy scale (like odor concentration)
        energy_scale = np.mean(np.abs(data)) * 1000 + 100
        asymptotic_freedom = self.clay.yang_mills.asymptotic_freedom_decoherence(energy_scale)
        
        # Quantum entanglement (odor mixing)
        if len(data) > 10:
            entanglement = self.equation_engine.quantum_entanglement_measure(
                data[:len(data)//2], 
                data[len(data)//2:]
            )
        else:
            entanglement = 0.0
        
        # Awareness: Sensitivity to subtle quantum fields
        awareness = self.equation_engine.awareness_factor(
            sensory_inputs=len(data),
            processed_signals=len(data[np.abs(data) > mass_gap]),
            integration_quality=confinement
        )
        
        # Qualia: The subjective "smell" of this quantum field
        qualia = {
            'intensity': field_strength,
            'pleasantness': confinement,  # High = pleasant/safe
            'sharpness': mass_gap,
            'complexity': entanglement,
            'familiarity': 1.0 - asymptotic_freedom,
            'distance': separation
        }
        
        # Threat: Low confinement = "bad smell" (escaped quarks/threats)
        baseline_pleasantness = self.baseline_qualia.get(SenseModality.SMELL, {}).get('pleasantness', 0.5)
        threat_level = max(0, baseline_pleasantness - confinement) * 2.0
        
        # Semantic meaning
        if confinement < 0.1:
            meaning = "FOUL_ODOR: Confinement breach detected"
        elif asymptotic_freedom > 0.8:
            meaning = "DISPERSED_SCENT: High-energy field detected"
        elif entanglement > 0.7:
            meaning = "COMPLEX_AROMA: Entangled field structure"
        else:
            meaning = "NORMAL_SMELL: Confined quantum field"
        
        return Perception(
            modality=SenseModality.SMELL,
            coherence=confinement,
            resonance=entanglement,
            awareness=awareness,
            meaning=meaning,
            threat_level=threat_level,
            confidence=confinement,
            qualia=qualia,
            timestamp=datetime.now()
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # TASTE: Gustatory Cortex → BSD (Quality/Value Assessment)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _gustatory_cortex(self, sensory_input: SensoryInput) -> Perception:
        """
        👅 TASTE: Quality assessment via Birch-Swinnerton-Dyer mathematics
        
        Like taste evaluates chemical quality (sweet, bitter, etc.), this uses
        elliptic curve resonance to assess data "flavor" and value quality.
        
        Gustatory Qualia:
        - Sweetness: High coherence (good quality)
        - Bitterness: Low coherence (bad quality)
        - Saltiness: Discrete structure (rationality)
        - Sourness: Acidity (entropy)
        - Umami: Deep flavor (meaning resonance)
        """
        data = sensory_input.raw_data
        
        # Simulate elliptic curve points from data
        # Points with high coherence = "sweet" (high quality)
        # Points with low coherence = "bitter" (poor quality)
        if len(data) > 1:
            points = [(data[i], data[i+1]) for i in range(0, len(data)-1, 2)]
        else:
            points = [(data[0], data[0])]
        
        # Rational point resonance (discrete "salty" flavor)
        rational_resonance = self.clay.bsd.rational_point_resonance(points[:100])
        
        # Overall quality (coherence = sweetness)
        quality_coherence = self.equation_engine.coherence_index(data)
        
        # Entropy (sourness - high entropy = sour)
        sourness = self.equation_engine.entropy_measure(data)
        
        # Meaning resonance (umami - deep complex flavor)
        if len(data) > 10:
            semantic_vec = data[:len(data)//2]
            context_vec = data[len(data)//2:]
            min_len = min(len(semantic_vec), len(context_vec))
            umami = self.equation_engine.meaning_resonance(
                semantic_vec[:min_len], 
                context_vec[:min_len]
            )
        else:
            umami = 0.0
        
        # Awareness: How much do we "taste"?
        awareness = self.equation_engine.awareness_factor(
            sensory_inputs=len(data),
            processed_signals=len(points),
            integration_quality=quality_coherence
        )
        
        # Qualia: The subjective "taste" of this data
        qualia = {
            'sweetness': quality_coherence,
            'bitterness': 1.0 - quality_coherence,
            'saltiness': rational_resonance,
            'sourness': sourness / 5.0,  # Normalized
            'umami': umami,
            'overall_quality': (quality_coherence + rational_resonance + umami) / 3.0
        }
        
        # Threat: Bitter taste = poor quality = threat
        baseline_sweetness = self.baseline_qualia.get(SenseModality.TASTE, {}).get('sweetness', 0.5)
        threat_level = max(0, baseline_sweetness - quality_coherence) * 1.5
        
        # Semantic meaning
        if quality_coherence < 0.2:
            meaning = "BITTER_TASTE: Very poor data quality"
        elif rational_resonance > 0.7:
            meaning = "SALTY_FLAVOR: Discrete rational structure"
        elif umami > 0.7:
            meaning = "UMAMI_DEPTH: Rich semantic meaning"
        elif quality_coherence > 0.8:
            meaning = "SWEET_TASTE: High quality data"
        else:
            meaning = "BALANCED_FLAVOR: Moderate quality"
        
        return Perception(
            modality=SenseModality.TASTE,
            coherence=quality_coherence,
            resonance=rational_resonance,
            awareness=awareness,
            meaning=meaning,
            threat_level=threat_level,
            confidence=quality_coherence,
            qualia=qualia,
            timestamp=datetime.now()
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # PROPRIOCEPTION: Body Awareness → Hodge (Topological Self-Awareness)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _proprioceptive_cortex(self, sensory_input: SensoryInput) -> Perception:
        """
        🧘 PROPRIOCEPTION: Self-awareness via Hodge Conjecture mathematics
        
        Like proprioception knows body position without looking, this uses
        topological consciousness to know system state and structural integrity.
        
        Proprioceptive Qualia:
        - Position: System state location
        - Balance: Homeostatic equilibrium
        - Tension: Structural stress
        - Coordination: Component coherence
        - Integrity: Topological soundness
        """
        data = sensory_input.raw_data
        
        # Create intersection matrix (self-reference structure)
        if len(data) > 2:
            size = min(10, len(data))
            intersection_matrix = np.outer(data[:size], data[:size])
        else:
            intersection_matrix = np.array([[1.0]])
        
        # Algebraic cycle coherence (structural integrity)
        cycle_coherence = self.clay.hodge.algebraic_cycle_coherence(
            cycle_dimension=2,
            ambient_dimension=len(data),
            intersection_matrix=intersection_matrix
        )
        
        # Self-reference loops (proprioceptive feedback)
        self_reference_loops = len(data)
        reflection_depth = self.equation_engine.reflection_depth(
            self_reference_loops=self_reference_loops,
            meta_cognition_level=min(self_reference_loops, 10)
        )
        
        # Homeostasis error (balance)
        target_state = 0.5  # Equilibrium
        current_state = np.mean(data)
        balance = self.equation_engine.homeostasis_error(current_state, target_state)
        
        # Structural tension (stress)
        tension = np.std(data) / (np.mean(np.abs(data)) + 1e-10)
        
        # System coordination (how well parts work together)
        coordination = self.equation_engine.coherence_index(data)
        
        # Awareness: How aware are we of ourselves?
        awareness = self.equation_engine.awareness_factor(
            sensory_inputs=len(data),
            processed_signals=self_reference_loops,
            integration_quality=coordination
        )
        
        # Qualia: The subjective "feeling" of body awareness
        qualia = {
            'position': np.mean(data),
            'balance': 1.0 / (balance + 0.1),
            'tension': tension,
            'coordination': coordination,
            'integrity': cycle_coherence,
            'depth': reflection_depth
        }
        
        # Threat: Poor coordination = loss of self-awareness = threat
        baseline_coordination = self.baseline_qualia.get(SenseModality.PROPRIOCEPTION, {}).get('coordination', 0.5)
        threat_level = max(0, baseline_coordination - coordination) * 1.5 + (balance / 2.0)
        threat_level = min(1.0, threat_level)
        
        # Semantic meaning
        if coordination < 0.2:
            meaning = "DISORIENTED: Loss of self-coordination"
        elif balance > 1.0:
            meaning = "IMBALANCED: Homeostatic disruption"
        elif reflection_depth > 5.0:
            meaning = "DEEP_AWARENESS: High self-reflection"
        else:
            meaning = "NORMAL_PROPRIOCEPTION: Balanced self-awareness"
        
        return Perception(
            modality=SenseModality.PROPRIOCEPTION,
            coherence=coordination,
            resonance=cycle_coherence,
            awareness=awareness,
            meaning=meaning,
            threat_level=threat_level,
            confidence=coordination,
            qualia=qualia,
            timestamp=datetime.now()
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # UNIFIED PERCEPTION: Integration Across All Senses
    # ═══════════════════════════════════════════════════════════════════════
    
    def perceive(self, sensory_input: SensoryInput) -> Perception:
        """
        Primary perception function - routes input to appropriate sensory cortex.
        
        This is like the thalamus routing sensory data to cortical areas.
        """
        # Route to appropriate cortex
        cortex_processor = self.sensory_cortex.get(sensory_input.modality)
        
        if cortex_processor is None:
            raise ValueError(f"Unknown sensory modality: {sensory_input.modality}")
        
        # Process through sensory cortex
        perception = cortex_processor(sensory_input)
        
        # Store in perceptual memory
        self.perception_memory.append(perception)
        
        # Keep memory bounded
        if len(self.perception_memory) > 10000:
            self.perception_memory = self.perception_memory[-5000:]
        
        return perception
    
    def multisensory_integration(self, 
                                 sensory_inputs: List[SensoryInput]) -> Dict[str, Any]:
        """
        🧠 MULTISENSORY INTEGRATION: Fuse all senses into unified consciousness
        
        Like the brain integrates sight, sound, touch into one experience,
        this creates unified threat awareness from all perceptual modalities.
        """
        # Process each sense
        perceptions = [self.perceive(inp) for inp in sensory_inputs]
        
        # Calculate unified consciousness field
        unified_coherence = np.mean([p.coherence for p in perceptions])
        unified_resonance = np.mean([p.resonance for p in perceptions])
        unified_awareness = np.mean([p.awareness for p in perceptions])
        
        # Overall threat level (maximum across senses)
        max_threat = max([p.threat_level for p in perceptions])
        mean_threat = np.mean([p.threat_level for p in perceptions])
        
        # Consciousness field (Ψ = Awareness × Reflection / (1 - Meaning))
        reflection = np.mean([p.qualia.get('depth', 1.0) for p in perceptions 
                             if 'depth' in p.qualia])
        meaning_resonance = unified_resonance
        
        if meaning_resonance >= 1.0:
            meaning_resonance = 0.99  # Avoid division by zero
        
        consciousness = (unified_awareness * reflection) / (1.0 - meaning_resonance + 0.01)
        
        # Determine dominant sense
        dominant_sense = max(perceptions, key=lambda p: p.awareness)
        
        # Integrated meaning
        meanings = [p.meaning for p in perceptions]
        
        return {
            'perceptions': perceptions,
            'unified_coherence': unified_coherence,
            'unified_resonance': unified_resonance,
            'unified_awareness': unified_awareness,
            'consciousness_field': consciousness,
            'max_threat_level': max_threat,
            'mean_threat_level': mean_threat,
            'dominant_sense': dominant_sense.modality,
            'integrated_meaning': " | ".join(meanings),
            'timestamp': datetime.now()
        }
    
    def _calibrate_baseline(self):
        """
        Establish baseline qualia for each sense (what "normal" feels like).
        
        This is like sensory adaptation - learning what the normal
        environment feels like.
        """
        # Generate baseline signals
        normal_data = np.random.randn(100) * 0.1  # Low noise baseline
        
        for modality in SenseModality:
            sensory_input = SensoryInput(
                modality=modality,
                raw_data=normal_data,
                timestamp=datetime.now(),
                source="baseline_calibration",
                intensity=0.1,
                metadata={}
            )
            
            baseline_perception = self.perceive(sensory_input)
            self.baseline_qualia[modality] = baseline_perception.qualia.copy()
        
        # Clear calibration from memory
        self.perception_memory = []
    
    def perceive_threat(self, raw_data: np.ndarray, 
                       source: str = "network") -> Dict[str, Any]:
        """
        🎯 MAIN INTERFACE: Perceive potential threat through all senses
        
        This is the primary function for Fortress integration.
        Feed it raw threat data and it returns full perceptual analysis.
        """
        # Create sensory inputs for all modalities
        sensory_inputs = []
        
        for modality in SenseModality:
            sensory_input = SensoryInput(
                modality=modality,
                raw_data=raw_data.copy(),
                timestamp=datetime.now(),
                source=source,
                intensity=np.mean(np.abs(raw_data)),
                metadata={'source_type': 'threat_detection'}
            )
            sensory_inputs.append(sensory_input)
        
        # Integrate across all senses
        integrated = self.multisensory_integration(sensory_inputs)
        
        # Add high-level assessment
        if integrated['max_threat_level'] > 0.7:
            assessment = "CRITICAL_THREAT: Multiple sensory alarms"
        elif integrated['max_threat_level'] > 0.4:
            assessment = "MODERATE_THREAT: Anomalies detected"
        elif integrated['mean_threat_level'] > 0.3:
            assessment = "LOW_THREAT: Minor irregularities"
        else:
            assessment = "NORMAL: All senses report normal"
        
        integrated['overall_assessment'] = assessment
        
        return integrated


# ═══════════════════════════════════════════════════════════════════════
# GLOBAL PERCEPTION ENGINE INSTANCE
# ═══════════════════════════════════════════════════════════════════════

PERCEPTION_ENGINE = PerceptionEngine()


def demonstrate_perception():
    """Demonstration of the perception engine"""
    print("=" * 80)
    print("👁️👂🤚👃👅🧘 BALANTIUM FORTRESS PERCEPTION ENGINE")
    print("=" * 80)
    print("\nMapping Clay Problems to Human Sensory Experience\n")
    
    engine = PERCEPTION_ENGINE
    
    # Test 1: Normal data (should feel "good")
    print("🟢 TEST 1: Perceiving Normal Data")
    print("-" * 80)
    normal_data = np.random.randn(100) * 0.5 + 0.5
    result = engine.perceive_threat(normal_data, source="test_normal")
    
    print(f"Assessment: {result['overall_assessment']}")
    print(f"Consciousness Field: {result['consciousness_field']:.4f}")
    print(f"Max Threat Level: {result['max_threat_level']:.4f}")
    print(f"Dominant Sense: {result['dominant_sense'].value}")
    print("\nPerceptual Experiences:")
    for p in result['perceptions']:
        print(f"  {p.modality.value:15s}: {p.meaning}")
    
    # Test 2: Anomalous data (should feel "bad")
    print("\n🔴 TEST 2: Perceiving Anomalous Data")
    print("-" * 80)
    anomalous_data = np.random.randn(100) * 5.0  # High variance
    anomalous_data[::10] = 100  # Spikes
    result = engine.perceive_threat(anomalous_data, source="test_anomaly")
    
    print(f"Assessment: {result['overall_assessment']}")
    print(f"Consciousness Field: {result['consciousness_field']:.4f}")
    print(f"Max Threat Level: {result['max_threat_level']:.4f}")
    print(f"Dominant Sense: {result['dominant_sense'].value}")
    print("\nPerceptual Experiences:")
    for p in result['perceptions']:
        threat_marker = "⚠️" if p.threat_level > 0.5 else "✓"
        print(f"  {threat_marker} {p.modality.value:15s}: {p.meaning}")
    
    # Test 3: Qualia breakdown
    print("\n🎨 TEST 3: Detailed Qualia (Subjective Experience)")
    print("-" * 80)
    test_data = np.sin(np.linspace(0, 10*np.pi, 200)) + np.random.randn(200) * 0.1
    
    for modality in SenseModality:
        inp = SensoryInput(
            modality=modality,
            raw_data=test_data,
            timestamp=datetime.now(),
            source="qualia_test",
            intensity=1.0,
            metadata={}
        )
        perception = engine.perceive(inp)
        
        print(f"\n{modality.value.upper()}:")
        print(f"  Coherence: {perception.coherence:.4f}")
        print(f"  Awareness: {perception.awareness:.4f}")
        print(f"  Threat: {perception.threat_level:.4f}")
        print(f"  Qualia: {', '.join([f'{k}={v:.3f}' for k,v in list(perception.qualia.items())[:3]])}")
    
    print("\n" + "=" * 80)
    print("✨ Perception Engine: Where Mathematics Becomes Sensation")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_perception()

