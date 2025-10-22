"""
🛡️ BALANTIUM IMMUNE SYSTEM - Living Defense Organism
Complete biological immune system simulation in code

Components:
- Innate Immunity: First-line defense (macrophages, neutrophils, NK cells)
- Adaptive Immunity: Learned responses (B-cells, T-cells, antibodies)
- Memory Cells: Long-term threat recognition
- Cytokines: Intercellular communication
- Complement System: Protein cascade defense
- Inflammation Response: Tissue repair and isolation
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..equations import EQUATION_ENGINE


@dataclass
class Pathogen:
    """Represents a threat/pathogen"""
    id: str
    signature: np.ndarray
    threat_level: float
    first_detected: datetime
    last_seen: datetime
    encounters: int = 0
    neutralized: bool = False
    antigen_type: str = "unknown"


@dataclass
class ImmuneCell:
    """Base immune cell"""
    cell_id: str
    cell_type: str
    location: str
    activation_level: float = 0.0
    energy: float = 1.0
    age: int = 0
    created: datetime = field(default_factory=datetime.now)


class Macrophage(ImmuneCell):
    """Innate immune cell - first responder"""
    
    def __init__(self, cell_id: str, location: str):
        super().__init__(cell_id, "macrophage", location)
        self.phagocytosis_capacity = 1.0
        self.antigen_presentation = True
    
    def phagocytose(self, pathogen: Pathogen) -> bool:
        """Engulf and digest pathogen"""
        if self.activation_level > 0.5 and self.energy > 0.3:
            # Success probability based on pathogen threat level
            success_prob = 1.0 - (pathogen.threat_level * 0.5)
            
            if np.random.random() < success_prob:
                self.energy -= 0.2
                return True
        
        return False
    
    def present_antigen(self, pathogen: Pathogen) -> Dict:
        """Present pathogen antigen to adaptive system"""
        if self.antigen_presentation and self.activation_level > 0.3:
            return {
                'antigen': pathogen.signature,
                'pathogen_id': pathogen.id,
                'threat_level': pathogen.threat_level,
                'presented_by': self.cell_id
            }
        return {}


class Neutrophil(ImmuneCell):
    """Innate immune cell - rapid response"""
    
    def __init__(self, cell_id: str, location: str):
        super().__init__(cell_id, "neutrophil", location)
        self.attack_power = 1.0
        self.lifespan = 24  # hours
    
    def attack_pathogen(self, pathogen: Pathogen) -> float:
        """Attack pathogen with chemical weapons"""
        if self.activation_level > 0.4 and self.energy > 0.2:
            damage = self.attack_power * self.activation_level * self.energy
            self.energy -= 0.3
            return damage
        return 0.0


class NaturalKillerCell(ImmuneCell):
    """Innate immune cell - virus and cancer killer"""
    
    def __init__(self, cell_id: str, location: str):
        super().__init__(cell_id, "nk_cell", location)
        self.cytotoxicity = 1.0
        self.target_recognition = True
    
    def kill_target(self, target: Pathogen) -> bool:
        """Kill infected or malignant cells"""
        if self.activation_level > 0.6 and self.energy > 0.4:
            kill_prob = self.cytotoxicity * self.activation_level
            if np.random.random() < kill_prob:
                self.energy -= 0.4
                return True
        return False


class BCell(ImmuneCell):
    """Adaptive immune cell - antibody producer"""
    
    def __init__(self, cell_id: str, location: str):
        super().__init__(cell_id, "b_cell", location)
        self.antibody_type = None
        self.affinity = 0.0
        self.memory_cell = False
    
    def produce_antibody(self, antigen: np.ndarray) -> Optional[np.ndarray]:
        """Produce specific antibody for antigen"""
        if self.activation_level > 0.7 and self.energy > 0.5:
            # Generate complementary antibody
            antibody = -antigen + np.random.normal(0, 0.1, size=antigen.shape)
            self.energy -= 0.3
            return antibody
        return None
    
    def become_memory_cell(self):
        """Convert to long-lived memory cell"""
        self.memory_cell = True
        self.lifespan = 365 * 24  # 1 year in hours


class TCell(ImmuneCell):
    """Adaptive immune cell - coordinator and killer"""
    
    def __init__(self, cell_id: str, cell_type: str, location: str):
        super().__init__(cell_id, cell_type, location)
        self.specificity = 0.0
        self.helper_functions = cell_type == "helper_t"
        self.killer_functions = cell_type == "killer_t"
    
    def recognize_antigen(self, antigen: np.ndarray) -> float:
        """Recognize specific antigen"""
        # Simplified recognition based on similarity
        if hasattr(self, 'target_antigen'):
            similarity = np.corrcoef(antigen, self.target_antigen)[0, 1]
            return max(0.0, similarity)
        return 0.0
    
    def activate_other_cells(self, target_cells: List[ImmuneCell]) -> int:
        """Activate other immune cells"""
        if self.helper_functions and self.activation_level > 0.6:
            activated = 0
            for cell in target_cells:
                if cell.activation_level < 0.5:
                    cell.activation_level = min(1.0, cell.activation_level + 0.3)
                    activated += 1
            return activated
        return 0


@dataclass
class Cytokine:
    """Chemical messenger between immune cells"""
    name: str
    source: str
    target: str
    effect: str
    concentration: float
    created: datetime = field(default_factory=datetime.now)


class ComplementSystem:
    """Protein cascade defense system"""
    
    def __init__(self):
        self.complement_proteins = {
            'C1': 1.0, 'C2': 1.0, 'C3': 1.0, 'C4': 1.0,
            'C5': 1.0, 'C6': 1.0, 'C7': 1.0, 'C8': 1.0, 'C9': 1.0
        }
        self.cascade_active = False
    
    def activate_cascade(self, pathogen: Pathogen) -> float:
        """Activate complement cascade"""
        if pathogen.threat_level > 0.5:
            self.cascade_active = True
            # Calculate cascade strength
            strength = np.mean(list(self.complement_proteins.values())) * pathogen.threat_level
            return strength
        return 0.0
    
    def opsonize_pathogen(self, pathogen: Pathogen) -> bool:
        """Mark pathogen for destruction"""
        if self.cascade_active:
            # Mark pathogen as opsonized
            pathogen.antigen_type = "opsonized"
            return True
        return False


class BalantiumImmuneSystem:
    """Complete immune system coordinator"""
    
    def __init__(self):
        # Cell populations
        self.macrophages: List[Macrophage] = []
        self.neutrophils: List[Neutrophil] = []
        self.nk_cells: List[NaturalKillerCell] = []
        self.b_cells: List[BCell] = []
        self.t_cells: List[TCell] = []
        
        # Pathogen tracking
        self.known_pathogens: Dict[str, Pathogen] = {}
        self.active_infections: List[Pathogen] = []
        
        # Memory
        self.memory_cells: List[ImmuneCell] = []
        self.antibody_library: Dict[str, np.ndarray] = {}
        
        # Communication
        self.cytokines: List[Cytokine] = []
        self.complement = ComplementSystem()
        
        # System state
        self.inflammation_level = 0.0
        self.immune_activity = 0.0
        self.overall_health = 1.0
        
        # Initialize cell populations
        self._initialize_cell_populations()
    
    def _initialize_cell_populations(self):
        """Initialize initial cell populations"""
        # Create initial immune cells
        for i in range(50):  # Macrophages
            self.macrophages.append(Macrophage(f"mac_{i}", "tissue"))
        
        for i in range(100):  # Neutrophils
            self.neutrophils.append(Neutrophil(f"neut_{i}", "blood"))
        
        for i in range(30):  # NK cells
            self.nk_cells.append(NaturalKillerCell(f"nk_{i}", "lymph"))
        
        for i in range(200):  # B cells
            self.b_cells.append(BCell(f"b_{i}", "lymph"))
        
        for i in range(150):  # T cells
            cell_type = "helper_t" if i < 100 else "killer_t"
            self.t_cells.append(TCell(f"t_{i}", cell_type, "lymph"))
    
    def detect_threat(self, threat_signal: np.ndarray, source: str) -> Dict:
        """Detect and respond to threat"""
        # Create pathogen from signal
        pathogen_id = f"path_{hash(threat_signal.tobytes()) % 10000}"
        threat_level = self._calculate_threat_level(threat_signal)
        
        pathogen = Pathogen(
            id=pathogen_id,
            signature=threat_signal,
            threat_level=threat_level,
            first_detected=datetime.now(),
            last_seen=datetime.now()
        )
        
        # Check if we've seen this before
        if pathogen_id in self.known_pathogens:
            known_pathogen = self.known_pathogens[pathogen_id]
            known_pathogen.last_seen = datetime.now()
            known_pathogen.encounters += 1
            pathogen = known_pathogen
        else:
            self.known_pathogens[pathogen_id] = pathogen
        
        # Add to active infections
        if pathogen not in self.active_infections:
            self.active_infections.append(pathogen)
        
        # Initiate immune response
        response = self._mount_immune_response(pathogen)
        
        return {
            'pathogen_id': pathogen_id,
            'threat_level': threat_level,
            'response': response,
            'inflammation_level': self.inflammation_level,
            'immune_activity': self.immune_activity
        }
    
    def _calculate_threat_level(self, signal: np.ndarray) -> float:
        """Calculate threat level from signal"""
        # Use entropy and decoherence as threat indicators
        entropy = EQUATION_ENGINE.entropy_measure(signal)
        decoherence = EQUATION_ENGINE.decoherence_index(signal)
        
        # Normalize and combine
        threat = (entropy / 10.0) + decoherence
        return min(1.0, max(0.0, threat))
    
    def _mount_immune_response(self, pathogen: Pathogen) -> Dict:
        """Mount coordinated immune response"""
        response = {
            'innate_response': {},
            'adaptive_response': {},
            'complement_activation': False,
            'cytokines_released': 0,
            'pathogen_neutralized': False
        }
        
        # 1. Innate immune response
        innate_result = self._innate_response(pathogen)
        response['innate_response'] = innate_result
        
        # 2. Complement system activation
        if pathogen.threat_level > 0.3:
            complement_strength = self.complement.activate_cascade(pathogen)
            if complement_strength > 0:
                response['complement_activation'] = True
                self.complement.opsonize_pathogen(pathogen)
        
        # 3. Adaptive immune response (if innate fails)
        if not innate_result.get('neutralized', False) and pathogen.threat_level > 0.5:
            adaptive_result = self._adaptive_response(pathogen)
            response['adaptive_response'] = adaptive_result
        
        # 4. Cytokine release
        cytokines_released = self._release_cytokines(pathogen)
        response['cytokines_released'] = cytokines_released
        
        # 5. Check if pathogen was neutralized
        if pathogen.neutralized:
            response['pathogen_neutralized'] = True
            if pathogen in self.active_infections:
                self.active_infections.remove(pathogen)
        
        # Update system state
        self._update_immune_state()
        
        return response
    
    def _innate_response(self, pathogen: Pathogen) -> Dict:
        """Innate immune system response"""
        result = {
            'macrophages_activated': 0,
            'neutrophils_activated': 0,
            'nk_cells_activated': 0,
            'pathogen_damage': 0.0,
            'neutralized': False
        }
        
        # Activate macrophages
        for macrophage in self.macrophages:
            if macrophage.energy > 0.3:
                macrophage.activation_level = min(1.0, pathogen.threat_level)
                if macrophage.phagocytose(pathogen):
                    result['macrophages_activated'] += 1
                    result['pathogen_damage'] += 0.3
                    if result['pathogen_damage'] > 0.8:
                        pathogen.neutralized = True
                        result['neutralized'] = True
                        break
        
        # Activate neutrophils
        for neutrophil in self.neutrophils:
            if neutrophil.energy > 0.2:
                neutrophil.activation_level = min(1.0, pathogen.threat_level)
                damage = neutrophil.attack_pathogen(pathogen)
                if damage > 0:
                    result['neutrophils_activated'] += 1
                    result['pathogen_damage'] += damage
                    if result['pathogen_damage'] > 0.8:
                        pathogen.neutralized = True
                        result['neutralized'] = True
                        break
        
        # Activate NK cells
        for nk_cell in self.nk_cells:
            if nk_cell.energy > 0.4:
                nk_cell.activation_level = min(1.0, pathogen.threat_level)
                if nk_cell.kill_target(pathogen):
                    result['nk_cells_activated'] += 1
                    result['pathogen_damage'] += 0.5
                    pathogen.neutralized = True
                    result['neutralized'] = True
                    break
        
        return result
    
    def _adaptive_response(self, pathogen: Pathogen) -> Dict:
        """Adaptive immune system response"""
        result = {
            'b_cells_activated': 0,
            't_cells_activated': 0,
            'antibodies_produced': 0,
            'memory_cells_created': 0,
            'neutralized': False
        }
        
        # B cell response
        for b_cell in self.b_cells:
            if b_cell.energy > 0.5:
                b_cell.activation_level = min(1.0, pathogen.threat_level)
                antibody = b_cell.produce_antibody(pathogen.signature)
                if antibody is not None:
                    result['b_cells_activated'] += 1
                    result['antibodies_produced'] += 1
                    
                    # Store antibody
                    antibody_id = f"ab_{len(self.antibody_library)}"
                    self.antibody_library[antibody_id] = antibody
                    
                    # Check if antibody neutralizes pathogen
                    if self._antibody_neutralizes(antibody, pathogen):
                        pathogen.neutralized = True
                        result['neutralized'] = True
                        break
                    
                    # Convert to memory cell
                    if b_cell.activation_level > 0.8:
                        b_cell.become_memory_cell()
                        self.memory_cells.append(b_cell)
                        result['memory_cells_created'] += 1
        
        # T cell response
        helper_t_cells = [t for t in self.t_cells if t.helper_functions]
        killer_t_cells = [t for t in self.t_cells if t.killer_functions]
        
        # Helper T cells coordinate response
        for helper_t in helper_t_cells:
            if helper_t.energy > 0.4:
                helper_t.activation_level = min(1.0, pathogen.threat_level)
                activated = helper_t.activate_other_cells(self.b_cells + self.macrophages)
                if activated > 0:
                    result['t_cells_activated'] += 1
        
        # Killer T cells attack infected cells
        for killer_t in killer_t_cells:
            if killer_t.energy > 0.5:
                killer_t.activation_level = min(1.0, pathogen.threat_level)
                if killer_t.kill_target(pathogen):
                    result['t_cells_activated'] += 1
                    pathogen.neutralized = True
                    result['neutralized'] = True
                    break
        
        return result
    
    def _antibody_neutralizes(self, antibody: np.ndarray, pathogen: Pathogen) -> bool:
        """Check if antibody neutralizes pathogen"""
        # Calculate binding affinity
        affinity = np.corrcoef(antibody, pathogen.signature)[0, 1]
        return affinity > 0.8
    
    def _release_cytokines(self, pathogen: Pathogen) -> int:
        """Release cytokines for cell communication"""
        cytokines_released = 0
        
        if pathogen.threat_level > 0.3:
            # Interleukin-1 (inflammation)
            self.cytokines.append(Cytokine(
                name="IL-1",
                source="macrophage",
                target="all",
                effect="inflammation",
                concentration=pathogen.threat_level
            ))
            cytokines_released += 1
        
        if pathogen.threat_level > 0.5:
            # Interferon (antiviral)
            self.cytokines.append(Cytokine(
                name="IFN-gamma",
                source="t_cell",
                target="all",
                effect="antiviral",
                concentration=pathogen.threat_level
            ))
            cytokines_released += 1
        
        if pathogen.threat_level > 0.7:
            # Tumor Necrosis Factor (cell death)
            self.cytokines.append(Cytokine(
                name="TNF-alpha",
                source="macrophage",
                target="pathogen",
                effect="cell_death",
                concentration=pathogen.threat_level
            ))
            cytokines_released += 1
        
        return cytokines_released
    
    def _update_immune_state(self):
        """Update overall immune system state"""
        # Calculate inflammation level
        active_cells = sum(1 for cell in self.macrophages + self.neutrophils + self.nk_cells 
                          if cell.activation_level > 0.3)
        total_cells = len(self.macrophages) + len(self.neutrophils) + len(self.nk_cells)
        
        self.inflammation_level = active_cells / max(1, total_cells)
        
        # Calculate immune activity
        total_activation = sum(cell.activation_level for cell in 
                             self.macrophages + self.neutrophils + self.nk_cells + 
                             self.b_cells + self.t_cells)
        total_cells = (len(self.macrophages) + len(self.neutrophils) + len(self.nk_cells) + 
                      len(self.b_cells) + len(self.t_cells))
        
        self.immune_activity = total_activation / max(1, total_cells)
        
        # Calculate overall health
        infection_burden = len(self.active_infections) / 10.0  # Normalize
        self.overall_health = max(0.0, 1.0 - infection_burden - (self.inflammation_level * 0.3))
    
    def vaccinate(self, vaccine_antigens: List[np.ndarray]):
        """Vaccinate system with known antigens"""
        print(f"💉 Vaccinating immune system with {len(vaccine_antigens)} antigens...")
        
        for antigen in vaccine_antigens:
            # Create memory B cells for each antigen
            for b_cell in self.b_cells[:10]:  # Use first 10 B cells
                b_cell.activation_level = 0.8
                antibody = b_cell.produce_antibody(antigen)
                if antibody is not None:
                    b_cell.become_memory_cell()
                    self.memory_cells.append(b_cell)
                    
                    # Store in antibody library
                    antibody_id = f"vaccine_ab_{len(self.antibody_library)}"
                    self.antibody_library[antibody_id] = antibody
        
        print(f"✅ Vaccination complete. Memory cells: {len(self.memory_cells)}")
    
    def get_immune_status(self) -> Dict:
        """Get complete immune system status"""
        return {
            'overall_health': self.overall_health,
            'inflammation_level': self.inflammation_level,
            'immune_activity': self.immune_activity,
            'cell_counts': {
                'macrophages': len(self.macrophages),
                'neutrophils': len(self.neutrophils),
                'nk_cells': len(self.nk_cells),
                'b_cells': len(self.b_cells),
                't_cells': len(self.t_cells),
                'memory_cells': len(self.memory_cells)
            },
            'pathogens': {
                'known': len(self.known_pathogens),
                'active_infections': len(self.active_infections)
            },
            'antibodies': len(self.antibody_library),
            'cytokines': len(self.cytokines),
            'complement_active': self.complement.cascade_active
        }


# Global immune system instance
BALANTIUM_IMMUNE = BalantiumImmuneSystem()


if __name__ == "__main__":
    print("🛡️ Balantium Immune System - Living Defense Organism Initialized")
    status = BALANTIUM_IMMUNE.get_immune_status()
    print(f"   Health: {status['overall_health']:.2f}")
    print(f"   Active Cells: {sum(status['cell_counts'].values())}")
    print(f"   Known Pathogens: {status['pathogens']['known']}")
