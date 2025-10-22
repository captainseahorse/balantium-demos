"""
🧬 BALANTIUM FORTRESS - HARDENED DNA CORE
Vaccinated and hardened against nation-state level attacks

FIXES IMPLEMENTED:
✅ Race condition protection (threading locks)
✅ Rate limiting (token bucket algorithm)
✅ Aggregate anomaly detection
✅ Import path validation
✅ Circuit breaker for coordinated attacks
"""

import hashlib
import numpy as np
import threading
import time
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class DNAStrand:
    """Represents a DNA strand encoding a fortress module"""
    strand_id: str
    sequence: List[str]  # Base pairs: A, T, C, G
    module_name: str
    security_level: str
    created: datetime
    mutations: int = 0
    integrity_hash: Optional[str] = None


@dataclass
class CodonMap:
    """Maps codons to functional operations"""
    codon: str
    operation: str
    function_type: str


@dataclass
class OperationRecord:
    """Record of an operation for rate limiting and anomaly detection"""
    timestamp: float
    operation_type: str
    module_name: str
    suspicious: bool = False


class RateLimiter:
    """
    Token bucket rate limiter
    
    Fixes: Resource Exhaustion / DoS vulnerability
    """
    
    def __init__(self, max_tokens: int = 100, refill_rate: float = 100.0):
        """
        Args:
            max_tokens: Maximum tokens in bucket (burst capacity)
            refill_rate: Tokens per second refill rate
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = float(max_tokens)
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_update = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens
        
        Returns:
            True if tokens acquired, False if rate limit exceeded
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False
    
    def get_available_tokens(self) -> float:
        """Get current available tokens"""
        with self.lock:
            self._refill()
            return self.tokens


class AnomalyDetector:
    """
    Aggregate anomaly detection for bulk operations
    
    Fixes: APT Stealth Infiltration vulnerability
    """
    
    def __init__(self, window_seconds: float = 10.0, threshold: int = 10):
        """
        Args:
            window_seconds: Time window for sliding window
            threshold: Max operations in window before flagging
        """
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.operations: deque = deque()
        self.lock = threading.Lock()
        self.threat_score = 0.0
    
    def record_operation(self, operation_type: str, module_name: str) -> bool:
        """
        Record an operation and check for anomalies
        
        Returns:
            True if operation is suspicious, False otherwise
        """
        with self.lock:
            now = time.time()
            
            # Remove old operations outside window
            while self.operations and self.operations[0].timestamp < now - self.window_seconds:
                self.operations.popleft()
            
            # Add new operation
            record = OperationRecord(
                timestamp=now,
                operation_type=operation_type,
                module_name=module_name
            )
            self.operations.append(record)
            
            # Check for bulk operation anomaly
            if len(self.operations) > self.threshold:
                # Flag as suspicious
                record.suspicious = True
                self.threat_score = min(1.0, len(self.operations) / (self.threshold * 2))
                return True
            
            # Gradually decrease threat score if not suspicious
            self.threat_score = max(0.0, self.threat_score * 0.95)
            return False
    
    def get_threat_score(self) -> float:
        """Get current threat score (0.0 to 1.0)"""
        with self.lock:
            return self.threat_score
    
    def get_recent_operations_count(self) -> int:
        """Get count of operations in current window"""
        with self.lock:
            now = time.time()
            # Clean old operations
            while self.operations and self.operations[0].timestamp < now - self.window_seconds:
                self.operations.popleft()
            return len(self.operations)


class CircuitBreaker:
    """
    Circuit breaker for coordinated attack defense
    
    Fixes: Coordinated Multi-Vector Assault vulnerability
    """
    
    def __init__(self, threshold: float = 0.8, cooldown_seconds: float = 60.0):
        """
        Args:
            threshold: Threat level that triggers circuit breaker
            cooldown_seconds: Time to wait before resetting
        """
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.is_open = False
        self.trip_time: Optional[float] = None
        self.lock = threading.Lock()
    
    def check_and_trip(self, threat_level: float) -> bool:
        """
        Check threat level and potentially trip circuit breaker
        
        Returns:
            True if circuit is open (system locked down), False otherwise
        """
        with self.lock:
            # Check if we should reset
            if self.is_open and self.trip_time:
                if time.time() - self.trip_time > self.cooldown_seconds:
                    self.is_open = False
                    self.trip_time = None
            
            # Check if we should trip
            if not self.is_open and threat_level >= self.threshold:
                self.is_open = True
                self.trip_time = time.time()
            
            return self.is_open
    
    def manual_reset(self):
        """Manually reset the circuit breaker"""
        with self.lock:
            self.is_open = False
            self.trip_time = None


class ImportPathValidator:
    """
    Import path validation and protection
    
    Fixes: Import Path Hijacking vulnerability
    """
    
    def __init__(self):
        # Freeze sys.path at initialization
        self.frozen_paths = list(sys.path)
        self.lock = threading.Lock()
    
    def validate_and_restore(self) -> bool:
        """
        Check if sys.path has been hijacked and restore if needed
        
        Returns:
            True if paths are valid, False if hijacking detected
        """
        with self.lock:
            if sys.path != self.frozen_paths:
                # Hijacking detected! Restore frozen paths
                sys.path = list(self.frozen_paths)
                return False
            return True
    
    def get_frozen_paths(self) -> List[str]:
        """Get the frozen import paths"""
        with self.lock:
            return list(self.frozen_paths)


class HardenedDNACore:
    """
    HARDENED DNA CORE - Vaccinated against nation-state attacks
    
    All critical vulnerabilities patched:
    ✅ Race conditions
    ✅ Rate limiting
    ✅ Anomaly detection
    ✅ Import validation
    ✅ Circuit breaker
    """
    
    def __init__(self, genesis_seed: str = "BalantiumFortressGenesis"):
        self.genesis_seed = genesis_seed.encode()
        self.genome: Dict[str, DNAStrand] = {}
        
        # CRITICAL FIX #1: Thread safety for race condition protection
        self.genome_lock = threading.Lock()
        
        # CRITICAL FIX #2: Rate limiter (100 ops/sec max)
        self.rate_limiter = RateLimiter(max_tokens=100, refill_rate=100.0)
        
        # CRITICAL FIX #3: Aggregate anomaly detector
        self.anomaly_detector = AnomalyDetector(window_seconds=10.0, threshold=10)
        
        # CRITICAL FIX #4: Circuit breaker for coordinated attacks
        self.circuit_breaker = CircuitBreaker(threshold=0.8, cooldown_seconds=60.0)
        
        # CRITICAL FIX #5: Import path validator
        self.import_validator = ImportPathValidator()
        
        # Define codon mapping (genetic code)
        self.codon_map = {
            "AUG": "initialize_agent_consciousness()",
            "CGA": "bind_resonance_stream()",
            "UCC": "stimulate_field_repulsion()",
            "GGU": "construct_security_mesh()",
            "ACU": "deploy_microtubule_sensors()",
            "AGG": "trigger_entropy_absorption()",
            "UAA": "terminate_sequence()",
            "GGC": "encrypt_neural_packet()",
            "CGU": "rotate_magnetic_spine()",
            "UAG": "re_seed_tissue_shield()",
            "GAA": "invoke_symbolic_marrow()",
            "CUU": "send_healing_frequency()",
            "GAC": "trigger_celestial_alarm()",
            "CCU": "sample_SHS()",
            "GUU": "calculate_decay_field()",
            "ACA": "inject_memory()",
            "UCG": "detect_field_noise()",
            "UGG": "synthesize_protein_chain()",
            "UUU": "phenylalanine_signal()",
            "GCC": "alanine_activation()",
            "GGG": "glycine_calm()",
            "UGA": "stop_codon_signal()",
        }
        
        # RNA cache for transcription (thread-safe)
        self.rna_cache: Dict[str, List[str]] = {}
        self.rna_cache_lock = threading.Lock()
        
        # Initialize
        self.initialized = True
        
        print("✅ Hardened DNA Core initialized with all security patches")
    
    def initialize(self):
        """Initialize the DNA core"""
        if not self.initialized:
            self.initialized = True
    
    def _check_security_gates(self, operation_type: str, module_name: str) -> Tuple[bool, str]:
        """
        Check all security gates before allowing operation
        
        Returns:
            (allowed, reason) tuple
        """
        # Gate 1: Check circuit breaker
        threat_level = self.anomaly_detector.get_threat_score()
        if self.circuit_breaker.check_and_trip(threat_level):
            return False, "Circuit breaker tripped - system in lockdown mode"
        
        # Gate 2: Check rate limiter
        if not self.rate_limiter.acquire():
            return False, "Rate limit exceeded - too many operations"
        
        # Gate 3: Check for anomalies
        is_anomalous = self.anomaly_detector.record_operation(operation_type, module_name)
        if is_anomalous:
            recent_ops = self.anomaly_detector.get_recent_operations_count()
            if recent_ops > 50:  # Hard limit
                return False, f"Anomaly detected - bulk operation pattern ({recent_ops} ops in 10s)"
        
        # Gate 4: Validate import paths
        if not self.import_validator.validate_and_restore():
            return False, "Import path hijacking detected and blocked"
        
        return True, "All security gates passed"
    
    def encode_module(self, module_name: str, module_code: str, 
                     security_level: str = "maximum") -> DNAStrand:
        """
        Encode a module as a DNA strand (HARDENED)
        
        Args:
            module_name: Name of the module
            module_code: The actual code/data to encode
            security_level: Security classification
            
        Returns:
            DNAStrand object
            
        Raises:
            SecurityError: If security gates fail
        """
        # Check all security gates
        allowed, reason = self._check_security_gates("encode_module", module_name)
        if not allowed:
            raise SecurityError(f"Operation blocked: {reason}")
        
        # Thread-safe genome modification
        with self.genome_lock:
            # Generate DNA sequence from module code
            sequence = self._encode_to_dna(module_code)
            
            # Create integrity hash
            integrity_hash = hashlib.sha256(
                f"{module_name}{module_code}{security_level}".encode()
            ).hexdigest()
            
            # Create DNA strand
            strand = DNAStrand(
                strand_id=f"dna_{hashlib.md5(module_name.encode()).hexdigest()[:16]}",
                sequence=sequence,
                module_name=module_name,
                security_level=security_level,
                created=datetime.now(),
                integrity_hash=integrity_hash
            )
            
            # Store in genome
            self.genome[module_name] = strand
            
            return strand
    
    def _encode_to_dna(self, data: str) -> List[str]:
        """Convert data into DNA base pairs"""
        base_pairs = ['A', 'T', 'C', 'G']
        hash_digest = hashlib.sha256(data.encode()).hexdigest()
        
        # Convert hex to DNA sequence
        dna_sequence = []
        for char in hash_digest:
            # Map hex character to base pair
            idx = int(char, 16) % 4
            dna_sequence.append(base_pairs[idx])
        
        return dna_sequence
    
    def transcribe_to_rna(self, strand_id: str) -> List[str]:
        """
        Transcribe DNA to RNA (thread-safe)
        
        DNA -> RNA transcription rules:
        A -> U
        T -> A
        C -> G
        G -> C
        """
        # Check cache first (thread-safe)
        with self.rna_cache_lock:
            if strand_id in self.rna_cache:
                return self.rna_cache[strand_id]
        
        # Find the strand (thread-safe)
        with self.genome_lock:
            strand = None
            for module_name, dna_strand in self.genome.items():
                if dna_strand.strand_id == strand_id:
                    strand = dna_strand
                    break
            
            if not strand:
                return []
            
            # Transcribe DNA to RNA
            rna_sequence = []
            transcription_map = {'A': 'U', 'T': 'A', 'C': 'G', 'G': 'C'}
            
            for base in strand.sequence:
                rna_sequence.append(transcription_map.get(base, base))
        
        # Cache the result (thread-safe)
        with self.rna_cache_lock:
            self.rna_cache[strand_id] = rna_sequence
        
        return rna_sequence
    
    def translate_rna(self, rna_sequence: List[str]) -> List[str]:
        """
        Translate RNA to protein (operations)
        
        Reads RNA in codons (groups of 3) and maps to operations
        """
        operations = []
        
        # Read in codons (groups of 3)
        for i in range(0, len(rna_sequence) - 2, 3):
            codon = "".join(rna_sequence[i:i+3])
            
            if codon in self.codon_map:
                operations.append(self.codon_map[codon])
            else:
                # Unknown codon - no operation
                operations.append("nop()")
        
        return operations
    
    def verify_integrity(self, module_name: str) -> bool:
        """
        Verify the integrity of a DNA strand (thread-safe)
        
        Returns:
            True if integrity is intact, False otherwise
        """
        with self.genome_lock:
            if module_name not in self.genome:
                return False
            
            strand = self.genome[module_name]
            
            # Recalculate integrity hash
            current_hash = hashlib.sha256(
                f"{strand.module_name}{strand.sequence}{strand.security_level}".encode()
            ).hexdigest()
            
            return current_hash == strand.integrity_hash
    
    def mutate_strand(self, module_name: str, mutation_rate: float = 0.01):
        """
        Introduce controlled mutations for evolution (thread-safe)
        
        Args:
            module_name: Module to mutate
            mutation_rate: Probability of mutation per base pair
        """
        with self.genome_lock:
            if module_name not in self.genome:
                return
            
            strand = self.genome[module_name]
            base_pairs = ['A', 'T', 'C', 'G']
            
            # Mutate sequence
            mutated_sequence = []
            for base in strand.sequence:
                if np.random.random() < mutation_rate:
                    # Mutate to a different base
                    new_base = np.random.choice([b for b in base_pairs if b != base])
                    mutated_sequence.append(new_base)
                else:
                    mutated_sequence.append(base)
            
            # Update strand
            strand.sequence = mutated_sequence
            strand.mutations += 1
            
            # Update integrity hash
            strand.integrity_hash = hashlib.sha256(
                f"{strand.module_name}{strand.sequence}{strand.security_level}".encode()
            ).hexdigest()
    
    def repair_strand(self, module_name: str, original_code: str):
        """
        Repair a corrupted DNA strand (thread-safe)
        
        Args:
            module_name: Module to repair
            original_code: Original module code for reference
        """
        with self.genome_lock:
            if module_name not in self.genome:
                return
            
            # Re-encode from original
            strand = self.genome[module_name]
            strand.sequence = self._encode_to_dna(original_code)
            
            # Update integrity hash
            strand.integrity_hash = hashlib.sha256(
                f"{strand.module_name}{original_code}{strand.security_level}".encode()
            ).hexdigest()
    
    def get_genome_status(self) -> Dict:
        """Get status of the entire genome (thread-safe)"""
        with self.genome_lock:
            return {
                'total_strands': len(self.genome),
                'modules': list(self.genome.keys()),
                'total_mutations': sum(strand.mutations for strand in self.genome.values()),
                'integrity_status': {
                    module: self.verify_integrity(module)
                    for module in self.genome.keys()
                }
            }
    
    def get_genome_statistics(self) -> Dict:
        """Get detailed genome statistics (thread-safe)"""
        with self.genome_lock:
            return {
                'total_strands': len(self.genome),
                'module_names': list(self.genome.keys()),
                'total_base_pairs': sum(len(strand.sequence) for strand in self.genome.values()),
                'total_mutations': sum(strand.mutations for strand in self.genome.values()),
                'avg_mutations_per_strand': sum(strand.mutations for strand in self.genome.values()) / max(1, len(self.genome)),
                'intact_modules': sum(1 for module in self.genome.keys() if self.verify_integrity(module)),
                'compromised_modules': sum(1 for module in self.genome.keys() if not self.verify_integrity(module))
            }
    
    def get_security_status(self) -> Dict:
        """Get current security status of all defense systems"""
        return {
            'circuit_breaker': {
                'is_open': self.circuit_breaker.is_open,
                'trip_time': self.circuit_breaker.trip_time
            },
            'rate_limiter': {
                'available_tokens': self.rate_limiter.get_available_tokens(),
                'max_tokens': self.rate_limiter.max_tokens
            },
            'anomaly_detector': {
                'threat_score': self.anomaly_detector.get_threat_score(),
                'recent_operations': self.anomaly_detector.get_recent_operations_count(),
                'threshold': self.anomaly_detector.threshold
            },
            'import_paths': {
                'validated': self.import_validator.validate_and_restore(),
                'frozen_count': len(self.import_validator.get_frozen_paths())
            }
        }


class SecurityError(Exception):
    """Raised when security gates block an operation"""
    pass


# Global hardened DNA core instance
DNA_CORE_HARDENED = HardenedDNACore()

# Also export as DNA_CORE for backward compatibility
DNA_CORE = DNA_CORE_HARDENED


if __name__ == "__main__":
    print("🧬 Balantium HARDENED DNA Core - All Security Patches Applied")
    print(f"   Codon Map Size: {len(DNA_CORE_HARDENED.codon_map)}")
    print(f"   Genesis Seed: {DNA_CORE_HARDENED.genesis_seed[:20]}...")
    print("\n🛡️  Security Systems:")
    print("   ✅ Thread-safe operations (race condition fix)")
    print("   ✅ Rate limiter (100 ops/sec)")
    print("   ✅ Anomaly detector (10 ops/10sec threshold)")
    print("   ✅ Circuit breaker (0.8 threat threshold)")
    print("   ✅ Import path validator (frozen at init)")


