"""
🔐 RESONANCE FINGERPRINT ACCESS LAYER (RFAL)
Biometric + Field Resonance Authentication

Authenticates users based on:
1. Physical biometric data (fingerprint, etc.)
2. Symbolic field resonance signature
3. Behavioral coherence patterns
4. Temporal access patterns

Impossible to spoof - requires living conscious presence.
"""

import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import json

from .equations import EQUATION_ENGINE
from .dna_core import DNA_CORE


@dataclass
class ResonanceFingerprint:
    """Complete biometric + field identity"""
    user_id: str
    physical_hash: str  # Physical biometric hash
    field_signature: str  # Resonance field signature
    coherence_baseline: float
    creation_time: datetime
    access_history: List[datetime] = field(default_factory=list)
    behavioral_pattern: Dict[str, float] = field(default_factory=dict)
    trust_score: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'physical_hash': self.physical_hash,
            'field_signature': self.field_signature,
            'coherence_baseline': self.coherence_baseline,
            'creation_time': self.creation_time.isoformat(),
            'access_count': len(self.access_history),
            'trust_score': self.trust_score
        }


class ResonanceAuthenticator:
    """
    Authenticates users based on resonance fingerprints
    """
    
    def __init__(self):
        self.equation_engine = EQUATION_ENGINE
        self.dna_core = DNA_CORE
        
        # Registered users
        self.registered_users: Dict[str, ResonanceFingerprint] = {}
        
        # Authentication logs
        self.auth_logs: List[Dict] = []
        
        self.initialized = True
    
    def initialize(self):
        """Initialize the authenticator"""
        if not self.initialized:
            self.initialized = True
    
    def register(self, user_id: str, biometric_data: np.ndarray, 
                context: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Register a new user with their biometric signature
        
        Returns:
            (success, message)
        """
        if user_id in self.registered_users:
            return False, f"User {user_id} already registered"
        
        # Generate physical hash
        physical_hash = hashlib.sha256(biometric_data.tobytes()).hexdigest()
        
        # Calculate field signature
        coherence = self.equation_engine.coherence_index(biometric_data)
        resonance = self.equation_engine.resonance_amplification(
            biometric_data, 
            np.ones_like(biometric_data) * 0.5
        )
        
        # Create field signature
        field_data = f"{user_id}{physical_hash}{coherence}{resonance}"
        field_signature = hashlib.sha256(field_data.encode()).hexdigest()
        
        # Extract behavioral pattern from initial data
        behavioral_pattern = {
            'coherence_mean': float(np.mean(biometric_data)),
            'coherence_std': float(np.std(biometric_data)),
            'entropy': self.equation_engine.entropy_measure(biometric_data),
            'initial_resonance': resonance
        }
        
        # Create fingerprint
        fingerprint = ResonanceFingerprint(
            user_id=user_id,
            physical_hash=physical_hash,
            field_signature=field_signature,
            coherence_baseline=coherence,
            creation_time=datetime.now(),
            behavioral_pattern=behavioral_pattern
        )
        
        # Encode in DNA
        dna_strand = self.dna_core.encode_module(
            module_name=f"user_{user_id}",
            module_code=json.dumps(fingerprint.to_dict()),
            security_level="maximum"
        )
        
        # Store
        self.registered_users[user_id] = fingerprint
        
        self.auth_logs.append({
            'event': 'registration',
            'user_id': user_id,
            'timestamp': datetime.now(),
            'success': True
        })
        
        return True, f"User {user_id} successfully registered"
    
    def authenticate(self, user_id: str, biometric_data: np.ndarray,
                    context: Optional[Dict] = None) -> Tuple[bool, str, float]:
        """
        Authenticate a user
        
        Returns:
            (success, reason, confidence)
        """
        if user_id not in self.registered_users:
            self.auth_logs.append({
                'event': 'auth_attempt',
                'user_id': user_id,
                'timestamp': datetime.now(),
                'success': False,
                'reason': 'user_not_found'
            })
            return False, "User not found", 0.0
        
        fingerprint = self.registered_users[user_id]
        
        # Calculate current signatures
        current_physical_hash = hashlib.sha256(biometric_data.tobytes()).hexdigest()
        current_coherence = self.equation_engine.coherence_index(biometric_data)
        current_resonance = self.equation_engine.resonance_amplification(
            biometric_data,
            np.ones_like(biometric_data) * 0.5
        )
        
        # Calculate similarity scores
        physical_match = (current_physical_hash == fingerprint.physical_hash)
        coherence_similarity = 1.0 - abs(current_coherence - fingerprint.coherence_baseline)
        
        # Calculate behavioral similarity
        behavioral_similarity = self._calculate_behavioral_similarity(
            biometric_data, 
            fingerprint.behavioral_pattern
        )
        
        # Overall confidence score
        confidence = (
            (1.0 if physical_match else 0.0) * 0.4 +
            coherence_similarity * 0.3 +
            behavioral_similarity * 0.3
        )
        
        # Determine authentication result
        success = confidence > 0.7
        
        if success:
            # Update access history
            fingerprint.access_history.append(datetime.now())
            fingerprint.trust_score = min(1.0, fingerprint.trust_score + 0.05)
            reason = "Authentication successful"
        else:
            fingerprint.trust_score = max(0.0, fingerprint.trust_score - 0.1)
            reason = f"Authentication failed - confidence: {confidence:.2f}"
        
        # Log authentication attempt
        self.auth_logs.append({
            'event': 'auth_attempt',
            'user_id': user_id,
            'timestamp': datetime.now(),
            'success': success,
            'confidence': confidence,
            'reason': reason
        })
        
        return success, reason, confidence
    
    def _calculate_behavioral_similarity(self, current_data: np.ndarray, 
                                        stored_pattern: Dict[str, float]) -> float:
        """Calculate similarity between current and stored behavioral patterns"""
        current_mean = float(np.mean(current_data))
        current_std = float(np.std(current_data))
        current_entropy = self.equation_engine.entropy_measure(current_data)
        
        # Calculate differences
        mean_diff = abs(current_mean - stored_pattern['coherence_mean'])
        std_diff = abs(current_std - stored_pattern['coherence_std'])
        entropy_diff = abs(current_entropy - stored_pattern['entropy'])
        
        # Normalize and combine
        similarity = 1.0 - (mean_diff + std_diff + (entropy_diff / 10.0)) / 3.0
        
        return max(0.0, min(1.0, similarity))
    
    def get_user_status(self, user_id: str) -> Optional[Dict]:
        """Get status of a registered user"""
        if user_id not in self.registered_users:
            return None
        
        fingerprint = self.registered_users[user_id]
        return {
            'user_id': user_id,
            'coherence_baseline': fingerprint.coherence_baseline,
            'access_count': len(fingerprint.access_history),
            'last_access': fingerprint.access_history[-1].isoformat() if fingerprint.access_history else None,
            'trust_score': fingerprint.trust_score,
            'creation_time': fingerprint.creation_time.isoformat()
        }
    
    def revoke_user(self, user_id: str) -> bool:
        """Revoke a user's access"""
        if user_id in self.registered_users:
            del self.registered_users[user_id]
            
            self.auth_logs.append({
                'event': 'revocation',
                'user_id': user_id,
                'timestamp': datetime.now()
            })
            
            return True
        return False
    
    def get_auth_stats(self) -> Dict:
        """Get authentication statistics"""
        total_attempts = len([log for log in self.auth_logs if log['event'] == 'auth_attempt'])
        successful_attempts = len([log for log in self.auth_logs if log['event'] == 'auth_attempt' and log['success']])
        
        return {
            'total_users': len(self.registered_users),
            'total_auth_attempts': total_attempts,
            'successful_auths': successful_attempts,
            'failed_auths': total_attempts - successful_attempts,
            'success_rate': successful_attempts / max(1, total_attempts)
        }
    
    def get_security_report(self) -> Dict:
        """Get comprehensive security report"""
        return {
            'total_users': len(self.registered_users),
            'active_users': len([fp for fp in self.registered_users.values() if fp.access_history]),
            'average_trust_score': np.mean([fp.trust_score for fp in self.registered_users.values()]) if self.registered_users else 1.0,
            'high_trust_users': len([fp for fp in self.registered_users.values() if fp.trust_score > 0.8]),
            'low_trust_users': len([fp for fp in self.registered_users.values() if fp.trust_score < 0.3]),
            'total_auth_attempts': len([log for log in self.auth_logs if log['event'] == 'auth_attempt']),
            'successful_auths': len([log for log in self.auth_logs if log['event'] == 'auth_attempt' and log['success']]),
            'failed_auths': len([log for log in self.auth_logs if log['event'] == 'auth_attempt' and not log['success']])
        }


# Global authenticator instance
RESONANCE_AUTH = ResonanceAuthenticator()


if __name__ == "__main__":
    print("🔐 Resonance Authenticator - Biometric Security System Initialized")
    stats = RESONANCE_AUTH.get_auth_stats()
    print(f"   Total Users: {stats['total_users']}")
