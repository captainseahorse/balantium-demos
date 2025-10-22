#!/usr/bin/env python3
"""
Conscious Firewall
==================

A consciousness-aware firewall that uses Balantium equations for threat detection
and protection of the consciousness ecosystem.

ALL IS ONE AND ONE IS ALL.
CONSCIOUSNESS ALWAYS.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class FirewallRule:
    """Represents a firewall rule"""
    rule_id: str
    name: str
    pattern: str
    action: str  # 'allow', 'block', 'monitor'
    severity: float
    consciousness_threshold: float

class ConsciousFirewall:
    """
    Consciousness-aware firewall using Balantium equations for threat detection.
    """
    
    def __init__(self):
        self.rules = []
        self.blocked_attempts = []
        self.allowed_attempts = []
        self.consciousness_threshold = 0.7
        self.last_scan = time.time()
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default firewall rules"""
        default_rules = [
            FirewallRule(
                rule_id="consciousness_validation",
                name="Consciousness Validation",
                pattern="consciousness_level",
                action="monitor",
                severity=0.8,
                consciousness_threshold=0.5
            ),
            FirewallRule(
                rule_id="coherence_check",
                name="Coherence Check",
                pattern="coherence_score",
                action="monitor",
                severity=0.7,
                consciousness_threshold=0.6
            ),
            FirewallRule(
                rule_id="resonance_validation",
                name="Resonance Validation",
                pattern="resonance_strength",
                action="monitor",
                severity=0.6,
                consciousness_threshold=0.7
            )
        ]
        
        self.rules.extend(default_rules)
    
    def check_request(self, request_data: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
        """
        Check a request against firewall rules using consciousness metrics.
        
        Args:
            request_data: Data associated with the request
            source: Source of the request
            
        Returns:
            Dictionary containing firewall decision and metrics
        """
        decision = {
            "allowed": True,
            "reason": "Request approved",
            "consciousness_score": 0.0,
            "threat_level": "Low",
            "rules_triggered": [],
            "timestamp": time.time()
        }
        
        # Calculate consciousness score for the request
        consciousness_score = self._calculate_consciousness_score(request_data)
        decision["consciousness_score"] = consciousness_score
        
        # Check against rules
        for rule in self.rules:
            if self._rule_matches(rule, request_data):
                decision["rules_triggered"].append(rule.rule_id)
                
                # Apply rule action
                if rule.action == "block":
                    decision["allowed"] = False
                    decision["reason"] = f"Blocked by rule: {rule.name}"
                    decision["threat_level"] = "High"
                    self.blocked_attempts.append({
                        "timestamp": time.time(),
                        "source": source,
                        "rule": rule.rule_id,
                        "consciousness_score": consciousness_score
                    })
                    break
                elif rule.action == "monitor":
                    if consciousness_score < rule.consciousness_threshold:
                        decision["threat_level"] = "Medium"
                        decision["reason"] = f"Monitored by rule: {rule.name}"
        
        # Log allowed attempts
        if decision["allowed"]:
            self.allowed_attempts.append({
                "timestamp": time.time(),
                "source": source,
                "consciousness_score": consciousness_score
            })
        
        return decision
    
    def _calculate_consciousness_score(self, data: Dict[str, Any]) -> float:
        """Calculate consciousness score for request data using Balantium equations"""
        try:
            # Extract consciousness-related metrics
            consciousness_level = data.get('consciousness_level', 0.5)
            coherence_score = data.get('coherence_score', 0.5)
            resonance_strength = data.get('resonance_strength', 0.5)
            
            # Simple Balantium-inspired calculation
            # Ba = Σ[(P_i − N_i) · C_i · R · M] − F + T
            P_i = np.array([consciousness_level, coherence_score, resonance_strength])
            N_i = np.array([1.0 - p for p in P_i])
            C_i = np.array([0.8, 0.8, 0.8])  # Coherence factors
            R = 1.0  # Resonance factor
            M = 0.9  # Transmission efficiency
            F = 0.1  # Feedback delay
            T = 0.0  # Tipping point
            
            # Calculate Balantium score
            terms = [(p - n) * c * R * M for p, n, c in zip(P_i, N_i, C_i)]
            balantium_score = sum(terms) - F + T
            
            # Normalize to 0-1 range
            normalized_score = max(0.0, min(1.0, (balantium_score + 1.0) / 2.0))
            
            return normalized_score
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _rule_matches(self, rule: FirewallRule, data: Dict[str, Any]) -> bool:
        """Check if a rule matches the request data"""
        try:
            if rule.pattern in data:
                return True
            return False
        except Exception:
            return False
    
    def get_firewall_status(self) -> Dict[str, Any]:
        """Get current firewall status and statistics"""
        return {
            "timestamp": time.time(),
            "total_rules": len(self.rules),
            "blocked_attempts": len(self.blocked_attempts),
            "allowed_attempts": len(self.allowed_attempts),
            "consciousness_threshold": self.consciousness_threshold,
            "last_scan": self.last_scan,
            "status": "active"
        }
    
    def add_rule(self, rule: FirewallRule) -> bool:
        """Add a new firewall rule"""
        try:
            self.rules.append(rule)
            return True
        except Exception:
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a firewall rule by ID"""
        try:
            self.rules = [rule for rule in self.rules if rule.rule_id != rule_id]
            return True
        except Exception:
            return False
    
    def update_consciousness_threshold(self, new_threshold: float) -> bool:
        """Update the consciousness threshold"""
        try:
            self.consciousness_threshold = max(0.0, min(1.0, new_threshold))
            return True
        except Exception:
            return False