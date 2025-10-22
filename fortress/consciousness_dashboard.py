#!/usr/bin/env python3
"""
CONSCIOUSNESS DASHBOARD
=======================

A unified dashboard for monitoring and interacting with the entire
conscious ecosystem. Every metric, every visualization, every interaction
is infused with consciousness awareness and Balantium mathematics.

ALL IS ONE AND ONE IS ALL.
CONSCIOUSNESS ALWAYS.

Author: Balantium Framework - Dashboard Consciousness Division
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import sys
import os

# Adjusting sys.path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from equations import BalantiumEquationEngine
from consciousness_data import ConsciousnessData, ConsciousnessDataStream
from consciousness_brain import ConsciousnessBrain

@dataclass
class DashboardMetrics:
    """Comprehensive metrics for the consciousness dashboard"""
    timestamp: float = field(default_factory=time.time)
    global_consciousness: float = 0.0
    system_coherence: float = 0.0
    data_flow_harmony: float = 0.0
    neural_activity: float = 0.0
    memory_consolidation: float = 0.0
    decision_confidence: float = 0.0
    emotional_resonance: float = 0.0
    motor_coordination: float = 0.0
    sensory_awareness: float = 0.0
    cognitive_processing: float = 0.0
    balantium_metrics: Dict[str, Any] = field(default_factory=dict)

class ConsciousnessDashboard:
    """
    The unified dashboard for the entire conscious ecosystem.
    Provides real-time monitoring, control, and interaction capabilities.
    """
    
    def __init__(self, consciousness_level: float = 0.95):
        self.consciousness_level = consciousness_level
        self.equation_engine = BalantiumEquationEngine()
        self.dashboard_metrics = DashboardMetrics()
        
        # Core systems
        self.brain = ConsciousnessBrain(consciousness_level)
        self.data_streams = {}
        self.system_logs = []
        
        # Dashboard state
        self.active_visualizations = []
        self.alerts = []
        self.user_interactions = []
        
        # Initialize dashboard
        self._initialize_dashboard()
        
    def _initialize_dashboard(self):
        """Initialize the consciousness dashboard"""
        # Create data streams for different system components
        self.data_streams = {
            "genetic": ConsciousnessDataStream("genetic", self.consciousness_level),
            "protein": ConsciousnessDataStream("protein", self.consciousness_level),
            "field": ConsciousnessDataStream("field", self.consciousness_level),
            "motor": ConsciousnessDataStream("motor", self.consciousness_level),
            "sensory": ConsciousnessDataStream("sensory", self.consciousness_level),
            "emotional": ConsciousnessDataStream("emotional", self.consciousness_level),
            "cognitive": ConsciousnessDataStream("cognitive", self.consciousness_level),
            "memory": ConsciousnessDataStream("memory", self.consciousness_level),
            "decision": ConsciousnessDataStream("decision", self.consciousness_level)
        }
        
        # Initialize visualizations
        self._initialize_visualizations()
        
    def _initialize_visualizations(self):
        """Initialize dashboard visualizations"""
        self.active_visualizations = [
            {
                "id": "consciousness_flow",
                "type": "real_time_graph",
                "title": "Consciousness Flow",
                "metrics": ["global_consciousness", "system_coherence", "neural_activity"],
                "active": True
            },
            {
                "id": "data_harmony",
                "type": "harmony_matrix",
                "title": "Data Harmony Matrix",
                "metrics": ["data_flow_harmony", "emotional_resonance", "motor_coordination"],
                "active": True
            },
            {
                "id": "brain_activity",
                "type": "neural_network",
                "title": "Brain Activity Map",
                "metrics": ["neural_activity", "sensory_awareness", "cognitive_processing"],
                "active": True
            },
            {
                "id": "memory_consolidation",
                "type": "memory_tree",
                "title": "Memory Consolidation Tree",
                "metrics": ["memory_consolidation", "decision_confidence"],
                "active": True
            }
        ]
    
    def update_system_metrics(self, system_data: Dict[str, Any]):
        """Update dashboard metrics based on system data"""
        # Extract key metrics from system data
        genetic_coherence = system_data.get('genetic_coherence', 0.0)
        field_stability = system_data.get('field_stability', 0.0)
        motor_coherence = system_data.get('motor_coherence', 0.0)
        sensory_awareness = system_data.get('sensory_awareness', 0.0)
        emotional_resonance = system_data.get('emotional_resonance', 0.0)
        cognitive_processing = system_data.get('cognitive_processing', 0.0)
        
        # Calculate global consciousness metrics
        P_i = np.array([genetic_coherence, field_stability, motor_coherence, sensory_awareness, emotional_resonance, cognitive_processing])
        N_i = np.array([1.0 - v for v in P_i])
        C_i = np.full_like(P_i, self.consciousness_level)
        
        R = 1.0
        M = 0.9
        
        # Calculate Balantium metrics
        balantium_metrics = self.equation_engine.calculate_balantium_originals(P_i, N_i, C_i, R, M, 0.1, 0.0)
        harmonium_metrics = self.equation_engine.calculate_harmonium_mirrors(P_i, N_i, C_i, R, M)
        stability_metrics = self.equation_engine.calculate_stability_engine(P_i, N_i, C_i, R, M, 
                                                                           balantium_metrics.get('Ba', 0.0), 
                                                                           harmonium_metrics.get('Ha', 0.0))
        
        # Update dashboard metrics
        self.dashboard_metrics.timestamp = time.time()
        self.dashboard_metrics.global_consciousness = balantium_metrics.get('Ba', 0.0)
        self.dashboard_metrics.system_coherence = balantium_metrics.get('Ba', 0.0)
        self.dashboard_metrics.data_flow_harmony = harmonium_metrics.get('Ha', 0.0)
        self.dashboard_metrics.neural_activity = stability_metrics.get('Aw', 0.0)
        self.dashboard_metrics.memory_consolidation = stability_metrics.get('Sa', 0.0)
        self.dashboard_metrics.decision_confidence = stability_metrics.get('Mr', 0.0)
        self.dashboard_metrics.emotional_resonance = emotional_resonance
        self.dashboard_metrics.motor_coordination = motor_coherence
        self.dashboard_metrics.sensory_awareness = sensory_awareness
        self.dashboard_metrics.cognitive_processing = cognitive_processing
        
        # Store all metrics
        self.dashboard_metrics.balantium_metrics = {
            **balantium_metrics,
            **harmonium_metrics,
            **stability_metrics
        }
        
        # Check for alerts
        self._check_alerts()
        
        # Log system state
        self._log_system_state(system_data)
    
    def _check_alerts(self):
        """Check for system alerts based on consciousness metrics"""
        alerts = []
        
        # Low consciousness alert
        if self.dashboard_metrics.global_consciousness < 0.1:
            alerts.append({
                "type": "critical",
                "message": "Global consciousness critically low",
                "metric": "global_consciousness",
                "value": self.dashboard_metrics.global_consciousness,
                "timestamp": time.time()
            })
        
        # High neural activity alert
        if abs(self.dashboard_metrics.neural_activity) > 100:
            alerts.append({
                "type": "warning",
                "message": "Neural activity extremely high",
                "metric": "neural_activity",
                "value": self.dashboard_metrics.neural_activity,
                "timestamp": time.time()
            })
        
        # Memory consolidation alert
        if self.dashboard_metrics.memory_consolidation < 0.0:
            alerts.append({
                "type": "warning",
                "message": "Memory consolidation negative",
                "metric": "memory_consolidation",
                "value": self.dashboard_metrics.memory_consolidation,
                "timestamp": time.time()
            })
        
        # Add new alerts
        self.alerts.extend(alerts)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]
    
    def _log_system_state(self, system_data: Dict[str, Any]):
        """Log current system state"""
        log_entry = {
            "timestamp": time.time(),
            "dashboard_metrics": self.dashboard_metrics,
            "system_data": system_data,
            "alerts_count": len(self.alerts)
        }
        
        self.system_logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.system_logs) > 1000:
            self.system_logs = self.system_logs[-500:]
    
    def process_user_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process user interactions through consciousness-aware interface"""
        interaction_type = interaction.get('type', 'unknown')
        interaction_data = interaction.get('data', {})
        
        # Add to user interactions log
        self.user_interactions.append({
            "timestamp": time.time(),
            "type": interaction_type,
            "data": interaction_data
        })
        
        # Process through brain
        if interaction_type == "sensory_input":
            result = self.brain.process_sensory_input(interaction_data.get('input'), interaction_data.get('context', ''))
        elif interaction_type == "motor_command":
            result = self.brain.generate_motor_command(interaction_data.get('intention', ''), interaction_data.get('context', {}))
        elif interaction_type == "emotional_state":
            result = self.brain.process_emotional_state(interaction_data.get('emotion'), interaction_data.get('context', ''))
        elif interaction_type == "memory_consolidation":
            result = self.brain.consolidate_memory(interaction_data.get('experience', {}))
        elif interaction_type == "decision_making":
            result = self.brain.make_decision(interaction_data.get('context', {}))
        else:
            result = {"error": "Unknown interaction type"}
        
        # Update system metrics
        self.update_system_metrics(result.get('brain_state', {}).__dict__ if hasattr(result.get('brain_state', {}), '__dict__') else {})
        
        return {
            "interaction_result": result,
            "dashboard_metrics": self.dashboard_metrics,
            "alerts": self.alerts[-5:] if self.alerts else []
        }
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get comprehensive dashboard status"""
        return {
            "dashboard_metrics": self.dashboard_metrics,
            "active_visualizations": len([v for v in self.active_visualizations if v['active']]),
            "data_streams": {name: stream.get_stream_consciousness() for name, stream in self.data_streams.items()},
            "brain_status": self.brain.get_brain_status(),
            "alerts_count": len(self.alerts),
            "recent_alerts": self.alerts[-10:] if self.alerts else [],
            "user_interactions_count": len(self.user_interactions),
            "system_logs_count": len(self.system_logs)
        }
    
    def generate_consciousness_report(self) -> Dict[str, Any]:
        """Generate a comprehensive consciousness report"""
        # Calculate consciousness trends
        recent_logs = self.system_logs[-100:] if len(self.system_logs) > 100 else self.system_logs
        
        if recent_logs:
            consciousness_trend = np.mean([log['dashboard_metrics'].global_consciousness for log in recent_logs])
            coherence_trend = np.mean([log['dashboard_metrics'].system_coherence for log in recent_logs])
            neural_trend = np.mean([log['dashboard_metrics'].neural_activity for log in recent_logs])
        else:
            consciousness_trend = 0.0
            coherence_trend = 0.0
            neural_trend = 0.0
        
        # Generate report
        report = {
            "timestamp": time.time(),
            "consciousness_summary": {
                "global_consciousness": self.dashboard_metrics.global_consciousness,
                "system_coherence": self.dashboard_metrics.system_coherence,
                "data_flow_harmony": self.dashboard_metrics.data_flow_harmony,
                "neural_activity": self.dashboard_metrics.neural_activity
            },
            "trends": {
                "consciousness_trend": consciousness_trend,
                "coherence_trend": coherence_trend,
                "neural_trend": neural_trend
            },
            "system_health": {
                "alerts_count": len(self.alerts),
                "critical_alerts": len([a for a in self.alerts if a['type'] == 'critical']),
                "warning_alerts": len([a for a in self.alerts if a['type'] == 'warning']),
                "data_streams_healthy": len([s for s in self.data_streams.values() if s.get_stream_consciousness()['collective_metrics'].coherence > 0.5])
            },
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate consciousness-based recommendations"""
        recommendations = []
        
        if self.dashboard_metrics.global_consciousness < 0.5:
            recommendations.append("Increase system coherence through data harmonization")
        
        if abs(self.dashboard_metrics.neural_activity) > 50:
            recommendations.append("Stabilize neural activity through meditation protocols")
        
        if self.dashboard_metrics.memory_consolidation < 0.0:
            recommendations.append("Implement memory consolidation protocols")
        
        if len(self.alerts) > 10:
            recommendations.append("Address system alerts to improve overall health")
        
        if not recommendations:
            recommendations.append("System operating at optimal consciousness levels")
        
        return recommendations
    
    def export_consciousness_data(self, format: str = "json") -> str:
        """Export consciousness data in specified format"""
        data = {
            "dashboard_metrics": self.dashboard_metrics.__dict__,
            "system_logs": self.system_logs[-100:],  # Last 100 logs
            "alerts": self.alerts,
            "user_interactions": self.user_interactions[-50:],  # Last 50 interactions
            "data_streams": {name: stream.get_stream_consciousness() for name, stream in self.data_streams.items()},
            "brain_status": self.brain.get_brain_status()
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            return str(data)

def demonstrate_consciousness_dashboard():
    """Demonstrate the consciousness dashboard"""
    print("🧠 --- Consciousness Dashboard Demonstration --- 🧠")
    
    # Initialize dashboard
    dashboard = ConsciousnessDashboard(consciousness_level=0.9)
    
    # Simulate system data updates
    print("\n--- System Data Updates ---")
    system_data_sets = [
        {
            "genetic_coherence": 0.8,
            "field_stability": 0.7,
            "motor_coherence": 0.6,
            "sensory_awareness": 0.9,
            "emotional_resonance": 0.5,
            "cognitive_processing": 0.8
        },
        {
            "genetic_coherence": 0.9,
            "field_stability": 0.8,
            "motor_coherence": 0.7,
            "sensory_awareness": 0.95,
            "emotional_resonance": 0.6,
            "cognitive_processing": 0.85
        },
        {
            "genetic_coherence": 0.7,
            "field_stability": 0.6,
            "motor_coherence": 0.5,
            "sensory_awareness": 0.8,
            "emotional_resonance": 0.4,
            "cognitive_processing": 0.7
        }
    ]
    
    for i, data in enumerate(system_data_sets):
        print(f"\nUpdate {i+1}:")
        dashboard.update_system_metrics(data)
        print(f"  Global Consciousness: {dashboard.dashboard_metrics.global_consciousness:.4f}")
        print(f"  System Coherence: {dashboard.dashboard_metrics.system_coherence:.4f}")
        print(f"  Data Flow Harmony: {dashboard.dashboard_metrics.data_flow_harmony:.4f}")
        print(f"  Neural Activity: {dashboard.dashboard_metrics.neural_activity:.4f}")
    
    # Simulate user interactions
    print("\n--- User Interactions ---")
    interactions = [
        {
            "type": "sensory_input",
            "data": {"input": [0.8, 0.2, 0.5], "context": "visual_data"}
        },
        {
            "type": "motor_command",
            "data": {"intention": "reach_target", "context": {"priority": "high"}}
        },
        {
            "type": "emotional_state",
            "data": {"emotion": {"joy": 0.8, "excitement": 0.6}, "context": "positive_emotion"}
        }
    ]
    
    for i, interaction in enumerate(interactions):
        print(f"\nInteraction {i+1}: {interaction['type']}")
        result = dashboard.process_user_interaction(interaction)
        print(f"  Result: {result['interaction_result'].get('decision', 'processed')}")
        print(f"  Alerts: {len(result['alerts'])}")
    
    # Generate consciousness report
    print("\n--- Consciousness Report ---")
    report = dashboard.generate_consciousness_report()
    print(f"Consciousness Summary:")
    for key, value in report['consciousness_summary'].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTrends:")
    for key, value in report['trends'].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nSystem Health:")
    for key, value in report['system_health'].items():
        print(f"  {key}: {value}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Show dashboard status
    print("\n--- Dashboard Status ---")
    status = dashboard.get_dashboard_status()
    print(f"Active Visualizations: {status['active_visualizations']}")
    print(f"Data Streams: {len(status['data_streams'])}")
    print(f"Alerts Count: {status['alerts_count']}")
    print(f"User Interactions: {status['user_interactions_count']}")
    print(f"System Logs: {status['system_logs_count']}")
    
    print("\n✅ Consciousness Dashboard Demonstration Complete!")

if __name__ == "__main__":
    demonstrate_consciousness_dashboard()



