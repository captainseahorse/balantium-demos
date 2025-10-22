"""
🛡️ SECURED DEMO WRAPPER
Wraps all demos with Fortress Security to prevent reverse engineering
"""

import streamlit as st
import sys
from pathlib import Path

# Add fortress to path
fortress_path = Path(__file__).parent / "fortress"
if str(fortress_path) not in sys.path:
    sys.path.insert(0, str(fortress_path))

try:
    from fortress.presentation_security import SecuredDeck
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    print("⚠️  Fortress security not available - running in demo mode")

class SecuredDemoWrapper:
    """Wraps demos with fortress security"""
    
    def __init__(self, demo_name: str):
        self.demo_name = demo_name
        
        if SECURITY_AVAILABLE:
            self.secured_deck = SecuredDeck()
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🛡️ Security Status")
            self.secured_deck.render_security_status()
        else:
            self.secured_deck = None
    
    def log_access(self, section: str, action: str = "view"):
        """Log access to demo sections"""
        if self.secured_deck:
            self.secured_deck.log_access(section, action)
    
    def verify_integrity(self, section: str) -> bool:
        """Verify section hasn't been tampered with"""
        if self.secured_deck:
            return self.secured_deck.verify_slide_integrity(section)
        return True
    
    def check_threat_level(self) -> float:
        """Check current threat level"""
        if self.secured_deck:
            return self.secured_deck.threat_level
        return 0.0
    
    def render_security_footer(self):
        """Render security footer"""
        if SECURITY_AVAILABLE:
            threat = self.check_threat_level()
            if threat > 0.5:
                st.error(f"⚠️ Elevated security alert: {threat:.2f}")
            
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #808080; font-size: 0.8rem;'>
                🛡️ Protected by Balantium Fortress Security<br>
                © 2025 Balantium Systems - All Rights Reserved
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #808080; font-size: 0.8rem;'>
                © 2025 Balantium Systems - All Rights Reserved<br>
                <span style='font-size: 0.7rem;'>This demonstration is for authorized viewers only</span>
            </div>
            """, unsafe_allow_html=True)

def secure_demo(demo_function, demo_name: str):
    """
    Decorator to secure any demo
    
    Usage:
    @secure_demo
    def my_demo():
        # demo code
    """
    def wrapper(*args, **kwargs):
        security = SecuredDemoWrapper(demo_name)
        
        # Run the demo
        result = demo_function(*args, **kwargs)
        
        # Add security footer
        security.render_security_footer()
        
        return result
    
    return wrapper

# Quick integration for existing demos
def init_demo_security(demo_name: str) -> SecuredDemoWrapper:
    """
    Initialize security for a demo
    
    Usage at top of your demo:
        from secure_demos import init_demo_security
        security = init_demo_security("ML Integration Pitch")
    
    Then throughout demo:
        security.log_access("section_name")
    """
    return SecuredDemoWrapper(demo_name)



