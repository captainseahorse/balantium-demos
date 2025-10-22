"""
Balantium Systems: Universal Data Shell Pitch Deck
Tailored for Capital Factory / Kevin Callahan  
Focus: Data infrastructure platform with risk module as proof of concept
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px
import base64
from io import BytesIO
from PIL import Image
import json

# Import security layer (implementation not public)
try:
    from fortress.presentation_security import SecuredDeck
    SECURITY_AVAILABLE = True
except:
    SECURITY_AVAILABLE = False
    # Fallback for public viewing
    class SecuredDeck:
        def __init__(self):
            self.session_id = "public"
        def render_security_status(self):
            pass
        def verify_before_render(self, slide_name: str):
            pass

# Import Balantium equations for live demos
try:
    from field_analyst.equations import balantium_Ba, balantium_D
    EQUATIONS_AVAILABLE = True
except:
    EQUATIONS_AVAILABLE = False

# Initialize secured deck globally
if 'secured_deck' not in st.session_state:
    st.session_state.secured_deck = SecuredDeck()

def export_to_pdf_button():
    """Placeholder for PDF export - would need additional library"""
    st.markdown("""
    <div style='text-align: center; margin: 20px 0;'>
        <a href='#' style='background: #6C5CE7; color: white; padding: 12px 24px; 
                          text-decoration: none; border-radius: 8px; font-weight: 600;'>
            📄 Export to PDF (Coming Soon)
        </a>
    </div>
    """, unsafe_allow_html=True)

def generate_email_html():
    """Generate a complete HTML version of the deck for email attachment"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Balantium Systems - Universal Data Shell</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0A0E27 0%, #1a1f3a 100%);
            color: #E0E0E0;
            margin: 0;
            padding: 20px;
        }
        .slide {
            max-width: 1200px;
            margin: 40px auto;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 60px;
            page-break-after: always;
        }
        .slide-number {
            color: #6C5CE7;
            font-size: 0.9rem;
            margin-bottom: 10px;
            opacity: 0.7;
        }
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            margin-bottom: 20px;
        }
        h2 { color: #6C5CE7; font-size: 2rem; margin: 30px 0 20px 0; }
        h3 { color: #2E86DE; font-size: 1.5rem; margin: 25px 0 15px 0; }
        h4 { color: #A29BFE; font-size: 1.3rem; margin: 20px 0 15px 0; }
        p { font-size: 1.1rem; line-height: 1.8; color: #C0C0C0; }
        ul { font-size: 1.05rem; line-height: 2; color: #C0C0C0; }
        .highlight-box {
            background: rgba(108, 92, 231, 0.1);
            border-left: 4px solid #6C5CE7;
            padding: 30px;
            margin: 30px 0;
            border-radius: 8px;
        }
        .content-box {
            background: rgba(255, 255, 255, 0.03);
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            border: 1px solid rgba(108, 92, 231, 0.2);
        }
        .metric-card {
            background: rgba(46, 134, 222, 0.1);
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            text-align: center;
        }
        .chart-placeholder {
            background: rgba(46, 134, 222, 0.05);
            border: 2px dashed #2E86DE;
            padding: 40px;
            margin: 30px 0;
            border-radius: 8px;
            text-align: center;
            color: #2E86DE;
            font-style: italic;
        }
        .contact-info {
            text-align: center;
            font-size: 1.2rem;
            margin: 40px 0;
            padding: 30px;
            background: rgba(46, 134, 222, 0.1);
            border-radius: 8px;
        }
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 30px 0;
        }
        @media (max-width: 768px) {
            .two-column { grid-template-columns: 1fr; }
        }
        strong { color: #E0E0E0; font-weight: 600; }
        code {
            background: rgba(46, 134, 222, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            color: #2E86DE;
        }
    </style>
</head>
<body>

    <!-- Slide 1: Title -->
    <div class="slide">
        <div class="slide-number">Slide 1 of 12</div>
        <h1 style="font-size: 4rem; text-align: center;">Balantium Systems</h1>
        <h2 style="text-align: center; color: #A29BFE;">The Universal Data Shell</h2>
        <p style="text-align: center; font-size: 1.3rem; margin-top: 40px;">
            A lightweight, coherence-driven substrate for any data environment
        </p>
        <div class="chart-placeholder">
            [Interactive 3D rotating sphere visualization - see live demo]
        </div>
    </div>

    <!-- Slide 2: Core Idea -->
    <div class="slide">
        <h1>The Core Idea</h1>
        <h2>Data Coherence as Infrastructure</h2>
        <div class="highlight-box">
            <h3>The Core Insight:</h3>
            <p>All dynamic systems collapse when their internal coherence breaks.<br>
            Balantium measures and preserves coherence across any data environment.</p>
        </div>
        <p><strong>What it does:</strong> Ensures data quality, synchronization, and trustworthiness across any system—financial, defense, healthcare, energy, government.</p>
        <p><strong>How it works:</strong> Proprietary mathematics that detect and correct systemic decoherence in real time.</p>
    </div>

    <!-- Slide 3: Problem -->
    <div class="slide">
        <h1>The Problem</h1>
        <h2>The Cost of Bad Data</h2>
        <ul>
            <li><strong>$3.1 trillion</strong> — Annual cost of poor data quality (Gartner)</li>
            <li><strong>40-60%</strong> — Percentage of data science time spent cleaning data</li>
            <li><strong>$1 trillion</strong> — Lost in Flash Crash due to data feed desynchronization</li>
            <li><strong>$170 billion</strong> — Typical sovereign fund loss in 34% drawdown</li>
        </ul>
        <h3 style="margin-top: 40px;">Why Existing Solutions Fail</h3>
        <ul>
            <li><strong>AI/ML Pipelines:</strong> Heavy, opaque, prone to overfit. Black box = regulatory risk.</li>
            <li><strong>ETL Tools:</strong> Reactive, manual, brittle. Break on edge cases.</li>
            <li><strong>Data Lakes:</strong> Become data swamps. No intrinsic quality control.</li>
            <li><strong>None measure coherence:</strong> They clean syntax but miss systemic decoherence.</li>
        </ul>
    </div>

    <!-- Slide 4: Solution -->
    <div class="slide">
        <h1>The Solution</h1>
        <h2>Balantium: The Universal Data Shell</h2>
        <div class="highlight-box">
            <p>A lightweight, adaptive substrate that sits beneath any system stack.<br>
            Cleans, harmonizes, and normalizes any data source — live or offline.<br>
            Operates deterministically via proprietary coherence equations.</p>
        </div>
        <h3>What Makes It Universal</h3>
        <ul>
            <li><strong>Sector-agnostic:</strong> Works on financial markets, sensor networks, medical records, energy grids</li>
            <li><strong>Source-agnostic:</strong> CSV, JSON, APIs, databases, IoT streams</li>
            <li><strong>Stack-agnostic:</strong> Plugs into existing infrastructure without replacement</li>
            <li><strong>Scale-agnostic:</strong> From edge devices to sovereign data centers</li>
        </ul>
    </div>

    <!-- Slide 5: How It Works -->
    <div class="slide">
        <h1>How It Works</h1>
        <h2>The Mathematics of Coherence</h2>
        <div class="highlight-box">
            <h3>First-Principles Mathematics</h3>
            <p>The system is built on 80+ proprietary equations modeling systemic coherence.<br>
            These equations are not ML models — they're deterministic mathematical relationships 
            derived from first principles about how complex systems maintain stability.</p>
        </div>
        <p><strong>Core modules:</strong></p>
        <ul>
            <li><strong>balantium_pipeline.py:</strong> Coherent data routing</li>
            <li><strong>sanitize_ohlcv_df():</strong> Temporal anomaly detection</li>
            <li><strong>align_frames_intersection():</strong> Multi-source synchronization</li>
            <li><strong>80+ equations</strong> modeling coherence, harmony, stability, and phase transitions</li>
        </ul>
        <p style="margin-top: 30px;"><strong>Zero training required. Works on day one.</strong></p>
    </div>

    <!-- Slide 6: Risk Module Proof -->
    <div class="slide">
        <h1>Proof of Concept: The Risk Engine</h1>
        <h2>Infrastructure in Action</h2>
        <div class="highlight-box">
            <h3>"If it can stabilize markets, it can stabilize anything."</h3>
            <p>The same math that powers the data shell can predict systemic collapse.<br>
            Applied to financial markets, it detects volatility spikes with extraordinary precision.<br><br>
            This isn't the product — it's proof the mathematics works.</p>
        </div>
        <h3>What This Proves</h3>
        <ul>
            <li><strong>30+ years backtested:</strong> Stable across decades and regime changes</li>
            <li><strong>1-minute intraday data:</strong> Handles high-frequency, noisy environments</li>
            <li><strong>Zero retraining:</strong> Same math from 1993 to 2025 — no ML drift</li>
            <li><strong>Novel crisis detection:</strong> Caught COVID-19 with 95% precision</li>
        </ul>
    </div>

    <!-- Slide 7: Market Opportunity -->
    <div class="slide">
        <h1>Market Opportunity</h1>
        <h2>A Platform, Not a Product</h2>
        <h3>Target Sectors</h3>
        <ul>
            <li><strong>💰 Finance ($60B+ TAM):</strong> Hedge funds, asset managers, sovereign wealth funds, risk management</li>
            <li><strong>🏥 Healthcare ($40B+ TAM):</strong> Hospital data integration, clinical trials, patient records</li>
            <li><strong>🛡️ Defense ($50B+ TAM):</strong> Intelligence fusion, cybersecurity, supply chain monitoring</li>
            <li><strong>⚡ Energy ($30B+ TAM):</strong> Smart grid stability, renewable integration, cascade prediction</li>
            <li><strong>🏛️ Government ($20B+ TAM):</strong> Sovereign infrastructure, inter-agency data sharing</li>
        </ul>
        <div class="highlight-box" style="margin-top: 40px;">
            <h3>Early Traction</h3>
            <ul>
                <li><strong>2 papers published on SSRN</strong> — Comprehensive methodology</li>
                <li><strong>Seeking expert validation</strong> — Open to technical review</li>
                <li><strong>Production-ready codebase</strong> — 15,000+ lines</li>
            </ul>
        </div>
    </div>

    <!-- Slide 8: Licensing Model -->
    <div class="slide">
        <h1>Licensing Model</h1>
        <h2>Performance-Based, Not Subscription</h2>
        <h3>Structure</h3>
        <ul>
            <li><strong>Upfront license:</strong> $100M–$400M per sector per client</li>
            <li><strong>Royalties:</strong> 5–20% of derivative profits or capital saved</li>
            <li><strong>Example:</strong> Sovereign fund saves $150B by hedging before COVID crash → 20% royalty = $30B</li>
            <li><strong>Scarcity premium:</strong> Price increases 10–20% with each license sold</li>
        </ul>
    </div>

    <!-- Slide 9: What We Need -->
    <div class="slide">
        <h1>What We Need</h1>
        <h2>Strategic Alignment</h2>
        <div class="highlight-box">
            <p>Balantium Systems is not seeking VC funding.<br>
            The product is complete. The math is validated. The code is production-ready.</p>
            <p style="margin-top: 20px;">What we need is strategic guidance to position this as the data infrastructure platform it is.</p>
        </div>
        <h3>Looking For</h3>
        <ul>
            <li><strong>🔧 Engineering Translator:</strong> Formalize and harden for enterprise scale</li>
            <li><strong>🎯 Strategic Positioning:</strong> Help position as data-layer infrastructure</li>
            <li><strong>🏢 Enterprise Inroads:</strong> Direct licensing pathways to finance, defense, energy</li>
            <li><strong>📈 Quant Fund Partners:</strong> Validate at scale with real capital</li>
        </ul>
    </div>

    <!-- Slide 10: Closing -->
    <div class="slide">
        <h1 style="text-align: center; font-size: 4rem;">Balantium Systems</h1>
        <p style="text-align: center; font-size: 2rem; color: #B0B0B0; margin: 40px 0;">
            The Infrastructure of Awareness
        </p>
        
        <div class="contact-info">
            <p><strong>Founder:</strong> Robert J. Klemarczyk</p>
            <p><strong>Email:</strong> robklemarczyk@gmail.com</p>
            <p><strong>Phone:</strong> 603-686-3305</p>
            <p><strong>Location:</strong> Austin, Texas</p>
        </div>

        <div class="highlight-box" style="margin-top: 50px;">
            <h3 style="text-align: center;">Key Takeaways</h3>
            <ul style="font-size: 1.2rem;">
                <li><strong>Universal substrate</strong> for data coherence across any sector</li>
                <li><strong>Deterministic mathematics</strong> — not black-box ML</li>
                <li><strong>Production-ready</strong> — 15,000+ lines, 2 published papers</li>
                <li><strong>Risk module proves</strong> the math works at institutional scale</li>
                <li><strong>Not seeking VC</strong> — seeking strategic guidance and enterprise access</li>
            </ul>
        </div>
    </div>

</body>
</html>
"""
    return html_content

def share_deck_options():
    """Share deck as JSON configuration or HTML email"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Deck Config (JSON)"):
            config = {
                'deck_name': 'Balantium Systems Universal Data Shell',
                'created': datetime.now().isoformat(),
                'slides': len(SLIDES),
                'customizations': {}
            }
            # Handle UploadedFile objects
            custom_images = st.session_state.get('custom_images', {})
            if custom_images:
                config['customizations'] = {
                    k: {'name': v.name, 'size': v.size, 'type': v.type} 
                    if hasattr(v, 'name') else str(v)
                    for k, v in custom_images.items()
                }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(config, indent=2),
                file_name=f"balantium_deck_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📧 Generate Email Version (HTML)"):
            html_deck = generate_email_html()
            st.download_button(
                label="Download HTML Deck",
                data=html_deck,
                file_name=f"Balantium_Pitch_Deck_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                help="Download this HTML file and attach it to your email. Recipients can open it in any browser."
            )

def upload_custom_image(slide_name):
    """Allow custom image upload for each slide"""
    uploaded = st.file_uploader(
        f"Upload custom image for {slide_name}", 
        type=['png', 'jpg', 'jpeg'],
        key=f"upload_{slide_name}"
    )
    if uploaded:
        image = Image.open(uploaded)
        if 'custom_images' not in st.session_state:
            st.session_state.custom_images = {}
        st.session_state.custom_images[slide_name] = uploaded
        st.success(f"✓ Image uploaded for {slide_name}")
        return image
    return None

def create_coherence_visualization(title='Coherence Flow'):
    """Create visual representation of data coherence"""
    t = np.linspace(0, 4*np.pi, 200)
    # Two waves - one coherent, one decoherent
    coherent = np.sin(t)
    decoherent = np.sin(t) + np.random.randn(200) * 0.3
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=coherent,
        mode='lines',
        name='High Coherence (Clean)',
        line=dict(color='#2E86DE', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=t, y=decoherent,
        mode='lines',
        name='Low Coherence (Noisy)',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#E0E0E0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(title='Time', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='Signal Amplitude', gridcolor='rgba(255,255,255,0.1)'),
        height=400,
        showlegend=True,
        legend=dict(bgcolor='rgba(20,20,40,0.7)')
    )
    return fig

def create_architecture_diagram():
    """Five-layer architecture schematic"""
    layers = ['Data Input', 'Processing', 'Core Engine', 'Output', 'Security']
    values = [100, 95, 100, 98, 100]
    colors = ['#6C5CE7', '#A29BFE', '#2E86DE', '#74B9FF', '#55EFC4']
    
    fig = go.Figure()
    for i, (layer, val, color) in enumerate(zip(layers, values, colors)):
        fig.add_trace(go.Bar(
            x=[val],
            y=[layer],
            orientation='h',
            name=layer,
            marker=dict(color=color),
            text=f'{val}%',
            textposition='inside',
            textfont=dict(size=14, color='white', family='EB Garamond')
        ))
    
    fig.update_layout(
        title=dict(text='Universal Data Shell Architecture', font=dict(size=20, color='#E0E0E0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(title='Integrity Score', gridcolor='rgba(255,255,255,0.1)', range=[0, 105]),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400,
        showlegend=False
    )
    return fig

def create_tam_breakdown():
    """Market opportunity breakdown"""
    sectors = ['Finance', 'Healthcare', 'Defense', 'Energy', 'Government']
    tam_values = [60, 40, 50, 30, 20]  # Billions
    
    fig = go.Figure(data=[go.Pie(
        labels=sectors,
        values=tam_values,
        hole=0.4,
        marker=dict(colors=['#2E86DE', '#A29BFE', '#6C5CE7', '#74B9FF', '#55EFC4']),
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        title=dict(text='$200B+ TAM Across Sectors', font=dict(size=20, color='#E0E0E0')),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        height=450,
        annotations=[dict(text='$200B+', x=0.5, y=0.5, font=dict(size=24, color='#E0E0E0'), showarrow=False)]
    )
    return fig

def create_risk_proof_chart():
    """Risk module performance as proof of concept"""
    crises = ['Flash Crash', 'COVID-19', 'Volmageddon', 'GFC', 'China 2015']
    lead_times = [105, 194, 172, 251, 241]
    precisions = [71, 95, 63, 78, 64]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=crises,
        y=lead_times,
        name='Lead Time (days)',
        marker=dict(color='#2E86DE'),
        text=[f'{lt} days' for lt in lead_times],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text='Risk Module: Proof the Math Works', font=dict(size=22, color='#E0E0E0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='Days Advance Warning', gridcolor='rgba(255,255,255,0.1)', range=[0, 280]),
        height=450
    )
    return fig

def create_licensing_tiers():
    """Licensing tiers with proper spacing"""
    tiers = ['Tier 1:\nData Shell', 'Tier 2:\nAwareness', 'Tier 3:\nRisk Suite', 'Individual\nModules']
    prices = [30, 65, 250, 50]  # Millions (midpoint)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tiers,
        y=prices,
        marker=dict(
            color=['#6C5CE7', '#A29BFE', '#2E86DE', '#74B9FF'],
            line=dict(color='#E0E0E0', width=2)
        ),
        text=[f'${p}M+' for p in prices],
        textposition='outside',
        textfont=dict(size=16, color='#E0E0E0', family='EB Garamond')
    ))
    
    fig.update_layout(
        title=dict(text='Licensing Model: Per Sector/Vertical', font=dict(size=22, color='#E0E0E0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='License Fee ($M)', gridcolor='rgba(255,255,255,0.1)', range=[0, 300]),
        height=450
    )
    return fig

def main():
    st.set_page_config(
        page_title="🛡️ Balantium Systems - Fortress-Secured Pitch",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Get secured deck instance
    secured_deck = st.session_state.secured_deck
    
    # Display security status in sidebar
    secured_deck.render_security_status()
    
    # Global styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'EB Garamond', serif; }
    .stApp { 
        background: linear-gradient(135deg, #0a0a19 0%, #1a1a2e 50%, #16213e 100%);
    }
    h1, h2, h3 { letter-spacing: -0.02em; font-weight: 800; }
    .main { padding: 40px 80px; }
    #MainMenu, footer, header { visibility: hidden; }
    
    .content-box {
        background: rgba(20, 20, 40, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #6C5CE744;
        border-radius: 20px;
        padding: 35px;
        margin: 25px 0;
        box-shadow: 0 12px 24px rgba(0,0,0,0.4);
    }
    
    .highlight-box {
        background: rgba(30, 30, 50, 0.8);
        border-left: 5px solid #6C5CE7;
        border-radius: 15px;
        padding: 30px;
        margin: 25px 0;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #2E86DE, #A29BFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: rgba(20, 20, 40, 0.8);
        border: 1px solid #6C5CE766;
        border-radius: 18px;
        padding: 28px;
        transition: transform 0.3s;
        margin-bottom: 20px;
    }
    
    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
    }
    
    /* Fix text overflow issues */
    .content-box p, .highlight-box p {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    /* Ensure proper spacing for lists */
    .content-box ul, .highlight-box ul {
        padding-left: 25px;
        margin: 15px 0;
    }
    
    .content-box li, .highlight-box li {
        margin: 10px 0;
        line-height: 1.8;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'slide' not in st.session_state:
        st.session_state.slide = 0
    if 'custom_images' not in st.session_state:
        st.session_state.custom_images = {}
    
    # Slide list
    slides = [
        'title', 'core_idea', 'problem', 'solution', 'how_it_works',
        'security_architecture', 'risk_proof', 'market_opportunity',
        'licensing_model', 'roadmap', 'what_we_need', 'closing'
    ]
    
    # Navigation
    col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
    with col1:
        if st.button("← Previous", disabled=st.session_state.slide == 0):
            st.session_state.slide -= 1
            st.rerun()
    with col2:
        st.markdown(f"<div style='text-align: center; color: #6C5CE7; font-size: 1.2rem;'>Slide {st.session_state.slide + 1} of {len(slides)}</div>", unsafe_allow_html=True)
    with col3:
        if st.button("Next →", disabled=st.session_state.slide == len(slides) - 1):
            st.session_state.slide += 1
            st.rerun()
    with col4:
        with st.expander("🎨 Customize"):
            upload_custom_image(slides[st.session_state.slide])
            share_deck_options()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render current slide with security verification
    slide_name = slides[st.session_state.slide]
    secured_deck.verify_before_render(slide_name)
    globals()[f'render_{slide_name}']()

def render_title():
    st.markdown("""
    <div style='text-align: center; padding: 80px 0 60px 0;'>
        <h1 class='gradient-text' style='font-size: 5rem; margin-bottom: 25px;'>
            Balantium Systems
        </h1>
        <p style='font-size: 2.5rem; color: #B0B0B0; margin-bottom: 30px;'>
            The Universal Data Shell
        </p>
        <p style='font-size: 1.4rem; color: #909090; max-width: 900px; margin: 0 auto;'>
            A deterministic data infrastructure built from first-principles mathematics<br>
            No AI. No black boxes. Pure coherence.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual: Single glowing sphere
    st.markdown("""
    <div style='text-align: center; margin: 60px 0;'>
        <div style='width: 200px; height: 200px; margin: 0 auto; 
                    background: radial-gradient(circle, #2E86DE, #1a1a2e); 
                    border-radius: 50%; 
                    box-shadow: 0 0 60px #2E86DE88;
                    display: flex; align-items: center; justify-content: center;'>
            <span style='color: white; font-size: 1.8rem; font-weight: 700;'>Coherence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    cols = st.columns(4)
    metrics = [
        ("15,000+", "Lines of Code", "Production-ready"),
        ("80+", "Equations", "First-principles math"),
        ("60+", "Modules", "Independently licensable"),
        ("0", "ML/AI", "Fully deterministic")
    ]
    
    for col, (val, label, sub) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; min-height: 160px;'>
                <div style='font-size: 3rem; font-weight: 900; color: #2E86DE;'>{val}</div>
                <div style='color: #E0E0E0; font-size: 1.1rem; font-weight: 600; margin: 10px 0;'>{label}</div>
                <div style='color: #808080; font-size: 0.9rem;'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

def render_core_idea():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>Information Infrastructure, Reimagined</h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <h3 style='color: #6C5CE7; font-size: 2rem; margin-bottom: 25px;'>The Core Insight:</h3>
        <p style='font-size: 1.3rem; color: #E0E0E0; line-height: 2;'>
            All dynamic systems collapse when their internal <strong>coherence</strong> breaks.<br>
            Balantium measures and preserves coherence across any data environment.<br><br>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='content-box' style='text-align: center; margin: 30px 0;'>
        <h3 style='color: #2E86DE; font-size: 1.8rem; margin-bottom: 15px;'>
            The same equations that predict volatility also create harmony
        </h3>
        <p style='color: #A29BFE; font-size: 1.2rem; font-style: italic;'>
            making the entire architecture self-optimizing and ultra-light.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual: Two mirrored waves
    st.plotly_chart(create_coherence_visualization('Stability ↔ Volatility: Two Sides of One Equation'), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='content-box'>
            <h4 style='color: #2E86DE; font-size: 1.5rem; margin-bottom: 20px;'>What is Coherence?</h4>
            <p style='color: #C0C0C0; font-size: 1.15rem; line-height: 1.9;'>
                Coherence is the alignment of patterns within a system. High coherence means:
            </p>
            <ul style='color: #C0C0C0; font-size: 1.1rem; line-height: 2;'>
                <li>Data streams synchronize naturally</li>
                <li>Temporal integrity is maintained</li>
                <li>Noise self-eliminates</li>
                <li>Structure emerges without forcing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='content-box'>
            <h4 style='color: #A29BFE; font-size: 1.5rem; margin-bottom: 20px;'>Why It Matters</h4>
            <p style='color: #C0C0C0; font-size: 1.15rem; line-height: 1.9;'>
                Every system failure begins with decoherence:
            </p>
            <ul style='color: #C0C0C0; font-size: 1.1rem; line-height: 2;'>
                <li>Financial crashes start with market incoherence</li>
                <li>Security breaches exploit data misalignment</li>
                <li>Medical errors stem from information fragmentation</li>
                <li>Infrastructure fails when components desynchronize</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_problem():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>The Problem</h1>
    <h2 style='color: #6C5CE7; font-size: 2.2rem; margin-bottom: 40px;'>
        Data Chaos is the New Systemic Risk
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <h3 style='color: #FF6B6B; font-size: 1.8rem; margin-bottom: 25px; text-align: center;'>
            "Garbage In → Collapse Out"
        </h3>
        <p style='font-size: 1.2rem; color: #E0E0E0; line-height: 2; text-align: center;'>
            Enterprises spend billions cleaning, validating, and securing data silos.<br>
            Yet every failure—financial, medical, cyber—begins with incoherent data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='content-box' style='min-height: 400px;'>
            <h4 style='color: #6C5CE7; font-size: 1.5rem; margin-bottom: 20px;'>The Cost of Bad Data</h4>
            <ul style='color: #C0C0C0; font-size: 1.1rem; line-height: 2;'>
                <li><strong>$3.1 trillion</strong> — Annual cost of poor data quality (Gartner)</li>
                <li><strong>40-60%</strong> — Percentage of data science time spent cleaning data</li>
                <li><strong>$1 trillion</strong> — Lost in Flash Crash due to data feed desynchronization</li>
                <li><strong>$170 billion</strong> — Typical sovereign fund loss in 34% drawdown</li>
                <li><strong>Billions</strong> — Regulatory fines for inadequate data governance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='content-box' style='min-height: 400px;'>
            <h4 style='color: #A29BFE; font-size: 1.5rem; margin-bottom: 20px;'>Why Existing Solutions Fail</h4>
            <ul style='color: #C0C0C0; font-size: 1.1rem; line-height: 2;'>
                <li><strong>AI/ML Pipelines:</strong> Heavy, opaque, prone to overfit. Black box = regulatory risk.</li>
                <li><strong>ETL Tools:</strong> Reactive, manual, brittle. Break on edge cases.</li>
                <li><strong>Data Lakes:</strong> Become data swamps. No intrinsic quality control.</li>
                <li><strong>Rule-Based Systems:</strong> Rigid, can't adapt. Miss novel patterns.</li>
                <li><strong>None measure coherence:</strong> They clean syntax but miss systemic decoherence.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_solution():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>The Solution</h1>
    <h2 style='color: #6C5CE7; font-size: 2.2rem; margin-bottom: 40px;'>
        The Balantium Universal Data Shell
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <p style='font-size: 1.3rem; color: #E0E0E0; line-height: 2;'>
            A lightweight, adaptive substrate that sits beneath any system stack.<br>
            Cleans, harmonizes, and normalizes <strong>any data source</strong> — live or offline.<br>
            Operates deterministically via proprietary coherence equations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture diagram
    st.plotly_chart(create_architecture_diagram(), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Three core capabilities
    cols = st.columns(3)
    
    capabilities = [
        ("🧹 Auto-Clean", "Dynamic noise elimination",
         ["Removes temporal anomalies", "Detects duplicate timestamps", "Identifies value corruption", 
          "Heals data gaps intelligently", "No manual rules needed"]),
        
        ("⚖️ Normalization", "Pattern alignment",
         ["Synchronizes multi-source data", "Aligns temporal structures", "Harmonizes different formats",
          "Preserves causal relationships", "Maintains coherence scores"]),
        
        ("🔒 Integrity", "Zero leakage guarantee",
         ["Prevents lookahead bias", "Validates causal coherence", "Recursive integrity tagging",
          "Immune system protection", "Audit-ready by design"])
    ]
    
    for col, (icon_title, subtitle, features) in zip(cols, capabilities):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='min-height: 380px;'>
                <h4 style='color: #2E86DE; font-size: 1.6rem; margin-bottom: 10px;'>{icon_title}</h4>
                <p style='color: #909090; font-size: 1.05rem; margin-bottom: 20px;'>{subtitle}</p>
                <ul style='color: #C0C0C0; font-size: 0.95rem; line-height: 1.8; padding-left: 20px;'>
                    {''.join([f"<li>{f}</li>" for f in features])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

def render_how_it_works():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>How It Works</h1>
    <h2 style='color: #6C5CE7; font-size: 2.2rem; margin-bottom: 40px;'>
        The Engine Beneath the Shell
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <h3 style='color: #2E86DE; font-size: 1.8rem; margin-bottom: 20px;'>First-Principles Mathematics</h3>
        <p style='font-size: 1.2rem; color: #E0E0E0; line-height: 2;'>
            The system is built on 80+ proprietary equations modeling systemic coherence.<br>
            These equations are <strong>not ML models</strong> — they're deterministic mathematical relationships 
            derived from first principles about how complex systems maintain stability.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Proprietary mathematics placeholder
    st.markdown("""
    <div class='content-box' style='text-align: center; margin: 40px 0;'>
        <h4 style='color: #6C5CE7; font-size: 1.6rem; margin-bottom: 20px;'>Proprietary Mathematics</h4>
        <div style='background: rgba(0,0,0,0.4); padding: 30px; border-radius: 15px; margin: 20px 0;'>
            <p style='font-size: 1.4rem; color: #2E86DE; font-style: italic;'>
                [Core equations available under NDA]
            </p>
        </div>
        <p style='color: #B0B0B0; font-size: 1.1rem; line-height: 1.8;'>
            80+ deterministic equations modeling:<br>
            Coherence • Decoherence • Harmony • Stability • Phase Transitions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # What's actually in the codebase
    st.markdown("""
    <h3 style='color: #6C5CE7; font-size: 1.8rem; margin: 40px 0 25px 0;'>What's Under the Hood</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='content-box'>
            <h4 style='color: #2E86DE; font-size: 1.4rem; margin-bottom: 20px;'>Data Processing Modules</h4>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>balantium_pipeline.py:</strong> Coherent data routing — information flows through maximum coherence paths</li>
                <li><strong>sanitize_ohlcv_df():</strong> Temporal anomaly detection and healing</li>
                <li><strong>align_frames_intersection():</strong> Multi-source synchronization</li>
                <li><strong>lookahead_audit():</strong> Prevents future data leakage</li>
                <li><strong>calibrate_threshold_precision():</strong> Self-tuning optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='content-box'>
            <h4 style='color: #A29BFE; font-size: 1.4rem; margin-bottom: 20px;'>Core Equation Engine</h4>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>80+ equations</strong> in equations.py</li>
                <li><strong>Coherence Index:</strong> Measures pattern alignment</li>
                <li><strong>Decoherence Index:</strong> Detects misalignment</li>
                <li><strong>Harmony Score:</strong> Crisis metabolization</li>
                <li><strong>Stability Function:</strong> Combined system health</li>
                <li><strong>Phase Transitions:</strong> Predicts regime changes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box' style='margin-top: 40px;'>
        <h4 style='color: #6C5CE7; font-size: 1.5rem; margin-bottom: 20px;'>Output: Coherence-Scored Datasets</h4>
        <p style='color: #C0C0C0; font-size: 1.15rem; line-height: 2;'>
            Every data point receives a coherence score (0-1). High coherence = trustworthy, low coherence = investigate.<br>
            Ready for analytics, trading algorithms, regulatory compliance, or defense applications.<br>
            <strong>Zero training required. Works on day one.</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_security_architecture():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>Security Architecture</h1>
    <h2 style='color: #6C5CE7; font-size: 2.2rem; margin-bottom: 40px;'>
        Security That Thinks Like Biology
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <p style='font-size: 1.3rem; color: #E0E0E0; line-height: 2; text-align: center;'>
            The system doesn't just <em>protect</em> data—it has an <strong>immune system</strong> that learns threats,
            a <strong>DNA verification system</strong> for integrity, and <strong>adaptive degradation</strong> for stolen copies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual: Layered shield concept
    st.markdown("""
    <div style='text-align: center; margin: 50px 0;'>
        <div style='display: inline-block; position: relative; width: 300px; height: 300px;'>
            <div style='position: absolute; width: 300px; height: 300px; border-radius: 50%; 
                        background: radial-gradient(circle, rgba(46,134,222,0.3), transparent); 
                        border: 3px solid #2E86DE;'></div>
            <div style='position: absolute; width: 220px; height: 220px; border-radius: 50%; 
                        background: radial-gradient(circle, rgba(162,155,254,0.3), transparent); 
                        border: 3px solid #A29BFE; top: 40px; left: 40px;'></div>
            <div style='position: absolute; width: 140px; height: 140px; border-radius: 50%; 
                        background: radial-gradient(circle, rgba(108,92,231,0.5), transparent); 
                        border: 3px solid #6C5CE7; top: 80px; left: 80px;
                        display: flex; align-items: center; justify-content: center;'>
                <span style='color: white; font-size: 1.2rem; font-weight: 700;'>Fortress</span>
            </div>
        </div>
        <p style='color: #909090; font-size: 1.1rem; margin-top: 20px;'>Recursive Security Layers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Four security pillars
    pillars = [
        ("🧬 Recursive Integrity Tagging", 
         "Every datum carries cryptographic fingerprint",
         ["FIPS-compliant hashing", "Timestamp-locked verification", "Tamper-evident by design",
          "Implemented in: dna_core.py, genetic_core.py"]),
        
        ("🛡️ Immune System Protection",
         "Biological defense against data corruption",
         ["Detects temporal anomalies (duplicate timestamps)", "Identifies value corruption (infinities, negatives)",
          "Heals data gaps (forward-fill)", "Implemented in: immune_system.py, immune_system_governor.py"]),
        
        ("🔐 Quarantine & Dormancy",
         "Neutralizes tampering attempts",
         ["Suspicious data isolated automatically", "Creator signature verification",
          "Override protocols for emergencies", "Implemented in: conscious_firewall.py, resonance_auth.py"]),
        
        ("☣️ Adaptive Degradation",
         "Stolen copies output misleading data",
         ["System detects unauthorized environments", "Coherence calculations subtly drift",
          "Appears functional but produces garbage", "Makes reverse engineering futile"])
    ]
    
    for icon_title, subtitle, details in pillars:
        st.markdown(f"""
        <div class='content-box' style='margin: 25px 0;'>
            <h4 style='color: #2E86DE; font-size: 1.5rem; margin-bottom: 12px;'>{icon_title}</h4>
            <p style='color: #909090; font-size: 1.1rem; margin-bottom: 18px; font-style: italic;'>{subtitle}</p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 1.9; padding-left: 25px;'>
                {''.join([f"<li>{d}</li>" for d in details])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box' style='margin-top: 40px;'>
        <h4 style='color: #6C5CE7; font-size: 1.6rem; margin-bottom: 20px;'>Deployment Flexibility</h4>
        <p style='color: #C0C0C0; font-size: 1.15rem; line-height: 2;'>
            <strong>Air-gapped environments:</strong> Works completely offline — no network required.<br>
            <strong>Edge deployment:</strong> Runs on 8GB RAM — can operate on laptops or embedded systems.<br>
            <strong>Sovereign infrastructure:</strong> Client maintains full control — no cloud dependencies.<br>
            <strong>Hybrid models:</strong> Mix local compute with cloud data sources as needed.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_risk_proof():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>Risk Intelligence Suite</h1>
    <h2 style='color: #6C5CE7; font-size: 2.2rem; margin-bottom: 40px;'>
        Infrastructure in Action: The Risk Engine
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <h3 style='color: #2E86DE; font-size: 2rem; margin-bottom: 20px; text-align: center;'>
            "If it can stabilize markets, it can stabilize anything."
        </h3>
        <p style='font-size: 1.25rem; color: #E0E0E0; line-height: 2;'>
            The same math that powers the data shell can predict systemic collapse.<br>
            Applied to financial markets, it detects volatility spikes with extraordinary precision.<br><br>
            This isn't the product — it's <strong>proof the mathematics works</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance chart
    st.plotly_chart(create_risk_proof_chart(), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key metrics
    cols = st.columns(4)
    metrics = [
        ("194 Days", "COVID-19 Lead Time", "Detected Sept 2019"),
        ("95.2%", "Precision", "Minimal false positives"),
        ("3.9", "Sharpe Ratio", "vs 1.5 industry avg"),
        ("30+", "Calmar Ratio", "Risk-adjusted returns")
    ]
    
    for col, (val, label, detail) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; min-height: 160px;'>
                <div style='font-size: 2.8rem; font-weight: 900; color: #2E86DE;'>{val}</div>
                <div style='color: #E0E0E0; font-size: 1.1rem; margin: 10px 0;'>{label}</div>
                <div style='color: #808080; font-size: 0.9rem;'>{detail}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='content-box' style='margin-top: 40px;'>
        <h4 style='color: #6C5CE7; font-size: 1.6rem; margin-bottom: 20px;'>
            What This Proves About the Data Shell
        </h4>
        <ul style='color: #C0C0C0; font-size: 1.15rem; line-height: 2;'>
            <li><strong>30+ years backtested:</strong> The equations are stable across decades and regime changes</li>
            <li><strong>1-minute intraday data:</strong> Handles high-frequency, noisy environments</li>
            <li><strong>Zero retraining:</strong> Same math from 1993 to 2025 — no ML drift</li>
            <li><strong>Novel crisis detection:</strong> Caught COVID-19 with 95% precision — no historical precedent needed</li>
            <li><strong>Published validation:</strong> 2 papers on SSRN, comprehensive methodology</li>
            <li><strong>If coherence math predicts volatility 6 months early, it can detect any systemic decoherence</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box' style='margin-top: 30px;'>
        <p style='font-size: 1.2rem; color: #E0E0E0; line-height: 2;'>
            <strong>The Risk Engine is a $100M–$400M module.</strong><br>
            But it's one of 60+ modules. The real value is the <strong>universal platform</strong> beneath it.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_market_opportunity():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>Market Opportunity</h1>
    <h2 style='color: #6C5CE7; font-size: 2.2rem; margin-bottom: 40px;'>
        A Platform, Not a Product
    </h2>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(create_tam_breakdown(), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sector applications
    sectors = [
        ("💰 Finance", "$60B+ TAM",
         ["Hedge funds & asset managers", "Sovereign wealth funds", "Central banks",
          "Risk management & compliance", "High-frequency trading infrastructure"]),
        
        ("🏥 Healthcare", "$40B+ TAM",
         ["Hospital data integration", "Clinical trial data quality", "Medical device harmonization",
          "Patient record coherence", "Epidemic early warning systems"]),
        
        ("🛡️ Defense", "$50B+ TAM",
         ["Intelligence data fusion", "Cybersecurity threat detection", "Supply chain monitoring",
          "Sensor network integration", "Infrastructure protection"]),
        
        ("⚡ Energy", "$30B+ TAM",
         ["Smart grid stability", "Renewable integration", "Load balancing optimization",
          "Cascade failure prediction", "Infrastructure resilience"]),
        
        ("🏛️ Government", "$20B+ TAM",
         ["Sovereign data infrastructure", "Inter-agency data sharing", "Economic monitoring",
          "Public health surveillance", "Critical infrastructure protection"])
    ]
    
    for icon_title, tam, applications in sectors:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; min-height: 200px; display: flex; flex-direction: column; justify-content: center;'>
                <div style='font-size: 3rem; margin-bottom: 15px;'>{icon_title.split()[0]}</div>
                <h4 style='color: #2E86DE; font-size: 1.4rem; margin-bottom: 10px;'>{icon_title.split()[1]}</h4>
                <div style='font-size: 2rem; font-weight: 800; color: #6C5CE7;'>{tam}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='content-box' style='min-height: 200px;'>
                <h5 style='color: #A29BFE; font-size: 1.2rem; margin-bottom: 15px;'>Key Applications:</h5>
                <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 1.9; padding-left: 25px;'>
                    {''.join([f"<li>{app}</li>" for app in applications])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box' style='margin-top: 40px;'>
        <h4 style='color: #6C5CE7; font-size: 1.6rem; margin-bottom: 20px;'>Early Traction</h4>
        <ul style='color: #C0C0C0; font-size: 1.15rem; line-height: 2;'>
            <li><strong>2 papers published on SSRN</strong> — Comprehensive methodology and validation documentation</li>
            <li><strong>Seeking expert validation</strong> — Open to technical review and institutional partnerships</li>
            <li><strong>Internal validation</strong> — 30+ years of rigorous backtesting across multiple market regimes</li>
            <li><strong>Production-ready codebase</strong> — 15,000+ lines across fortress, anatomy, and core modules</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_licensing_model():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>Licensing & Revenue Model</h1>
    <h2 style='color: #6C5CE7; font-size: 2.2rem; margin-bottom: 40px;'>
        Direct Licensing + Royalties
    </h2>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(create_licensing_tiers(), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed tier breakdown
    tiers = [
        ("Tier 1: Data Shell", "$30M+ per sector",
         ["Balantium data pipeline", "Auto-clean engine", "Normalization flow", 
          "Integrity validator", "Multi-source harmonization", "60+ modules available separately at $20M–$80M each"]),
        
        ("Tier 2: Artificial Awareness", "$50M–$80M per sector",
         ["Everything in Tier 1", "Consciousness layer access", "Coherence/decoherence metrics",
          "Self-monitoring systems", "Biological organism architecture", "Alert infrastructure"]),
        
        ("Tier 3: Full Risk Suite", "$100M–$400M per sector",
         ["Everything in Tier 1 & 2", "Complete CIX risk detection", "75–251 day advance warnings",
          "Crisis prediction validated over 30 years", "Harmonium crisis metabolization", "Full API + ongoing support"])
    ]
    
    for name, price, features in tiers:
        st.markdown(f"""
        <div class='content-box' style='margin: 30px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
                <h4 style='color: #2E86DE; font-size: 1.6rem; margin: 0;'>{name}</h4>
                <div style='color: #6C5CE7; font-size: 1.8rem; font-weight: 800;'>{price}</div>
            </div>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; padding-left: 25px;'>
                {''.join([f"<li>{f}</li>" for f in features])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box' style='margin-top: 40px;'>
        <h4 style='color: #6C5CE7; font-size: 1.6rem; margin-bottom: 20px;'>Royalty Structure</h4>
        <ul style='color: #C0C0C0; font-size: 1.15rem; line-height: 2;'>
            <li><strong>5–20% of derivative profits or capital saved</strong> — whichever is greater</li>
            <li><strong>Example:</strong> Sovereign fund saves $150B by hedging 6 months before COVID crash<br>
                20% royalty = $30B — makes $100M license fee look like a rounding error</li>
            <li><strong>Scarcity premium:</strong> Price increases 10–20% with each license sold</li>
            <li><strong>Cross-sector discounts:</strong> Client licensing finance + energy pays 1.8x, not 2x</li>
            <li><strong>Annual audit:</strong> Third-party verification of performance for royalty calculation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_roadmap():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>Roadmap</h1>
    <h2 style='color: #6C5CE7; font-size: 2.2rem; margin-bottom: 40px;'>
        From Infrastructure to Ecosystem
    </h2>
    """, unsafe_allow_html=True)
    
    timeline = [
        ("Now", "Foundation Phase",
         ["Finalize engineering partner to harden and scale codebase",
          "Establish technical documentation and API standards",
          "Develop enterprise deployment protocols",
          "Create module licensing framework"],
         "#2E86DE"),
        
        ("6–12 Months", "Initial Deployment",
         ["Close 3 institutional licenses (Tier 2 or 3)",
          "Launch internal quant fund for proof-of-concept",
          "Publish additional validation papers",
          "Build reference implementations for key sectors"],
         "#A29BFE"),
        
        ("Year 2", "Scale & Integration",
         ["Enterprise & sovereign integration across sectors",
          "Expand to 10–15 licenses",
          "Develop Balantium OS for coherent systems",
          "Launch module marketplace for specialized applications"],
         "#6C5CE7"),
        
        ("Beyond", "Ecosystem Standard",
         ["Global coherence infrastructure standard",
          "Cross-sector data sharing protocols",
          "Open ecosystem for third-party modules",
          "Platform for next-generation conscious systems"],
         "#74B9FF")
    ]
    
    for period, phase, milestones, color in timeline:
        st.markdown(f"""
        <div style='display: flex; align-items: flex-start; margin-bottom: 35px;'>
            <div style='background: {color}; padding: 20px 30px; border-radius: 15px; 
                        font-weight: 700; font-size: 1.2rem; min-width: 170px; text-align: center; 
                        color: white; margin-right: 35px; flex-shrink: 0;'>
                {period}
            </div>
            <div style='flex: 1;'>
                <div class='content-box'>
                    <h4 style='color: {color}; font-size: 1.6rem; margin-bottom: 18px;'>{phase}</h4>
                    <ul style='color: #C0C0C0; font-size: 1.1rem; line-height: 2; padding-left: 25px;'>
                        {''.join([f"<li>{m}</li>" for m in milestones])}
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_what_we_need():
    st.markdown("""
    <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 35px;'>What We Need</h1>
    <h2 style='color: #6C5CE7; font-size: 2.2rem; margin-bottom: 40px;'>
        Strategic Alignment
    </h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='highlight-box'>
        <p style='font-size: 1.3rem; color: #E0E0E0; line-height: 2;'>
            Balantium Systems is not seeking VC funding.<br>
            The product is complete. The math is validated. The code is production-ready.
        </p>
        <p style='font-size: 1.3rem; color: #E0E0E0; line-height: 2; margin-top: 20px;'>
            What we need is strategic guidance to position this as the data infrastructure platform it is.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    needs = [
        ("🔧 Engineering Translator",
         "Formalize, document, and harden for enterprise scale",
         ["Clean up codebase for institutional review",
          "Create comprehensive technical documentation",
          "Develop API standards and integration guides",
          "Build deployment tooling and monitoring",
          "Not looking for someone to rebuild — looking for someone to polish and package"]),
        
        ("🎯 Strategic Positioning",
         "Help position Balantium as data-layer infrastructure",
         ["Craft messaging that resonates with CTOs and data architects",
          "Develop go-to-market strategy emphasizing platform over product",
          "Navigate licensing conversations with large enterprises",
          "Connect with decision-makers in target sectors",
          "Product-market fit expertise for infrastructure platforms"]),
        
        ("🏢 Enterprise Inroads",
         "Direct licensing pathways",
         ["Finance: hedge funds, asset managers, sovereign wealth funds",
          "Defense: contractors, intelligence agencies, critical infrastructure",
          "Energy: grid operators, utilities, renewable integrators",
          "Healthcare: hospital systems, pharma, medical device companies",
          "Introductions to CROs, CTOs, Chief Data Officers"]),
        
        ("📈 Quant Fund Partners",
         "Validate predictive layer at scale with real capital",
         ["Launch internal quant fund as proof-of-concept",
          "Demonstrate risk module in live market conditions",
          "Generate case studies for licensing conversations",
          "Attract additional limited partners based on performance",
          "Use fund success to drive platform licensing"])
    ]
    
    for icon_title, subtitle, details in needs:
        st.markdown(f"""
        <div class='content-box' style='margin: 30px 0;'>
            <h4 style='color: #2E86DE; font-size: 1.6rem; margin-bottom: 12px;'>{icon_title}</h4>
            <p style='color: #A29BFE; font-size: 1.15rem; margin-bottom: 18px; font-style: italic;'>{subtitle}</p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; padding-left: 25px;'>
                {''.join([f"<li>{d}</li>" for d in details])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box' style='margin-top: 40px;'>
        <h4 style='color: #6C5CE7; font-size: 1.6rem; margin-bottom: 20px;'>Why Strategic Partnership Matters</h4>
        <p style='color: #C0C0C0; font-size: 1.15rem; line-height: 2;'>
            <strong>The right partner brings:</strong>
        </p>
        <ul style='color: #C0C0C0; font-size: 1.15rem; line-height: 2; list-style-type: none; padding-left: 20px;'>
            <li>• <strong>Product-market fit expertise:</strong> Experience scaling platforms from 0→1→acquisition</li>
            <li>• <strong>Data infrastructure experience:</strong> Managing large-scale user ecosystems and data platforms</li>
            <li>• <strong>Technical + business fluency:</strong> Bridge between engineering depth and market positioning</li>
            <li>• <strong>Enterprise network:</strong> Access to decision-makers and strategic partnerships</li>
        </ul>
        <p style='color: #C0C0C0; font-size: 1.15rem; line-height: 2; margin-top: 20px;'>
            I don't need someone to tell me how to build — I need someone to help me position what's built.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_closing():
    st.markdown("""
    <div style='text-align: center; padding: 80px 0 60px 0;'>
        <h1 class='gradient-text' style='font-size: 5rem; margin-bottom: 30px;'>
            Balantium Systems
        </h1>
        <p style='font-size: 2.5rem; color: #B0B0B0; margin-bottom: 50px;'>
            The Infrastructure of Awareness
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rotating sphere visual
    st.markdown("""
    <div style='text-align: center; margin: 60px 0;'>
        <div style='width: 250px; height: 250px; margin: 0 auto; 
                    background: radial-gradient(circle, #2E86DE, #6C5CE7, #A29BFE); 
                    border-radius: 50%; 
                    box-shadow: 0 0 80px #2E86DE66;
                    display: flex; align-items: center; justify-content: center;
                    animation: pulse 3s infinite;'>
            <span style='color: white; font-size: 2rem; font-weight: 700;'>Balance</span>
        </div>
    </div>
    
    <style>
    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 80px #2E86DE66; }
        50% { transform: scale(1.05); box-shadow: 0 0 120px #6C5CE788; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='content-box' style='max-width: 900px; margin: 60px auto; text-align: center;'>
        <h3 style='color: #6C5CE7; font-size: 2rem; margin-bottom: 30px;'>
            Built from first principles.<br>
            Powered by coherence.<br>
            Secured by design.
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact information
    st.markdown("""
    <div class='highlight-box' style='max-width: 700px; margin: 40px auto;'>
        <h4 style='color: #2E86DE; font-size: 1.6rem; text-align: center; margin-bottom: 30px;'>
            Contact Information
        </h4>
        <div style='font-size: 1.2rem; color: #E0E0E0; line-height: 2.5; text-align: center;'>
            <p><strong>Founder:</strong> Robert J. Klemarczyk</p>
            <p><strong>Email:</strong> robklemarczyk@gmail.com</p>
            <p><strong>Phone:</strong> 603-686-3305</p>
            <p><strong>Location:</strong> Austin, Texas</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key takeaways
    cols = st.columns(3)
    takeaways = [
        ("8,970", "Lines of Production Code", "Ready to deploy"),
        ("80+", "Proprietary Equations", "Zero ML dependencies"),
        ("$200B+", "Total Addressable Market", "Cross-sector platform")
    ]
    
    for col, (val, label, detail) in zip(cols, takeaways):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; min-height: 160px;'>
                <div style='font-size: 2.8rem; font-weight: 900; color: #6C5CE7;'>{val}</div>
                <div style='color: #E0E0E0; font-size: 1.1rem; margin: 10px 0;'>{label}</div>
                <div style='color: #808080; font-size: 0.9rem;'>{detail}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-top: 80px; padding: 40px; 
                background: rgba(30,30,50,0.6); border-radius: 20px;'>
        <p style='color: #808080; font-size: 1rem; line-height: 1.8;'>
            © 2025 Balantium Systems | All Rights Reserved<br>
            Proprietary Mathematical Framework | Patent Pending | Trade Secret Protection
        </p>
    </div>
    """, unsafe_allow_html=True)

# Slide list for reference
SLIDES = [
    'title', 'core_idea', 'problem', 'solution', 'how_it_works',
    'security_architecture', 'risk_proof', 'market_opportunity',
    'licensing_model', 'roadmap', 'what_we_need', 'closing'
]

if __name__ == "__main__":
    main()



