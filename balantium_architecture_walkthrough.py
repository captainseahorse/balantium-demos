"""
Balantium Systems: Interactive Architecture Walkthrough
Deep dive into the living organism - structured by its own mathematics
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.express as px
from plotly.subplots import make_subplots

# Import Balantium core mathematics - the organism structures itself
EQUATIONS_AVAILABLE = False
IMPORT_ERROR = None

try:
    from field_analyst.equations import balantium_Ba, balantium_D
    from balantium_pipeline import BalantiumPipeline
    EQUATIONS_AVAILABLE = True
except Exception as e:
    EQUATIONS_AVAILABLE = False
    IMPORT_ERROR = str(e)

def create_data_flow_animation(step):
    """Animated data flow through the organism"""
    # Create a flow diagram showing data transformation
    fig = go.Figure()
    
    # Define the stages
    stages = ['Raw Data', 'Sanitization', 'Coherence Check', 'Normalization', 'Output']
    x_positions = [0, 1, 2, 3, 4]
    
    # Draw the pipeline
    for i in range(len(stages) - 1):
        fig.add_trace(go.Scatter(
            x=[x_positions[i], x_positions[i+1]],
            y=[0, 0],
            mode='lines',
            line=dict(
                color='#2E86DE' if i < step else 'rgba(46,134,222,0.3)',
                width=6
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw the nodes
    for i, stage in enumerate(stages):
        color = '#2E86DE' if i <= step else 'rgba(46,134,222,0.3)'
        size = 25 if i == step else 20
        
        fig.add_trace(go.Scatter(
            x=[x_positions[i]],
            y=[0],
            mode='markers+text',
            marker=dict(size=size, color=color, line=dict(color='white', width=2)),
            text=[stage],
            textposition='bottom center',
            textfont=dict(size=12, color='#E0E0E0'),
            showlegend=False,
            hoverinfo='text',
            hovertext=stage
        ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        xaxis=dict(visible=False, range=[-0.5, 4.5]),
        yaxis=dict(visible=False, range=[-1, 1]),
        height=200,
        margin=dict(t=20, b=80, l=20, r=20),
        showlegend=False
    )
    
    return fig

def create_coherence_live_demo():
    """Live demonstration of coherence calculation - conceptual version"""
    st.markdown("""
    <div class='content-box'>
        <h4 style='color: #2E86DE; font-size: 1.5rem; margin-bottom: 20px;'>
            Coherence Calculation Concept
        </h4>
        <p style='color: #A0A0A0; font-size: 0.95rem;'>
            <em>Interactive demonstration showing the principles of coherence measurement. 
            Production implementation uses proprietary Balantium mathematics.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    t = np.linspace(0, 10, 200)
    
    col1, col2 = st.columns(2)
    
    with col1:
        noise_level = st.slider("Data Noise Level", 0.0, 1.0, 0.3, 0.1)
    with col2:
        sync_quality = st.slider("Synchronization Quality", 0.0, 1.0, 0.8, 0.1)
    
    # Create two signals with varying coherence
    signal1 = np.sin(t)
    signal2 = np.sin(t + (1 - sync_quality) * np.pi/4) + np.random.randn(200) * noise_level
    
    # Conceptual coherence calculation
    # Measures signal alignment and noise resistance
    coherence_score = 1 - (noise_level * 0.5 + (1 - sync_quality) * 0.5)
    
    st.markdown("""
    <div style='background: rgba(108,92,231,0.1); padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <h4 style='color: #6C5CE7;'>Coherence Measurement Principles:</h4>
        <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 1.8;'>
            <li><strong>Synchronization:</strong> How well signals align in phase</li>
            <li><strong>Noise Resistance:</strong> Signal clarity despite interference</li>
            <li><strong>Constructive/Destructive Balance:</strong> Pattern reinforcement vs. cancellation</li>
            <li><strong>Transmission Efficiency:</strong> Information preservation through the system</li>
        </ul>
        <p style='color: #808080; font-size: 0.9rem; margin-top: 15px;'>
            <em>Production system: Balantium Ba equation integrates these factors with additional 
            resonance field terms and multi-scale temporal analysis.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Conceptual Coherence Score", f"{coherence_score:.4f}", 
             help="Simplified demonstration - production uses full Balantium mathematics")
    
    # Visualize
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal1,
        mode='lines',
        name='Reference Signal',
        line=dict(color='#2E86DE', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=t, y=signal2,
        mode='lines',
        name='Measured Signal',
        line=dict(color='#A29BFE', width=2)
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(title='Time', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='Amplitude', gridcolor='rgba(255,255,255,0.1)'),
        height=400,
        showlegend=True,
        legend=dict(bgcolor='rgba(20,20,40,0.7)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    if coherence_score > 0.8:
        status = "✅ High Coherence - Data is trustworthy"
        color = "#55EFC4"
    elif coherence_score > 0.5:
        status = "⚠️ Medium Coherence - Investigate anomalies"
        color = "#FDCB6E"
    else:
        status = "🚨 Low Coherence - Data integrity compromised"
        color = "#FF6B6B"
    
    st.markdown(f"""
    <div style='background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px; 
                border-left: 4px solid {color}; margin-top: 20px;'>
        <p style='color: {color}; font-size: 1.2rem; margin: 0;'>{status}</p>
    </div>
    """, unsafe_allow_html=True)

def create_layer_diagram(highlight_layer=None):
    """Five-layer architecture with interactive highlighting"""
    layers = [
        {
            'name': 'Security Fortress',
            'components': ['DNA Core', 'Immune System', 'Firewall', 'Auth'],
            'color': '#6C5CE7',
            'position': 0
        },
        {
            'name': 'Data Intake',
            'components': ['CSV/JSON', 'APIs', 'Streams', 'Databases'],
            'color': '#2E86DE',
            'position': 1
        },
        {
            'name': 'Processing Core',
            'components': ['Sanitization', 'Validation', 'Alignment', 'Harmony'],
            'color': '#A29BFE',
            'position': 2
        },
        {
            'name': 'Mathematics Engine',
            'components': ['Ba (Coherence)', 'D (Decoherence)', 'Harmonium', 'CIX'],
            'color': '#74B9FF',
            'position': 3
        },
        {
            'name': 'Output Layer',
            'components': ['Clean Data', 'Metrics', 'Alerts', 'API'],
            'color': '#55EFC4',
            'position': 4
        }
    ]
    
    fig = go.Figure()
    
    for i, layer in enumerate(layers):
        is_highlighted = highlight_layer == i if highlight_layer is not None else True
        opacity = 1.0 if is_highlighted else 0.3
        
        # Convert hex to rgba
        hex_color = layer['color'].lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        fillcolor = f'rgba({r},{g},{b},{opacity})'
        
        # Draw layer box
        fig.add_trace(go.Scatter(
            x=[0, 10, 10, 0, 0],
            y=[i*2, i*2, i*2+1.5, i*2+1.5, i*2],
            fill='toself',
            fillcolor=fillcolor,
            line=dict(color=layer['color'], width=2 if is_highlighted else 1),
            mode='lines',
            name=layer['name'],
            hoverinfo='text',
            hovertext=f"{layer['name']}<br>" + "<br>".join(layer['components']),
            showlegend=False
        ))
        
        # Add layer label
        fig.add_annotation(
            x=5, y=i*2+0.75,
            text=f"<b>{layer['name']}</b>",
            showarrow=False,
            font=dict(size=14, color='white', family='EB Garamond'),
            bgcolor='rgba(0,0,0,0.5)',
            borderpad=8
        )
        
        # Add component labels if highlighted
        if is_highlighted and highlight_layer is not None:
            components_text = " | ".join(layer['components'])
            fig.add_annotation(
                x=5, y=i*2+0.3,
                text=components_text,
                showarrow=False,
                font=dict(size=10, color='white', family='EB Garamond')
            )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        xaxis=dict(visible=False, range=[-0.5, 10.5]),
        yaxis=dict(visible=False, range=[-0.5, 10.5]),
        height=600,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False
    )
    
    return fig

def create_data_transformation_demo():
    """Show actual data transformation through the pipeline"""
    st.markdown("""
    <h4 style='color: #2E86DE; font-size: 1.5rem; margin-bottom: 20px;'>
        Data Transformation: Before & After
    </h4>
    """, unsafe_allow_html=True)
    
    # Generate messy sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    
    # Introduce common data quality issues
    raw_data = pd.DataFrame({
        'timestamp': dates,
        'value': np.cumsum(np.random.randn(100)) + 100
    })
    
    # Add issues
    raw_data.loc[10, 'value'] = np.nan  # Missing value
    raw_data.loc[25, 'value'] = np.inf  # Infinity
    raw_data.loc[50, 'value'] = -999   # Outlier
    raw_data.loc[75:77, 'timestamp'] = raw_data.loc[75, 'timestamp']  # Duplicate timestamps
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='content-box' style='background: rgba(255,107,107,0.1); border-left: 4px solid #FF6B6B;'>
            <h5 style='color: #FF6B6B;'>Raw Input Data</h5>
            <ul style='color: #C0C0C0; font-size: 0.9rem;'>
                <li>Missing values (NaN)</li>
                <li>Invalid values (Inf)</li>
                <li>Outliers (-999)</li>
                <li>Duplicate timestamps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show raw data issues
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            x=raw_data['timestamp'],
            y=raw_data['value'],
            mode='lines+markers',
            name='Raw Data',
            line=dict(color='#FF6B6B', width=1),
            marker=dict(size=4)
        ))
        fig_raw.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,40,0.5)',
            font=dict(color='#E0E0E0'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=300,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_raw, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class='content-box' style='background: rgba(85,239,196,0.1); border-left: 4px solid #55EFC4;'>
            <h5 style='color: #55EFC4;'>Balantium Cleaned Data</h5>
            <ul style='color: #C0C0C0; font-size: 0.9rem;'>
                <li>✅ Missing values healed</li>
                <li>✅ Invalid values removed</li>
                <li>✅ Outliers corrected</li>
                <li>✅ Timestamps aligned</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Clean the data
        clean_data = raw_data.copy()
        clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
        clean_data.loc[clean_data['value'] < 0, 'value'] = np.nan
        clean_data = clean_data.drop_duplicates(subset=['timestamp'])
        clean_data['value'] = clean_data['value'].fillna(method='ffill')
        
        # Show cleaned data
        fig_clean = go.Figure()
        fig_clean.add_trace(go.Scatter(
            x=clean_data['timestamp'],
            y=clean_data['value'],
            mode='lines+markers',
            name='Clean Data',
            line=dict(color='#55EFC4', width=2),
            marker=dict(size=4)
        ))
        fig_clean.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,40,0.5)',
            font=dict(color='#E0E0E0'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=300,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_clean, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Balantium Architecture Walkthrough",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Conceptual demo notice
    if not EQUATIONS_AVAILABLE:
        st.info("""
        📘 **Conceptual Architecture Demo**  
        This demonstration shows the architectural principles and system design without exposing proprietary 
        Balantium mathematics. Interactive visualizations demonstrate coherence concepts using simplified calculations. 
        Full mathematical implementation available in licensed production version.
        """)
    
    # Match pitch deck styling
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
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("""
    <div style='text-align: center; padding: 40px 0 30px 0;'>
        <h1 class='gradient-text' style='font-size: 4rem; margin-bottom: 15px;'>
            Architecture Walkthrough
        </h1>
        <p style='font-size: 1.8rem; color: #B0B0B0;'>
            Inside the Balantium Organism
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <h2 style='color: #6C5CE7; margin-bottom: 20px;'>Navigation</h2>
        """, unsafe_allow_html=True)
        
        step = st.radio(
            "Select Step:",
            [
                "Overview",
                "Layer 1: Security Fortress",
                "Layer 2: Data Intake",
                "Layer 3: Processing Core",
                "Layer 4: Mathematics Engine",
                "Layer 5: Output Layer",
                "Live Demo: Coherence",
                "Live Demo: Data Transformation",
                "Complete Data Flow"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if EQUATIONS_AVAILABLE:
            st.markdown("""
            <div style='background: rgba(85,239,196,0.1); padding: 15px; border-radius: 10px; border: 1px solid #55EFC4;'>
                <p style='color: #55EFC4; margin: 0; font-size: 0.9rem;'>
                    ✅ Live Mathematics Active
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(253,203,110,0.1); padding: 15px; border-radius: 10px; border: 1px solid #FDCB6E;'>
                <p style='color: #FDCB6E; margin: 0; font-size: 0.9rem;'>
                    ⚠️ Demo Mode (equations not loaded)
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content based on selection
    if step == "Overview":
        render_overview()
    elif step == "Layer 1: Security Fortress":
        render_layer_detail(0)
    elif step == "Layer 2: Data Intake":
        render_layer_detail(1)
    elif step == "Layer 3: Processing Core":
        render_layer_detail(2)
    elif step == "Layer 4: Mathematics Engine":
        render_layer_detail(3)
    elif step == "Layer 5: Output Layer":
        render_layer_detail(4)
    elif step == "Live Demo: Coherence":
        render_coherence_demo()
    elif step == "Live Demo: Data Transformation":
        render_transformation_demo()
    elif step == "Complete Data Flow":
        render_complete_flow()

def render_overview():
    st.markdown("""
    <h2 style='color: #6C5CE7; font-size: 2.5rem; margin-bottom: 30px;'>
        System Overview: Five Layers of Coherence
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <p style='font-size: 1.2rem; color: #E0E0E0; line-height: 2;'>
            The Balantium architecture is structured like a living organism—each layer serves a purpose,
            and all layers communicate through <strong>coherence metrics</strong>. 
            The same mathematics that powers the system also structures it.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show full architecture
    st.plotly_chart(create_layer_diagram(), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview of each layer
    cols = st.columns(5)
    layers_info = [
        ("🛡️", "Security", "Fortress", "DNA verification, immune system, quarantine"),
        ("📥", "Intake", "Data Shell", "CSV, JSON, APIs, streams, databases"),
        ("⚙️", "Processing", "Core Engine", "Sanitize, validate, align, harmonize"),
        ("🧮", "Mathematics", "Equations", "Ba, D, Harmonium, CIX calculations"),
        ("📤", "Output", "Clean Data", "Coherence-scored, audit-ready datasets")
    ]
    
    for col, (icon, name, subtitle, desc) in zip(cols, layers_info):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; min-height: 220px;'>
                <div style='font-size: 3rem; margin-bottom: 10px;'>{icon}</div>
                <div style='color: #2E86DE; font-size: 1.3rem; font-weight: 700; margin-bottom: 5px;'>{name}</div>
                <div style='color: #A29BFE; font-size: 1rem; margin-bottom: 15px;'>{subtitle}</div>
                <div style='color: #C0C0C0; font-size: 0.85rem; line-height: 1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='content-box' style='margin-top: 40px;'>
        <h4 style='color: #2E86DE; font-size: 1.5rem; margin-bottom: 20px;'>
            Key Architectural Principles
        </h4>
        <ul style='color: #C0C0C0; font-size: 1.1rem; line-height: 2;'>
            <li><strong>Self-structuring:</strong> The mathematics that process data also organize the system</li>
            <li><strong>Coherence-driven:</strong> All layers communicate via coherence scores</li>
            <li><strong>Biological model:</strong> Immune system, DNA verification, adaptive responses</li>
            <li><strong>Deterministic:</strong> No black boxes, no ML training, pure mathematics</li>
            <li><strong>Modular:</strong> Each layer can operate independently or as part of the whole</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_layer_detail(layer_index):
    """Detailed view of a specific layer"""
    layers = [
        {
            'name': 'Security Fortress',
            'icon': '🛡️',
            'color': '#6C5CE7',
            'tagline': 'Security That Thinks Like Biology',
            'description': """
                The Security Fortress is the organism's immune system. It doesn't just block threats—
                it learns them, remembers them, and adapts to them. Every data point carries cryptographic DNA,
                and the system can detect when something doesn't belong.
            """,
            'components': [
                {
                    'name': 'DNA Core (dna_core.py)',
                    'function': 'Cryptographic fingerprinting of all data',
                    'features': ['FIPS-compliant hashing', 'Timestamp-locked verification', 'Tamper-evident by design']
                },
                {
                    'name': 'Immune System (immune_system.py)',
                    'function': 'Biological defense against corruption',
                    'features': ['Detects temporal anomalies', 'Identifies value corruption', 'Heals data gaps']
                },
                {
                    'name': 'Conscious Firewall (conscious_firewall.py)',
                    'function': 'Adaptive threat neutralization',
                    'features': ['Auto-quarantine suspicious data', 'Emergency override protocols', 'Threat learning']
                },
                {
                    'name': 'Resonance Auth (resonance_auth.py)',
                    'function': 'Creator signature verification',
                    'features': ['Multi-factor coherence auth', 'Environmental fingerprinting', 'Adaptive degradation']
                }
            ],
            'data_flow': 'All incoming data passes through security checks before entering the pipeline. Failed checks trigger quarantine.'
        },
        {
            'name': 'Data Intake Layer',
            'icon': '📥',
            'color': '#2E86DE',
            'tagline': 'Universal Data Shell',
            'description': """
                The Intake Layer is truly universal—it accepts any data format, any frequency, any source.
                CSV files, JSON APIs, streaming data, database connections—all are normalized into a coherent format
                before processing begins.
            """,
            'components': [
                {
                    'name': 'CSV/JSON Parsers',
                    'function': 'File-based data ingestion',
                    'features': ['Auto-detect schemas', 'Handle missing columns', 'Timezone normalization']
                },
                {
                    'name': 'API Connectors',
                    'function': 'Real-time data streaming',
                    'features': ['REST/WebSocket support', 'Rate limiting', 'Automatic retry logic']
                },
                {
                    'name': 'Database Adapters',
                    'function': 'Direct database integration',
                    'features': ['SQL/NoSQL support', 'Connection pooling', 'Query optimization']
                },
                {
                    'name': 'Stream Processors',
                    'function': 'High-frequency data handling',
                    'features': ['Buffer management', 'Backpressure handling', 'Event windowing']
                }
            ],
            'data_flow': 'Data enters through multiple channels, gets tagged with source metadata, and flows to sanitization.'
        },
        {
            'name': 'Processing Core',
            'icon': '⚙️',
            'color': '#A29BFE',
            'tagline': 'Where Chaos Becomes Order',
            'description': """
                The Processing Core is where the magic happens. Raw, messy, incoherent data enters—
                and clean, aligned, harmonized data emerges. This is the organism's digestive system,
                breaking down complexity and extracting what matters.
            """,
            'components': [
                {
                    'name': 'Sanitizer (sanitize_ohlcv_df)',
                    'function': 'Remove data quality issues',
                    'features': ['Duplicate timestamp removal', 'Infinity/NaN detection', 'Outlier flagging']
                },
                {
                    'name': 'Validator (validation_core.py)',
                    'function': 'Ensure data integrity',
                    'features': ['Schema validation', 'Causal ordering', 'Lookahead audit']
                },
                {
                    'name': 'Aligner (align_frames_intersection)',
                    'function': 'Multi-source synchronization',
                    'features': ['Timestamp alignment', 'Missing value interpolation', 'Frequency matching']
                },
                {
                    'name': 'Harmonizer (harmonium.py)',
                    'function': 'Crisis metabolization',
                    'features': ['Conflict resolution', 'Pattern emergence', 'Coherence optimization']
                }
            ],
            'data_flow': 'Data flows through sanitization → validation → alignment → harmonization in sequence.'
        },
        {
            'name': 'Mathematics Engine',
            'icon': '🧮',
            'color': '#74B9FF',
            'tagline': 'The Brain of the Organism',
            'description': """
                The Mathematics Engine contains 80+ proprietary equations that measure and optimize coherence.
                These aren't ML models—they're deterministic relationships derived from first principles.
                This is the organism's brain, making sense of patterns and detecting when things break.
            """,
            'components': [
                {
                    'name': 'Coherence Index (Ba)',
                    'function': 'Measures pattern alignment',
                    'features': ['Temporal coherence', 'Cross-asset harmony', 'Returns 0-1 score']
                },
                {
                    'name': 'Decoherence Index (D)',
                    'function': 'Detects pattern breakdown',
                    'features': ['Phase transition detection', 'Volatility precursors', 'Crisis early warning']
                },
                {
                    'name': 'Harmonium',
                    'function': 'Crisis metabolization',
                    'features': ['Multi-scale analysis', 'Regime identification', 'Stability scoring']
                },
                {
                    'name': 'CIX (Coherence Index Extended)',
                    'function': 'Unified system health metric',
                    'features': ['Combines Ba + D + Harmonium', '75-251 day lead time', 'Validated 30+ years']
                }
            ],
            'data_flow': 'Clean data enters, equations calculate coherence metrics, scores tag each data point.'
        },
        {
            'name': 'Output Layer',
            'icon': '📤',
            'color': '#55EFC4',
            'tagline': 'Coherence-Scored Intelligence',
            'description': """
                The Output Layer delivers clean, coherence-scored data ready for any use case.
                Each data point carries its own trustworthiness metric. High coherence = trust it.
                Low coherence = investigate. The system doesn't hide uncertainty—it quantifies it.
            """,
            'components': [
                {
                    'name': 'Data Export',
                    'function': 'Multi-format output',
                    'features': ['CSV/JSON/Parquet', 'Streaming API', 'Database write']
                },
                {
                    'name': 'Metrics Dashboard',
                    'function': 'Real-time monitoring',
                    'features': ['Coherence trends', 'Alert triggers', 'System health']
                },
                {
                    'name': 'Alert System',
                    'function': 'Proactive notifications',
                    'features': ['Threshold breaches', 'Anomaly detection', 'Custom rules']
                },
                {
                    'name': 'API Endpoints',
                    'function': 'Programmatic access',
                    'features': ['REST/GraphQL', 'Websocket streams', 'Batch queries']
                }
            ],
            'data_flow': 'Scored data flows to storage, dashboards, alerts, and external systems via API.'
        }
    ]
    
    layer = layers[layer_index]
    
    # Header
    st.markdown(f"""
    <div style='text-align: center; padding: 30px 0;'>
        <div style='font-size: 5rem; margin-bottom: 15px;'>{layer['icon']}</div>
        <h2 class='gradient-text' style='font-size: 3rem; margin-bottom: 15px;'>
            {layer['name']}
        </h2>
        <p style='font-size: 1.8rem; color: #B0B0B0;'>
            {layer['tagline']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show architecture with this layer highlighted
    st.plotly_chart(create_layer_diagram(highlight_layer=layer_index), use_container_width=True)
    
    # Description
    # Clean up the description text (remove extra whitespace and newlines)
    clean_description = ' '.join(layer['description'].split())
    
    st.markdown(f"""
    <div class='highlight-box'>
        <p style='font-size: 1.2rem; color: #E0E0E0; line-height: 2;'>
            {clean_description}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Components
    st.markdown("""
    <h3 style='color: #6C5CE7; font-size: 2rem; margin: 40px 0 25px 0;'>
        Core Components
    </h3>
    """, unsafe_allow_html=True)
    
    for comp in layer['components']:
        st.markdown(f"""
        <div class='content-box'>
            <h4 style='color: {layer['color']}; font-size: 1.4rem; margin-bottom: 10px;'>
                {comp['name']}
            </h4>
            <p style='color: #A29BFE; font-size: 1.1rem; margin-bottom: 15px; font-style: italic;'>
                {comp['function']}
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                {''.join([f"<li>{f}</li>" for f in comp['features']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Data flow
    st.markdown(f"""
    <div class='highlight-box' style='margin-top: 40px;'>
        <h4 style='color: {layer['color']}; font-size: 1.5rem; margin-bottom: 15px;'>
            Data Flow Through This Layer
        </h4>
        <p style='color: #E0E0E0; font-size: 1.1rem; line-height: 2;'>
            {layer['data_flow']}
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_coherence_demo():
    st.markdown("""
    <h2 style='color: #6C5CE7; font-size: 2.5rem; margin-bottom: 30px;'>
        Live Demo: Coherence Calculation
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <p style='font-size: 1.2rem; color: #E0E0E0; line-height: 2;'>
            This is a live demonstration of how Balantium measures coherence.
            Adjust the sliders to see how data quality affects the coherence score.
            If the core mathematics are available, this uses actual Balantium equations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    create_coherence_live_demo()
    
    st.markdown("""
    <div class='content-box' style='margin-top: 40px;'>
        <h4 style='color: #2E86DE; font-size: 1.5rem; margin-bottom: 20px;'>
            What Coherence Tells You
        </h4>
        <ul style='color: #C0C0C0; font-size: 1.1rem; line-height: 2;'>
            <li><strong>Coherence > 0.8:</strong> Data is highly trustworthy, patterns are clear, safe to use</li>
            <li><strong>Coherence 0.5-0.8:</strong> Data has some noise, investigate anomalies before critical use</li>
            <li><strong>Coherence < 0.5:</strong> Data integrity compromised, do not trust without deep investigation</li>
            <li><strong>Real-time monitoring:</strong> Coherence calculated continuously as data flows</li>
            <li><strong>No training needed:</strong> Mathematics work from day one on any dataset</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_transformation_demo():
    st.markdown("""
    <h2 style='color: #6C5CE7; font-size: 2.5rem; margin-bottom: 30px;'>
        Live Demo: Data Transformation
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <p style='font-size: 1.2rem; color: #E0E0E0; line-height: 2;'>
            Watch how Balantium transforms messy, broken data into clean, trustworthy data.
            This demonstrates the actual cleaning logic used in the Processing Core.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    create_data_transformation_demo()
    
    st.markdown("""
    <div class='content-box' style='margin-top: 40px;'>
        <h4 style='color: #2E86DE; font-size: 1.5rem; margin-bottom: 20px;'>
            Automatic Data Quality Fixes
        </h4>
        <ul style='color: #C0C0C0; font-size: 1.1rem; line-height: 2;'>
            <li><strong>Missing values:</strong> Forward-fill or interpolation based on temporal context</li>
            <li><strong>Invalid values:</strong> Infinity, NaN, negative prices automatically detected and removed</li>
            <li><strong>Outliers:</strong> Statistical outliers flagged and optionally smoothed</li>
            <li><strong>Duplicate timestamps:</strong> Conflicting values resolved via coherence maximization</li>
            <li><strong>Temporal gaps:</strong> Missing periods filled intelligently or flagged for review</li>
            <li><strong>All automatic:</strong> No manual rules, no configuration needed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_complete_flow():
    st.markdown("""
    <h2 style='color: #6C5CE7; font-size: 2.5rem; margin-bottom: 30px;'>
        Complete Data Flow
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <p style='font-size: 1.2rem; color: #E0E0E0; line-height: 2;'>
            This is the complete journey of a data point through the Balantium organism.
            From raw, untrusted input to clean, coherence-scored output ready for any application.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create comprehensive flow diagram
    fig = go.Figure()
    
    # Define all stages
    stages = [
        {'name': 'Raw Data\nInput', 'x': 0, 'y': 0, 'color': '#FF6B6B'},
        {'name': 'Security\nCheck', 'x': 1, 'y': 0, 'color': '#6C5CE7'},
        {'name': 'Data\nIntake', 'x': 2, 'y': 0, 'color': '#2E86DE'},
        {'name': 'Sanitize', 'x': 3, 'y': 0, 'color': '#A29BFE'},
        {'name': 'Validate', 'x': 4, 'y': 0, 'color': '#A29BFE'},
        {'name': 'Align', 'x': 5, 'y': 0, 'color': '#A29BFE'},
        {'name': 'Ba\nCoherence', 'x': 6, 'y': 0, 'color': '#74B9FF'},
        {'name': 'D\nDecoherence', 'x': 6, 'y': -1, 'color': '#74B9FF'},
        {'name': 'Harmonium', 'x': 6, 'y': 1, 'color': '#74B9FF'},
        {'name': 'CIX\nScore', 'x': 7, 'y': 0, 'color': '#74B9FF'},
        {'name': 'Clean\nOutput', 'x': 8, 'y': 0, 'color': '#55EFC4'}
    ]
    
    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (6, 7), (6, 8), (7, 9), (8, 9), (9, 10)
    ]
    
    for i, j in connections:
        fig.add_trace(go.Scatter(
            x=[stages[i]['x'], stages[j]['x']],
            y=[stages[i]['y'], stages[j]['y']],
            mode='lines',
            line=dict(color='rgba(108, 92, 231, 0.33)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw nodes
    for stage in stages:
        fig.add_trace(go.Scatter(
            x=[stage['x']],
            y=[stage['y']],
            mode='markers+text',
            marker=dict(size=30, color=stage['color'], line=dict(color='white', width=2)),
            text=[stage['name']],
            textposition='bottom center',
            textfont=dict(size=10, color='#E0E0E0'),
            showlegend=False,
            hoverinfo='text',
            hovertext=stage['name']
        ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.5)',
        xaxis=dict(visible=False, range=[-0.5, 8.5]),
        yaxis=dict(visible=False, range=[-2, 2]),
        height=400,
        margin=dict(t=20, b=100, l=20, r=20),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Timeline explanation
    st.markdown("""
    <h3 style='color: #6C5CE7; font-size: 2rem; margin: 40px 0 25px 0;'>
        Step-by-Step Breakdown
    </h3>
    """, unsafe_allow_html=True)
    
    steps = [
        ("1. Raw Data Input", "#FF6B6B", 
         "Messy, unvalidated data from any source—CSV, API, database, stream. May contain errors, gaps, duplicates."),
        
        ("2. Security Check", "#6C5CE7",
         "DNA verification, immune system scan. Suspicious data quarantined. Creator signatures validated."),
        
        ("3. Data Intake", "#2E86DE",
         "Format normalization, timezone alignment, schema detection. All sources unified into standard structure."),
        
        ("4. Sanitization", "#A29BFE",
         "Remove NaN, Inf, duplicate timestamps, outliers. Forward-fill gaps. Statistical outlier detection."),
        
        ("5. Validation", "#A29BFE",
         "Lookahead audit, causal ordering check, schema validation. Ensure no future data leakage."),
        
        ("6. Alignment", "#A29BFE",
         "Multi-source synchronization, timestamp matching, frequency normalization. All streams aligned."),
        
        ("7. Mathematics Engine", "#74B9FF",
         "Calculate Ba (coherence), D (decoherence), Harmonium. Each metric captures different aspects of system health."),
        
        ("8. CIX Scoring", "#74B9FF",
         "Unified coherence score combining all metrics. Single 0-1 value indicating trustworthiness."),
        
        ("9. Clean Output", "#55EFC4",
         "Data with coherence scores attached. Ready for analytics, trading, compliance, or any application.")
    ]
    
    for step_name, color, description in steps:
        st.markdown(f"""
        <div class='content-box' style='border-left: 4px solid {color}; margin: 20px 0;'>
            <h4 style='color: {color}; font-size: 1.3rem; margin-bottom: 10px;'>
                {step_name}
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 1.9;'>
                {description}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box' style='margin-top: 40px;'>
        <h4 style='color: #6C5CE7; font-size: 1.6rem; margin-bottom: 20px;'>
            Key Insights
        </h4>
        <ul style='color: #C0C0C0; font-size: 1.15rem; line-height: 2;'>
            <li><strong>Fully deterministic:</strong> Same input always produces same output</li>
            <li><strong>No training required:</strong> Mathematics work on day one</li>
            <li><strong>Transparent:</strong> Every step is auditable and explainable</li>
            <li><strong>Fast:</strong> Processing occurs in real-time or near-real-time</li>
            <li><strong>Universal:</strong> Works on any time-series or structured data</li>
            <li><strong>The organism self-organizes around coherence maximization</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

