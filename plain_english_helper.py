"""
Plain English Helper Module
Making technical concepts accessible to non-technical audiences
"""

import streamlit as st

class PlainEnglish:
    """Helper class to render plain-English explanations of technical concepts"""
    
    @staticmethod
    def coherence_explanation():
        """Simple explanation of what coherence means"""
        st.markdown("""
        <div style='background: rgba(85,239,196,0.1); border-left: 4px solid #55EFC4; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #55EFC4; margin-bottom: 15px;'>
                🎯 What "Coherence" Actually Means (In Plain English)
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                Think of coherence like <strong>music playing in harmony</strong>:
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>High coherence:</strong> Like an orchestra where everyone's in sync - patterns are clear, predictable, trustworthy</li>
                <li><strong>Low coherence:</strong> Like musicians playing different songs - chaotic, unpredictable, don't trust it</li>
            </ul>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                In data terms: Coherence measures if your data "makes sense" - are the patterns stable? 
                Is it following predictable rules? Or is it just random noise?
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def lookahead_explanation():
        """Simple explanation of the lookahead problem"""
        st.markdown("""
        <div style='background: rgba(255,107,107,0.1); border-left: 4px solid #FF6B6B; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #FF6B6B; margin-bottom: 15px;'>
                ⚠️ The "Lookahead" Problem (Why Most Backtests Are Fake)
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                Imagine you're taking a test where you accidentally see tomorrow's answers today. 
                You'd score 100%, but that doesn't mean you actually know the material.
            </p>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>In trading:</strong> "Lookahead bias" is when your model secretly uses future data 
                to make past predictions. It's like:
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li>"I predict the stock will crash on Monday" ... but you're saying this on Tuesday when you already know it crashed</li>
                <li>Your backtest looks perfect, but it's cheating</li>
            </ul>
            <p style='color: #55EFC4; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>How we prove we don't cheat:</strong> Our predictions only use data from BEFORE the prediction date. 
                We literally shift everything forward so it's impossible to peek at the future.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def walk_forward_explanation():
        """Simple explanation of walk-forward testing"""
        st.markdown("""
        <div style='background: rgba(108,92,231,0.1); border-left: 4px solid #6C5CE7; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #6C5CE7; margin-bottom: 15px;'>
                🚶 Walk-Forward Testing (How We Prove It's Not Curve-Fit)
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                Think of it like a weather forecast:
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>Bad way:</strong> Look at last year's weather, then "predict" last year's weather. Obviously you'll be right.</li>
                <li><strong>Good way:</strong> Look at Jan-Nov, predict December. Then look at Jan-Dec, predict January. Keep walking forward.</li>
            </ul>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                That's walk-forward testing. We train on past data, predict the future (which hasn't happened yet), 
                then check if we were right. Over and over, walking through time.
            </p>
            <p style='color: #55EFC4; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>Why it matters:</strong> If it works walking forward through 30 years of data, 
                it's not just lucky curve-fitting. It's actually predicting.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def parameter_stability_explanation():
        """Simple explanation of parameter stability"""
        st.markdown("""
        <div style='background: rgba(116,185,255,0.1); border-left: 4px solid #74B9FF; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #74B9FF; margin-bottom: 15px;'>
                ⚖️ Parameter Stability (Proof It's Not Fragile)
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                Imagine you're baking cookies:
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>Fragile recipe:</strong> Exactly 350°F for exactly 12 minutes. Off by 1°, they burn. Off by 1 minute, they're raw.</li>
                <li><strong>Stable recipe:</strong> 325-375°F for 10-15 minutes all work fine. Forgiving recipe.</li>
            </ul>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                In modeling: If changing your "window size" from 250 days to 260 days breaks everything, 
                you've overfit (found noise, not signal). It's fragile.
            </p>
            <p style='color: #55EFC4; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>Our system:</strong> Works across a wide range of parameters (180-365 days, different thresholds, etc). 
                That means we found real patterns, not random noise.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def cix_calculation_explanation():
        """Simple explanation of CIX calculation"""
        st.markdown("""
        <div style='background: rgba(162,155,254,0.1); border-left: 4px solid #A29BFE; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #A29BFE; margin-bottom: 15px;'>
                📊 CIX Calculation (What The Math Actually Does)
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                CIX (Coherence Index) is a score from 0 to 1 that answers: <strong>"How stable is the market right now?"</strong>
            </p>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>It combines four things:</strong>
            </p>
            <ol style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>Price patterns:</strong> Are stocks moving together logically?</li>
                <li><strong>Volatility patterns:</strong> Is volatility stable or spiking?</li>
                <li><strong>Volume patterns:</strong> Is trading volume normal or weird?</li>
                <li><strong>Cross-correlations:</strong> Are relationships between assets stable?</li>
            </ol>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>The result:</strong>
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>CIX > 0.8:</strong> Market is coherent (stable, predictable, safe to trade)</li>
                <li><strong>CIX 0.5-0.8:</strong> Market is okay (be cautious, watch closely)</li>
                <li><strong>CIX < 0.5:</strong> Market is breaking down (get defensive, reduce exposure)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def data_cleaning_explanation():
        """Simple explanation of data cleaning"""
        st.markdown("""
        <div style='background: rgba(253,203,110,0.1); border-left: 4px solid #FDCB6E; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #FDCB6E; margin-bottom: 15px;'>
                🧹 Data Cleaning (Why It Matters)
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                Think of data like vegetables from the store. They look clean, but...
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li>Some have dirt on them (bad data points)</li>
                <li>Some are rotten (outliers from errors)</li>
                <li>Some are missing (gaps in the data)</li>
                <li>Some are duplicates (same entry twice)</li>
            </ul>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                You wouldn't cook with dirty vegetables. Don't model with dirty data.
            </p>
            <p style='color: #55EFC4; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>What we do:</strong> Automatically detect and fix:
            </p>
            <ul style='color: #55EFC4; font-size: 1.05rem; line-height: 2;'>
                <li>✅ Remove duplicate timestamps</li>
                <li>✅ Fill missing values intelligently</li>
                <li>✅ Detect and handle outliers</li>
                <li>✅ Fix data type errors (Inf, NaN, etc.)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def regime_detection_explanation():
        """Simple explanation of regime detection"""
        st.markdown("""
        <div style='background: rgba(85,239,196,0.1); border-left: 4px solid #55EFC4; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #55EFC4; margin-bottom: 15px;'>
                📈 Regime Detection (Why Your Strategy Stops Working)
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                Markets have different "moods" or "regimes":
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>Bull market:</strong> Everything goes up, risk is rewarded</li>
                <li><strong>Bear market:</strong> Everything goes down, fear dominates</li>
                <li><strong>Sideways/choppy:</strong> No clear direction, mean-reversion works</li>
                <li><strong>Crisis:</strong> Correlations break, normal rules don't apply</li>
            </ul>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>The problem:</strong> Your strategy might work great in bull markets but fail in crises.
            </p>
            <p style='color: #55EFC4; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>What we do:</strong> Detect regime changes BEFORE they destroy your portfolio. 
                When coherence drops, we know the market is changing regimes. Time to get defensive.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def confidence_scoring_explanation():
        """Simple explanation of confidence scoring"""
        st.markdown("""
        <div style='background: rgba(108,92,231,0.1); border-left: 4px solid #6C5CE7; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #6C5CE7; margin-bottom: 15px;'>
                🎯 Confidence Scoring (Not All Signals Are Equal)
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                Imagine your GPS says "turn left" but the confidence is only 20%. Would you turn? Probably not.
            </p>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                Same with trading signals. Your model might say "BUY", but:
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li>Is the data it's using trustworthy? (Data quality)</li>
                <li>Is the market in a regime where this works? (Market coherence)</li>
                <li>Are correlations stable? (Cross-asset alignment)</li>
            </ul>
            <p style='color: #55EFC4; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>What we provide:</strong> A confidence score (0-1) for each signal based on:
            </p>
            <ul style='color: #55EFC4; font-size: 1.05rem; line-height: 2;'>
                <li>Data quality × Market stability × Regime alignment = Confidence</li>
                <li>Use it to size positions: High confidence = full size, low confidence = small size (or skip)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def ml_integration_explanation():
        """Simple explanation of how Balantium complements ML"""
        st.markdown("""
        <div style='background: rgba(46,134,222,0.1); border-left: 4px solid #2E86DE; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #2E86DE; margin-bottom: 15px;'>
                🤝 How Balantium + ML Work Together
            </h4>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                Think of it like a pilot and co-pilot:
            </p>
            <ul style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>Your ML model is the pilot:</strong> It's flying the plane (making trading decisions based on sentiment, patterns, etc.)</li>
                <li><strong>Balantium is the co-pilot:</strong> It's monitoring the instruments (is data clean? Is weather stable? Are we off course?)</li>
            </ul>
            <p style='color: #C0C0C0; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>What Balantium adds:</strong>
            </p>
            <ol style='color: #C0C0C0; font-size: 1.05rem; line-height: 2;'>
                <li><strong>Pre-flight check:</strong> Clean your data before feeding it to ML</li>
                <li><strong>Weather report:</strong> Tell you if market conditions support your strategy</li>
                <li><strong>Risk management:</strong> Adjust position size based on confidence</li>
                <li><strong>Emergency alerts:</strong> Warn you when regimes are changing</li>
            </ol>
            <p style='color: #55EFC4; font-size: 1.05rem; line-height: 2; margin-top: 15px;'>
                <strong>Result:</strong> Your ML keeps doing what it does best. Balantium just makes sure 
                you're flying with good data, in good conditions, with appropriate risk management.
            </p>
        </div>
        """, unsafe_allow_html=True)


# Quick access functions for common use cases
def show_all_basics():
    """Show all basic explanations"""
    PlainEnglish.coherence_explanation()
    PlainEnglish.cix_calculation_explanation()
    PlainEnglish.data_cleaning_explanation()

def show_all_validation():
    """Show all validation explanations"""
    PlainEnglish.lookahead_explanation()
    PlainEnglish.walk_forward_explanation()
    PlainEnglish.parameter_stability_explanation()

def show_all_advanced():
    """Show all advanced explanations"""
    PlainEnglish.regime_detection_explanation()
    PlainEnglish.confidence_scoring_explanation()
    PlainEnglish.ml_integration_explanation()

def show_everything():
    """Show all explanations"""
    show_all_basics()
    show_all_validation()
    show_all_advanced()



