"""
WhatsApp Message Classification System
AI-Powered Institutional Message Categorization using TF-IDF and Logistic Regression
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import io
import base64

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="WhatsApp Message Classifier",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR GLASSMORPHISM AND DARK THEME
# ============================================================================

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background: linear-gradient(135deg, #0E1117 0%, #161B22 100%);
        color: #C9D1D9;
    }
    
    /* Main container styling */
    .main {
        background: transparent;
        padding: 0;
    }
    
    /* Custom card styling for glassmorphism */
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.3);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        margin: 10px 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(0, 168, 107, 0.15) 0%, rgba(30, 144, 255, 0.15) 100%);
        border: 2px solid #00A86B;
        border-radius: 15px;
        padding: 30px;
        backdrop-filter: blur(10px);
        box-shadow: 0 12px 48px 0 rgba(0, 168, 107, 0.2);
        text-align: center;
    }
    
    .status-card {
        background: rgba(22, 27, 34, 0.8);
        border-left: 4px solid #00A86B;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
    }
    
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.3);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #00A86B 0%, #1E90FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 20px 0 10px 0;
    }
    
    .header-subtitle {
        font-size: 1.1em;
        color: #8B949E;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00A86B 0%, #1E90FF 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 168, 107, 0.3);
    }
    
    /* Input area styling */
    .stTextArea textarea {
        background: rgba(22, 27, 34, 0.5);
        border: 2px solid rgba(48, 54, 61, 0.3);
        border-radius: 8px;
        color: #C9D1D9;
        font-size: 1em;
    }
    
    .stTextArea textarea:focus {
        border-color: #00A86B;
        box-shadow: 0 0 20px rgba(0, 168, 107, 0.2);
    }
    
    /* Table styling */
    .dataframe {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.3);
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(13, 17, 23, 0.95) 0%, rgba(22, 27, 34, 0.95) 100%);
    }
    
    .sidebar-section {
        background: rgba(30, 144, 255, 0.1);
        border-left: 4px solid #1E90FF;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* Success message */
    .success-badge {
        display: inline-block;
        background: #00A86B;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: 600;
    }
    
    /* Error message */
    .error-badge {
        display: inline-block;
        background: #FF4444;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: 600;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background: #00A86B;
        box-shadow: 0 0 8px #00A86B;
    }
    
    .status-inactive {
        background: #FF4444;
        box-shadow: 0 0 8px #FF4444;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING FUNCTIONS FOR MODEL AND VECTORIZER LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the pre-trained logistic regression model"""
    try:
        with open("logistic_regression_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found: logistic_regression_model.pkl")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_vectorizer():
    """Load the pre-trained TF-IDF vectorizer"""
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except FileNotFoundError:
        st.error("❌ Vectorizer file not found: tfidf_vectorizer.pkl")
        return None
    except Exception as e:
        st.error(f"❌ Error loading vectorizer: {str(e)}")
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def analyze_text(text):
    """Analyze text and return statistics"""
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len([s for s in text.split('.') if s.strip()])
    
    return {
        "characters": char_count,
        "words": word_count,
        "sentences": sentence_count if sentence_count > 0 else 1
    }

def predict_category(text, model, vectorizer):
    """Predict category and confidence"""
    try:
        # Transform text using TF-IDF vectorizer
        text_vectorized = vectorizer.transform([text])
        
        # Get prediction
        prediction = model.predict(text_vectorized)[0]
        
        # Get confidence scores
        prediction_proba = model.predict_proba(text_vectorized)[0]
        confidence = float(np.max(prediction_proba) * 100)
        
        # Get all class probabilities
        class_probabilities = dict(zip(model.classes_, prediction_proba * 100))
        
        return {
            "category": prediction,
            "confidence": confidence,
            "all_probabilities": class_probabilities,
            "success": True
        }
    except Exception as e:
        return {
            "category": None,
            "confidence": 0,
            "all_probabilities": {},
            "success": False,
            "error": str(e)
        }

def create_confidence_chart(probabilities):
    """Create a confidence visualization chart"""
    df = pd.DataFrame(list(probabilities.items()), columns=['Category', 'Confidence'])
    df = df.sort_values('Confidence', ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Category'],
            x=df['Confidence'],
            orientation='h',
            marker=dict(
                color=df['Confidence'],
                colorscale='Viridis',
                showscale=False
            ),
            text=df['Confidence'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Category Confidence Distribution",
        xaxis_title="Confidence (%)",
        yaxis_title="Category",
        template="plotly_dark",
        height=400,
        showlegend=False,
        margin=dict(l=150),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)"
    )
    
    return fig

def export_to_csv(history_df):
    """Create CSV download link"""
    csv = history_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return b64

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0

if "session_predictions" not in st.session_state:
    st.session_state.session_predictions = 0

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 PROJECT INFORMATION")
    
    with st.expander("ℹ️ About Project", expanded=False):
        st.markdown("""
        This system classifies WhatsApp institutional messages using a machine learning model trained with TF-IDF and Logistic Regression.
        
        **Purpose:** Automatically categorize incoming institutional messages from WhatsApp into predefined categories for better organization and notification management.
        
        **Technology:** Uses Natural Language Processing (NLP) to understand message content and predict the most likely category.
        """)
    
    with st.expander("🤖 Model Information", expanded=False):
        st.markdown("""
        **Algorithm:** Logistic Regression
        
        **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
        
        **Why Logistic Regression?**
        - Fast inference for real-time predictions
        - Interpretable results
        - Low computational overhead
        - Excellent for text classification
        
        **Why TF-IDF?**
        - Converts text to numerical features
        - Reduces impact of common words
        - Captures important terms in messages
        """)
    
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        **Step 1:** Paste your WhatsApp message in the text area
        
        **Step 2:** Click the "🔍 Predict Category" button
        
        **Step 3:** View the prediction results with confidence score
        
        **Step 4:** Check the confidence distribution chart
        
        **Step 5:** Download your prediction history as CSV
        
        **Tips:**
        - Use complete, natural messages for best results
        - The system works best with institutional messages
        - Confidence > 80% indicates high certainty
        """)
    
    with st.expander("📊 Project Statistics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", st.session_state.total_predictions)
        with col2:
            st.metric("Session Predictions", st.session_state.session_predictions)
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("[GitHub](https://github.com)")
    with col2:
        st.markdown("[LinkedIn](https://linkedin.com)")
    with col3:
        st.markdown("[Portfolio](https://portfolio.com)")

# ============================================================================
# HEADER SECTION
# ============================================================================

st.markdown("""
<div class="header-title">
    💬 WhatsApp Message Classification System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-subtitle">
    AI-Powered Institutional Message Categorization
</div>
""", unsafe_allow_html=True)

# ============================================================================
# STATUS INDICATORS
# ============================================================================

# Load model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

col1, col2, col3 = st.columns(3)

with col1:
    if model is not None:
        st.markdown("""
        <div class="status-card">
            <span class="status-indicator status-active"></span>
            <strong>Model Status</strong><br>
            <span style="color: #00A86B; font-weight: 600;">✓ Loaded</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <span class="status-indicator status-inactive"></span>
            <strong>Model Status</strong><br>
            <span style="color: #FF4444; font-weight: 600;">✗ Failed</span>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if vectorizer is not None:
        st.markdown("""
        <div class="status-card">
            <span class="status-indicator status-active"></span>
            <strong>Vectorizer Status</strong><br>
            <span style="color: #00A86B; font-weight: 600;">✓ Loaded</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <span class="status-indicator status-inactive"></span>
            <strong>Vectorizer Status</strong><br>
            <span style="color: #FF4444; font-weight: 600;">✗ Failed</span>
        </div>
        """, unsafe_allow_html=True)

with col3:
    if model is not None and vectorizer is not None:
        st.markdown("""
        <div class="status-card">
            <span class="status-indicator status-active"></span>
            <strong>Prediction Engine</strong><br>
            <span style="color: #00A86B; font-weight: 600;">✓ Ready</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <span class="status-indicator status-inactive"></span>
            <strong>Prediction Engine</strong><br>
            <span style="color: #FF4444; font-weight: 600;">✗ Not Ready</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# MAIN INPUT SECTION
# ============================================================================

st.markdown("### 📝 Input Message")
message_input = st.text_area(
    "Paste WhatsApp Message",
    placeholder="Internal exam scheduled on Monday\nTCS placement drive tomorrow\nSubmit assignment before Friday",
    height=150,
    key="message_input"
)

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("🔍 Predict Category", use_container_width=True, key="predict_btn")

with col2:
    clear_button = st.button("🗑️ Clear Input", use_container_width=True, key="clear_btn")

if clear_button:
    st.session_state.message_input = ""
    st.rerun()

# ============================================================================
# PREDICTION LOGIC
# ============================================================================

if predict_button:
    if not message_input.strip():
        st.warning("⚠️ Please enter a message before predicting.")
    elif model is None or vectorizer is None:
        st.error("❌ Model or Vectorizer not loaded. Cannot make predictions.")
    else:
        with st.spinner("🔄 Analyzing message..."):
            start_time = datetime.now()
            
            # Make prediction
            result = predict_category(message_input, model, vectorizer)
            
            end_time = datetime.now()
            prediction_time = (end_time - start_time).total_seconds()
            
            if result["success"]:
                # Get text statistics
                stats = analyze_text(message_input)
                
                # Display prediction card
                st.markdown("### 🎯 Prediction Result")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style="margin: 0; color: #00A86B;">📌 {result['category'].upper()}</h2>
                        <p style="margin: 10px 0; font-size: 0.9em; color: #8B949E;">Predicted Category</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                
                with col3:
                    st.metric("Time", f"{prediction_time:.3f}s")
                
                # Add to history
                st.session_state.prediction_history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Message": message_input[:50] + "..." if len(message_input) > 50 else message_input,
                    "Category": result["category"],
                    "Confidence": f"{result['confidence']:.1f}%"
                })
                
                st.session_state.session_predictions += 1
                st.session_state.total_predictions += 1
                
                st.success("✅ Prediction completed successfully!")
                
                # Message Analytics
                st.markdown("### 📊 Message Analytics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin: 0; color: #1E90FF;">{stats['characters']}</h3>
                        <p style="margin: 5px 0; font-size: 0.9em; color: #8B949E;">Characters</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin: 0; color: #00A86B;">{stats['words']}</h3>
                        <p style="margin: 5px 0; font-size: 0.9em; color: #8B949E;">Words</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin: 0; color: #FFD700;">{stats['sentences']}</h3>
                        <p style="margin: 5px 0; font-size: 0.9em; color: #8B949E;">Sentences</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence visualization
                st.markdown("### 📈 Confidence Distribution")
                fig = create_confidence_chart(result["all_probabilities"])
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence progress bar
                st.markdown("### 📊 Confidence Level")
                st.progress(result['confidence'] / 100.0)
                st.caption(f"Model confidence: {result['confidence']:.1f}% - {'High certainty' if result['confidence'] > 80 else 'Medium certainty' if result['confidence'] > 60 else 'Low certainty'}")
                
            else:
                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

# ============================================================================
# PREDICTION HISTORY SECTION
# ============================================================================

st.markdown("---")
st.markdown("### 📜 Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Display table
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv_data = history_df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv"><button style="background: linear-gradient(135deg, #00A86B 0%, #1E90FF 100%); color: white; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; width: 100%;">📥 Download as CSV</button></a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("📭 No predictions made yet. Start by entering a message and clicking predict!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8B949E; font-size: 0.9em; padding: 20px 0;">
    <p>Built with <span style="color: #FF4444;">❤️</span> using Streamlit, TF-IDF & Logistic Regression</p>
    <p>© 2024 WhatsApp Message Classification System | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)