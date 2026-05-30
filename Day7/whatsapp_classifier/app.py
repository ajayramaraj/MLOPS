"""
WhatsApp Message Classification Application
AI-Powered Institutional Message Categorization System
Built with Streamlit, scikit-learn, and TF-IDF
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import io
import csv
import sys
import warnings

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

try:
    import joblib
except ImportError:
    joblib = None


# Configure Streamlit Page
st.set_page_config(
    page_title="WhatsApp Message Classifier",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "WhatsApp Message Classification System - AI-Powered Institutional Message Categorizer"
    }
)

# Apply Custom CSS for Modern UI
st.markdown("""
<style>
    :root {
        --primary-color: #25D366;
        --secondary-color: #1F2937;
        --dark-bg: #0F1419;
        --card-bg: #1a1f2e;
        --border-color: #2d3748;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background-color: var(--dark-bg);
        color: var(--text-primary);
    }
    
    .stApp {
        background-color: var(--dark-bg);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(26, 31, 46, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        background: rgba(26, 31, 46, 0.95);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(37, 211, 102, 0.3);
    }
    
    /* Status Indicators */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px;
    }
    
    .status-active {
        background-color: rgba(37, 211, 102, 0.2);
        color: #25D366;
        border: 1px solid #25D366;
    }
    
    .status-inactive {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(37, 211, 102, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 2px solid rgba(37, 211, 102, 0.3);
        border-radius: 16px;
        padding: 28px;
        margin: 20px 0;
        box-shadow: 0 12px 40px rgba(37, 211, 102, 0.15);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(26, 31, 46, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #25D366 0%, #1abc9c 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
        margin: 8px 0;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1abc9c 0%, #25D366 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(37, 211, 102, 0.3);
    }
    
    /* Text Input */
    .stTextArea > div > div > textarea {
        background-color: rgba(26, 31, 46, 0.8) !important;
        border: 1px solid rgba(37, 211, 102, 0.3) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text-primary);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--card-bg);
    }
    
    /* Info/Success/Warning Messages */
    .stInfo {
        background-color: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.5);
        border-radius: 8px;
    }
    
    .stSuccess {
        background-color: rgba(37, 211, 102, 0.2);
        border: 1px solid rgba(37, 211, 102, 0.5);
        border-radius: 8px;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.2);
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-radius: 8px;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 8px;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 20px 0;
    }
    
    /* Center text */
    .center-text {
        text-align: center;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #25D366 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8em;
        font-weight: 800;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
def initialize_session_state():
    """Initialize all session state variables"""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "total_predictions" not in st.session_state:
        st.session_state.total_predictions = 0
    if "session_predictions" not in st.session_state:
        st.session_state.session_predictions = 0


initialize_session_state()


# =====================================================================
# MODEL LOADING WITH CACHING
# =====================================================================
@st.cache_resource
def load_model():
    """Load logistic regression model from pickle file with multiple encoding fallbacks"""
    try:
        model_path = Path(__file__).parent / "logistic_regression_model.pkl"
        
        if not model_path.exists():
            return None, False, f"Model file not found at: {model_path}"
        
        # Suppress warnings during loading
        warnings.filterwarnings('ignore')
        
        # Try standard pickle loading
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model, True, None
        except Exception as e1:
            pass
        
        # Try with latin1 encoding
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f, encoding='latin1')
            return model, True, None
        except Exception as e2:
            pass
        
        # Try with bytes encoding
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f, encoding='bytes')
            return model, True, None
        except Exception as e3:
            pass
        
        # Try with fix_imports
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f, fix_imports=True, encoding='latin1')
            return model, True, None
        except Exception as e4:
            pass
        
        # Try joblib if available
        if joblib is not None:
            try:
                model = joblib.load(model_path)
                return model, True, None
            except Exception as e5:
                pass
        
        # If all fail, return detailed error
        return None, False, "Model file is corrupted or incompatible. Please regenerate the model using Python 3.8+ and scikit-learn 1.3.2"
    
    except Exception as e:
        return None, False, f"Error loading model: {str(e)}"


@st.cache_resource
def load_vectorizer():
    """Load TF-IDF vectorizer from pickle file with multiple encoding fallbacks"""
    try:
        vectorizer_path = Path(__file__).parent / "tfidf_vectorizer.pkl"
        
        if not vectorizer_path.exists():
            return None, False, f"Vectorizer file not found at: {vectorizer_path}"
        
        # Suppress warnings during loading
        warnings.filterwarnings('ignore')
        
        # Try standard pickle loading
        try:
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)
            return vectorizer, True, None
        except Exception as e1:
            pass
        
        # Try with latin1 encoding
        try:
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f, encoding='latin1')
            return vectorizer, True, None
        except Exception as e2:
            pass
        
        # Try with bytes encoding
        try:
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f, encoding='bytes')
            return vectorizer, True, None
        except Exception as e3:
            pass
        
        # Try with fix_imports
        try:
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f, fix_imports=True, encoding='latin1')
            return vectorizer, True, None
        except Exception as e4:
            pass
        
        # Try joblib if available
        if joblib is not None:
            try:
                vectorizer = joblib.load(vectorizer_path)
                return vectorizer, True, None
            except Exception as e5:
                pass
        
        # If all fail, return detailed error
        return None, False, "Vectorizer file is corrupted or incompatible. Please regenerate the vectorizer using Python 3.8+ and scikit-learn 1.3.2"
    
    except Exception as e:
        return None, False, f"Error loading vectorizer: {str(e)}"


# Load model and vectorizer
model, model_loaded, model_error = load_model()
vectorizer, vectorizer_loaded, vectorizer_error = load_vectorizer()

# Determine overall system status
system_ready = model_loaded and vectorizer_loaded


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def get_system_status():
    """Get system status indicators"""
    statuses = {
        "Model": (model_loaded, model_error),
        "Vectorizer": (vectorizer_loaded, vectorizer_error),
        "Prediction Engine": (system_ready, None if system_ready else "Waiting for model and vectorizer")
    }
    return statuses


def analyze_text(text: str) -> dict:
    """Analyze text properties"""
    analysis = {
        "characters": len(text),
        "words": len(text.split()),
        "sentences": len([s for s in text.split('.') if s.strip()]),
    }
    return analysis


def predict_category(text: str) -> dict:
    """Predict message category and confidence"""
    try:
        if not system_ready:
            return {
                "success": False,
                "error": "Model or vectorizer not loaded"
            }
        
        if not text or len(text.strip()) == 0:
            return {
                "success": False,
                "error": "Input text cannot be empty"
            }
        
        # Transform text using vectorizer
        text_vectorized = vectorizer.transform([text])
        
        # Get prediction and probability
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get confidence score
        confidence = float(np.max(probabilities)) * 100
        
        # Get prediction time
        prediction_time = datetime.now()
        
        return {
            "success": True,
            "category": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "classes": model.classes_,
            "time": prediction_time
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }


def add_to_history(message: str, category: str, confidence: float, timestamp: datetime):
    """Add prediction to history"""
    history_entry = {
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Message": message[:50] + "..." if len(message) > 50 else message,
        "Category": category,
        "Confidence": f"{confidence:.2f}%"
    }
    st.session_state.prediction_history.append(history_entry)
    st.session_state.total_predictions += 1
    st.session_state.session_predictions += 1


def export_history_to_csv() -> str:
    """Export prediction history to CSV format"""
    if not st.session_state.prediction_history:
        return ""
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Timestamp", "Message", "Category", "Confidence"])
    writer.writeheader()
    writer.writerows(st.session_state.prediction_history)
    return output.getvalue()


# =====================================================================
# MAIN UI - HEADER
# =====================================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<h1 class="main-title">💬 WhatsApp Message Classifier</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #a0aec0; font-size: 1.1em; margin-bottom: 24px;'>
AI-Powered Institutional Message Categorization
</div>
""", unsafe_allow_html=True)

# System Status
st.markdown("### 🔧 System Status")
col1, col2, col3 = st.columns(3)

statuses = get_system_status()
with col1:
    status_text = "✅ Loaded" if statuses["Model"][0] else "❌ Failed"
    st.markdown(f"<div style='text-align: center;'><div class='status-badge {('status-active' if statuses['Model'][0] else 'status-inactive')}'>📦 Model: {status_text}</div></div>", unsafe_allow_html=True)

with col2:
    status_text = "✅ Loaded" if statuses["Vectorizer"][0] else "❌ Failed"
    st.markdown(f"<div style='text-align: center;'><div class='status-badge {('status-active' if statuses['Vectorizer'][0] else 'status-inactive')}'>🔢 Vectorizer: {status_text}</div></div>", unsafe_allow_html=True)

with col3:
    status_text = "✅ Ready" if statuses["Prediction Engine"][0] else "❌ Not Ready"
    st.markdown(f"<div style='text-align: center;'><div class='status-badge {('status-active' if statuses['Prediction Engine'][0] else 'status-inactive')}'>⚙️ Engine: {status_text}</div></div>", unsafe_allow_html=True)

st.divider()

# Show error if system not ready
if not system_ready:
    st.error("❌ **System Error**")
    if model_error:
        st.error(f"Model Issue: {model_error}")
    if vectorizer_error:
        st.error(f"Vectorizer Issue: {vectorizer_error}")
    st.stop()


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### 📋 Navigation")
    
    # About Project
    with st.expander("ℹ️ About Project", expanded=False):
        st.markdown("""
        **WhatsApp Message Classification System**
        
        This intelligent system classifies WhatsApp institutional messages 
        using advanced machine learning techniques.
        
        **How it works:**
        1. User provides a WhatsApp message
        2. Message is transformed using TF-IDF
        3. Logistic Regression model predicts the category
        4. Confidence score and prediction details are displayed
        
        **Use Cases:**
        - Automatic message categorization
        - Institutional communication management
        - Message routing and organization
        - Analytics and insights
        """)
    
    # Model Information
    with st.expander("🤖 Model Information", expanded=False):
        st.markdown(f"""
        **Model Details**
        
        - **Algorithm:** Logistic Regression
        - **Feature Extraction:** TF-IDF Vectorizer
        - **Input:** Text messages
        - **Output:** Message category + confidence
        
        **Categories Found:**
        """)
        
        if model_loaded and hasattr(model, 'classes_'):
            categories = list(model.classes_)
            for i, cat in enumerate(categories, 1):
                st.write(f"{i}. `{cat}`")
        else:
            st.warning("Categories not available")
    
    # Usage Instructions
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        **Step-by-Step Guide:**
        
        1. **Enter Message**
           - Paste or type a WhatsApp message in the text area
        
        2. **Make Prediction**
           - Click "🔮 Predict Category" button
        
        3. **View Results**
           - See predicted category with confidence
           - Check message analytics
           - View confidence visualization
        
        4. **Manage History**
           - View all predictions in the history table
           - Export history as CSV
        
        5. **Clear**
           - Use "🗑️ Clear Input" to reset
        """)
    
    # Project Statistics
    with st.expander("📊 Project Statistics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Total Predictions",
                st.session_state.total_predictions,
                delta=None
            )
        
        with col2:
            st.metric(
                "Session Predictions",
                st.session_state.session_predictions,
                delta=None
            )
        
        st.divider()
        
        st.markdown("**System Info:**")
        st.info("""
        - **Python ML Framework:** scikit-learn
        - **Frontend:** Streamlit
        - **Model Type:** Classification
        - **Vectorization:** TF-IDF
        """)


# =====================================================================
# MAIN CONTENT - INPUT SECTION
# =====================================================================
st.markdown("### 📝 Input Message")

user_message = st.text_area(
    label="Paste WhatsApp Message",
    placeholder="Examples:\n• Internal exam scheduled on Monday\n• TCS placement drive tomorrow\n• Submit assignment before Friday\n• Daily attendance check",
    height=150,
    label_visibility="collapsed"
)

# Action Buttons
col1, col2, col3, col4 = st.columns(4)

predict_button = col1.button("🔮 Predict Category", use_container_width=True, key="predict")
clear_button = col2.button("🗑️ Clear Input", use_container_width=True, key="clear")

if clear_button:
    st.session_state.clear_input = True
    st.rerun()

if st.session_state.get("clear_input", False):
    user_message = ""
    st.session_state.clear_input = False


# =====================================================================
# MESSAGE ANALYTICS
# =====================================================================
if user_message:
    st.markdown("### 📊 Message Analytics")
    
    analysis = analyze_text(user_message)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 2em; color: #25D366; font-weight: bold;'>
                {}
            </div>
            <div style='color: #a0aec0; font-size: 0.9em; margin-top: 8px;'>
                Characters
            </div>
        </div>
        """.format(analysis["characters"]), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 2em; color: #3b82f6; font-weight: bold;'>
                {}
            </div>
            <div style='color: #a0aec0; font-size: 0.9em; margin-top: 8px;'>
                Words
            </div>
        </div>
        """.format(analysis["words"]), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 2em; color: #f59e0b; font-weight: bold;'>
                {}
            </div>
            <div style='color: #a0aec0; font-size: 0.9em; margin-top: 8px;'>
                Sentences
            </div>
        </div>
        """.format(analysis["sentences"]), unsafe_allow_html=True)


# =====================================================================
# PREDICTION SECTION
# =====================================================================
if predict_button and user_message:
    with st.spinner("🔄 Analyzing message..."):
        result = predict_category(user_message)
    
    if result["success"]:
        # Add to history
        add_to_history(
            user_message,
            result["category"],
            result["confidence"],
            result["time"]
        )
        
        # Display prediction card
        st.markdown("### 🎯 Prediction Result")
        
        st.markdown(f"""
        <div class='prediction-card'>
            <div style='text-align: center;'>
                <div style='font-size: 1.3em; color: #a0aec0; margin-bottom: 12px;'>
                    Predicted Category
                </div>
                <div style='font-size: 2.5em; font-weight: bold; color: #25D366; margin-bottom: 16px;'>
                    {result['category'].upper()}
                </div>
                <div style='display: flex; justify-content: space-around; margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);'>
                    <div>
                        <div style='color: #a0aec0; font-size: 0.9em;'>Confidence</div>
                        <div style='font-size: 1.8em; color: #25D366; font-weight: bold;'>{result['confidence']:.2f}%</div>
                    </div>
                    <div>
                        <div style='color: #a0aec0; font-size: 0.9em;'>Time</div>
                        <div style='font-size: 1.1em; color: #a0aec0; font-weight: 600;'>{result['time'].strftime("%H:%M:%S")}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ Prediction completed successfully!")
        
        # Confidence Visualization
        st.markdown("### 📈 Confidence Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        # Progress Bar
        with col1:
            st.markdown("**Confidence Score:**")
            st.progress(result["confidence"] / 100.0)
            st.markdown(f"<div style='text-align: center; color: #a0aec0;'>{result['confidence']:.2f}%</div>", unsafe_allow_html=True)
        
        # Category Probabilities Chart
        with col2:
            if len(result["classes"]) > 0:
                # Create dataframe for visualization
                prob_df = pd.DataFrame({
                    "Category": result["classes"],
                    "Probability": result["probabilities"] * 100
                }).sort_values("Probability", ascending=True)
                
                # Try to use plotly if available, otherwise use streamlit bar chart
                if go is not None:
                    try:
                        # Create bar chart with plotly
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    y=prob_df["Category"],
                                    x=prob_df["Probability"],
                                    orientation='h',
                                    marker=dict(
                                        color=prob_df["Probability"],
                                        colorscale='Greens',
                                        showscale=False
                                    ),
                                    text=[f"{p:.2f}%" for p in prob_df["Probability"]],
                                    textposition='auto'
                                )
                            ]
                        )
                        
                        fig.update_layout(
                            title="Category Probabilities",
                            xaxis_title="Probability (%)",
                            yaxis_title="Category",
                            height=300,
                            template="plotly_dark",
                            showlegend=False,
                            margin=dict(l=100, r=20, t=40, b=20),
                            plot_bgcolor='rgba(26, 31, 46, 0.5)',
                            paper_bgcolor='rgba(26, 31, 46, 0)',
                            font=dict(color='#a0aec0')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        # Fallback to streamlit bar chart
                        st.bar_chart(prob_df.set_index("Category"), use_container_width=True)
                else:
                    # Use streamlit native bar chart
                    st.bar_chart(prob_df.set_index("Category"), use_container_width=True)
    
    else:
        st.error(f"❌ Prediction Failed: {result['error']}")

elif predict_button and not user_message:
    st.warning("⚠️ Please enter a message before predicting!")


# =====================================================================
# PREDICTION HISTORY
# =====================================================================
if st.session_state.prediction_history:
    st.divider()
    st.markdown("### 📜 Prediction History")
    
    # Display history table
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.TextColumn("Timestamp", width="small"),
            "Message": st.column_config.TextColumn("Message", width="medium"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
        }
    )
    
    # Export button
    csv_data = export_history_to_csv()
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()


# =====================================================================
# FOOTER
# =====================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.85em; margin-top: 30px;'>
    <p>🚀 WhatsApp Message Classification System v1.0</p>
    <p>Built with ❤️ using Streamlit, scikit-learn, and TF-IDF</p>
    <p>© 2024 AI-Powered Message Classification</p>
</div>
""", unsafe_allow_html=True)
