# 💬 WhatsApp Message Classification System

An AI-powered application that classifies WhatsApp institutional messages using Machine Learning. Built with Streamlit, TF-IDF vectorization, and Logistic Regression.

## 🎯 Project Overview

This system automatically categorizes incoming institutional WhatsApp messages into predefined categories such as:
- Exam Notifications
- Placement Tests
- Assignment Notices
- Attendance Notices
- Admin Notices
- Event Notices

The application uses a pre-trained Logistic Regression model with TF-IDF feature extraction for fast and accurate predictions.

## ✨ Features

✅ **Real-time Message Classification** - Instant prediction of message categories  
✅ **Confidence Scores** - See how confident the model is about its prediction  
✅ **Beautiful UI** - Dark theme with glassmorphism design  
✅ **Text Analytics** - Analyze character, word, and sentence counts  
✅ **Confidence Visualization** - Interactive charts showing category probabilities  
✅ **Prediction History** - Track all predictions made during the session  
✅ **Export to CSV** - Download prediction history as a CSV file  
✅ **Status Indicators** - Real-time model and vectorizer status  
✅ **Responsive Design** - Works on desktop and mobile devices  
✅ **Error Handling** - Comprehensive error handling and user-friendly messages  

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python 3.8+
- **ML Model:** Logistic Regression
- **Vectorization:** TF-IDF
- **Visualization:** Plotly
- **Data Processing:** Pandas, NumPy
- **Serialization:** Pickle

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual Environment (recommended)

## 💻 Installation

### 1. Clone or Download the Project

```bash
# If cloning from git
git clone https://github.com/your-username/whatsapp-classifier.git
cd whatsapp_classifier
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Model Files

Ensure these files exist in the project root: