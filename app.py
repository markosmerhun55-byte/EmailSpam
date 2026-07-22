import streamlit as st
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Page Title & Layout
st.set_page_config(page_title="Email Spam Detector", page_icon="📩", layout="centered")

MODEL_FILE = "best_sms_spam_model.joblib"

@st.cache_resource
def load_or_train_model():
    """Loads saved model or automatically trains and saves one on startup."""
    try:
        pipeline = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        st.info("First-time setup: Training the Linear SVM model...")
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
            ('classifier', LinearSVC(C=1.0, random_state=42))
        ])
        pipeline.fit(df['message'], df['label_num'])
        joblib.dump(pipeline, MODEL_FILE)
    return pipeline

# Load model
pipeline = load_or_train_model()

# User Interface Header
st.title("📩 Email Spam Classifier")
st.write("Type or paste an Email text message below to classify it in real-time.")

# User Input Text Area
user_input = st.text_area("Enter Email Message:", height=100, placeholder="e.g. Bantayehu Sileshi...")

# Predict Button
if st.button("Classify Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message to analyze.")
    else:
        prediction = pipeline.predict([user_input])[0]
        score = pipeline.decision_function([user_input])[0]
        
        st.divider()
        if prediction == 1:
            st.error("🚨 **Prediction: SPAM**")
            st.write(f"**Confidence Score:** `{score:.2f}` (Positive score indicates spam)")
        else:
            st.success("✅ **Prediction: HAM (Legitimate)**")
            st.write(f"**Confidence Score:** `{score:.2f}` (Negative score indicates ham)")

# --- FOOTER ---
st.divider()
st.caption("© 2026 Email Spam Detector | Powered by Merhun Markos")
