import os
import re
import json
import pickle
import datetime
import string
import pandas as pd

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from train_model import train_model, clean_and_tokenize, preprocess_text, SPAM_KEYWORDS

# Set page config with custom title and dark theme settings
st.set_page_config(
    page_title="Spam Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom Stylesheet
def load_css(css_file):
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# --- Helper Functions ---
@st.cache_resource
def load_classifier_assets():
    """Loads the classifier, vectorizer, and metrics. Retrains if missing."""
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    metrics_path = os.path.join("models", "metrics.json")
    
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        with st.spinner("Compiling community model..."):
            train_model()
            
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
        
    return model, vectorizer, metrics_data

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "input_email" not in st.session_state:
    st.session_state.input_email = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Load model assets
try:
    model, vectorizer, metrics_data = load_classifier_assets()
except Exception as e:
    st.error(f"Failed to load or compile model: {e}")
    model, vectorizer, metrics_data = None, None, None

# --- Custom Sidebar ---
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #30363d; margin-bottom: 20px;">
        <h3 style="margin: 0; color: #ffffff; font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace;">
            <span style="background: #14599f; width: 8px; height: 16px; border-radius: 2px; display: inline-block;"></span>
            Spam Scanner
        </h3>
        <span style="color: #ff9800; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace;">CLASSIFICATION ENGINE</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<div style='font-size: 11px; font-weight: 600; color: #8b949e; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px; padding-left: 10px; font-family: \"JetBrains Mono\", monospace;'>Workspace</div>", unsafe_allow_html=True)

nav_options = ["🛡️ Spam Detector", "📊 Diagnostics"]
selected_page = st.sidebar.radio("Navigation", nav_options, label_visibility="collapsed")

st.sidebar.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size: 11px; font-weight: 600; color: #8b949e; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px; padding-left: 10px; font-family: \"JetBrains Mono\", monospace;'>Model Specs</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"""
    <div style="background: rgba(255,255,255,0.01); border: 1px solid #30363d; border-radius: 4px; padding: 12px; font-size: 12px; font-family: 'JetBrains Mono', monospace;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #8b949e;">Accuracy:</span>
            <span style="color: #00c853; font-weight: 700;">{metrics_data['accuracy']*100:.2f}%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #8b949e;">F1 Index:</span>
            <span style="color: #ffffff;">{metrics_data['f1_score']*100:.2f}%</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #8b949e;">Vocab Size:</span>
            <span style="color: #ffffff;">{metrics_data['hyperparameters']['max_features']} words</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="position: fixed; bottom: 20px; font-size: 10px; color: #8b949e; padding-left: 10px; font-family: 'JetBrains Mono', monospace;">
        Press <kbd style="background: #21262d; border: 1px solid #30363d; border-radius: 3px; padding: 1px 4px; font-family: monospace;">R</kbd> to compile page
    </div>
    """,
    unsafe_allow_html=True
)

# --- 1. EMAIL SCANNER PAGE (HOME) ---
if selected_page == "🛡️ Spam Detector":
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="hero-tagline" style="font-size: 32px !important; text-align: left; margin-bottom: 6px;"><span class="text-blue">Spam Scanner</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtext" style="font-size: 13px !important; text-align: left; margin-left: 0; margin-bottom: 24px; max-width: 100%; font-family: \'JetBrains Mono\', monospace;">Type text or upload files to scan for spam risks and inspect threat indicators in real-time.</div>', unsafe_allow_html=True)
    
    # Pre-configured templates map
    templates = {
        "--- Select a pre-configured template to test ---": "",
        "Ham: Meeting Invitation": "Hi David,\n\nAre you available for a brief sync tomorrow at 10 AM to discuss the product roadmap and user research findings? Let me know if that works for you.\n\nBest,\nSarah",
        "Ham: Dinner Plans": "Hey buddy, don't forget we have dinner scheduled for this Friday at 7 PM. Let me know if you can make it, and let me know if we need to make reservations.\n\nCheers,\nJohn",
        "Spam: Mega Jackpot Winner": "CONGRATULATIONS! You have been selected as the WINNER of our weekly $500,000 cash sweepstakes. Claim your prize now by verifying your card details at http://verifiedprize.net/claim. Reply urgently to secure your winnings.",
        "Spam: Urgent Bank Reset Alert": "URGENT ACCOUNT VERIFICATION: Your online bank profile has been temporarily locked due to suspicious login attempts. To restore access and verify your password, click here: http://secure-verify-account.com/auth. Failure to verify within 24 hours will result in permanent account suspension."
    }
    
    # Inputs Area Row (Templates + Upload)
    col_input1, col_input2 = st.columns([1, 1])
    
    with col_input1:
        selected_tpl = st.selectbox(
            "Load Sample Template Email", 
            options=list(templates.keys()),
            help="Select a standard ham or spam template to prefill the text field."
        )
        if selected_tpl != "--- Select a pre-configured template to test ---":
            st.session_state.input_email = templates[selected_tpl]
            st.rerun()
            
    with col_input2:
        uploaded_file = st.file_uploader(
            "Or upload a .txt email file", 
            type=["txt"], 
            help="Drag and drop or upload a plain text file."
        )
        if uploaded_file is not None:
            try:
                string_data = uploaded_file.getvalue().decode("utf-8")
                st.session_state.input_email = string_data
                st.toast("Loaded email text from file!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Email text area
    email_text = st.text_area(
        "Email content to check", 
        value=st.session_state.input_email, 
        placeholder="Paste your email text here...",
        height=180,
        key="main_email_input"
    )
    
    # Run Actions
    col_btns1, col_btns2, col_btns3 = st.columns([1, 1, 4])
    with col_btns1:
        analyze_btn = st.button("Scan Email", key="scan_btn")
    with col_btns2:
        clear_btn = st.button("Clear Text", key="clear_btn")
        if clear_btn:
            st.session_state.input_email = ""
            st.session_state.analysis_result = None
            st.rerun()
            
    # Analyzer Execution
    if analyze_btn and email_text.strip():
        if model is None or vectorizer is None:
            st.error("Engine failed to load assets. Try retraining under Diagnostics.")
        else:
            with st.spinner("Analyzing email..."):
                cleaned = preprocess_text(email_text)
                if not cleaned.strip():
                    cleaned = "empty"
                
                # Predict
                vec_text = vectorizer.transform([cleaned])
                pred = model.predict(vec_text)[0]
                proba = model.predict_proba(vec_text)[0]
                
                spam_prob = proba[1]
                risk_score = int(spam_prob * 100)
                classification = "Spam" if pred == 1 else "Not Spam"
                
                # Word checks
                original_words = re.findall(r'\b\w+\b', email_text)
                suspicious_matches = []
                for w in original_words:
                    clean_w = w.lower()
                    matched = False
                    if clean_w in SPAM_KEYWORDS:
                        matched = True
                    else:
                        for stem in SPAM_KEYWORDS:
                            if len(stem) > 3 and clean_w.startswith(stem) and len(clean_w) - len(stem) <= 3:
                                matched = True
                                break
                    if matched:
                        suspicious_matches.append(w)
                
                # HTML Highlights
                safe_original = email_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                highlighted_html = safe_original
                for match in sorted(list(set(suspicious_matches)), key=len, reverse=True):
                    highlighted_html = re.sub(
                        r'\b(' + re.escape(match) + r')\b',
                        r'<span class="spam-highlight">\1</span>',
                        highlighted_html,
                        flags=re.IGNORECASE
                    )
                
                # Store
                st.session_state.analysis_result = {
                    "text": email_text,
                    "cleaned": cleaned,
                    "classification": classification,
                    "spam_probability": spam_prob,
                    "risk_score": risk_score,
                    "highlighted_html": highlighted_html,
                    "suspicious_matches": list(set([m.lower() for m in suspicious_matches]))
                }
                
                # Append session history
                new_log = {
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Text Preview": email_text[:55] + "..." if len(email_text) > 55 else email_text,
                    "Result": classification,
                    "Risk Score": risk_score,
                    "Spam Prob (%)": f"{spam_prob*100:.1f}%"
                }
                st.session_state.history.append(new_log)
                st.toast("Scan complete!")
                st.rerun()

    # Results Rendering
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3>Scan Findings</h3>", unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            badge_html = ""
            if res["classification"] == "Spam":
                badge_html = '<div class="spam-badge"><span class="badge-dot"></span>CRITICAL RISK: SPAM</div>'
            else:
                badge_html = '<div class="ham-badge"><span class="badge-dot"></span>SECURE: NOT SPAM</div>'
                
            st.markdown(
                f"""
                <div class="linear-card linear-card-glow">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <span style="font-size: 13px; color: #8b949e; font-weight: 500; font-family: 'JetBrains Mono', monospace;">Verdict</span>
                        {badge_html}
                    </div>
                    <div style="margin-bottom: 24px;">
                        <div class="metric-title">Risk Index Score</div>
                        <div style="display: flex; align-items: baseline; gap: 8px;">
                            <span class="metric-value" style="font-size: 40px; color: {'#ff9800' if res['classification']=='Spam' else '#00c853'};">{res['risk_score']}</span>
                            <span style="color: #484f58; font-size: 14px; font-family: 'JetBrains Mono', monospace;">/ 100</span>
                        </div>
                    </div>
                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace;">
                            <span style="color: #8b949e;">Spam Probability Confidence:</span>
                            <span style="color: #ffffff; font-weight: 700;">{res['spam_probability']*100:.1f}%</span>
                        </div>
                        <div style="background-color: #21262d; height: 6px; border-radius: 3px; overflow: hidden;">
                            <div style="background-color: {'#ff9800' if res['classification']=='Spam' else '#00c853'}; width: {res['risk_score']}%; height: 100%;"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_res2:
            if res['suspicious_matches']:
                trigger_badges = " ".join([
                    f'<span style="background: rgba(255,152,0,0.12); border: 1px dashed rgba(255,152,0,0.3); color: #ff9800; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600; font-family: monospace;">{m}</span>' 
                    for m in res['suspicious_matches']
                ])
            else:
                trigger_badges = '<span style="color: #484f58; font-size: 12px; font-family: monospace;">No high-risk word markers found.</span>'
                
            st.markdown(
                f"""
                <div class="linear-card" style="height: 100%;">
                    <div style="font-size: 13px; color: #8b949e; font-weight: 500; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace;">Trigger Word Flags ({len(res['suspicious_matches'])})</div>
                    <p style="font-size: 12px; color: #8b949e; margin-bottom: 16px; font-family: 'JetBrains Mono', monospace;">We found these suspicious terms linked heavily to phishing templates:</p>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        {trigger_badges}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            
        # Text highlight display
        st.markdown("<div style='margin-bottom: 8px; font-size: 13px; color: #8b949e; font-weight: 500; font-family: \"JetBrains Mono\", monospace;'>Flag Highlight layer</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="highlight-container">{res["highlighted_html"]}</div>', unsafe_allow_html=True)
        
        # Collapsible technical drawer for advanced users
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        with st.expander("🛠️ Advanced NLP Analysis (Technical Details)"):
            st.markdown("<h5 style='margin-top: 10px;'>Preprocessing Pipeline Transformation</h5>", unsafe_allow_html=True)
            
            # Simple flow diagram
            st.markdown(
                f"""
                <div class="nlp-flow-container" style="margin-bottom: 20px;">
                    <div class="nlp-step-card"><b>Input</b><br><span style="font-size:10px; color:#8b949e;">Raw String</span></div>
                    <div class="nlp-arrow">→</div>
                    <div class="nlp-step-card"><b>Normalization</b><br><span style="font-size:10px; color:#8b949e;">Regex filters</span></div>
                    <div class="nlp-arrow">→</div>
                    <div class="nlp-step-card"><b>Stopwords</b><br><span style="font-size:10px; color:#8b949e;">Noise removed</span></div>
                    <div class="nlp-arrow">→</div>
                    <div class="nlp-step-card"><b>Lemmatization</b><br><span style="font-size:10px; color:#8b949e;">Stemmed tokens</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            col_ad1, col_ad2 = st.columns([1, 1])
            with col_ad1:
                st.markdown("**Normalized Stems (Preprocessed Tokens):**")
                st.code(res["cleaned"] if res["cleaned"] else "[All words removed as grammar stopwords]")
                
                # Doc metrics
                word_c = len(res["text"].split())
                char_c = len(res["text"])
                link_c = len(re.findall(r'https?://\S+|www\.\S+', res["text"]))
                spec_c = len([c for c in res["text"] if c in string.punctuation])
                
                st.markdown(
                    f"""
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 15px; font-family: 'JetBrains Mono', monospace;">
                        <div style="background: rgba(255,255,255,0.01); border: 1px solid #30363d; padding: 8px; border-radius: 4px;">
                            <span style="font-size:10px; color:#8b949e;">Words</span><br><b>{word_c}</b>
                        </div>
                        <div style="background: rgba(255,255,255,0.01); border: 1px solid #30363d; padding: 8px; border-radius: 4px;">
                            <span style="font-size:10px; color:#8b949e;">Characters</span><br><b>{char_c}</b>
                        </div>
                        <div style="background: rgba(255,255,255,0.01); border: 1px solid #30363d; padding: 8px; border-radius: 4px;">
                            <span style="font-size:10px; color:#8b949e;">Hyperlinks</span><br><b style="color: {'#ff9800' if link_c>0 else '#ffffff'}">{link_c}</b>
                        </div>
                        <div style="background: rgba(255,255,255,0.01); border: 1px solid #30363d; padding: 8px; border-radius: 4px;">
                            <span style="font-size:10px; color:#8b949e;">Special Characters</span><br><b>{spec_c}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with col_ad2:
                st.markdown("**TF-IDF Weights (Tokens Impact):**")
                tf_idf_vector = vectorizer.transform([res["cleaned"]])
                feature_names = vectorizer.get_feature_names_out()
                dense = tf_idf_vector.todense().tolist()[0]
                phrase_weights = []
                for i, val in enumerate(dense):
                    if val > 0:
                        phrase_weights.append({"Token": feature_names[i], "Weight": val})
                        
                if phrase_weights:
                    df_weights = pd.DataFrame(phrase_weights).sort_values(by="Weight", ascending=False).reset_index(drop=True)
                    
                    fig = px.bar(
                        df_weights.head(6),
                        x="Weight",
                        y="Token",
                        orientation="h",
                        color_discrete_sequence=["#14599f"],
                        height=160
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8b949e",
                        margin=dict(l=0, r=0, t=5, b=5),
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No vocabulary words matched.")
                    
    # Feature Grid collapsed inside expander
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    with st.expander("ℹ️ About Spam Scanner & Key Features"):
        st.markdown(
            """
            <div class="feature-grid" style="margin-top: 10px;">
                <div class="feature-card">
                    <div class="feature-icon">🛡️</div>
                    <div class="feature-title">Naive Bayes Classifier</div>
                    <div class="feature-desc">Utilizes multinomial distribution probabilities to evaluate and score incoming email content.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">Microsecond Inference</div>
                    <div class="feature-desc">Engineered for real-time analysis, compiling word weights in milliseconds.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚙️</div>
                    <div class="feature-title">Tuneable Brain</div>
                    <div class="feature-desc">Re-compile vocabulary sizes, modify Alpha Laplace values, and recalibrate models directly under Campus Diagnostics.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- 2. CAMPUS DIAGNOSTICS PAGE ---
elif selected_page == "📊 Diagnostics":
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2>Diagnostics & Metrics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8b949e; font-size: 14px; margin-bottom: 30px; font-family: \"JetBrains Mono\", monospace;'>Evaluate classifier accuracy indices, dataset token distributions, and audit scan logs.</p>", unsafe_allow_html=True)
    
    # Model evaluation metrics row
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 24px;">
            <div class="linear-card" style="padding: 16px;">
                <div class="metric-title">Validation Accuracy</div>
                <div class="metric-value">{metrics_data['accuracy']*100:.2f}%</div>
                <div class="metric-delta" style="color: #00c853;">↑ Naive Bayes Engine</div>
            </div>
            <div class="linear-card" style="padding: 16px;">
                <div class="metric-title">Precision</div>
                <div class="metric-value">{metrics_data['precision']*100:.2f}%</div>
                <div class="metric-delta" style="color: #00c853;">Correctly identified spam</div>
            </div>
            <div class="linear-card" style="padding: 16px;">
                <div class="metric-title">Recall</div>
                <div class="metric-value">{metrics_data['recall']*100:.2f}%</div>
                <div class="metric-delta" style="color: #00c853;">Total spam caught</div>
            </div>
            <div class="linear-card" style="padding: 16px;">
                <div class="metric-title">F1 Index</div>
                <div class="metric-value">{metrics_data['f1_score']*100:.2f}%</div>
                <div class="metric-delta" style="color: #00c853;">Weighted harmonic accuracy</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_db1, col_db2 = st.columns([1, 1])
    
    with col_db1:
        st.markdown("<h4>Corpus Classification Breakdown</h4>", unsafe_allow_html=True)
        total_spam = metrics_data.get('total_spam', 747)
        total_ham = metrics_data.get('total_ham', 4827)
        
        labels = ['Ham (Normal)', 'Spam']
        values = [total_ham, total_spam]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.5,
            marker_colors=['#21262d', '#14599f'],
            textinfo='percent+value'
        )])
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8b949e",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_db2:
        st.markdown("<h4>Confusion Matrix (Validation split)</h4>", unsafe_allow_html=True)
        cm = metrics_data['confusion_matrix']
        x_lbl = ['Pred Ham', 'Pred Spam']
        y_lbl = ['True Ham', 'True Spam']
        
        fig_hm = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="True Label", color="Samples"),
            x=x_lbl,
            y=y_lbl,
            color_continuous_scale=[[0, '#161b22'], [0.5, '#14599f'], [1, '#1f75cb']],
            text_auto=True
        )
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8b949e",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        
    # Session Log History
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4>Scan History Logs</h4>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No scans executed in current session.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        
        # Download buttons
        col_ex1, col_ex2 = st.columns([1, 4])
        with col_ex1:
            csv_data = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button("Export CSV Log", data=csv_data, file_name="spam_scanner_logs.csv", mime="text/csv")
        with col_ex2:
            json_data = json.dumps(st.session_state.history, indent=4)
            st.download_button("Export JSON Log", data=json_data, file_name="spam_scanner_logs.json", mime="application/json")
            
    # Tuning Panel Expander
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    with st.expander("⚙️ Adjust Parameters & Retrain Model"):
        st.markdown("<h5 style='margin-top: 5px;'>Adjust Model Parameters</h5>", unsafe_allow_html=True)
        
        csv_exists = os.path.exists(os.path.join("dataset", "spam.csv"))
        if csv_exists:
            df_full = pd.read_csv(os.path.join("dataset", "spam.csv"))
            st.markdown(f"**Loaded Dataset**: `dataset/spam.csv` | **Entries**: {len(df_full)} emails (Spam ratio: {sum(df_full['label']=='spam')/len(df_full)*100:.1f}%)")
        else:
            st.warning("Database CSV file not detected in project folder.")
            
        curr_features = metrics_data['hyperparameters']['max_features'] if metrics_data else 5000
        curr_alpha = metrics_data['hyperparameters']['alpha'] if metrics_data else 1.0
        
        max_features = st.slider("Max Vocabulary Features size", min_value=500, max_value=10000, value=int(curr_features), step=500)
        alpha = st.slider("NB Laplace Smoothing (Alpha)", min_value=0.1, max_value=10.0, value=float(curr_alpha), step=0.1)
        
        retrain_btn = st.button("Re-compile Classifier")
        if retrain_btn:
            with st.spinner("Re-compiling Naive Bayes Model..."):
                try:
                    new_metrics = train_model(max_features=max_features, alpha=alpha)
                    load_classifier_assets.clear()
                    model, vectorizer, metrics_data = load_classifier_assets()
                    st.toast("Model updated successfully!")
                    st.success("Trained successfully! Metrics reloaded.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Retraining error: {e}")
                    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
