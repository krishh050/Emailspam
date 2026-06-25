# Linear-Inspired Email Spam Detection System 🛡️

A production-grade, premium email risk scanning and classifier web application built in Python using Scikit-Learn (Multinomial Naive Bayes), Natural Language Processing (NLP), and Streamlit. The user interface mimics the visual aesthetic, layouts, typography, and dark-theme style of the Linear application (`linear.app`).

---

## Features

1. **Linear Landing Page**: Clean, high-impact dark hero layout with active grids showcasing the core engine capabilities.
2. **Email Risk Profiler**:
   - Real-time classification badge (`SPAM` or `SECURE`).
   - Risk Index Meter scaled from `0-100` and percentage confidence scores.
   - Flag word visual highlighter (highlights threat indicators in a custom red glow container).
   - Drag-and-drop support for `.txt` files.
   - Preset templates for instant sandbox testing.
3. **NLP Preprocessing Inspection**:
   - Interactive, step-by-step breakdown tracing raw text transformation: Normalization → Tokenization → Stopwords Filtering → Lemmatization → TF-IDF Weights.
   - Interactive TF-IDF weights chart and token tables.
   - Document stats: word count, character count, links, and special character signatures.
4. **Performance Dashboard**:
   - Sleek widget cards displaying Validation Accuracy, Precision, Recall, and F1 Score.
   - Interactive Plotly figures (Pie distribution of Dataset, Confusion Matrix Heatmap).
   - Live Session Audit log with single-click exports to CSV or JSON formats.
5. **Model Controls (Settings)**:
   - Live dataset metrics (total entries, spam/ham samples, ratio).
   - Real-time hyperparameter adjustments: adjust the TF-IDF feature space size and Naive Bayes smoothing value (alpha) with instant retraining capability.

---

## Directory Structure

```
email-spam-detector/
│
├── dataset/
│   └── spam.csv              # Downloaded/generated training data
├── models/
│   ├── model.pkl             # Serialized Naive Bayes classifier
│   ├── vectorizer.pkl        # Serialized TF-IDF vectorizer
│   └── metrics.json          # Compiled evaluation metrics
├── app.py                    # Core Streamlit web application
├── train_model.py            # Model downloader & training script
├── styles.css                # Custom CSS styling (Linear-inspired dark theme)
├── requirements.txt          # Python library dependencies
└── README.md                 # User guide (this file)
```

---

## Installation & Setup

Follow these steps to run the application on your local machine.

### Prerequisites
Make sure you have **Python 3.9+** and `pip` installed.

### 1. Clone or Move to Workspace
Open a terminal in the folder containing these files:
```bash
cd "c:\Users\FATHIMA RIFDA\Desktop\email"
```

### 2. Install Dependencies
Run the command to install packages specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Run Training Pipeline
The app will automatically train the model on its first launch if model files are missing. However, to run the training script manually and inspect evaluation outputs:
```bash
python train_model.py
```
This script will:
- Download the SMS Spam Collection dataset (`dataset/spam.csv`). If offline, it generates a high-quality fallback corpus of ~300 entries.
- Download the required NLTK corpuses (`punkt`, `stopwords`, `wordnet`) with robust fallbacks.
- Preprocess the text (clean, tokenize, remove stopwords, lemmatize).
- Train a Multinomial Naive Bayes classifier.
- Save serialization files to the `models/` directory.

### 4. Start Web Application
Run the Streamlit application using:
```bash
streamlit run app.py
```
This command starts a local server. A browser window should open automatically at `http://localhost:8501`.

---

## Technology Stack
- **Frontend**: Streamlit, Custom CSS (Linear-style glassmorphism & overlays)
- **Machine Learning**: Scikit-Learn (TF-IDF Vectorizer + Multinomial Naive Bayes Classifier)
- **Data Engineering**: Pandas, NumPy, Regular Expressions
- **NLP**: NLTK (Tokenization, Stopwords filtering, WordNet Lemmatization)
- **Visuals**: Plotly Express, Plotly Graph Objects (Themed in purple/charcoal)
