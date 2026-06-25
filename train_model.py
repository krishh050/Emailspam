import os
import re
import json
import pickle
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Initialize directories
os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Robust NLTK setup
import nltk
NLTK_RESOURCES = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
for res in NLTK_RESOURCES:
    try:
        # Check if already available
        if res == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif res == 'stopwords':
            nltk.data.find('corpora/stopwords')
        elif res == 'wordnet':
            nltk.data.find('corpora/wordnet')
        else:
            nltk.data.find(f'corpora/{res}')
    except LookupError:
        try:
            print(f"Downloading NLTK resource: {res}...")
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Skipping download for '{res}' (will use local fallback if needed): {e}")

# Import NLTK components safely
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    STOPWORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    STOPWORDS = set()
    LEMMATIZER = None

# Custom Fallback stopword list (covering most English stopwords)
FALLBACK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
    "wouldn't"
}

def clean_and_tokenize(text):
    """
    Cleans email text and tokenizes it.
    If NLTK is not available or errors, uses a regex fallback.
    """
    if not isinstance(text, str):
        return []
    
    # Lowercase & strip HTML tag patterns
    text = text.lower()
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' [url] ', text)
    
    # Strip Email addresses
    text = re.sub(r'\S+@\S+', ' [email] ', text)
    
    # Strip numbers / digits (often high in spam but can skew basic models)
    text = re.sub(r'\b\d+\b', ' [number] ', text)
    
    tokens = []
    if HAS_NLTK:
        try:
            # Tokenize using NLTK
            words = word_tokenize(text)
            # Filter and Lemmatize
            for word in words:
                # Remove punctuation
                word = "".join([char for char in word if char not in string.punctuation])
                if word.strip():
                    if word not in STOPWORDS:
                        tokens.append(LEMMATIZER.lemmatize(word))
        except Exception:
            # Fallback to regex tokenizer
            tokens = regex_fallback_tokenizer(text)
    else:
        tokens = regex_fallback_tokenizer(text)
        
    return tokens

def regex_fallback_tokenizer(text):
    tokens = []
    words = re.findall(r'\b\w+\b', text)
    for w in words:
        if w not in FALLBACK_STOPWORDS:
            # Simple stemming rule (crude Porter approximation)
            if len(w) > 4:
                if w.endswith('ing'): w = w[:-3]
                elif w.endswith('ly'): w = w[:-2]
                elif w.endswith('ed'): w = w[:-2]
                elif w.endswith('es') and not w.endswith('ses'): w = w[:-2]
                elif w.endswith('s') and not w.endswith('ss'): w = w[:-1]
            tokens.append(w)
    return tokens

def preprocess_text(text):
    """Returns a joined preprocessed string for vectorization."""
    return " ".join(clean_and_tokenize(text))

# List of typical spam keywords to highlight/analyse
SPAM_KEYWORDS = {
    'free', 'winner', 'win', 'prize', 'claim', 'urgent', 'cash', 'money', 'credit', 'card', 
    'offer', 'guaranteed', 'billions', 'dollars', 'investment', 'selected', 'reply', 'stop', 
    'unsubscribed', 'restricted', 'suspended', 'account', 'verify', 'password', 'click', 
    'link', 'inherit', 'estate', 'lotto', 'lottery', 'crypto', 'bitcoin', 'loan', 'refund',
    'congratulations', 'claims', 'special', 'gift', 'award', 'exclusive', 'dear friend', 'earn'
}

def generate_synthetic_dataset():
    """Generates a high-quality synthetic dataset if online sources fail."""
    print("Generating fallback synthetic spam/ham dataset...")
    ham_templates = [
        "Hi Team, just checking in on the status of the project. Can we meet at 3 PM today?",
        "Hey, are you free for lunch tomorrow? Let me know.",
        "Please find attached the quarterly financial reports for your review. Let me know if you have questions.",
        "Your appointment with Dr. Smith is confirmed for Thursday at 10:00 AM.",
        "Thanks for the update. I will review the document and send you my feedback by end of day.",
        "Can you send me the slide deck for tomorrow's presentation? Thanks!",
        "Great job on the release yesterday! The client was very pleased with the new features.",
        "Hey! Don't forget we have family dinner this Sunday. See you there.",
        "Could you please sign and return the NDA as soon as possible?",
        "Hi, I'm running 5 minutes late for our sync, please start without me.",
        "Let's schedule a call next week to discuss the product roadmap.",
        "Hi, your package from Amazon has been delivered to the front desk.",
        "Reviewing the resume you forwarded. They seem like a solid candidate.",
        "Can we push the meeting to tomorrow morning? Let me know what works.",
        "Happy Birthday! Hope you have a wonderful day ahead."
    ]
    
    spam_templates = [
        "CONGRATULATIONS! You have been selected as the WINNER of our weekly $1,000,000 cash prize! Claim now!",
        "URGENT: Your account has been suspended due to suspicious activity. Verify your password here.",
        "Get rich quick! Earn up to $5000 a day working from home. No experience needed. Click link!",
        "Special offer: Buy one get one free on all brand name prescription medications. Guaranteed delivery.",
        "Dear Friend, I am a barrister and I have an inheritance of $10.5 million waiting for you. Reply urgently.",
        "Win a brand new iPhone 15! Complete this short survey to claim your exclusive gift card reward now.",
        "Guaranteed loan approval with low interest rates. No credit check required. Apply within 24 hours.",
        "Claim your free spins and bonus match up to $500 at the best online crypto casino today!",
        "Unsubscribed? To stop receiving these promotional notifications, click here to unsubscribe.",
        "Refund notification: You have an outstanding tax refund of $450. Click link to verify bank account.",
        "Double your bitcoin in 24 hours! Safe and secure cryptocurrency investment plans starting now.",
        "Congratulations! You won a free holiday trip to Hawaii. Click here to claim your tickets.",
        "Final Warning: Your credit card payment is overdue. Pay immediately to avoid legal actions.",
        "Exclusive deal: Save 80% on anti-aging skin products. Limited stock available, order today!",
        "Access your funds now. Fill out the attached document to release your wire transfer."
    ]
    
    # Replicate template data to create a decent-sized dataset (~300 rows)
    rows = []
    for _ in range(15):
        for ham in ham_templates:
            rows.append({'label': 'ham', 'text': ham})
        for spam in spam_templates:
            rows.append({'label': 'spam', 'text': spam})
            
    # Add a bit of random variation to avoid identical rows
    df = pd.DataFrame(rows)
    df['text'] = df['text'] + df.index.map(lambda i: f" (Ref: #{i:04d})" if i % 3 == 0 else "")
    return df

def download_and_load_dataset():
    """Downloads the spam dataset from public URLs. If both fail, uses fallback data."""
    csv_path = os.path.join("dataset", "spam.csv")
    
    # URLs to try
    url_stedy = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/sms_spam.csv"
    url_mohit = "https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
    
    # Try downloading from stedy first (standard clean format)
    try:
        print(f"Attempting to download from primary source: {url_stedy}")
        urllib.request.urlretrieve(url_stedy, csv_path)
        df = pd.read_csv(csv_path)
        if 'type' in df.columns and 'text' in df.columns:
            df = df.rename(columns={'type': 'label'})
            df.to_csv(csv_path, index=False)
            print("Successfully downloaded and normalized primary dataset.")
            return df
    except Exception as e:
        print(f"Failed to fetch primary dataset: {e}")
        
    # Try downloading from mohitgupta-1O1 (alternative SMS collection)
    try:
        print(f"Attempting to download from secondary source: {url_mohit}")
        urllib.request.urlretrieve(url_mohit, csv_path)
        # It has latin-1 encoding and some unnamed columns
        df = pd.read_csv(csv_path, encoding='latin-1')
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
            df.to_csv(csv_path, index=False)
            print("Successfully downloaded and normalized secondary dataset.")
            return df
    except Exception as e:
        print(f"Failed to fetch secondary dataset: {e}")
        
    # Fallback to synthetic if offline/downloads fail
    df = generate_synthetic_dataset()
    df.to_csv(csv_path, index=False)
    return df

def train_model(max_features=5000, alpha=1.0):
    """
    Trains TF-IDF + Naive Bayes Classifier on the spam dataset.
    Saves model, vectorizer, and metrics.
    """
    print("Loading dataset...")
    if os.path.exists(os.path.join("dataset", "spam.csv")):
        try:
            df = pd.read_csv(os.path.join("dataset", "spam.csv"))
            # Make sure it contains required columns
            if not ('label' in df.columns and 'text' in df.columns):
                raise ValueError("CSV column format mismatch")
        except Exception:
            df = download_and_load_dataset()
    else:
        df = download_and_load_dataset()
        
    print(f"Dataset stats: {len(df)} samples | Spam: {sum(df['label'] == 'spam')} | Ham: {sum(df['label'] == 'ham')}")
    
    # Preprocess corpus
    print("Preprocessing text corpus (this might take a few seconds)...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Filter empty rows after preprocessing
    df = df[df['cleaned_text'].str.strip() != ""]
    
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_text'])
    
    # Map label: ham -> 0, spam -> 1
    y = df['label'].map({'ham': 0, 'spam': 1}).values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit Classifier
    classifier = MultinomialNB(alpha=alpha)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    
    cm = metrics.confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON serialization
    
    # Save artifacts
    print("Saving trained model assets...")
    with open(os.path.join("models", "model.pkl"), "wb") as f:
        pickle.dump(classifier, f)
    with open(os.path.join("models", "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
        
    # Compile performance report
    perf_metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "total_spam": int(sum(df['label'] == 'spam')),
        "total_ham": int(sum(df['label'] == 'ham')),
        "hyperparameters": {
            "max_features": max_features,
            "alpha": alpha
        }
    }
    
    with open(os.path.join("models", "metrics.json"), "w") as f:
        json.dump(perf_metrics, f, indent=4)
        
    print("Training successfully complete!")
    print(f"Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")
    return perf_metrics

if __name__ == "__main__":
    train_model()
