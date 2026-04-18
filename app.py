import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="wide")

# ---------------- LAVENDER THEME CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #E6E6FA, #F8F0FF);
}

/* Title */
h1 {
    text-align: center;
    color: #4B0082;
    font-weight: 700;
}

/* Subheaders */
h2, h3 {
    color: #5D3FD3;
}

/* Input Box */
textarea {
    border-radius: 12px !important;
    border: 1px solid #C8A2C8 !important;
    background-color: #FFFFFF !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #B57EDC, #9370DB);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    transition: 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9370DB, #7B68EE);
}

/* Result Card */
.result-box {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #D8BFD8, #E6E6FA);
    color: #2c2c2c;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

/* Cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🧠 Sentiment & Emotion Analyzer</h1>", unsafe_allow_html=True)

# ---------------- NLTK FIX ----------------
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    train_df = pd.read_csv("training.csv")
    val_df = pd.read_csv("validation.csv")
    return train_df, val_df

train_df, val_df = load_data()

# ---------------- MAPPING ----------------
mapping = {
    0: "😢 sadness",
    1: "😄 joy",
    2: "❤️ love",
    3: "😡 anger",
    4: "😨 fear",
    5: "😲 surprise"
}

def emotion_to_sentiment(emotion):
    if emotion in ["joy", "love"]:
        return "positive"
    elif emotion in ["sadness", "anger", "fear"]:
        return "negative"
    else:
        return "neutral"

train_df["emotion_name"] = train_df["label"].map(mapping)
val_df["emotion_name"] = val_df["label"].map(mapping)

train_df["sentiment"] = train_df["emotion_name"].apply(lambda x: emotion_to_sentiment(x.split()[1]))
val_df["sentiment"] = val_df["emotion_name"].apply(lambda x: emotion_to_sentiment(x.split()[1]))

# ---------------- TEXT CLEANING ----------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

train_df["clean_text"] = train_df["text"].apply(clean_text)
val_df["clean_text"] = val_df["text"].apply(clean_text)

# ---------------- MODEL ----------------
@st.cache_resource
def train_model():
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = tfidf.fit_transform(train_df["clean_text"])

    emotion_model = LogisticRegression(max_iter=1000)
    emotion_model.fit(X_train, train_df["label"])

    sentiment_model = LogisticRegression(max_iter=1000)
    sentiment_model.fit(X_train, train_df["sentiment"])

    return tfidf, emotion_model, sentiment_model

tfidf, emotion_model, sentiment_model = train_model()

# ---------------- INPUT CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("💬 Enter your text")
user_input = st.text_area("Type something...", height=150)

if st.button("🚀 Analyze"):
    clean = clean_text(user_input)
    vec = tfidf.transform([clean])

    emotion_pred = emotion_model.predict(vec)[0]
    sentiment_pred = sentiment_model.predict(vec)[0]

    emotion_name = mapping[emotion_pred]

    st.markdown(f"""
    <div class="result-box">
        🎯 Emotion: {emotion_name} <br><br>
        📊 Sentiment: {sentiment_pred.upper()}
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- CHARTS ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Emotion Distribution")
    fig, ax = plt.subplots()
    train_df["label"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Sentiment Distribution")
    fig2, ax2 = plt.subplots()
    train_df["sentiment"].value_counts().plot(kind="bar", ax=ax2)
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)
