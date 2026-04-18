import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix NLTK issue
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

st.title("🧠 Sentiment & Emotion Analyzer")

# Load data
@st.cache_data
def load_data():
    train_df = pd.read_csv("training.csv")
    val_df = pd.read_csv("validation.csv")
    return train_df, val_df

train_df, val_df = load_data()

# Mapping
mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
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

train_df["sentiment"] = train_df["emotion_name"].apply(emotion_to_sentiment)
val_df["sentiment"] = val_df["emotion_name"].apply(emotion_to_sentiment)

# Text cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

train_df["clean_text"] = train_df["text"].apply(clean_text)
val_df["clean_text"] = val_df["text"].apply(clean_text)

# Train model
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

# User input
user_input = st.text_area("Enter your text")

if st.button("Analyze"):
    clean = clean_text(user_input)
    vec = tfidf.transform([clean])

    emotion_pred = emotion_model.predict(vec)[0]
    sentiment_pred = sentiment_model.predict(vec)[0]

    st.subheader("Result")
    st.write("Emotion:", mapping[emotion_pred])
    st.write("Sentiment:", sentiment_pred)

# Charts
st.subheader("Emotion Distribution")
fig, ax = plt.subplots()
train_df["label"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)
