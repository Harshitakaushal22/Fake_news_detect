import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def predict_news(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean]).toarray()
    pred = model.predict(vec)
    return "Real" if pred[0] == 1 else "Fake"

# Streamlit app UI
st.title("Fake News Detection")

user_input = st.text_area("Enter the news article text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        result = predict_news(user_input)
        if result == "Fake":
            st.error("⚠️ This news article is predicted to be: FAKE")
        else:
            st.success("✅ This news article is predicted to be: REAL")
