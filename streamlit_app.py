import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Load model and vectorizer
try:
    vectorizer = joblib.load("count_vectorizer.pkl")
    model = joblib.load("logistic_regression_model.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    words = word_tokenize(text)

    processed_words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word.lower() not in stop_words
    ]

    return ' '.join(processed_words)

# UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below and the model will predict whether it is Positive or Negative.")

user_input = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Type or paste your movie review here..."
)

if st.button("Analyze Sentiment"):

    if not user_input.strip():
        st.warning("Please enter a movie review.")
    else:
        try:
            processed_input = preprocess_text(user_input)

            input_vectorized = vectorizer.transform([processed_input])

            prediction = model.predict(input_vectorized)[0]

            if prediction.lower() == "positive":
                st.success("😊 Positive Review")
            elif prediction.lower() == "negative":
                st.error("😞 Negative Review")
            else:
                st.info(f"Prediction: {prediction}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# Footer
st.markdown("---")
st.caption("Built using Streamlit, NLTK, Scikit-learn, and Logistic Regression")
