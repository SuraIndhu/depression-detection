import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Use the same cleaning logic from your notebook
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Streamlit UI
st.title("Depression Detection Analysis")
st.write("Enter text below to predict the sentiment.")

user_input = st.text_area("Enter tweet or post text:")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        
        if prediction[0] == 1:
            st.warning("Prediction: Potential Depressive Content")
        else:
            st.success("Prediction: Non-Depressive Content")
    else:
        st.write("Please enter some text.")