import streamlit as st
import pickle
import re
import string
import numpy as np

# Step 1: Load the trained model and vectorizer from .pkl files
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Step 2: Function to clean the input text (same as the notebook's clean_text function)
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    return text

# Step 3: Set up the Streamlit UI
st.title("Spam or Ham Message Classifier")

st.write("""
This is a simple spam detection app. 
Enter a message below, and the model will classify it as either **Spam** or **Ham** (not Spam).
""")

# Step 4: Create an input field for the user to enter a message
user_input = st.text_area("Enter the message:")

# Step 5: Predict if the message is spam or ham
if user_input:
    cleaned_input = clean_text(user_input)  
    vectorized_input = vectorizer.transform([cleaned_input])  

    prediction = model.predict(vectorized_input)
    
    if prediction == 1:
        st.write("**Prediction: Spam**")
    else:
        st.write("**Prediction: Ham**")