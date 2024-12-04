# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 2: Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Step 3: Remove unnecessary columns
df = df[['v1', 'v2']]  # Keep only the 'v1' (label) and 'v2' (message) columns

# Optional: Inspect the first few rows of the dataset
print(df.head())

# Step 4: Preprocessing the text data
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Apply text cleaning
df['cleaned_text'] = df['v2'].apply(clean_text)

# Step 5: Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])

# Step 6: Encode the labels (Spam = 1, Ham = 0)
y = np.where(df['v1'] == 'spam', 1, 0)

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Save the trained model and vectorizer as .pkl files
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved as spam_model.pkl and vectorizer.pkl")