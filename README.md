# PhishGuard AI: SMS Phishing Detection

Project Live Link: https://sms-phishing-data-sec2024.streamlit.app/

**PhishGuard AI** is an innovative tool designed to combat SMS phishing (smishing) by leveraging machine learning and providing a real-time chatbot interface for phishing detection. The project aims to enhance user awareness and provide an effective solution for identifying phishing attempts in SMS messages.

## Features

- **Machine Learning-Powered Detection**: 
  - Utilizes Logistic Regression with TF-IDF vectorization for high accuracy (98.5%) in classifying SMS messages as phishing (spam) or legitimate (ham).
- **Interactive Chatbot**:
  - Built with Streamlit, the chatbot provides an intuitive and user-friendly interface for real-time SMS classification.
- **Simulation Environment**:
  - A controlled testing environment using Android Studio Emulator to replicate real-world smishing scenarios.
- **Educational Value**:
  - Educates users about common phishing tactics and promotes cybersecurity awareness.

## Technical Overview

- **Dataset**: SMS Spam Collection Dataset (5,572 messages; 13.4% spam, 86.6% ham).
- **Preprocessing**:
  - Text cleaning, tokenization, and TF-IDF vectorization.
- **Model**: Logistic Regression.
  - Achieved F1 Score: 95.5%, Precision: 96.3%, Recall: 94.7%.
- **Deployment**: 
  - Streamlit interface integrated with a pre-trained model (`spam_model.pkl` and `vectorizer.pkl`).
- **Simulation**:
  - Tested with Android Studio Emulator for realistic phishing scenarios.

## Tools and Libraries

- **Programming Language**: Python
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Deployment**: Streamlit
- **Simulation**: Android Studio Emulator

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/PhishGuard-AI.git
   cd PhishGuard-AI
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Chatbot**:
   ```bash
   streamlit run app.py
   ```
4. **Classify an SMS**:
   - Enter the SMS message in the chatbot interface.
   - View the classification as "Phishing" or "Legitimate."

## Future Enhancements

- Multilingual support for detecting phishing in non-English messages.
- Advanced deep learning models (e.g., transformers).
- Integration of multimedia phishing detection (e.g., URLs, QR codes).
- Mobile app for on-the-go phishing detection.
