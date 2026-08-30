import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Sample Dataset of Emails (Text & Labels)
data = {
    'email_text': [
        "URGENT: Your account has been suspended! Click here to verify your identity.",
        "Congratulations! You won a $1000 gift card. Claim your reward immediately.",
        "Security Alert: Unusual login attempt detected. Reset your password now at http://fake-bank-login.com",
        "Dear customer, update your billing details right away to avoid service cancellation.",
        "Verify your banking details within 24 hours to prevent account closure.",
        "Hey, are we still meeting for lunch today at the cafeteria?",
        "Please find the quarterly financial report attached for your review.",
        "Team meeting is rescheduled to 3 PM tomorrow. See you in the conference room.",
        "Thanks for sending the project files. I will review them and get back to you.",
        "Can you send me the lecture notes for today's computer science class?"
    ],
    'label': [
        'Phishing', 'Phishing', 'Phishing', 'Phishing', 'Phishing',
        'Safe', 'Safe', 'Safe', 'Safe', 'Safe'
    ]
}

# 2. DataFrame and Split
df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(
    df['email_text'], df['label'], test_size=0.3, random_state=42
)

# 3. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Model (Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Predictions & Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=['Safe', 'Phishing'])

print("=" * 55)
print("      PHISHING EMAIL DETECTION MODEL EVALUATION")
print("=" * 55)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print("                 Predicted Safe  Predicted Phishing")
print(f"Actual Safe:           {conf_matrix[0][0]:<15} {conf_matrix[0][1]}")
print(f"Actual Phishing:       {conf_matrix[1][0]:<15} {conf_matrix[1][1]}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Real-Time Classifier Function
def predict_email(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction

print("=" * 55)
print("                  TEST SAMPLE EMAILS")
print("=" * 55)
sample_1 = "Claim your lottery prize by clicking this link immediately!"
sample_2 = "Hey, let's submit the project report by Friday."

print(f"Email: '{sample_1}'\n-> Result: {predict_email(sample_1)}\n")
print(f"Email: '{sample_2}'\n-> Result: {predict_email(sample_2)}")
print("=" * 55)
