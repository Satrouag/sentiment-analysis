import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load IMDb dataset from local file
print("Loading dataset...")
df = pd.read_csv("IMDB Dataset.csv")

# Preprocessing function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)

# Apply text cleaning
print("Cleaning text data...")
df["cleaned_review"] = df["review"].apply(clean_text)

# Convert sentiment labels to binary (Positive → 1, Negative → 0)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_review"], df["sentiment"], test_size=0.2, random_state=42)

# Build Naive Bayes model with TF-IDF
print("Training model...")
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Model evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Save model
joblib.dump(model, "sentiment_model.pkl")
print("Model saved as 'sentiment_model.pkl'. Training complete!")
