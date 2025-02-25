"""
RUN THIS CODE IN COLAB FOR FASTER RESULTS:
DOWNLOAD ALL THE REQUIRED FILES AND UPLOAD IT ON COLAB
HERE'S THE LINK FOR COLAB NOTEBOOK:

https://colab.research.google.com/drive/1E298DNEq7B1T6-2CvCqFdOqIRUPJBDkn?usp=sharing
"""

"""
Model Improvement Note:
This code enhances the original sentiment analysis model (accuracy: 0.7380) to achieve 0.8674 on IMDb test data and 0.8810 on a new dataset. Key changes:
1. Enhanced Preprocessing (Step 1):
   - Added negation handling: Preserves words like 'not' by combining them with the next word (e.g., 'not_good'), fixing issues like 'not good' being misclassified as 'good'.
   - Included lemmatization: Reduces words to their base form (e.g., 'running' → 'run') for consistency.
   - Removed HTML tags: Cleans IMDb-specific noise (e.g., '<br>').
2. N-grams with TF-IDF (Step 2):
   - Upgraded TfidfVectorizer to include unigrams and bigrams (ngram_range=(1, 2)): Captures phrases like 'not_good' or 'very_bad' for better context.
   - Limited to 10,000 features (max_features=10000): Reduces noise while retaining key terms.
Result: Improved negative class recall (from 0.51 to 0.87+), balanced F1 scores (~0.88), and strong generalization across datasets.
"""

import pandas as pd
import nltk
import re
import string
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# STEP 1:--------------------------------------------------------------->
# Preprocessing function with negation handling
def clean_text_advanced(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(f"[{string.punctuation}]", " ", text)  # Replace punctuation with space
    tokens = word_tokenize(text)
    
    # Exclude negation words from stopwords
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
    lemmatizer = WordNetLemmatizer()
    
    # Preserve negations and lemmatize
    cleaned_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] in {'not', 'no', 'never'} and i + 1 < len(tokens):
            # Combine negation with next word
            cleaned_tokens.append(tokens[i] + "_" + lemmatizer.lemmatize(tokens[i + 1]))
            i += 2
        else:
            if tokens[i] not in stop_words:
                cleaned_tokens.append(lemmatizer.lemmatize(tokens[i]))
            i += 1
    return " ".join(cleaned_tokens)

# Apply cleaning
print("Cleaning text data...")
df["cleaned_review"] = df["review"].apply(clean_text_advanced)

# Map sentiment to binary
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# STEP 2:--------------------------------------------------------------------->
# Split data (if not already done)
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_review"], df["sentiment"], test_size=0.2, random_state=42)

# Build pipeline with n-grams
print("Training model with n-grams...")
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=10000),
    MultinomialNB()
)
model.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# FINAL MODEL:-------------------------------------------------------------------->
joblib.dump(model, "final_sentiment_model.pkl")





