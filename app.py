from flask import Flask, request, render_template, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    if text:
        score = sia.polarity_scores(text)
        if score['compound'] >= 0.05:
            sentiment = "Positive"
        elif score['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return jsonify({'sentiment': sentiment, 'score': score})
    return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
