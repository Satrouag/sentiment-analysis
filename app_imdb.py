from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load("sentiment_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')  # Renders the frontend page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the request
        data = request.get_json()
        user_text = data.get("text", "")

        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        # Predict sentiment using the trained model
        prediction = model.predict([user_text])  # Model expects a list
        sentiment = "Positive" if prediction[0] == 1 else "Negative"

        probabilities = model.predict_proba([user_text])[0]  # Get probabilities
        # sentiment = "Positive" if np.argmax(probabilities) == 1 else "Negative"

        pos_prob = round(probabilities[1] * 100, 2)  # Probability of positive sentiment
        neg_prob = round(probabilities[0] * 100, 2)  # Probability of negative sentiment


        return jsonify({
            "sentiment": sentiment,
            "positive_probability": pos_prob,
            "negative_probability": neg_prob
        })  # Send response to frontend

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)