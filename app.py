from flask import Flask, request, render_template
import joblib
import nltk
import numpy as np

# Download sentence tokenizer if not available
#nltk.download('punkt')
def sent_tokenize(text):
    return [s.strip() for s in text.split('.') if s.strip()]
#from nltk.tokenize import sent_tokenize

# Load the trained model and vectorizer
model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentence_results = []
    overall_sentiment = None

    if request.method == 'POST':
        user_input = request.form['user_input']
        sentences = sent_tokenize(user_input)

        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}
        predictions = []

        for sentence in sentences:
            vec = vectorizer.transform([sentence])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            sentiment_scores[pred] += 1
            predictions.append({
                'sentence': sentence,
                'prediction': pred,
                'probability': f"{np.max(prob)*100:.2f}%"
            })

        # Decide overall sentiment by majority
        overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)

        sentence_results = predictions

    return render_template('index.html', predictions=sentence_results, overall=overall_sentiment)

if __name__ == '__main__':
    app.run(debug=True)

