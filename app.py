from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.tokenize import word_tokenize
import nltk
from lime.lime_text import LimeTextExplainer
import numpy as np

nltk.download('punkt')

app = Flask(__name__)

model = load_model('fake_news_model3.h5')

with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

MAX_SEQUENCE_LENGTH = 100

stop_words = set(["the", "and", "to", "of", "a", "in", "it", "is", "was", "for", "on", "that", "with", "as", "at", "by"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def text_to_sequence(tokens):
    return [word_index.get(word, 0) for word in tokens]

def predict_fake_news(news_text):
    tokens = clean_text(news_text)
    sequence = text_to_sequence(tokens)
    padded = pad_sequences([sequence], maxlen=MAX_SEQUENCE_LENGTH)
    prediction = model.predict(padded)[0][0]
    label = "Правдива" if prediction >= 0.5 else "Фейкова"
    prob = prediction * 100
    return f"Ймовірність правдивості: {prob:.2f}% – {label}"

def explain_prediction(news_text):
    explainer = LimeTextExplainer(class_names=["Фейкова", "Правдива"])

    def predictor(texts):
        sequences = [text_to_sequence(clean_text(t)) for t in texts]
        padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        preds = model.predict(padded)
        return np.hstack([1 - preds, preds])

    explanation = explainer.explain_instance(news_text, predictor, num_features=10)
    return explanation

def highlight_lime_words(text, explanation):
    tokens = word_tokenize(text)
    word_weights = dict(explanation.as_list())

    highlighted = ""
    for word in tokens:
        weight = word_weights.get(word.lower(), 0)
        if abs(weight) > 0.05:
            color = "rgba(255,0,0,{})".format(min(abs(weight) * 2, 1.0)) if weight < 0 else \
                    "rgba(0,255,0,{})".format(min(weight * 2, 1.0))
            highlighted += f'<span class="highlight" title="Вплив: {weight:.3f}" style="background-color: {color}; padding:1px; border-radius:3px;">{word}</span> '
        else:
            highlighted += word + " "
    return highlighted.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    news_text = ''
    explanation = None
    highlighted_text = ''
    explanation_details = []

    if request.method == 'POST':
        news_text = request.form['text']
        result = predict_fake_news(news_text)
        explanation = explain_prediction(news_text)
        highlighted_text = highlight_lime_words(news_text, explanation)
        explanation_details = explanation.as_list()

    return render_template('index.html',
                           result=result,
                           news_text=news_text,
                           highlighted_text=highlighted_text,
                           explanation_details=explanation_details)

if __name__ == '__main__':
    app.run(debug=True)
