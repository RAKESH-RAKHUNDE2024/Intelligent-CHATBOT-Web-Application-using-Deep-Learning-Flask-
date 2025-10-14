# processor.py (snippet)
import os
import logging
import traceback
import json
import nltk
import numpy as np

# If using TF
from tensorflow.keras.models import load_model

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "chatbot_model.h5")

# minimal logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

# Ensure NLTK data is present (downloads silently if missing).
# List only the resources you actually need (e.g., punkt, wordnet)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
# add any other downloads you need: 'omw-1.4', 'stopwords', etc.

# Load model at import time and log success/failure
model = None
try:
    logging.info(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Failed to load model:\n" + traceback.format_exc())
    # Optionally re-raise to stop the app startup (so you notice immediately)
    # raise

def chatbot_response(question: str) -> str:
    try:
        if model is None:
            raise RuntimeError("Model not loaded")

        # --- your preprocessing/inference code here ---
        # Example placeholder:
        processed = question  # replace with actual preprocessing
        # prediction = model.predict(...)  # your real code
        # inference example:
        # result = decode_prediction(prediction)
        result = "This is a placeholder. Implement inference logic."

        return result

    except Exception as e:
        logging.error("Error in chatbot_response:\n" + traceback.format_exc())
        # Return a friendly message (and log the full stack trace)
        return "Sorry, something went wrong!"
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random

nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

model = tf.keras.models.load_model('chatbot_model.h5')
intents = json.loads(open('job_intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "You must ask the right questions"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)

    return res
